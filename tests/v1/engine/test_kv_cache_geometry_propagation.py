# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore resolves the KV-cache geometry once and publishes it.

`EngineCore._initialize_kv_caches` is the only place the scheduling and
prefix-cache-matching units are derived. It stamps them onto every
`KVCacheConfig` before the workers are handed theirs, so the scheduler and
every worker -- including one created later -- match prefixes at the same
boundaries. These tests pin that wiring; the resolver itself and the worker
side are covered in `tests/v1/core/test_kv_cache_utils.py` and
`tests/v1/worker/test_gpu_worker.py`.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

# A hybrid pair already aligned to a common page, as the platform does for
# full attention + mamba. The group block size is far coarser than the unit a
# worker would pick on its own, which is what makes the disagreement visible.
MODEL = "Qwen/Qwen3.5-0.8B"
BLOCK_SIZE = 32
NUM_WORKERS = 2


def _hybrid_specs():
    return {
        "model.full_attn": FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
        ),
        "model.mamba": MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
    }


@pytest.fixture
def engine_core(monkeypatch: pytest.MonkeyPatch):
    """The minimal EngineCore surface `_initialize_kv_caches` touches.

    Everything that would need a device is stubbed; the KV cache grouping and
    geometry resolution run for real so the stamped values are the ones a real
    engine would compute.
    """
    monkeypatch.setattr(
        "vllm.v1.engine.core.register_all_kvcache_specs", lambda *a, **k: None
    )
    executor = MagicMock()
    executor.get_kv_cache_specs.return_value = [_hybrid_specs()] * NUM_WORKERS
    executor.get_supported_kv_cache_layouts.return_value = [["NHD"]] * NUM_WORKERS
    executor.determine_available_memory.return_value = [1 << 30] * NUM_WORKERS
    return SimpleNamespace(
        model_executor=executor,
        collective_rpc=MagicMock(),
        available_gpu_memory_for_kv_cache=0,
    )


def test_engine_core_publishes_one_geometry_to_every_config(engine_core):
    """The scheduler's config and each worker's config carry the same pair.

    Without this, deleting the stamping in `_initialize_kv_caches` leaves the
    resolver and the worker-side adoption individually correct while nothing
    connects them.
    """
    vllm_config = create_vllm_config(model_name=MODEL, block_size=BLOCK_SIZE)
    vllm_config.cache_config.enable_prefix_caching = True

    scheduler_config = EngineCore._initialize_kv_caches(engine_core, vllm_config)

    hash_block_size = scheduler_config.get_hash_block_size()
    scheduler_block_size = scheduler_config.get_scheduler_block_size()
    assert hash_block_size > 0 and scheduler_block_size > 0

    worker_configs = engine_core.model_executor.initialize_from_config.call_args[0][0]
    assert len(worker_configs) == NUM_WORKERS
    for worker_config in worker_configs:
        assert worker_config.get_hash_block_size() == hash_block_size
        assert worker_config.get_scheduler_block_size() == scheduler_block_size


def test_engine_core_records_the_resolved_unit_in_its_own_config(engine_core):
    """The engine-core process reads the same unit it published, and the
    user's `prefix_match_unit` request is left alone."""
    vllm_config = create_vllm_config(model_name=MODEL, block_size=BLOCK_SIZE)
    vllm_config.cache_config.enable_prefix_caching = True
    assert vllm_config.cache_config.prefix_match_unit is None

    scheduler_config = EngineCore._initialize_kv_caches(engine_core, vllm_config)

    cache_config = vllm_config.cache_config
    assert (
        cache_config.get_resolved_hash_block_size()
        == scheduler_config.get_hash_block_size()
    )
    assert cache_config.prefix_match_unit is None


def test_workers_are_stamped_before_they_are_initialized(engine_core):
    """A worker must never see an unstamped config: it adopts the unit inside
    `initialize_from_config`, so stamping has to happen first."""
    seen: list[int | None] = []
    engine_core.model_executor.initialize_from_config.side_effect = (
        lambda configs: seen.extend(config.hash_block_size for config in configs)
    )
    vllm_config = create_vllm_config(model_name=MODEL, block_size=BLOCK_SIZE)
    vllm_config.cache_config.enable_prefix_caching = True

    EngineCore._initialize_kv_caches(engine_core, vllm_config)

    assert seen and all(unit is not None for unit in seen)
