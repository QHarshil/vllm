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
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

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


# The Qwen3.6-27B hybrid pair, as used in
# `tests/v1/core/test_mamba_align_chunk_split.py`. A fresh resolution of these
# groups returns (1600, 16), so stamping anything else makes a second
# resolution observable.
QWEN36_ATTN_BLOCK = 16
QWEN36_MAMBA_BLOCK = 1600
STAMPED_SCHEDULER_BLOCK = 3200
STAMPED_HASH_BLOCK = 8


class _SchedulerReached(Exception):
    """Raised once the scheduler is constructed, to stop `__init__` there."""


def test_engine_core_reads_back_the_published_geometry(monkeypatch):
    """`__init__` uses the geometry `_initialize_kv_caches` published rather
    than resolving it a second time.

    A second resolution returns the same pair for the same config, so the
    duplicate call is invisible in normal operation and the two sites can drift
    apart unnoticed. Stamping a pair the resolver would not produce for these
    groups makes the difference observable: reading back hands the scheduler
    the stamped pair, resolving again hands it (1600, 16).
    """
    vllm_config = create_vllm_config(model_name=MODEL, block_size=BLOCK_SIZE)
    vllm_config.cache_config.enable_prefix_caching = True

    published = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=QWEN36_ATTN_BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=QWEN36_MAMBA_BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
        scheduler_block_size=STAMPED_SCHEDULER_BLOCK,
        hash_block_size=STAMPED_HASH_BLOCK,
    )
    assert resolve_kv_cache_block_sizes(published, vllm_config) != (
        STAMPED_SCHEDULER_BLOCK,
        STAMPED_HASH_BLOCK,
    )

    captured: dict[str, int] = {}

    class _CapturingScheduler:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise _SchedulerReached

    monkeypatch.setattr("vllm.plugins.load_general_plugins", lambda: None)
    monkeypatch.setattr(
        EngineCore, "_initialize_kv_caches", lambda self, config: published
    )
    monkeypatch.setattr("vllm.v1.engine.core.StructuredOutputManager", MagicMock())
    monkeypatch.setattr(
        vllm_config.scheduler_config,
        "get_scheduler_cls",
        lambda: _CapturingScheduler,
    )

    with pytest.raises(_SchedulerReached):
        EngineCore(vllm_config, MagicMock(), log_stats=False)

    assert captured["block_size"] == STAMPED_SCHEDULER_BLOCK
    assert captured["hash_block_size"] == STAMPED_HASH_BLOCK
