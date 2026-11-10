"""Flushing and closing a Pipeline, independently of a backend."""

from typing import Any, Dict, List

import pytest

from scipion_bridge.core.streaming import CompiledPipeline, Pipeline, StageStats


class RecordingPipeline(CompiledPipeline):
    """Compiled pipeline recording the calls of the Pipeline."""

    def __init__(self, fail_stats: bool = False) -> None:
        self.calls: List[str] = []
        self.fail_stats = fail_stats

    def send(self, source_name: str, value: Any) -> None:
        self.calls.append(f"send:{source_name}")

    def flush(self) -> None:
        self.calls.append("flush")

    def stats(self) -> Dict[str, StageStats]:
        self.calls.append("stats")
        if self.fail_stats:
            raise ConnectionError("stage lost")
        else:
            return {"0:map": StageStats(items_in=len(self.calls))}

    def close(self) -> None:
        self.calls.append("close")


def test_the_context_manager_flushes_and_closes():
    compiled = RecordingPipeline()

    with Pipeline(compiled) as pipeline:
        pipeline.send(items=1)

    assert compiled.calls == ["send:items", "flush", "stats", "close"]


def test_stats_remain_available_after_closing():
    compiled = RecordingPipeline()

    with Pipeline(compiled) as pipeline:
        pipeline.send(items=1)

    assert pipeline.stats() == {"0:map": StageStats(items_in=3)}
    assert compiled.calls.count("stats") == 1


def test_an_error_releases_the_pipeline_without_flushing():
    compiled = RecordingPipeline()
    pipeline = Pipeline(compiled)

    with pytest.raises(ValueError, match="in the block"):
        with pipeline:
            raise ValueError("in the block")

    assert compiled.calls == ["close"]
    with pytest.raises(RuntimeError, match="released after a failure"):
        pipeline.stats()


def test_closing_twice_closes_once():
    compiled = RecordingPipeline()
    pipeline = Pipeline(compiled)

    pipeline.close()
    pipeline.close()

    assert compiled.calls == ["stats", "close"]


def test_closing_terminates_the_backend_when_its_stats_fail():
    compiled = RecordingPipeline(fail_stats=True)

    with pytest.raises(ConnectionError):
        Pipeline(compiled).close()

    assert compiled.calls == ["stats", "close"]


def test_a_closed_pipeline_rejects_items_and_flushes():
    pipeline = Pipeline(RecordingPipeline())
    pipeline.close()

    with pytest.raises(RuntimeError, match="closed"):
        pipeline.send(items=1)
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.flush()
