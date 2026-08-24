from pathlib import Path

import pytest

from football_vision_analytics.config import PipelineConfig


def test_pipeline_config_creates_output_directory(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    target = tmp_path / "outputs" / "result.mp4"
    source.write_bytes(b"video")

    config = PipelineConfig(source_video_path=source, target_video_path=target)

    config.validate(require_models=False)

    assert target.parent.exists()


def test_pipeline_config_rejects_missing_source(tmp_path: Path) -> None:
    config = PipelineConfig(
        source_video_path=tmp_path / "missing.mp4",
        target_video_path=tmp_path / "result.mp4",
    )

    with pytest.raises(FileNotFoundError):
        config.validate(require_models=False)
