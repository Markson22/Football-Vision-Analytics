from football_vision_analytics.classification.team import create_batches


def test_create_batches_yields_expected_chunks() -> None:
    assert list(create_batches([1, 2, 3, 4, 5], batch_size=2)) == [
        [1, 2],
        [3, 4],
        [5],
    ]


def test_create_batches_uses_minimum_batch_size() -> None:
    assert list(create_batches([1, 2], batch_size=0)) == [[1], [2]]
