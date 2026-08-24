import numpy as np

from football_vision_analytics.pitch import ViewTransformer


def test_view_transformer_identity_points() -> None:
    source = np.array(
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        dtype=np.float32,
    )
    transformer = ViewTransformer(source=source, target=source)

    points = np.array([[2, 3], [8, 9]], dtype=np.float32)

    np.testing.assert_allclose(transformer.transform_points(points), points, atol=1e-4)
