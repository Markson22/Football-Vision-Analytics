import numpy as np
import supervision as sv

from football_vision_analytics.utils import resolve_goalkeepers_team_id


def test_resolve_goalkeepers_team_id_uses_nearest_team_centroid() -> None:
    players = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 10, 10],
                [10, 0, 20, 10],
                [100, 0, 110, 10],
                [110, 0, 120, 10],
            ],
            dtype=float,
        )
    )
    players_team_id = np.array([0, 0, 1, 1])
    goalkeepers = sv.Detections(
        xyxy=np.array(
            [
                [2, 0, 12, 10],
                [105, 0, 115, 10],
            ],
            dtype=float,
        )
    )

    np.testing.assert_array_equal(
        resolve_goalkeepers_team_id(players, players_team_id, goalkeepers),
        np.array([0, 1]),
    )
