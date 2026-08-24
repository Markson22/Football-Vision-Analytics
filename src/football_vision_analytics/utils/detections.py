"""Utilitarios puros para recortes e pos-processamento de deteccoes."""

from __future__ import annotations

import numpy as np
import supervision as sv


def get_crops(frame: np.ndarray, detections: sv.Detections) -> list[np.ndarray]:
    """Extrai recortes de imagem a partir das caixas detectadas.

    Args:
        frame: Frame original em formato OpenCV.
        detections: Deteccoes com coordenadas `xyxy`.

    Returns:
        Lista de crops correspondentes as deteccoes recebidas.
    """

    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections,
) -> np.ndarray:
    """Associa goleiros ao time mais proximo no frame.

    A classificacao visual e treinada sobre crops de jogadores de linha. Para os
    goleiros, este projeto preserva a heuristica existente: calcula o centroide de
    cada time e atribui cada goleiro ao centroide mais proximo.

    Args:
        players: Deteccoes dos jogadores de linha.
        players_team_id: IDs de time preditos para `players`.
        goalkeepers: Deteccoes dos goleiros.

    Returns:
        Array com o ID de time de cada goleiro. Retorna vazio se nao houver goleiros.

        Se houver jogadores de apenas um time no frame, os goleiros sao associados a
        esse time como fallback.
    """

    if len(goalkeepers) == 0:
        return np.array([], dtype=int)

    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    if len(players_xy) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    if not np.any(players_team_id == 0):
        return np.ones(len(goalkeepers), dtype=int)
    if not np.any(players_team_id == 1):
        return np.zeros(len(goalkeepers), dtype=int)

    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)

    goalkeepers_team_id: list[int] = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id, dtype=int)
