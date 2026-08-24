"""Modos disponiveis para o pipeline de futebol."""

from enum import Enum


class Mode(str, Enum):
    """Modos de processamento suportados pelo exemplo de futebol."""

    PITCH_DETECTION = "PITCH_DETECTION"
    PLAYER_DETECTION = "PLAYER_DETECTION"
    BALL_DETECTION = "BALL_DETECTION"
    PLAYER_TRACKING = "PLAYER_TRACKING"
    TEAM_CLASSIFICATION = "TEAM_CLASSIFICATION"
    RADAR = "RADAR"
    PLAYER_SPEED_ESTIMATION = "PLAYER_SPEED_ESTIMATION"
