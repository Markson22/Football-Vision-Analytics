"""Estimativa de velocidade de jogadores a partir de posicoes no campo."""

from __future__ import annotations

from collections import deque

import numpy as np
import supervision as sv

from football_vision_analytics.pitch.transform import ViewTransformer


class SpeedEstimator:
    """Estima velocidade dos jogadores usando historico de posicoes transformadas.

    As coordenadas do frame sao transformadas para o plano do campo em centimetros.
    A distancia percorrida no historico recente e convertida para km/h e suavizada
    exponencialmente para reduzir oscilacoes visuais no label.

    Args:
        fps: Taxa de frames do video.
        buffer_size: Quantidade maxima de posicoes mantidas por jogador.
    """

    def __init__(self, fps: float, buffer_size: int = 30) -> None:
        self.fps = fps
        self.buffer_size = buffer_size
        self.positions: dict[int, deque[np.ndarray]] = {}
        self._smoothed_speeds: dict[int, float] = {}

    def update(self, detections: sv.Detections, transformer: ViewTransformer) -> None:
        """Atualiza o historico de posicoes dos jogadores rastreados.

        Args:
            detections: Deteccoes de jogadores com `tracker_id` preenchido.
            transformer: Transformador de perspectiva para coordenadas do campo.
        """

        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            return

        anchors = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        for tracker_id, xy in zip(tracker_ids, anchors):
            tracker_key = int(tracker_id)
            if tracker_key not in self.positions:
                self.positions[tracker_key] = deque(maxlen=self.buffer_size)

            transformed_xy = transformer.transform_points(points=np.array([xy]))[0]
            self.positions[tracker_key].append(transformed_xy)

    def get_speeds(self, detections: sv.Detections) -> list[str]:
        """Calcula labels de velocidade para as deteccoes recebidas.

        Args:
            detections: Deteccoes de jogadores na mesma ordem desejada para os labels.

        Returns:
            Lista de strings como `12.4km/h` ou `N/A` quando nao ha historico suficiente.
        """

        tracker_ids = detections.tracker_id
        if tracker_ids is None:
            return ["N/A"] * len(detections)

        speeds: list[str] = []
        for tracker_id in tracker_ids:
            tracker_key = int(tracker_id)
            if tracker_key not in self.positions or len(self.positions[tracker_key]) <= 1:
                speeds.append("N/A")
                continue

            history = self.positions[tracker_key]
            if len(history) > 3:
                pos_oldest = history[0]
                pos_recent = history[-1]
                frames_elapsed = len(history) - 1
            else:
                pos_oldest = history[-2]
                pos_recent = history[-1]
                frames_elapsed = 1

            distance_cm = np.linalg.norm(pos_recent - pos_oldest)
            distance_meters = distance_cm / 100
            time_seconds = frames_elapsed / self.fps
            speed_kmh = (distance_meters / time_seconds) * 3.6

            previous_speed = self._smoothed_speeds.get(tracker_key)
            if previous_speed is not None:
                speed_kmh = 0.7 * previous_speed + 0.3 * speed_kmh
            self._smoothed_speeds[tracker_key] = speed_kmh

            speeds.append(f"{speed_kmh:.1f}km/h")

        return speeds
