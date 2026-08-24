"""Pipeline principal de visao computacional para videos de futebol."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from football_vision_analytics.analytics import SpeedEstimator
from football_vision_analytics.classification import TeamClassifier
from football_vision_analytics.config import PipelineConfig
from football_vision_analytics.pipeline.modes import Mode
from football_vision_analytics.pitch import SoccerPitchConfiguration, ViewTransformer
from football_vision_analytics.tracking import BallAnnotator, BallTracker
from football_vision_analytics.utils import get_crops, resolve_goalkeepers_team_id
from football_vision_analytics.visualization import draw_pitch, draw_points_on_pitch

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]


class FootballAnalysisPipeline:
    """Orquestra os modos de analise de futebol implementados no projeto.

    Args:
        config: Configuracoes de entrada, saida, modelos e parametros do pipeline.
        pitch_config: Configuracao geometrica do campo de futebol.
    """

    def __init__(
        self,
        config: PipelineConfig,
        pitch_config: SoccerPitchConfiguration | None = None,
    ) -> None:
        self.config = config
        self.pitch_config = pitch_config or SoccerPitchConfiguration()

        color_palette = sv.ColorPalette.from_hex(COLORS)
        self.vertex_label_annotator = sv.VertexLabelAnnotator(
            color=[sv.Color.from_hex(color) for color in self.pitch_config.colors],
            text_color=sv.Color.from_hex("#FFFFFF"),
            border_radius=5,
            text_thickness=1,
            text_scale=0.5,
            text_padding=5,
        )
        self.box_annotator = sv.BoxAnnotator(color=color_palette, thickness=2)
        self.box_label_annotator = sv.LabelAnnotator(
            color=color_palette,
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_padding=5,
            text_thickness=1,
        )
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=color_palette, thickness=2
        )
        self.ellipse_label_annotator = sv.LabelAnnotator(
            color=color_palette,
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_padding=5,
            text_thickness=1,
            text_position=sv.Position.BOTTOM_CENTER,
        )

    def run(self, mode: Mode) -> None:
        """Executa um modo de analise e grava o video processado.

        Args:
            mode: Modo de processamento escolhido.

        Raises:
            NotImplementedError: Se o modo informado nao estiver mapeado.
            FileNotFoundError: Se video ou modelos obrigatorios nao existirem.
        """

        self.config.validate(require_models=True)
        frame_generator = self._build_frame_generator(mode)
        video_info = sv.VideoInfo.from_video_path(str(self.config.source_video_path))

        with sv.VideoSink(str(self.config.target_video_path), video_info) as sink:
            for frame in frame_generator:
                sink.write_frame(frame)
                if not self.config.display_preview:
                    continue

                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if self.config.display_preview:
            cv2.destroyAllWindows()

    def _build_frame_generator(self, mode: Mode) -> Iterator[np.ndarray]:
        """Seleciona o gerador de frames anotados para o modo solicitado."""

        if mode == Mode.PITCH_DETECTION:
            return self.run_pitch_detection()
        if mode == Mode.PLAYER_DETECTION:
            return self.run_player_detection()
        if mode == Mode.BALL_DETECTION:
            return self.run_ball_detection()
        if mode == Mode.PLAYER_TRACKING:
            return self.run_player_tracking()
        if mode == Mode.TEAM_CLASSIFICATION:
            return self.run_team_classification()
        if mode == Mode.RADAR:
            return self.run_radar()
        if mode == Mode.PLAYER_SPEED_ESTIMATION:
            return self.run_player_speed_estimation()
        raise NotImplementedError(f"Modo {mode} nao implementado.")

    def run_pitch_detection(self) -> Iterator[np.ndarray]:
        """Detecta pontos-chave do campo e retorna frames anotados."""

        pitch_detection_model = YOLO(
            self.config.model_paths.pitch_detection
        ).to(device=self.config.device)
        frame_generator = sv.get_video_frames_generator(
            source_path=str(self.config.source_video_path)
        )

        for frame in frame_generator:
            result = pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            annotated_frame = frame.copy()
            yield self.vertex_label_annotator.annotate(
                annotated_frame, keypoints, self.pitch_config.labels
            )

    def run_player_detection(self) -> Iterator[np.ndarray]:
        """Detecta jogadores, goleiros, arbitros e bola com o modelo de jogadores."""

        player_detection_model = YOLO(
            self.config.model_paths.player_detection
        ).to(device=self.config.device)
        frame_generator = sv.get_video_frames_generator(
            source_path=str(self.config.source_video_path)
        )

        for frame in frame_generator:
            result = player_detection_model(
                frame,
                imgsz=self.config.player_detection_image_size,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            annotated_frame = frame.copy()
            annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.box_label_annotator.annotate(
                annotated_frame, detections
            )
            yield annotated_frame

    def run_ball_detection(self) -> Iterator[np.ndarray]:
        """Detecta e rastreia a bola usando slicing para objetos pequenos."""

        ball_detection_model = YOLO(
            self.config.model_paths.ball_detection
        ).to(device=self.config.device)
        frame_generator = sv.get_video_frames_generator(
            source_path=str(self.config.source_video_path)
        )
        ball_tracker = BallTracker(buffer_size=self.config.ball_tracker_buffer_size)
        ball_annotator = BallAnnotator(
            radius=6, buffer_size=self.config.ball_annotator_buffer_size
        )

        def callback(image_slice: np.ndarray) -> sv.Detections:
            result = ball_detection_model(
                image_slice,
                imgsz=self.config.ball_detection_image_size,
                verbose=False,
            )[0]
            return sv.Detections.from_ultralytics(result)

        slicer = sv.InferenceSlicer(
            callback=callback,
            overlap_filter_strategy=sv.OverlapFilter.NONE,
            slice_wh=(640, 640),
        )

        for frame in frame_generator:
            detections = slicer(frame).with_nms(
                threshold=self.config.ball_nms_threshold
            )
            detections = ball_tracker.update(detections)
            annotated_frame = frame.copy()
            yield ball_annotator.annotate(annotated_frame, detections)

    def run_player_tracking(self) -> Iterator[np.ndarray]:
        """Rastreia deteccoes entre frames usando ByteTrack."""

        player_detection_model = YOLO(
            self.config.model_paths.player_detection
        ).to(device=self.config.device)
        frame_generator = sv.get_video_frames_generator(
            source_path=str(self.config.source_video_path)
        )
        tracker = sv.ByteTrack(
            minimum_consecutive_frames=(
                self.config.byte_track_minimum_consecutive_frames
            )
        )

        for frame in frame_generator:
            result = player_detection_model(
                frame,
                imgsz=self.config.player_detection_image_size,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)
            labels = [str(tracker_id) for tracker_id in detections.tracker_id]

            annotated_frame = frame.copy()
            annotated_frame = self.ellipse_annotator.annotate(
                annotated_frame, detections
            )
            annotated_frame = self.ellipse_label_annotator.annotate(
                annotated_frame, detections, labels=labels
            )
            yield annotated_frame

    def run_team_classification(self) -> Iterator[np.ndarray]:
        """Classifica jogadores por time e anota as deteccoes por cor."""

        player_detection_model = YOLO(
            self.config.model_paths.player_detection
        ).to(device=self.config.device)
        team_classifier = self._fit_team_classifier(player_detection_model)
        frame_generator = sv.get_video_frames_generator(
            source_path=str(self.config.source_video_path)
        )
        tracker = sv.ByteTrack(
            minimum_consecutive_frames=(
                self.config.byte_track_minimum_consecutive_frames
            )
        )

        for frame in frame_generator:
            detections = self._detect_and_track_players(
                frame, player_detection_model, tracker
            )
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(crops)
            annotated_frame = self._annotate_team_detections(
                frame, detections, players, players_team_id
            )
            yield annotated_frame

    def run_radar(self) -> Iterator[np.ndarray]:
        """Gera visualizacao radar com jogadores projetados no campo 2D."""

        player_detection_model = YOLO(
            self.config.model_paths.player_detection
        ).to(device=self.config.device)
        pitch_detection_model = YOLO(
            self.config.model_paths.pitch_detection
        ).to(device=self.config.device)
        team_classifier = self._fit_team_classifier(player_detection_model)
        frame_generator = sv.get_video_frames_generator(
            source_path=str(self.config.source_video_path)
        )
        tracker = sv.ByteTrack(
            minimum_consecutive_frames=(
                self.config.byte_track_minimum_consecutive_frames
            )
        )

        for frame in frame_generator:
            pitch_result = pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            detections = self._detect_and_track_players(
                frame, player_detection_model, tracker
            )
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(crops)
            annotated_frame, merged_detections, color_lookup = (
                self._annotate_team_detections(
                    frame,
                    detections,
                    players,
                    players_team_id,
                    return_metadata=True,
                )
            )

            h, w, _ = frame.shape
            radar = self.render_radar(merged_detections, keypoints, color_lookup)
            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(
                x=w // 2 - radar_w // 2,
                y=h - radar_h,
                width=radar_w,
                height=radar_h,
            )
            yield sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)

    def run_player_speed_estimation(self) -> Iterator[np.ndarray]:
        """Estima velocidades em km/h para jogadores rastreados."""

        player_detection_model = YOLO(
            self.config.model_paths.player_detection
        ).to(device=self.config.device)
        pitch_detection_model = YOLO(
            self.config.model_paths.pitch_detection
        ).to(device=self.config.device)

        video_info = sv.VideoInfo.from_video_path(str(self.config.source_video_path))
        frame_generator = sv.get_video_frames_generator(
            source_path=str(self.config.source_video_path)
        )
        tracker = sv.ByteTrack(
            minimum_consecutive_frames=(
                self.config.byte_track_minimum_consecutive_frames
            ),
            frame_rate=video_info.fps,
        )
        speed_estimator = SpeedEstimator(fps=video_info.fps)

        for frame in frame_generator:
            pitch_result = pitch_detection_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            detections = self._detect_and_track_players(
                frame, player_detection_model, tracker
            )
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            if len(players) == 0:
                yield frame
                continue

            transformer = self._build_view_transformer(keypoints)
            speed_estimator.update(players, transformer)
            speeds = speed_estimator.get_speeds(players)
            labels = [
                f"#{tracker_id} {speed}"
                for tracker_id, speed in zip(players.tracker_id, speeds)
            ]

            annotated_frame = frame.copy()
            annotated_frame = self.ellipse_annotator.annotate(
                annotated_frame, players
            )
            yield self.ellipse_label_annotator.annotate(
                annotated_frame, players, labels=labels
            )

    def render_radar(
        self,
        detections: sv.Detections,
        keypoints: sv.KeyPoints,
        color_lookup: np.ndarray,
    ) -> np.ndarray:
        """Renderiza as posicoes detectadas em um mini-campo tatico."""

        transformer = self._build_view_transformer(keypoints)
        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(points=xy)

        radar = draw_pitch(config=self.pitch_config)
        for team_id, color in enumerate(COLORS):
            radar = draw_points_on_pitch(
                config=self.pitch_config,
                xy=transformed_xy[color_lookup == team_id],
                face_color=sv.Color.from_hex(color),
                radius=20,
                pitch=radar,
            )
        return radar

    def _fit_team_classifier(self, player_detection_model: YOLO) -> TeamClassifier:
        """Coleta crops amostrados e ajusta o classificador visual de times."""

        frame_generator = sv.get_video_frames_generator(
            source_path=str(self.config.source_video_path),
            stride=self.config.team_classification_stride,
        )
        crops: list[np.ndarray] = []
        for frame in tqdm(frame_generator, desc="collecting crops"):
            result = player_detection_model(
                frame,
                imgsz=self.config.player_detection_image_size,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(result)
            crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

        team_classifier = TeamClassifier(device=self.config.device)
        team_classifier.fit(crops)
        return team_classifier

    def _detect_and_track_players(
        self,
        frame: np.ndarray,
        player_detection_model: YOLO,
        tracker: sv.ByteTrack,
    ) -> sv.Detections:
        """Executa deteccao de jogadores e atualiza o tracker."""

        result = player_detection_model(
            frame,
            imgsz=self.config.player_detection_image_size,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        return tracker.update_with_detections(detections)

    def _annotate_team_detections(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        players: sv.Detections,
        players_team_id: np.ndarray,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, sv.Detections, np.ndarray]:
        """Anota jogadores, goleiros e arbitros com cores por papel/time."""

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        merged_detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in merged_detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = self.ellipse_annotator.annotate(
            annotated_frame, merged_detections, custom_color_lookup=color_lookup
        )
        annotated_frame = self.ellipse_label_annotator.annotate(
            annotated_frame,
            merged_detections,
            labels,
            custom_color_lookup=color_lookup,
        )

        if return_metadata:
            return annotated_frame, merged_detections, color_lookup
        return annotated_frame

    def _build_view_transformer(self, keypoints: sv.KeyPoints) -> ViewTransformer:
        """Cria homografia usando apenas pontos-chave validos do campo."""

        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        return ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(self.pitch_config.vertices)[mask].astype(np.float32),
        )
