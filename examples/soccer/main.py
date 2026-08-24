import argparse  # Processa argumentos da linha de comando (paths dos vídeos, modo de operação)
from collections import deque  # Define o deque buffer
from enum import (  # Importa o enum
    Enum,
)  # Define o enum Mode com os 6 modos de operação (ex: PLAYER_DETECTION, RADAR)
from typing import (
    Iterator,
    List,
)  # . Eles melhoram a documentação, facilitam a detecção de erros e tornam o código mais legível. ( Ex O código fica auto-explicativo)

import os  # o Python é usado para interagir com o sistema operacional, incluindo manipulação de caminhos de arquivos. (evita erros de caminho)
import cv2  # OpenCV é uma biblioteca de visão computacional usada para processamento de imagens e vídeos.
import numpy as np  # NumPy é uma biblioteca de manipulação de matrizes e vetores numéricos em Python.
import supervision as sv  # Supervision é uma biblioteca para anotação e manipulação de dados visuais (imagens, vídeos).
from tqdm import tqdm  # Tqdm é uma biblioteca para criar barras de progresso em loops.
from ultralytics import (
    YOLO,
)  # Ultralytics é uma biblioteca que implementa modelos YOLO para detecção de objetos.

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
)  # Funções para desenhar o campo de futebol e pontos no campo.
from sports.common.ball import (
    BallTracker,
    BallAnnotator,
)  # Funções para detecção e anotação de bola.
from sports.common.team import (
    TeamClassifier,
)  # Funções para detecção e anotação de equipe.
from sports.common.view import ViewTransformer  # Funções para transformação de visão.
from sports.configs.soccer import (
    SoccerPitchConfiguration,
)  # Configuração do campo de futebol (dimensões, layout).

PARENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)  # Diretório pai do arquivo atual
PLAYER_DETECTION_MODEL_PATH = (
    os.path.join(  #   Caminho do modelo de detecção de jogadores
        PARENT_DIR, "data/football-player-detection.pt"
    )
)

PITCH_DETECTION_MODEL_PATH = os.path.join(
    PARENT_DIR, "data/football-pitch-detection.pt"
)
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, "data/football-ball-detection.pt")

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60  # Intervalo de quadros para amostragem durante a classificação da equipe
CONFIG = SoccerPitchConfiguration()  # Configuração do campo de futebol padrão

COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]
VERTEX_LABEL_ANNOTATOR = (
    sv.VertexLabelAnnotator(  # Anotador de rótulo de vértice para pontos-chave do campo
        color=[sv.Color.from_hex(color) for color in CONFIG.colors],
        text_color=sv.Color.from_hex("#FFFFFF"),
        border_radius=5,
        text_thickness=1,
        text_scale=0.5,
        text_padding=5,
    )
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(  # Anotador de borda para linhas do campo
    color=sv.Color.from_hex("#FF1493"),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = (
    sv.TriangleAnnotator(  # Anotador de triângulo para pontos-chave do campo
        color=sv.Color.from_hex("#FF1493"),
        base=20,
        height=15,
    )
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2
)  # Anotador de caixa para detecções de jogadores
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2
)
BOX_LABEL_ANNOTATOR = (
    sv.LabelAnnotator(  # Anotador de rótulo de caixa para detecções de jogadores
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
    )
)
ELLIPSE_LABEL_ANNOTATOR = (
    sv.LabelAnnotator(  # Anotador de rótulo de elipse para detecções de jogadores
        color=sv.ColorPalette.from_hex(COLORS),
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
        text_position=sv.Position.BOTTOM_CENTER,
    )
)


class Mode(Enum):
    """
    Classe Enum que representa diferentes modos de operação para análise de vídeo do Soccer AI.
    """

    PITCH_DETECTION = (
        "PITCH_DETECTION  "  # Identifica pontos-chave do campo (linhas, áreas)
    )
    PLAYER_DETECTION = "PLAYER_DETECTION"  # Detecta pessoas em campo
    BALL_DETECTION = "BALL_DETECTION"  # Detecta bola em campo
    PLAYER_TRACKING = (
        "PLAYER_TRACKING"  # Rastreia jogadores com IDs únicos entre frames
    )
    TEAM_CLASSIFICATION = "TEAM_CLASSIFICATION"  # Classifica jogadores por time usando suas cores de uniforme
    RADAR = "RADAR"  # Desenha um radar com base nas detecções de jogadores
    PLAYER_SPEED_ESTIMATION = (
        "PLAYER_SPEED_ESTIMATION"  # Estima a velocidade dos jogadores
    )


class SpeedEstimator:  # Estima a velocidade dos jogadores com base em suas posições ao longo do tempo
    def __init__(
        self, fps: int, buffer_size: int = 30
    ):  # Inicializa o estimador de velocidade com a taxa de quadros e o tamanho do buffer
        self.fps = fps
        self.positions = {}
        self.buffer_size = buffer_size
        self._smoothed_speeds = {}

    def update(
        self, detections: sv.Detections, transformer: ViewTransformer
    ):  # Atualiza as posições dos jogadores com base nas detecções atuais e na transformação de visão
        for tracker_id, xy in zip(
            detections.tracker_id,
            detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER),
        ):
            if tracker_id not in self.positions:
                self.positions[tracker_id] = deque(maxlen=self.buffer_size)

            transformed_xy = transformer.transform_points(points=np.array([xy]))[0]
            self.positions[tracker_id].append(transformed_xy)

    def get_speeds(
        self, detections: sv.Detections
    ) -> List[
        str
    ]:  # Calcula e retorna as velocidades dos jogadores como strings formatadas
        speeds = []
        for (
            tracker_id
        ) in detections.tracker_id:  # Para cada ID de rastreamento na detecção
            if tracker_id in self.positions and len(self.positions[tracker_id]) > 1:
                history = self.positions[tracker_id]
                if len(history) > 3:
                    pos_oldest = history[0]
                    pos_recent = history[-1]
                    frames_elapsed = len(history) - 1
                else:
                    pos_oldest = history[-2]
                    pos_recent = history[-1]
                    frames_elapsed = 1

                distance_cm = np.linalg.norm(pos_recent - pos_oldest)
                distance_meters = distance_cm / 100  # Converte de cm para metros
                time_seconds = frames_elapsed / self.fps

                speed_mps = (
                    distance_meters / time_seconds
                )  # velocidade em metros por segundo
                speed_kmh = speed_mps * 3.6  # velocidade em quilometros por hora

                previous_speed = self._smoothed_speeds.get(tracker_id)
                if previous_speed is not None:
                    # suavização exponencial para reduzir oscilações rápidas
                    speed_kmh = 0.7 * previous_speed + 0.3 * speed_kmh
                self._smoothed_speeds[tracker_id] = speed_kmh

                speeds.append(f"{speed_kmh:.1f}km/h")
            else:
                speeds.append("N/A")
        return speeds


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extraia recortes do quadro com base nas caixas delimitadoras detectadas.

    Argumentos:
        frame (np.ndarray): O quadro do qual serão extraídas as colheitas.
        detecções (sv.Detections): Objetos detectados com caixas delimitadoras.

    Retorna:
        List[np.ndarray]: Lista de imagens recortadas.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections, players_team_id: np.array, goalkeepers: sv.Detections
) -> np.ndarray:
    """
        Resolva os IDs dos times dos goleiros detectados com base na proximidade do time
        centróides.

        Argumentos:
            players (sv.Detections): Detecções de todos os jogadores.
            players_team_id (np.array): Matriz contendo IDs de times de jogadores detectados.
            goleiros (sv.Detections): Detecções de goleiros.

        Retorna:
            np.ndarray: Array contendo os IDs dos times dos goleiros detectados.

        Esta função calcula os centróides das duas equipes com base nas posições dos
    os jogadores. Em seguida, atribui cada goleiro ao centróide da equipe mais próxima,
        calculando a distância entre cada goleiro e os centróides das duas equipes.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(  # Desenha um radar com base nas detecções de jogadores
    detections: sv.Detections, keypoints: sv.KeyPoints, color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)  # Desenha o campo de futebol como fundo do radar
    radar = draw_points_on_pitch(
        config=CONFIG,
        xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]),
        radius=20,
        pitch=radar,
    )
    radar = draw_points_on_pitch(  # Desenha pontos no campo com base nas detecções de jogadores
        config=CONFIG,
        xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]),
        radius=20,
        pitch=radar,
    )
    radar = draw_points_on_pitch(  # Desenha pontos no campo com base nas detecções de jogadores
        config=CONFIG,
        xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]),
        radius=20,
        pitch=radar,
    )
    radar = draw_points_on_pitch(  # Desenha pontos no campo com base nas detecções de jogadores
        config=CONFIG,
        xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]),
        radius=20,
        pitch=radar,
    )
    return radar


def run_pitch_detection(
    source_video_path: str, device: str
) -> Iterator[np.ndarray]:  # Identifica pontos-chave do campo
    """

    IMPORTANTE Identifica pontos-chave do campo (linhas, áreas) em um vídeo e produz quadros anotados.

    Execute a detecção de pitch em um vídeo e produza quadros anotados.

     Argumentos:
         source_video_path (str): Caminho para o vídeo de origem.
         device (str): Dispositivo para executar o modelo (por exemplo, 'cpu', 'cuda').

     Rendimentos:
         Iterador[np.ndarray]: Iterador sobre quadros anotados.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(
        device=device
    )  # Carrega o modelo de detecção de campo
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(  # Desenha o campo de futebol como fundo do radar
            annotated_frame, keypoints, CONFIG.labels
        )
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Execute a detecção do player em um vídeo e produza quadros anotados.

      Argumentos:
          source_video_path (str): Caminho para o vídeo de origem.
          device (str): Dispositivo para executar o modelo (por exemplo, 'cpu', 'cuda').

      Rendimentos:
          Iterador[np.ndarray]: Iterador sobre quadros anotados.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Execute a detecção de bola em um vídeo e produza quadros anotados.

        Argumentos:
            source_video_path (str): Caminho para o vídeo de origem.
            device (str): Dispositivo para executar o modelo (por exemplo, 'cpu', 'cuda').

        Rendimentos:
            Iterador[np.ndarray]: Iterador sobre quadros anotados.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Execute o rastreamento de jogadores em um vídeo e produza quadros anotados com jogadores rastreados.

    Argumentos:
        source_video_path (str): Caminho para o vídeo de origem.
        device (str): Dispositivo para executar o modelo (por exemplo, 'cpu', 'cuda').

    Rendimentos:
        Iterador[np.ndarray]: Iterador sobre quadros anotados.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels
        )
        yield annotated_frame


def run_team_classification(
    source_video_path: str, device: str
) -> Iterator[np.ndarray]:
    """
    Execute a classificação da equipe em um vídeo e produza quadros anotados com as cores da equipe.

       Argumentos:
           source_video_path (str): Caminho para o vídeo de origem.
           device (str): Dispositivo para executar o modelo (por exemplo, 'cpu', 'cuda').

       Rendimentos:
           Iterador[np.ndarray]: Iterador sobre quadros anotados.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE
    )

    crops = []
    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )
        yield annotated_frame


def run_radar(
    source_video_path: str, device: str
) -> Iterator[
    np.ndarray
]:  # Coleta recortes de jogadores para treinamento do classificador de equipes
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE
    )

    crops = []
    for frame in tqdm(
        frame_generator, desc="collecting crops"
    ):  #  Coleta recortes de jogadores para treinamento do classificador de equipes
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path
    )  #  Gera quadros do vídeo de origem
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[
            detections.class_id == PLAYER_CLASS_ID
        ]  # Detecções de jogadores
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[
            detections.class_id == GOALKEEPER_CLASS_ID
        ]  # Detecções de goleiros
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge(
            [players, goalkeepers, referees]
        )  #  Mescla todas as detecções (jogadores, goleiros, árbitros) em um único objeto Detections
        color_lookup = np.array(
            players_team_id.tolist()
            + goalkeepers_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(  # Desenha elipses com base nas detecções de jogadores
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(  # Desenha o rosto com base nas detecções de jogadores
            annotated_frame, detections, labels, custom_color_lookup=color_lookup
        )

        h, w, _ = frame.shape
        radar = render_radar(
            detections, keypoints, color_lookup
        )  # Desenha o radar com base nas detecções de jogadores
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2, y=h - radar_h, width=radar_w, height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def run_player_speed_estimation(
    source_video_path: str, device: str
) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    tracker = sv.ByteTrack(minimum_consecutive_frames=3, frame_rate=video_info.fps)
    speed_estimator = SpeedEstimator(fps=video_info.fps)

    for frame in frame_generator:
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        if len(players) == 0:
            yield frame
            continue

        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32),
        )

        speed_estimator.update(players, transformer)
        speeds = speed_estimator.get_speeds(players)

        labels = [
            f"#{tracker_id} {speed}"
            for tracker_id, speed in zip(players.tracker_id, speeds)
        ]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, players)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, players, labels=labels
        )

        yield annotated_frame


def main(
    source_video_path: str, target_video_path: str, device: str, mode: Mode
) -> None:
    if mode == Mode.PITCH_DETECTION:  # Identifica pontos-chave do campo (linhas, áreas)
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device
        )
    elif mode == Mode.RADAR:
        frame_generator = run_radar(source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_SPEED_ESTIMATION:
        frame_generator = run_player_speed_estimation(
            source_video_path=source_video_path, device=device
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(
        source_video_path
    )  # Informações do vídeo de origem
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )  # Descrição do analisador de argumentos
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=Mode, default=Mode.PLAYER_DETECTION)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
    )
