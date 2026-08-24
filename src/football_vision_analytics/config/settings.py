"""Configuracoes centrais do projeto Football Vision Analytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOCCER_DATA_DIR = PROJECT_ROOT / "examples" / "soccer" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "videos"


@dataclass(frozen=True)
class ModelPaths:
    """Caminhos dos modelos utilizados pelo pipeline de futebol.

    Args:
        player_detection: Modelo YOLO para jogadores, goleiros e arbitros.
        pitch_detection: Modelo YOLO para pontos-chave do campo.
        ball_detection: Modelo YOLO para deteccao da bola.
    """

    player_detection: Path = DEFAULT_SOCCER_DATA_DIR / "football-player-detection.pt"
    pitch_detection: Path = DEFAULT_SOCCER_DATA_DIR / "football-pitch-detection.pt"
    ball_detection: Path = DEFAULT_SOCCER_DATA_DIR / "football-ball-detection.pt"


@dataclass(frozen=True)
class PipelineConfig:
    """Parametros configuraveis do pipeline de analise de futebol.

    Args:
        source_video_path: Caminho do video de entrada.
        target_video_path: Caminho do video processado.
        device: Dispositivo de inferencia aceito pelas bibliotecas de ML.
        model_paths: Caminhos dos modelos de deteccao.
        team_classification_stride: Intervalo de frames usado para coletar crops.
        player_detection_image_size: Tamanho de inferencia do detector de jogadores.
        ball_detection_image_size: Tamanho de inferencia do detector da bola.
        ball_nms_threshold: Limiar de NMS aplicado nas deteccoes da bola.
        ball_tracker_buffer_size: Tamanho do historico do rastreador da bola.
        ball_annotator_buffer_size: Tamanho do rastro visual da bola.
        byte_track_minimum_consecutive_frames: Minimo de frames para confirmar tracks.
        display_preview: Se verdadeiro, exibe uma janela OpenCV durante o processamento.
    """

    source_video_path: Path
    target_video_path: Path
    device: str = "cpu"
    model_paths: ModelPaths = ModelPaths()
    team_classification_stride: int = 60
    player_detection_image_size: int = 1280
    ball_detection_image_size: int = 640
    ball_nms_threshold: float = 0.1
    ball_tracker_buffer_size: int = 20
    ball_annotator_buffer_size: int = 10
    byte_track_minimum_consecutive_frames: int = 3
    display_preview: bool = True

    def validate(self, require_models: bool = True) -> None:
        """Valida arquivos de entrada e cria o diretorio de saida.

        Args:
            require_models: Quando verdadeiro, tambem valida os modelos YOLO.

        Raises:
            FileNotFoundError: Se o video de entrada ou algum modelo obrigatorio nao
                existir.
        """

        if not self.source_video_path.exists():
            raise FileNotFoundError(
                f"Video de entrada nao encontrado: {self.source_video_path}"
            )

        if require_models:
            for path in (
                self.model_paths.player_detection,
                self.model_paths.pitch_detection,
                self.model_paths.ball_detection,
            ):
                if not path.exists():
                    raise FileNotFoundError(f"Modelo nao encontrado: {path}")

        self.target_video_path.parent.mkdir(parents=True, exist_ok=True)
