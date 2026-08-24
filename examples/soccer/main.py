"""Entrypoint do exemplo de analise de futebol."""

from __future__ import annotations

import argparse
from pathlib import Path

from football_vision_analytics.config import ModelPaths, PipelineConfig
from football_vision_analytics.pipeline import FootballAnalysisPipeline, Mode


def parse_args() -> argparse.Namespace:
    """Processa argumentos de linha de comando do exemplo."""

    parser = argparse.ArgumentParser(
        description="Executa o pipeline Football Vision Analytics em um video."
    )
    parser.add_argument(
        "--source_video_path",
        type=Path,
        required=True,
        help="Caminho para o video de entrada.",
    )
    parser.add_argument(
        "--target_video_path",
        type=Path,
        required=True,
        help="Caminho onde o video processado sera salvo.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Dispositivo de inferencia, como cpu, cuda ou mps.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in Mode],
        default=Mode.PLAYER_DETECTION.value,
        help="Modo de analise a executar.",
    )
    parser.add_argument(
        "--player_model_path",
        type=Path,
        default=None,
        help="Caminho opcional para o modelo de deteccao de jogadores.",
    )
    parser.add_argument(
        "--pitch_model_path",
        type=Path,
        default=None,
        help="Caminho opcional para o modelo de deteccao do campo.",
    )
    parser.add_argument(
        "--ball_model_path",
        type=Path,
        default=None,
        help="Caminho opcional para o modelo de deteccao da bola.",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Desativa a janela OpenCV durante o processamento.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Monta a configuracao do pipeline a partir da CLI.

    Args:
        args: Argumentos retornados por `parse_args`.

    Returns:
        Configuracao validavel para o pipeline de futebol.
    """

    default_model_paths = ModelPaths()
    model_paths = ModelPaths(
        player_detection=args.player_model_path or default_model_paths.player_detection,
        pitch_detection=args.pitch_model_path or default_model_paths.pitch_detection,
        ball_detection=args.ball_model_path or default_model_paths.ball_detection,
    )
    return PipelineConfig(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        model_paths=model_paths,
        display_preview=not args.no_display,
    )


def main() -> None:
    """Executa o pipeline principal de analise de futebol."""

    args = parse_args()
    pipeline = FootballAnalysisPipeline(build_config(args))
    pipeline.run(Mode(args.mode))


if __name__ == "__main__":
    main()
