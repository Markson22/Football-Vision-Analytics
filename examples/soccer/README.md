# Exemplo: Soccer

Este exemplo executa os modos atuais do **Football Vision Analytics** em vídeos de futebol.

## Instalação

Na raiz do repositório:

```bash
pip install -e ".[dev]"
```

## Dados esperados

Por padrão, o exemplo procura vídeos e modelos em `examples/soccer/data/`.

```text
examples/soccer/data/
├── football-player-detection.pt
├── football-pitch-detection.pt
├── football-ball-detection.pt
└── 0bfacc_0.mp4
```

Arquivos `.pt` e `.mp4` são ignorados pelo Git por serem pesados.

## Executar

```bash
python examples/soccer/main.py \
  --source_video_path examples/soccer/data/0bfacc_0.mp4 \
  --target_video_path outputs/videos/player-detection.mp4 \
  --device cpu \
  --mode PLAYER_DETECTION \
  --no_display
```

## Modos

- `PITCH_DETECTION`
- `PLAYER_DETECTION`
- `BALL_DETECTION`
- `PLAYER_TRACKING`
- `TEAM_CLASSIFICATION`
- `RADAR`
- `PLAYER_SPEED_ESTIMATION`

## Créditos

Os notebooks e referências de datasets vieram do ecossistema Roboflow/Supervision e devem preservar os créditos originais. O projeto atual é desenvolvido e mantido por Markson Cesar.
