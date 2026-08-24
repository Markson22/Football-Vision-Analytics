# Guia de instalação e execução

Este guia detalha como preparar o ambiente e executar o exemplo de futebol.

## 1. Ambiente

Use Python >= 3.10. Em WSL, prefira criar o ambiente virtual dentro do próprio diretório Linux do projeto.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

No Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. Dados e modelos

Os arquivos `.mp4` e `.pt` são pesados e ficam fora do Git. O caminho padrão esperado pelo exemplo é:

```text
examples/soccer/data/
├── football-player-detection.pt
├── football-pitch-detection.pt
├── football-ball-detection.pt
└── 0bfacc_0.mp4
```

Você também pode informar caminhos personalizados com `--player_model_path`, `--pitch_model_path`, `--ball_model_path` e `--source_video_path`.

## 3. Execução rápida

```bash
python examples/soccer/main.py \
  --source_video_path examples/soccer/data/0bfacc_0.mp4 \
  --target_video_path outputs/videos/player-detection.mp4 \
  --device cpu \
  --mode PLAYER_DETECTION \
  --no_display
```

Remova `--no_display` se quiser ver a janela OpenCV durante o processamento.

## 4. Modos disponíveis

- `PITCH_DETECTION`: detecta pontos-chave do campo.
- `PLAYER_DETECTION`: detecta jogadores, goleiros, árbitros e bola.
- `BALL_DETECTION`: detecta e rastreia a bola.
- `PLAYER_TRACKING`: rastreia jogadores com IDs.
- `TEAM_CLASSIFICATION`: separa jogadores por time.
- `RADAR`: projeta jogadores em uma visão tática 2D.
- `PLAYER_SPEED_ESTIMATION`: estima velocidades em km/h.

## 5. Testes

```bash
pytest
```

Os testes atuais focam componentes independentes de GPU, pesos YOLO e vídeos reais.
