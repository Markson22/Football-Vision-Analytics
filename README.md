# ⚽ Football Vision Analytics

Projeto de Visão Computacional aplicado à análise de partidas de futebol.

## Sobre o projeto

O **Football Vision Analytics** organiza um pipeline Python para processar vídeos de futebol com modelos de visão computacional. O projeto detecta elementos da partida, rastreia jogadores e bola, classifica jogadores por time, projeta posições em uma visão tática do campo e estima velocidades dos jogadores.

O código foi reorganizado para separar responsabilidades em módulos reutilizáveis, mantendo o exemplo executável em `examples/soccer/main.py`.

## Funcionalidades

- Detecção de pontos-chave do campo de futebol.
- Detecção de jogadores, goleiros, árbitros e bola.
- Rastreamento de jogadores com ByteTrack.
- Rastreamento e anotação da bola com histórico visual.
- Classificação visual de jogadores em dois times usando crops de uniforme.
- Associação de goleiros ao time mais próximo.
- Visualização radar com transformação de perspectiva para o campo 2D.
- Estimativa de velocidade dos jogadores em km/h.
- Geração de vídeo processado com anotações.

## Tecnologias utilizadas

- Python
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLO
- Supervision
- Transformers
- SigLIP
- UMAP
- scikit-learn
- tqdm
- gdown

## Arquitetura

Fluxo geral do exemplo de futebol:

```text
Vídeo
  ↓
Leitura frame a frame
  ↓
Detecção com YOLO
  ↓
Tracking e pós-processamento
  ↓
Classificação / transformação de perspectiva / análise
  ↓
Anotação visual
  ↓
Vídeo processado
```

O entrypoint `examples/soccer/main.py` apenas lê argumentos, monta a configuração e executa `FootballAnalysisPipeline`. A lógica fica em `src/football_vision_analytics/`.

## Estrutura do projeto

```text
Football-Vision-Analytics/
├── assets/                         # Imagens e diagramas de apoio
├── docs/                           # Documentação técnica em português
├── examples/
│   └── soccer/
│       ├── data/                   # Vídeos/modelos locais ignorados pelo Git
│       ├── main.py                 # Entrypoint do exemplo
│       ├── requirements.txt        # Dependências extras históricas do exemplo
│       └── setup.sh                # Script original de download de dados/modelos
├── notebooks/                      # Notebooks de treino dos modelos
├── outputs/                        # Resultados gerados localmente
├── src/
│   └── football_vision_analytics/
│       ├── analytics/              # Cálculos derivados, como velocidade
│       ├── classification/         # Classificação visual dos times
│       ├── config/                 # Configuração e caminhos
│       ├── pipeline/               # Orquestração dos modos de análise
│       ├── pitch/                  # Campo e transformação de perspectiva
│       ├── tracking/               # Rastreamento da bola
│       ├── utils/                  # Utilitários de detecções
│       └── visualization/          # Desenho do campo e visualizações
└── tests/                          # Testes unitários de componentes puros
```

## Pré-requisitos

- Python >= 3.10
- Git
- Ambiente com suporte às dependências de Machine Learning do projeto
- CUDA ou MPS são opcionais; `cpu` funciona, mas tende a ser mais lento

Os vídeos e pesos `.pt` são arquivos pesados e ficam ignorados pelo Git. Para executar o pipeline completo, eles devem existir em `examples/soccer/data/` ou ser informados via argumentos CLI.

## Clonar

```bash
git clone https://github.com/Markson22/Football-Vision-Analytics.git
cd Football-Vision-Analytics
```

## Criar ambiente virtual

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Instalar dependências

```bash
pip install -e ".[dev]"
```

Se for usar apenas o exemplo histórico, `examples/soccer/requirements.txt` ainda lista dependências auxiliares do setup original.

## Executar

Exemplo sem janela OpenCV, salvando o resultado em `outputs/videos/`:

```bash
python examples/soccer/main.py \
  --source_video_path examples/soccer/data/0bfacc_0.mp4 \
  --target_video_path outputs/videos/player-detection.mp4 \
  --device cpu \
  --mode PLAYER_DETECTION \
  --no_display
```

Modos disponíveis:

```text
PITCH_DETECTION
PLAYER_DETECTION
BALL_DETECTION
PLAYER_TRACKING
TEAM_CLASSIFICATION
RADAR
PLAYER_SPEED_ESTIMATION
```

Também é possível informar modelos em outros caminhos:

```bash
python examples/soccer/main.py \
  --source_video_path caminho/video.mp4 \
  --target_video_path outputs/videos/resultado.mp4 \
  --player_model_path caminho/football-player-detection.pt \
  --pitch_model_path caminho/football-pitch-detection.pt \
  --ball_model_path caminho/football-ball-detection.pt \
  --mode RADAR \
  --no_display
```

## Documentação

- [Guia de instalação e execução](docs/getting-started.md)
- [Arquitetura interna](docs/architecture.md)
- [Pipeline de visão computacional](docs/pipeline.md)

## Roadmap

- Cálculo de distância percorrida por jogador.
- Posse de bola.
- Mapas de calor.
- Estatísticas agregadas da partida.
- Identificação mais robusta de árbitros e goleiros.
- Exportação de métricas em CSV/JSON.
- Testes de integração com vídeos pequenos de amostra.

## Créditos e código de terceiros

Este projeto utiliza bibliotecas e ideias do ecossistema de visão computacional, incluindo Roboflow, Supervision, Ultralytics YOLO e Transformers. Partes do código original foram inspiradas ou adaptadas do projeto `roboflow/sports`; os créditos e licenças desses projetos devem ser preservados.

O **Football Vision Analytics** é desenvolvido e mantido por Markson Cesar, sem reivindicar autoria sobre bibliotecas, modelos, datasets ou projetos de terceiros utilizados como base.

## 👨‍💻 Autor

**Markson Cesar**

Desenvolvedor do projeto Football Vision Analytics.

GitHub: [@Markson22](https://github.com/Markson22)

## Licença

Este repositório mantém a licença MIT presente no projeto. Verifique também as licenças dos modelos, datasets e bibliotecas externas utilizados no pipeline.
