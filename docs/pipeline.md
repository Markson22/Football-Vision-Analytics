# Pipeline de visão computacional

O pipeline atual processa vídeos frame a frame e produz um novo vídeo anotado.

```text
Entrada
 ↓
Leitura do vídeo
 ↓
Detecção
 ↓
Tracking / classificação / transformação
 ↓
Anotação
 ↓
Saída em vídeo
```

## Etapas

### Entrada

O vídeo é lido com `supervision.get_video_frames_generator`. O arquivo de saída usa as mesmas informações básicas do vídeo original por meio de `supervision.VideoInfo`.

### Detecção

Os modelos YOLO são carregados por modo:

- jogador/campo/árbitro/bola via `football-player-detection.pt`;
- pontos-chave do campo via `football-pitch-detection.pt`;
- bola via `football-ball-detection.pt`.

### Tracking

Jogadores usam `supervision.ByteTrack`. A bola usa `BallTracker`, que escolhe a detecção mais próxima do centroide recente para reduzir oscilações.

### Classificação dos times

`TeamClassifier` extrai crops dos jogadores, gera embeddings com SigLIP, reduz com UMAP e agrupa com KMeans em dois times.

### Transformação de perspectiva

`ViewTransformer` calcula uma homografia entre pontos-chave detectados no frame e os vértices configurados do campo. Essa transformação alimenta o radar e a estimativa de velocidade.

### Anotação

O pipeline usa anotadores do Supervision e funções próprias para desenhar:

- caixas e labels;
- elipses de jogadores;
- rastro da bola;
- pontos no mini-campo tático;
- labels de velocidade.

### Saída

Os frames anotados são gravados com `supervision.VideoSink`. Por padrão, também podem ser exibidos em uma janela OpenCV; para execução sem interface, use `--no_display`.
