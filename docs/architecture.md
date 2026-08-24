# Arquitetura interna

O projeto usa layout `src/` para separar o pacote Python dos exemplos e artefatos locais.

## Pacote principal

```text
src/football_vision_analytics/
├── analytics/
├── classification/
├── config/
├── pipeline/
├── pitch/
├── tracking/
├── utils/
└── visualization/
```

## Responsabilidades

- `config`: centraliza caminhos, parâmetros do pipeline e validação básica.
- `pipeline`: orquestra modelos, leitura de vídeo, modos de análise e gravação da saída.
- `classification`: classifica jogadores em times a partir de embeddings visuais.
- `tracking`: contém o rastreador e anotador específicos da bola.
- `pitch`: descreve a geometria do campo e a transformação de perspectiva.
- `analytics`: calcula métricas derivadas, como velocidade.
- `visualization`: desenha campo, pontos, trajetórias e visualizações táticas.
- `utils`: concentra funções pequenas reaproveitáveis para detecções.

## Entry point

`examples/soccer/main.py` é propositalmente pequeno. Ele:

1. Lê argumentos da linha de comando.
2. Monta `PipelineConfig`.
3. Instancia `FootballAnalysisPipeline`.
4. Executa o modo escolhido.

Essa separação facilita adicionar novos modos sem transformar o exemplo em um arquivo monolítico.

## Créditos

A base original tem relação com o projeto `roboflow/sports` e com bibliotecas como Supervision e Ultralytics. A reorganização mantém essa referência explícita e separa autoria do projeto atual de autoria de terceiros.
