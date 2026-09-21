# Experimentos

Um diretório por experimento, contendo o que é necessário para **reproduzir**
um resultado: o protocolo seguido, a configuração usada e os artefatos citados.

```text
experiments/
├── README.md
├── results/            # saida do `bench`/`compare` — fora do git (.gitignore)
└── <nome-do-experimento>/
    ├── README.md       # o que foi feito, quando, com que hardware
    ├── vision_config.json   # a calibracao exata usada
    └── resultados/     # apenas os JSONs citados em um artigo
```

`experiments/results/` está no `.gitignore` porque os resultados são regeráveis
a partir do dataset mais a configuração. Versionar à mão só o que for citado,
junto do commit exato que o gerou e do hardware em que rodou.

Protocolos: [`../docs/research/p1-illumination.md`](../docs/research/p1-illumination.md)
e [`../docs/research/p2-latency.md`](../docs/research/p2-latency.md).
