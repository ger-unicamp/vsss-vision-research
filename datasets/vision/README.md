# Dataset anotado — ground truth

Cada arquivo de mídia (`1.png`, `partida.mp4`, …) tem um **JSON irmão de mesmo
nome** com o estado verdadeiro do campo e as condições de captura. Mídia sem
JSON irmão é ignorada pelo carregador — dá para manter material bruto ainda não
anotado aqui dentro sem quebrar nada.

```text
datasets/vision/
├── exemplo.json           # template comentado (sem mídia irmã: não é carregado)
├── images/
│   ├── 1.png
│   ├── 1.json
│   └── ...
└── videos/
    ├── 1.mp4
    ├── 1.json
    └── ...
```

Carregado por `vsss_vision.benchmark.dataset.load_dataset`.

## Esquema

```jsonc
{
  "conditions": {
    "illuminance_lux": 420,          // medida com luximetro, no plano do campo
    "light_type": "cold",            // cold | warm | mixed | natural
    "uniform": false,                // ha sombra ou reflexo no campo?
    "notes": "sombra no canto inferior esquerdo"
  },
  "vision": {
    "units": "m",                    // "m" (preferido) ou "px"
    "robots": [
      { "id": 0, "team": "own",      "x": -0.30, "y":  0.12, "theta_deg": 35.0 },
      { "id": 1, "team": "opponent", "x":  0.44, "y": -0.05 }
    ],
    "ball": { "x": 0.0, "y": 0.0 }
  }
}
```

### `conditions`

Todos os campos são opcionais, mas **`illuminance_lux` é obrigatório na
prática** para a proposta P1: é a variável independente, e sem ela o `bench`
agrupa a cena como "iluminância desconhecida".

### `vision`

| Campo | Observação |
| :--- | :--- |
| `units` | `"m"` (origem no **centro** do campo, X para a direita, Y para cima) ou `"px"` (origem no canto superior-esquerdo, Y para baixo) |
| `robots[].id` | identidade esperada. É comparada **depois** da associação por posição — é o que permite contar trocas de identidade |
| `robots[].team` | `"own"` ou `"opponent"`. Padrão `"own"`. Robôs de times diferentes nunca são associados entre si |
| `robots[].theta_deg` | orientação em graus. Omitir quando não houver ground truth angular confiável — a métrica de orientação simplesmente não é calculada para esse robô |
| `ball` | omitir ou `null` quando a bola não estiver em campo |

#### Por que anotar em metros

Erro em pixel só é comparável dentro da mesma montagem de câmera e da mesma
resolução. Foi uma limitação declarada do trabalho de referência (ver
`docs/research/related-work.md`), e é o motivo de todas as métricas deste
repositório serem reportadas em centímetros e graus.

Quando `units` for `"px"`, a conversão usa a mesma escala do pipeline
(`geometry.pixel_to_meters`) e, portanto, **herda a incerteza do ROI**: o erro
residual da homografia passa a fazer parte do ground truth, não só da predição.
Aceitável para desenvolvimento, ruim para um resultado publicado.

### Vídeos

Anotar apenas os frames de interesse, indexados por número de frame:

```jsonc
{
  "conditions": { "illuminance_lux": 420, "light_type": "warm" },
  "vision": {
    "units": "m",
    "frames": {
      "0":   { "robots": [ /* ... */ ], "ball": { "x": 0.0, "y": 0.0 } },
      "150": { "robots": [ /* ... */ ], "ball": null }
    }
  }
}
```

O carregador busca exatamente esses frames (`CAP_PROP_POS_FRAMES`), não decodifica
o vídeo inteiro. `units` declarada no nível de `vision` vale para todos os
frames; um frame pode sobrescrevê-la.

## Antes de anotar: verificar a homografia

`vision/geometry.py` assume que o frame retificado cobre exatamente o retângulo
físico do campo (150 × 130 cm). Antes de qualquer coleta, posicionar um robô em
pontos conhecidos, comparar com a saída do pipeline e registrar o erro residual
— ele é o piso do erro de posição de qualquer detector e precisa constar do
artigo. Ver `docs/research/p1-illumination.md`.

## Rodando

```bash
uv run vsss-vision bench --dataset datasets/vision --out experiments/results/bench.json
uv run vsss-vision compare --dataset datasets/vision
```
