"""Fine-tuning de um YOLO sobre o dataset exportado, com os braços da proposta P1.

Dois braços de treino, e a diferença entre eles é um resultado, não uma
preferência de implementação:

  `--augment brightness` — jitter de brilho/saturação/matiz ligado. É a
      hipótese de que *augmentation* fotométrico substitui recalibração
      manual sob mudança de iluminação (P1, H1).
  `--augment none`       — tudo desligado. É o controle: sem ele, não dá
      para dizer se um ganho veio da arquitetura ou do augmentation.

Rodar os dois braços para cada variante. Treinar só o braço com
augmentation e comparar com o pipeline clássico não separa as duas causas.

Escolhas que não são óbvias e valem justificar no artigo:

- **`fliplr` é seguro**, o campo VSSS é simétrico no eixo longitudinal e
  espelhar uma cena produz uma cena possível. Mas espelhar **inverte a
  orientação**: com `task=pose`, `flip_idx` no `data.yaml` precisa estar
  correto, senão a rede aprende frente e trás trocadas.
- **`mosaic` é duvidoso aqui.** Ele cola pedaços de quatro imagens; num
  sistema de visão global com câmera fixa, isso gera composições que nunca
  ocorrem (dois campos na mesma imagem, bordas no meio do campo). Fica
  desligado por padrão; ligar só com justificativa e medindo.
- **`degrees` (rotação) é limitado.** A câmera é fixa acima do campo, então
  rotações grandes não representam nada real; pequenas absorvem erro de
  montagem do ROI.

Uso:
    uv run python -m tools.train_yolo --data datasets/yolo/data.yaml --model yolov8n-pose.pt
    uv run python -m tools.train_yolo --data datasets/yolo/data.yaml --augment none --name n_sem_aug
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("train_yolo")

_INSTALL_HINT = "Ultralytics nao esta instalado. Rode `uv sync --extra yolo`."

# Jitter fotométrico. hsv_v (brilho) e hsv_s (saturacao) sao os que
# importam para a P1; hsv_h fica baixo porque a matiz e justamente o
# sinal que distingue os times — destrui-la ensina a rede a ignorar a
# unica pista confiavel de qual robo e de quem.
AUGMENTATION_ARMS = {
    "brightness": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.6,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.1,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "erasing": 0.0,
    },
    "none": {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "erasing": 0.0,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("datasets/yolo/data.yaml"))
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="pesos de partida; a variante (n/s/m/l/x) e o eixo varrido na P2",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu | 0 | 0,1")
    parser.add_argument("--augment", choices=tuple(AUGMENTATION_ARMS), default="brightness")
    parser.add_argument("--name", default=None, help="nome da execucao (runs/<task>/<name>)")
    parser.add_argument("--project", type=Path, default=Path("experiments/training"))
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="early stopping; o trabalho de referencia varreu 100..1000 epocas a mao",
    )
    parser.add_argument("--seed", type=int, default=0, help="fixar para o treino ser reproduzivel")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError as error:  # pragma: no cover - depende do ambiente
        raise SystemExit(_INSTALL_HINT) from error

    if not args.data.exists():
        raise SystemExit(
            f"{args.data} nao existe. Gere o dataset antes: "
            "uv run python -m tools.export_yolo_dataset --out datasets/yolo"
        )

    augmentation = AUGMENTATION_ARMS[args.augment]
    name = args.name or f"{Path(args.model).stem}_{args.augment}_{args.imgsz}"
    logger.info("Treinando %s | braco de augmentation: %s | %d epocas", args.model, args.augment, args.epochs)

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        project=str(args.project),
        name=name,
        exist_ok=False,
        **augmentation,
    )

    weights = Path(args.project) / name / "weights" / "best.pt"
    logger.info("Pesos: %s", weights)
    logger.info(
        "Avaliar em escala real (mAP nao basta):\n"
        "  uv run vsss-vision bench --detector yolo --model %s --csv experiments/results/%s.csv",
        weights,
        name,
    )
    return 0 if results is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
