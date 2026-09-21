"""Gera um dataset em formato Ultralytics a partir das anotações deste repositório.

As anotações guardam centro em metros e ângulo; as caixas e os keypoints
de treino são derivados das dimensões físicas que as regras fixam (robô
7,5 cm, bola 42,7 mm) usando a escala px/m do frame retificado. Ver
`vsss_vision/benchmark/yolo_export.py` para o porquê.

Uso:
    uv run python -m tools.export_yolo_dataset --out datasets/yolo
    uv run python -m tools.export_yolo_dataset --out datasets/yolo --task detect --clean

A divisão treino/validação/teste é feita **por arquivo de origem**, nunca
por frame: frames vizinhos de um mesmo vídeo são quase idênticos, e
dividi-los aleatoriamente faz o teste medir memorização.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vsss_vision.benchmark.dataset import load_dataset
from vsss_vision.benchmark.yolo_export import (
    DEFAULT_BOX_MARGIN,
    class_names,
    export_dataset,
    split_by_source,
    summarize_splits,
)
from vsss_vision.vision.config import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("export_yolo_dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("datasets/vision"), help="dataset anotado de entrada")
    parser.add_argument("--out", type=Path, default=Path("datasets/yolo"), help="diretorio de saida")
    parser.add_argument("--task", choices=("pose", "detect"), default="pose")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.2, 0.1), metavar=("TREINO", "VAL", "TESTE"))
    parser.add_argument("--box-margin", type=float, default=DEFAULT_BOX_MARGIN, help="folga na caixa derivada")
    parser.add_argument("--clean", action="store_true", help="apaga o diretorio de saida antes de exportar")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config()

    dataset = load_dataset(args.dataset, config.field_length_m, config.field_width_m)
    if not len(dataset):
        logger.error("Nenhuma cena anotada em %s — ver datasets/vision/README.md.", args.dataset)
        return 1

    logger.info("Classes (de vision_config.json): %s", ", ".join(class_names(config)))
    logger.info("Divisao por arquivo de origem:\n%s", summarize_splits(split_by_source(dataset, tuple(args.ratios))))

    stats = export_dataset(
        dataset,
        output_root=args.out,
        config=config,
        task=args.task,
        ratios=tuple(args.ratios),
        box_margin=args.box_margin,
        clean=args.clean,
    )
    logger.info("Exportado para %s (task=%s):\n%s", args.out, args.task, stats.as_text())
    logger.info("Proximo passo: uv run python -m tools.train_yolo --data %s/data.yaml", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
