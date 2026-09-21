"""Interface de linha de comando: `vsss-vision <subcomando>`.

  `bench`   — avaliação offline de um detector sobre um dataset anotado.
  `compare` — vários detectores sobre exatamente as mesmas cenas (P1).
  `sweep`   — varredura de variantes e resoluções, produzindo a curva de
              precisão × latência (P2).
  `live`    — pipeline sobre a câmera, com latência instrumentada.

`compare` e `sweep` são os comandos que fecham os argumentos dos artigos.
O número absoluto de um detector isolado não diz nada; o que sustenta uma
conclusão é a diferença entre detectores sob as mesmas cenas, a mesma
calibração e a mesma instrumentação, ou a mesma configuração variando um
único eixo por vez.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Optional, Sequence

from vsss_vision.benchmark.dataset import Dataset, load_dataset
from vsss_vision.benchmark.report import (
    markdown_table,
    write_csv,
    write_latency_curve_csv,
    write_markdown,
)
from vsss_vision.benchmark.runner import RunResult, format_summary, run_benchmark
from vsss_vision.vision.config import VisionConfig, load_config
from vsss_vision.vision.detectors import available_detectors

DEFAULT_DATASET = Path("datasets/vision")
DEFAULT_OUT_DIR = Path("experiments/results")
logger = logging.getLogger("vsss_vision.cli")


# --- argumentos compartilhados ------------------------------------------


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="diretorio do dataset anotado")
    parser.add_argument("--no-tracker", action="store_true", help="desliga a suavizacao temporal (imagens soltas)")
    parser.add_argument("--match-radius-cm", type=float, default=10.0, help="raio de casamento predicao/ground truth")
    parser.add_argument("--warmup", type=int, default=0, help="frames iniciais descartados da medicao de latencia")


def _add_yolo_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=None, help="pesos do detector YOLO")
    parser.add_argument("--imgsz", type=int, default=None, help="resolucao de entrada da rede")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda:0")
    parser.add_argument("--conf", type=float, default=None, help="limiar de confianca")


def _add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="diretorio dos resultados")
    parser.add_argument("--csv", type=Path, default=None, help="tambem escreve um CSV agregado")
    parser.add_argument("--markdown", type=Path, default=None, help="tambem escreve uma tabela Markdown")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vsss-vision", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="log em nivel DEBUG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench", help="avaliacao offline sobre dataset anotado")
    _add_dataset_arguments(bench)
    _add_yolo_arguments(bench)
    _add_report_arguments(bench)
    bench.add_argument("--detector", choices=available_detectors(), default=None)
    bench.add_argument("--out", type=Path, default=None, help="arquivo JSON de resultado")

    compare = subparsers.add_parser("compare", help="varios detectores sobre o mesmo dataset")
    _add_dataset_arguments(compare)
    _add_yolo_arguments(compare)
    _add_report_arguments(compare)
    compare.add_argument("--detectors", nargs="+", default=available_detectors())

    sweep = subparsers.add_parser("sweep", help="curva precisao x latencia (variantes e resolucoes)")
    _add_dataset_arguments(sweep)
    _add_report_arguments(sweep)
    sweep.add_argument("--models", nargs="+", type=Path, required=True, help="pesos de cada variante")
    sweep.add_argument("--imgsz", nargs="+", type=int, default=[640], help="resolucoes a varrer")
    sweep.add_argument("--device", default=None, help="auto | cpu | cuda:0")
    sweep.add_argument(
        "--include-color",
        action="store_true",
        help="inclui o pipeline classico como ponto de referencia na curva",
    )

    live = subparsers.add_parser("live", help="pipeline sobre a camera, com latencia instrumentada")
    _add_yolo_arguments(live)
    live.add_argument("--detector", choices=available_detectors(), default=None)
    live.add_argument("--sink", choices=("none", "log", "zmq"), default="none")
    live.add_argument("--warmup", type=int, default=30)
    live.add_argument("--report-every", type=int, default=300, help="frames entre relatorios de latencia")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    handlers = {"bench": _run_bench, "compare": _run_compare, "sweep": _run_sweep, "live": _run_live}
    return handlers[args.command](args)


# --- subcomandos ---------------------------------------------------------


def _run_bench(args: argparse.Namespace) -> int:
    config = _config_from(args)
    dataset = _load(args.dataset, config)
    if dataset is None:
        return 1

    result = run_benchmark(
        dataset,
        detector_name=args.detector,
        config=config,
        use_tracker=not args.no_tracker,
        match_radius_cm=args.match_radius_cm,
        warmup_frames=args.warmup,
    )
    print(format_summary(result))
    out = args.out or (args.out_dir / f"{result.name}.json")
    print(f"resultado salvo em {result.write_json(out)}")
    _write_reports(args, [result])
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    config = _config_from(args)
    dataset = _load(args.dataset, config)
    if dataset is None:
        return 1

    results: list[RunResult] = []
    for detector_name in args.detectors:
        result = _safe_run(
            dataset,
            config=config,
            detector_name=detector_name,
            label=detector_name,
            use_tracker=not args.no_tracker,
            match_radius_cm=args.match_radius_cm,
            warmup_frames=args.warmup,
        )
        if result is None:
            continue
        results.append(result)
        print(format_summary(result))
        print(f"  -> {result.write_json(args.out_dir / f'{result.name}.json')}\n")

    if not results:
        return 1
    _write_reports(args, results)
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    """Varre variantes × resoluções e escreve a curva de compromisso da P2.

    Uma execução por ponto, sempre sobre o mesmo dataset e a mesma
    calibração: só o eixo varrido muda. O hardware precisa ser o mesmo
    durante toda a varredura, e isso é responsabilidade de quem roda —
    o resultado grava o `device`, mas não sabe em que máquina está.
    """

    base_config = _config_from(args)
    dataset = _load(args.dataset, base_config)
    if dataset is None:
        return 1

    results: list[RunResult] = []
    if args.include_color:
        reference = _safe_run(
            dataset,
            config=base_config,
            detector_name="color",
            label="color",
            use_tracker=not args.no_tracker,
            match_radius_cm=args.match_radius_cm,
            warmup_frames=args.warmup,
        )
        if reference is not None:
            results.append(reference)
            print(format_summary(reference))

    for model_path in args.models:
        for imgsz in args.imgsz:
            config = dataclasses.replace(base_config, yolo_model_path=str(model_path), yolo_imgsz=imgsz)
            label = f"{Path(model_path).stem}@{imgsz}"
            result = _safe_run(
                dataset,
                config=config,
                detector_name="yolo",
                label=label,
                use_tracker=not args.no_tracker,
                match_radius_cm=args.match_radius_cm,
                warmup_frames=args.warmup,
            )
            if result is None:
                continue
            results.append(result)
            print(format_summary(result))
            result.write_json(args.out_dir / f"{label}.json")

    if not results:
        print("Nenhum ponto da varredura foi concluido.")
        return 1

    curve = write_latency_curve_csv(results, args.out_dir / "curva_precisao_latencia.csv")
    print(f"\ncurva precisao x latencia: {curve}")
    _write_reports(args, results)
    return 0


def _run_live(args: argparse.Namespace) -> int:
    from vsss_vision.vision.live import run  # import tardio: só o modo live precisa de zmq ativo

    run(
        config=_config_from(args),
        detector_name=args.detector,
        sink=args.sink,
        warmup_frames=args.warmup,
        report_every=args.report_every,
    )
    return 0


# --- apoio ---------------------------------------------------------------


def _config_from(args: argparse.Namespace) -> VisionConfig:
    """Configuração do arquivo, com as sobrescritas da linha de comando."""

    config = load_config()
    overrides = {}
    if getattr(args, "model", None) is not None:
        overrides["yolo_model_path"] = str(args.model)
    if getattr(args, "imgsz", None) is not None and isinstance(args.imgsz, int):
        overrides["yolo_imgsz"] = args.imgsz
    if getattr(args, "device", None) is not None:
        overrides["yolo_device"] = args.device
    if getattr(args, "conf", None) is not None:
        overrides["yolo_confidence"] = args.conf
    return dataclasses.replace(config, **overrides) if overrides else config


def _load(dataset_root: Path, config: VisionConfig) -> Optional[Dataset]:
    dataset = load_dataset(dataset_root, config.field_length_m, config.field_width_m)
    if not len(dataset):
        print(
            f"Nenhuma cena anotada em {dataset_root} — cada midia precisa de um JSON irmao. "
            "Ver datasets/vision/README.md."
        )
        return None
    return dataset


def _safe_run(dataset: Dataset, **kwargs) -> Optional[RunResult]:
    """Roda um ponto; uma falha dele não derruba a execução inteira.

    Extra não instalado, pesos ausentes ou mapa de classes incompatível são
    problemas de um detector só — os demais pontos continuam válidos e
    devem ser reportados.
    """

    label = kwargs.get("label") or kwargs.get("detector_name")
    try:
        return run_benchmark(dataset, **kwargs)
    except (ImportError, FileNotFoundError, ValueError) as error:
        print(f"[pulado] {label}: {error}")
        return None


def _write_reports(args: argparse.Namespace, results: Sequence[RunResult]) -> None:
    if getattr(args, "csv", None):
        print(f"csv: {write_csv(results, args.csv)}")
    if getattr(args, "markdown", None):
        print(f"markdown: {write_markdown(results, args.markdown)}")
    if len(results) > 1 and not getattr(args, "markdown", None):
        print(markdown_table(results, split_by_condition=False))


if __name__ == "__main__":
    raise SystemExit(main())
