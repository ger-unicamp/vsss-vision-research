"""Interface de linha de comando: `vsss-vision <subcomando>`.

Três subcomandos, correspondendo aos três modos de uso do repositório:

  `bench`   — avaliação offline sobre um dataset anotado (propostas P1 e P2).
  `live`    — pipeline sobre a câmera, com latência instrumentada (P2).
  `compare` — roda vários detectores sobre o mesmo dataset e imprime lado a lado.

`compare` é o comando que fecha o argumento dos artigos: o que interessa
não é o número absoluto de um detector, e sim a diferença entre eles sob
*exatamente* as mesmas cenas, mesma calibração e mesma instrumentação.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vsss_vision.benchmark.dataset import load_dataset
from vsss_vision.benchmark.runner import format_summary, run_benchmark
from vsss_vision.vision.config import load_config
from vsss_vision.vision.detectors import available_detectors

DEFAULT_DATASET = Path("datasets/vision")


def _add_bench_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="diretorio do dataset anotado")
    parser.add_argument("--no-tracker", action="store_true", help="desliga a suavizacao temporal (imagens soltas)")
    parser.add_argument("--match-radius-cm", type=float, default=10.0, help="raio de casamento predicao/ground truth")
    parser.add_argument("--warmup", type=int, default=0, help="frames iniciais descartados da medicao de latencia")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vsss-vision", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="log em nivel DEBUG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench", help="avaliacao offline sobre dataset anotado")
    _add_bench_arguments(bench)
    bench.add_argument("--detector", choices=available_detectors(), default=None)
    bench.add_argument("--out", type=Path, default=None, help="arquivo JSON de resultado")

    compare = subparsers.add_parser("compare", help="roda varios detectores sobre o mesmo dataset")
    _add_bench_arguments(compare)
    compare.add_argument("--detectors", nargs="+", default=available_detectors())
    compare.add_argument("--out-dir", type=Path, default=Path("experiments/results"))

    live = subparsers.add_parser("live", help="pipeline sobre a camera, com latencia instrumentada")
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

    if args.command == "bench":
        return _run_bench(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "live":
        return _run_live(args)
    return 1


def _run_bench(args: argparse.Namespace) -> int:
    config = load_config()
    dataset = load_dataset(args.dataset, config.field_length_m, config.field_width_m)
    if not len(dataset):
        print(f"Nenhuma cena anotada em {args.dataset} — ver datasets/vision/README.md para o esquema JSON.")
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
    if args.out:
        print(f"resultado salvo em {result.write_json(args.out)}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    config = load_config()
    dataset = load_dataset(args.dataset, config.field_length_m, config.field_width_m)
    if not len(dataset):
        print(f"Nenhuma cena anotada em {args.dataset} — ver datasets/vision/README.md para o esquema JSON.")
        return 1

    failures = 0
    for detector_name in args.detectors:
        try:
            result = run_benchmark(
                dataset,
                detector_name=detector_name,
                config=config,
                use_tracker=not args.no_tracker,
                match_radius_cm=args.match_radius_cm,
                warmup_frames=args.warmup,
            )
        except NotImplementedError as error:
            # Um detector ainda não implementado não invalida a comparação
            # entre os demais — reportar e seguir.
            print(f"detector={detector_name}: {error}")
            failures += 1
            continue

        print(format_summary(result))
        print(f"  -> {result.write_json(args.out_dir / f'{detector_name}.json')}\n")

    return 0 if failures < len(args.detectors) else 1


def _run_live(args: argparse.Namespace) -> int:
    from vsss_vision.vision.live import run  # import tardio: só o modo live precisa de zmq ativo

    run(
        detector_name=args.detector,
        sink=args.sink,
        warmup_frames=args.warmup,
        report_every=args.report_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
