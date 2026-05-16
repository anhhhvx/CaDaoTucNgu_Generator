from __future__ import annotations

import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

RAW_DATASET = DATA_DIR / "dataset.txt"
NORMALIZED_DATASET = DATA_DIR / "dataset_normalized.txt"
TRAIN_DATASET = DATA_DIR / "train_data_seg.txt"
ARPA_FILE = MODEL_DIR / "model.arpa"
BINARY_FILE = MODEL_DIR / "model.bin"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N-gram Vietnamese poetry pipeline for VS Code.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser("normalize", help="Chuan hoa dataset tho.")
    normalize_parser.add_argument("--input", default=str(RAW_DATASET))
    normalize_parser.add_argument("--output", default=str(NORMALIZED_DATASET))

    tokenize_parser = subparsers.add_parser("tokenize", help="Tach tu dataset da chuan hoa.")
    tokenize_parser.add_argument("--input", default=str(NORMALIZED_DATASET))
    tokenize_parser.add_argument("--output", default=str(TRAIN_DATASET))

    train_parser = subparsers.add_parser("train", help="Train KenLM model tu file da tach tu.")
    train_parser.add_argument("--input", default=str(TRAIN_DATASET))
    train_parser.add_argument("--arpa", default=str(ARPA_FILE))
    train_parser.add_argument("--binary", default=str(BINARY_FILE))
    train_parser.add_argument("--order", type=int, default=5)
    train_parser.add_argument("--kenlm-bin-dir", default=None)

    generate_parser = subparsers.add_parser("generate", help="Sinh cau tho tu seed text.")
    generate_parser.add_argument("--input", required=True, help="Cum tu goi y.")
    generate_parser.add_argument("--model", default=str(BINARY_FILE))
    generate_parser.add_argument("--data", default=str(TRAIN_DATASET))
    generate_parser.add_argument("--order", type=int, default=5)
    generate_parser.add_argument("--top-k", type=int, default=5)

    evaluate_parser = subparsers.add_parser("evaluate", help="Danh gia mo hinh tren full dataset.")
    evaluate_parser.add_argument("--model", default=str(BINARY_FILE))
    evaluate_parser.add_argument("--data", default=str(TRAIN_DATASET))
    evaluate_parser.add_argument("--raw", default=str(RAW_DATASET))
    evaluate_parser.add_argument("--order", type=int, default=5)
    evaluate_parser.add_argument("--max-input-ratio", type=float, default=0.7)
    evaluate_parser.add_argument("--output-csv", default=str(BASE_DIR / "evaluation_results.csv"))

    full_parser = subparsers.add_parser("run-all", help="Chay normalize -> tokenize -> train.")
    full_parser.add_argument("--raw", default=str(RAW_DATASET))
    full_parser.add_argument("--normalized", default=str(NORMALIZED_DATASET))
    full_parser.add_argument("--train-data", default=str(TRAIN_DATASET))
    full_parser.add_argument("--arpa", default=str(ARPA_FILE))
    full_parser.add_argument("--binary", default=str(BINARY_FILE))
    full_parser.add_argument("--order", type=int, default=5)
    full_parser.add_argument("--kenlm-bin-dir", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "normalize":
        from src.normalize import normalize_step

        normalize_step(args.input, args.output)
        return

    if args.command == "tokenize":
        from src.tokenize_data import tokenize_step

        tokenize_step(args.input, args.output)
        return

    if args.command == "train":
        from src.train_model import train_kenlm_model

        train_kenlm_model(
            input_file=args.input,
            arpa_file=args.arpa,
            binary_file=args.binary,
            order=args.order,
            kenlm_bin_dir=args.kenlm_bin_dir,
        )
        return

    if args.command == "generate":
        from src.generator import BidirectionalBeamGenerator

        generator = BidirectionalBeamGenerator(args.model, args.data, n_gram_order=args.order)
        results = generator.generate_best_cases(args.input, num_results=args.top_k)
        if not results:
            print("Khong tim thay ket qua phu hop.")
            return

        print(f"\nTOP KET QUA CHO: '{args.input}'")
        print("=" * 60)
        for index, (score, text) in enumerate(results, start=1):
            print(f"Hang #{index} (Score: {score:.2f})")
            print(text)
            print("-" * 30)
        return

    if args.command == "evaluate":
        from src.evaluate import evaluate_pipeline

        result_df = evaluate_pipeline(
            model_file=args.model,
            data_file=args.data,
            raw_dataset=args.raw,
            n_gram_order=args.order,
            max_input_ratio=args.max_input_ratio,
        )
        output_csv = Path(args.output_csv)
        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"Da luu ket qua danh gia tai: {output_csv}")
        return

    if args.command == "run-all":
        from src.normalize import normalize_step
        from src.tokenize_data import tokenize_step
        from src.train_model import train_kenlm_model

        normalize_step(args.raw, args.normalized)
        tokenize_step(args.normalized, args.train_data)
        train_kenlm_model(
            input_file=args.train_data,
            arpa_file=args.arpa,
            binary_file=args.binary,
            order=args.order,
            kenlm_bin_dir=args.kenlm_bin_dir,
        )
        return

    parser.error("Lenh khong hop le.")


if __name__ == "__main__":
    main()
