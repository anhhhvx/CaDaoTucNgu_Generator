from __future__ import annotations

import difflib
import time
from pathlib import Path

import pandas as pd
from pyvi import ViTokenizer
from tqdm import tqdm

from src.generator import BidirectionalBeamGenerator


def create_full_stress_test(original_file: str | Path, max_input_ratio: float = 0.7) -> pd.DataFrame:
    original_file = Path(original_file)
    print(f"Dang xay dung ma tran test toan dien tu {original_file}...")

    if not original_file.exists():
        raise FileNotFoundError(f"Khong tim thay dataset goc: {original_file}")

    test_data: list[dict[str, str | int]] = []
    lines = [line.strip() for line in original_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    for line in tqdm(lines, desc="Processing Data"):
        tokenized_line = ViTokenizer.tokenize(line).lower()
        words = tokenized_line.split()
        word_count = len(words)

        if word_count < 3:
            continue

        ground_truth = " ".join(words)
        max_test_len = max(1, int(word_count * max_input_ratio))

        for size in range(1, max_test_len + 1):
            test_data.append(
                {
                    "Loai": "Start",
                    "Input Len": size,
                    "Input": " ".join(words[:size]),
                    "Ground_Truth": ground_truth,
                }
            )

        for size in range(1, max_test_len + 1):
            test_data.append(
                {
                    "Loai": "End",
                    "Input Len": size,
                    "Input": " ".join(words[-size:]),
                    "Ground_Truth": ground_truth,
                }
            )

        if word_count >= 5:
            for size in range(1, max_test_len + 1):
                for start_index in range(1, word_count - size):
                    test_data.append(
                        {
                            "Loai": "Mid",
                            "Input Len": size,
                            "Input": " ".join(words[start_index : start_index + size]),
                            "Ground_Truth": ground_truth,
                        }
                    )

    df = pd.DataFrame(test_data)
    print(f"Da tao {len(df)} mau test cases tu {len(lines)} cau goc.")
    return df


def evaluate_full_dataset(
    generator: BidirectionalBeamGenerator,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    print(f"\nBAT DAU DANH GIA TREN {len(test_df)} MAU...")

    results: list[dict[str, str | int | float]] = []
    correct_count = 0
    total_similarity = 0.0
    start_time = time.time()

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        user_input = row["Input"]
        truth = row["Ground_Truth"]

        try:
            gen_output = generator.generate_best_cases(user_input, num_results=1)
            if not gen_output:
                pred_clean = ""
                score = -999.0
            else:
                score, formatted_text = gen_output[0]
                pred_clean = formatted_text.replace("\n", " ").lower()
                pred_clean = pred_clean.replace(".", "").replace(",", "").strip()
                pred_clean = ViTokenizer.tokenize(pred_clean)
        except Exception as error:
            pred_clean = f"Error: {error}"
            score = -999.0

        truth_clean = str(truth).replace(".", "").strip()
        is_exact = 1 if pred_clean == truth_clean else 0
        similarity = difflib.SequenceMatcher(None, pred_clean, truth_clean).ratio()

        correct_count += is_exact
        total_similarity += similarity

        results.append(
            {
                "Loai": row["Loai"],
                "Input Len": row["Input Len"],
                "Input": user_input,
                "Ket qua Gen": pred_clean,
                "Dap an Goc": truth_clean,
                "Diem Model": score,
                "Dung": is_exact,
                "Do giong": similarity,
            }
        )

    total_time = time.time() - start_time
    res_df = pd.DataFrame(results)

    sample_count = len(test_df)
    accuracy = (correct_count / sample_count * 100) if sample_count else 0.0
    avg_similarity = (total_similarity / sample_count * 100) if sample_count else 0.0

    print("\n" + "=" * 60)
    print("BAO CAO HIEU NANG TOAN DIEN")
    print("=" * 60)
    if sample_count:
        print(f"Thoi gian: {total_time:.2f}s ({total_time / sample_count * 1000:.1f} ms/cau)")
    else:
        print(f"Thoi gian: {total_time:.2f}s")
    print(f"Do chinh xac tuyet doi (Exact Match): {accuracy:.2f}%")
    print(f"Do tuong dong trung binh (Similarity): {avg_similarity:.2f}%")

    if not res_df.empty:
        print("-" * 60)
        print("1. PHAN TICH THEO VI TRI (Start/Mid/End):")
        print((res_df.groupby("Loai")[["Dung", "Do giong"]].mean() * 100).round(2))

        print("-" * 60)
        print("2. PHAN TICH THEO DO DAI INPUT (10 do dai dau):")
        print((res_df.groupby("Input Len")[["Dung", "Do giong"]].mean().head(10) * 100).round(2))

    print("=" * 60)
    return res_df


def evaluate_pipeline(
    model_file: str | Path,
    data_file: str | Path,
    raw_dataset: str | Path,
    n_gram_order: int = 5,
    max_input_ratio: float = 0.7,
) -> pd.DataFrame:
    generator = BidirectionalBeamGenerator(model_file, data_file, n_gram_order=n_gram_order)
    full_test_df = create_full_stress_test(raw_dataset, max_input_ratio=max_input_ratio)
    return evaluate_full_dataset(generator, full_test_df)
