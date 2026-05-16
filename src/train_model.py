from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _resolve_binary(binary_name: str, kenlm_bin_dir: str | Path | None = None) -> str:
    if kenlm_bin_dir:
        candidate = Path(kenlm_bin_dir) / binary_name
        if os.name == "nt" and candidate.with_suffix(".exe").exists():
            return str(candidate.with_suffix(".exe"))
        if candidate.exists():
            return str(candidate)

    found = shutil.which(binary_name)
    if found:
        return found

    if os.name == "nt":
        found = shutil.which(f"{binary_name}.exe")
        if found:
            return found

    raise FileNotFoundError(
        f"Khong tim thay binary '{binary_name}'. "
        "Hay cai KenLM binary va them vao PATH, "
        "hoac truyen --kenlm-bin-dir."
    )


def train_kenlm_model(
    input_file: str | Path,
    arpa_file: str | Path,
    binary_file: str | Path,
    order: int = 5,
    kenlm_bin_dir: str | Path | None = None,
) -> Path:
    input_file = Path(input_file)
    arpa_file = Path(arpa_file)
    binary_file = Path(binary_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Khong tim thay file train: {input_file}")

    lmplz_bin = _resolve_binary("lmplz", kenlm_bin_dir)
    build_binary_bin = _resolve_binary("build_binary", kenlm_bin_dir)

    arpa_file.parent.mkdir(parents=True, exist_ok=True)
    binary_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dang huan luyen mo hinh {order}-gram tu file: {input_file}")

    subprocess.run(
        [
            lmplz_bin,
            "-o",
            str(order),
            "--text",
            str(input_file),
            "--arpa",
            str(arpa_file),
            "--discount_fallback",
        ],
        check=True,
    )

    print("Dang nen sang binary...")

    subprocess.run(
        [build_binary_bin, str(arpa_file), str(binary_file)],
        check=True,
    )

    if not binary_file.exists():
        raise RuntimeError("Train xong nhung khong tao duoc file model.bin")

    print(f"Thanh cong. Model binary da luu tai: {binary_file}")
    return binary_file
