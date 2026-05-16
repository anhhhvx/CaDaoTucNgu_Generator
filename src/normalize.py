from __future__ import annotations

import re
from pathlib import Path


def normalize_step(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Dang chuan hoa file: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Khong tim thay file dau vao: {input_path}")

    raw_content = input_path.read_text(encoding="utf-8")

    poems = re.split(r"\n\s*\n", raw_content.strip())
    clean_lines: list[str] = []

    for poem in poems:
        verses = poem.strip().splitlines()
        clean_verses = [verse.strip().lower() for verse in verses if verse.strip()]

        if not clean_verses:
            continue

        merged_line = ". ".join(clean_verses)
        if not merged_line.endswith("."):
            merged_line += "."

        while ".." in merged_line:
            merged_line = merged_line.replace("..", ".")

        clean_lines.append(merged_line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(clean_lines), encoding="utf-8")

    print(f"Da chuan hoa {len(clean_lines)} dong.")
    print(f"Luu tai: {output_path}")
    if clean_lines:
        print(f"Mau: {clean_lines[0]}")

    return output_path
