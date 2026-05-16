from __future__ import annotations

from pathlib import Path

from pyvi import ViTokenizer


def tokenize_step(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Dang tach tu file: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file input: {input_path}. Hay chay buoc normalize truoc."
        )

    lines = input_path.read_text(encoding="utf-8").splitlines()
    final_data: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        tokenized_line = ViTokenizer.tokenize(line)
        complete_line = f"<s> {tokenized_line} </s>"
        final_data.append(complete_line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(final_data), encoding="utf-8")

    print("Da tach tu va gan the xong.")
    print(f"File san sang train: {output_path}")
    if final_data:
        print("Mau ket qua:")
        print(final_data[0])

    return output_path
