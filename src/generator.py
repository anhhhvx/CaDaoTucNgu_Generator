from __future__ import annotations

from pathlib import Path

import kenlm
from pyvi import ViTokenizer


class BidirectionalBeamGenerator:
    def __init__(self, model_path: str | Path, train_data_path: str | Path, n_gram_order: int = 5):
        model_path = Path(model_path)
        train_data_path = Path(train_data_path)

        print("[Beam Search] Dang load mo hinh KenLM...")
        if not model_path.exists():
            raise FileNotFoundError(f"Khong tim thay file mo hinh: {model_path}")

        if not train_data_path.exists():
            raise FileNotFoundError(f"Khong tim thay file du lieu train: {train_data_path}")

        self.model = kenlm.Model(str(model_path))
        self.n_order = n_gram_order

        print("[Beam Search] Dang xay dung ban do tu vung...")
        self.fwd_map: dict[tuple[str, ...], set[str]] = {}
        self.bwd_map: dict[tuple[str, ...], set[str]] = {}

        with train_data_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                words = line.split()
                if len(words) < 2:
                    continue

                for index in range(len(words) - 1):
                    for offset in range(1, self.n_order):
                        if index + offset >= len(words):
                            continue

                        fwd_ctx = tuple(words[index : index + offset])
                        next_word = words[index + offset]
                        self.fwd_map.setdefault(fwd_ctx, set()).add(next_word)

                        prev_word = words[index]
                        bwd_ctx = tuple(words[index + 1 : index + 1 + offset])
                        self.bwd_map.setdefault(bwd_ctx, set()).add(prev_word)

    def expand_backward(self, seed_words: list[str], beam_width: int = 10) -> list[list[str]]:
        current_beams = [list(seed_words)]
        completed_prefixes: list[list[str]] = []

        for _ in range(50):
            next_beams: list[list[str]] = []

            for words in current_beams:
                if words[0] == "<s>":
                    completed_prefixes.append(words)
                    continue

                candidates = None
                search_len = min(len(words), self.n_order - 1)
                for context_size in range(search_len, 0, -1):
                    context = tuple(words[:context_size])
                    if context in self.bwd_map:
                        candidates = self.bwd_map[context]
                        break

                if not candidates:
                    continue

                for prev_word in candidates:
                    next_beams.append([prev_word] + words)

            if not next_beams:
                break

            current_beams = next_beams[: beam_width * 5]

        return completed_prefixes if completed_prefixes else [list(seed_words)]

    def expand_forward_beam(self, prefixes: list[list[str]], beam_width: int = 5) -> list[tuple[str, float]]:
        current_beams: list[tuple[float, list[str]]] = []
        for words in prefixes:
            score = self.model.score(" ".join(words))
            current_beams.append((score, words))

        current_beams.sort(key=lambda item: item[0], reverse=True)
        current_beams = current_beams[:beam_width]

        final_results: list[tuple[float, list[str]]] = []

        for _ in range(50):
            next_beams: list[tuple[float, list[str]]] = []
            all_ended = True

            for score, words in current_beams:
                if words[-1] == "</s>":
                    final_results.append((score, words))
                    continue

                all_ended = False
                candidates = None
                search_len = min(len(words), self.n_order - 1)
                for context_size in range(search_len, 0, -1):
                    context = tuple(words[-context_size:])
                    if context in self.fwd_map:
                        candidates = self.fwd_map[context]
                        break

                if not candidates:
                    continue

                for next_word in candidates:
                    new_words = words + [next_word]
                    new_score = self.model.score(" ".join(new_words))
                    next_beams.append((new_score, new_words))

            if all_ended or not next_beams:
                break

            next_beams.sort(key=lambda item: item[0], reverse=True)
            current_beams = next_beams[:beam_width]

        final_results.extend(current_beams)

        unique_map: dict[str, float] = {}
        for score, words in final_results:
            text = " ".join(words)
            if text not in unique_map or score > unique_map[text]:
                unique_map[text] = score

        sorted_results = sorted(unique_map.items(), key=lambda item: item[1], reverse=True)
        return sorted_results[:beam_width]

    @staticmethod
    def format_poetry(raw_text: str) -> str:
        clean_text = raw_text.replace("<s>", "").replace("</s>", "").strip()
        text = clean_text.replace("_", " ")
        verses = text.split(".")
        formatted_verses: list[str] = []

        for verse in verses:
            verse = verse.strip()
            if not verse:
                continue
            formatted_verses.append(verse[0].upper() + verse[1:])

        return "\n".join(formatted_verses)

    def generate_best_cases(self, seed_text: str, num_results: int = 5) -> list[tuple[float, str]]:
        tokenized = ViTokenizer.tokenize(seed_text).lower()
        seed_words = tokenized.split()

        print(f"Input: {seed_words}")
        print(f"Dang tim {num_results} ket qua tot nhat theo xac suat...")

        prefixes = self.expand_backward(seed_words, beam_width=num_results * 2)
        results = self.expand_forward_beam(prefixes, beam_width=num_results)

        formatted_results: list[tuple[float, str]] = []
        for text, score in results:
            pretty_text = self.format_poetry(text)
            formatted_results.append((score, pretty_text))

        return formatted_results
