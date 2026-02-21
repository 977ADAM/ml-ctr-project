# cli_args.py
from __future__ import annotations
import argparse
from pathlib import Path


class CTRTrainCLI:
    """
    Класс для парсинга аргументов командной строки
    для обучения CTR модели.
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="Train a click probability (CTR) model on dataset.csv"
        )
        self._add_arguments()

    def _add_arguments(self) -> None:
        """Добавление аргументов в парсер."""
        self.parser.add_argument("--data", type=Path, default=Path("data/dataset.csv"))
        self.parser.add_argument("--model-dir", type=Path, default=Path("models"))
        self.parser.add_argument("--model-name", type=str, default="ctr_model.cbm")
        self.parser.add_argument("--target", type=str, default="CTR")
        self.parser.add_argument("--random-state", type=int, default=42)

    def parse(self) -> argparse.Namespace:
        """Парсинг аргументов."""
        return self.parser.parse_args()
    
# Пример использования:
# python3 src/train.py --data data/dataset.csv --model-dir models --model-name ctr_model.cbm --target CTR
