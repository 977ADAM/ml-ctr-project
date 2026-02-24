from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FeatureSchema:
    numerical: List[str]
    categorical: List[str]
    drop_columns: List[str]
    target: str = None
    weight: str = None

FEATURE_SCHEMA = FeatureSchema(
    numerical=[
        "ID кампании",
        "ID баннера",
        "Показы",
    ],
    categorical=[
        "Тип баннера",
        "Тип устройства"
    ],
    drop_columns=[
        "CTR"
    ],
    target="CTR",
    weight="Показы"
)
