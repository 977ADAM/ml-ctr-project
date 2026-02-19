from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class FeatureSchema:
    numerical: List[str]
    categorical: List[str]
    drop_columns: List[str]

FEATURE_SCHEMA = FeatureSchema(
    numerical=[
        "Показы",
    ],
    categorical=[
        "Тип баннера",
        "Тип устройства"
    ],
    drop_columns=[
        "ID кампании",
        "ID баннера",
    ]
)