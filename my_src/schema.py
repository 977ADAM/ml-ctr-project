from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class FeatureSchema:
    categorical: List[str]
    drop_columns: List[str]

FEATURE_SCHEMA = FeatureSchema(
    categorical=[
        "Тип баннера",
        "Тип устройства"
    ],
    drop_columns=[
        "ID кампании",
        "ID баннера"
    ]
)