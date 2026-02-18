from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    version: str = '1.0.0'
    dataset: str = 'dataset.csv'
    target: str = 'CTR'
    test_size: float = 0.3
    random_state: int = 42
    iterations: int = 2000
    depth: int = 6
    learning_rate: float = 0.05
    random_seed: int = 42
    verbose: int = 100
    loss_function: str = 'RMSE'
    eval_metric: str = 'RMSE'
    early_stopping_rounds: int = 100

config = Config()