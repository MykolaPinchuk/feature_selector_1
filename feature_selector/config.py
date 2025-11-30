from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SplitConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    fs_holdout_frac: float = 0.25
    fs_eval_sample_size: int = 5000
    train_fs_subsample_size: Optional[int] = 15000

    def validate(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must equal 1.0")
        if not (0.05 <= self.fs_holdout_frac <= 0.5):
            raise ValueError("fs_holdout_frac should be between 0.05 and 0.5")


@dataclass
class ModelConfig:
    max_depth: int = 5
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.9
    reg_lambda: float = 2.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    eta: float = 0.1
    n_estimators: int = 350
    scale_pos_weight: float = 1.0
    auto_scale_pos_weight: bool = False
    early_stopping_rounds: int = 30
    tree_method: str = "hist"
    random_state_base: int = 42
    n_models: int = 3
    n_jobs: int = 4


@dataclass
class PermutationConfig:
    shap_top_k: int = 60
    shap_bottom_k: int = 60
    group_importance_tol: float = 0.0005
    min_positive_delta: float = 0.0005
    importance_norm_epsilon: float = 1e-6
    must_keep_threshold: float = 0.25
    must_keep_top_n: int = 20


@dataclass
class FeatureSelectorConfig:
    target_col: str
    time_col: str
    metric: str = "roc_auc"
    main_metric: str = "roc_auc"
    split_config: SplitConfig = field(default_factory=SplitConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    permutation_config: PermutationConfig = field(default_factory=PermutationConfig)
    leakage_features: Optional[List[str]] = None
    drop_features: Optional[List[str]] = None
    target_binarize_threshold: Optional[float] = None
    missingness_threshold: float = 0.99
    quasi_constant_threshold: float = 0.995
    fs_eval_sample_size: Optional[int] = None
    random_seed: int = 13
    dataset_name: Optional[str] = None
    extra_params: Dict[str, float] = field(default_factory=dict)

    def resolve_fs_eval_size(self) -> int:
        if self.fs_eval_sample_size is not None:
            return self.fs_eval_sample_size
        return self.split_config.fs_eval_sample_size
