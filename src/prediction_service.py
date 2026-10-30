"""Student Performance Prediction Service for EduPredict.


- Loads the 3 saved model pipelines (DecisionTree, RandomForestClassifier, RandomForestRegressor).
- Uses the embedded scikit-learn ColumnTransformer preprocessing in each artifact.
- Does not fit, retrain, or tune models during prediction.
- Validates that all 30 non-grade predictor features exist and have valid types.
- Strictly strips out and ignores target leakage features ('G1', 'G2', 'G3', 'at_risk')
  and database metadata ('_id', 'source_row_id', 'dataset_source').
- Generates:
    1. decision_tree_at_risk (0 or 1)
    2. random_forest_at_risk (0 or 1)
    3. random_forest_risk_probability (float 0.0-1.0)
    4. random_forest_estimated_g3 (float 0.0-20.0)
- Converts all NumPy scalars to standard Python primitives with timezone-aware UTC datetime.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FROM_FEATURES,
    NUMERIC_FEATURES,
)

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH_DT: Path = MODELS_DIR / "decision_tree_classifier.joblib"
MODEL_PATH_RF_CLF: Path = MODELS_DIR / "random_forest_classifier.joblib"
MODEL_PATH_RF_REG: Path = MODELS_DIR / "random_forest_regressor.joblib"

# Ordered list of all 30 predictor features
ALL_PREDICTOR_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class PredictionService:
    """Production prediction service for scoring student risk and performance."""

    def __init__(
        self,
        dt_path: Path = MODEL_PATH_DT,
        rf_clf_path: Path = MODEL_PATH_RF_CLF,
        rf_reg_path: Path = MODEL_PATH_RF_REG,
        model_version: str = "v1",
    ) -> None:
        """Initialize the service by loading fitted pipelines from disk.

        Args:
            dt_path: Path to saved DecisionTreeClassifier pipeline.
            rf_clf_path: Path to saved RandomForestClassifier pipeline.
            rf_reg_path: Path to saved RandomForestRegressor pipeline.
            model_version: String tag for model versioning (default: 'v1').

        Raises:
            FileNotFoundError: If any of the required joblib model files are missing.
        """
        for path, name in [
            (dt_path, "Decision Tree Classifier"),
            (rf_clf_path, "Random Forest Classifier"),
            (rf_reg_path, "Random Forest Regressor"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required model artifact '{name}' not found at: {path.resolve()}\n"
                    "Please ensure Phase 4 training has been run: python -m src.train_models"
                )

        self.model_version = model_version
        self.dt_path = dt_path
        self.rf_clf_path = rf_clf_path
        self.rf_reg_path = rf_reg_path

        # Load fitted model pipelines (including ColumnTransformer preprocessors)
        self.dt_pipeline = joblib.load(dt_path)
        self.rf_clf_pipeline = joblib.load(rf_clf_path)
        self.rf_reg_pipeline = joblib.load(rf_reg_path)

        self.artifact_names = [
            dt_path.name,
            rf_clf_path.name,
            rf_reg_path.name,
        ]

    def validate_and_clean_features(
        self,
        raw_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate input features and sanitize types for model inference.

        Ensures:
          1. All 30 required predictor features are present.
          2. Numeric features are convertible to float/int.
          3. Categorical features are strings.
          4. Leaked columns ('G1', 'G2', 'G3', 'at_risk') and database identifiers
             are never passed to model pipelines.

        Args:
            raw_record: Raw student dictionary (e.g. from MongoDB or API input).

        Returns:
            Dictionary containing exclusively the 30 validated predictor features.

        Raises:
            ValueError: If required features are missing or contain corrupt non-numeric data.
        """
        cleaned_features: Dict[str, Any] = {}
        missing_features: List[str] = []

        # 1. Check numeric features
        for num_col in NUMERIC_FEATURES:
            if num_col not in raw_record:
                missing_features.append(num_col)
                continue

            val = raw_record[num_col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cleaned_features[num_col] = np.nan  # Pipeline imputer will handle NaN with median
            else:
                try:
                    cleaned_features[num_col] = float(val)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Invalid numeric value for feature '{num_col}': {val} ({type(val).__name__})"
                    ) from exc

        # 2. Check categorical features
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col not in raw_record:
                missing_features.append(cat_col)
                continue

            val = raw_record[cat_col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                cleaned_features[cat_col] = None  # Pipeline imputer will handle with most_frequent
            else:
                cleaned_features[cat_col] = str(val).strip()

        if missing_features:
            raise ValueError(
                f"Student record is missing {len(missing_features)} required predictor features: {missing_features}"
            )

        return cleaned_features

    def predict_single(
        self,
        raw_record: Dict[str, Any],
        student_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate risk classifications, risk probability, and estimated final grade.

        Args:
            raw_record: Input dictionary containing student demographic & academic factors.
            student_id: Optional unique identifier string. If None, derived from raw_record.

        Returns:
            Structured prediction dictionary matching project schema with native Python types.
        """
        # Resolve student identifier
        sid = (
            student_id
            or str(raw_record.get("_id", ""))
            or f"student_{raw_record.get('source_row_id', 'unknown')}"
        )
        source_row_id = raw_record.get("source_row_id")

        # 1. Validate features (excludes G1, G2, G3, at_risk, _id)
        cleaned_dict = self.validate_and_clean_features(raw_record)

        # 2. Construct 1-row DataFrame in the exact feature order
        feature_df = pd.DataFrame([cleaned_dict])[ALL_PREDICTOR_FEATURES]

        # 3. Model 1: Decision Tree Classifier
        dt_pred_raw = self.dt_pipeline.predict(feature_df)[0]
        dt_at_risk = int(dt_pred_raw)

        # 4. Model 2: Random Forest Classifier + Risk Probability
        rf_pred_raw = self.rf_clf_pipeline.predict(feature_df)[0]
        rf_at_risk = int(rf_pred_raw)

        rf_proba_raw = self.rf_clf_pipeline.predict_proba(feature_df)[0]
        # Class 1 is at-risk
        rf_risk_prob = float(rf_proba_raw[1]) if len(rf_proba_raw) > 1 else float(rf_pred_raw)

        # 5. Model 3: Random Forest Regressor (Estimated G3)
        rf_reg_raw = self.rf_reg_pipeline.predict(feature_df)[0]
        estimated_g3 = round(float(rf_reg_raw), 2)
        # Clamp estimated grade to valid 0-20 scale
        estimated_g3 = max(0.0, min(20.0, estimated_g3))

        # 6. Build structured output with timezone-aware UTC datetime
        prediction_document: Dict[str, Any] = {
            "student_id": sid,
            "model_version": self.model_version,
            "predictions": {
                "decision_tree_at_risk": dt_at_risk,
                "random_forest_at_risk": rf_at_risk,
                "random_forest_risk_probability": round(rf_risk_prob, 4),
                "random_forest_estimated_g3": estimated_g3,
            },
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "metadata": {
                "models_used": self.artifact_names,
                "features_evaluated": len(ALL_PREDICTOR_FEATURES),
                "is_synthetic": True,
                "note": "Estimates based on non-grade behavioral/demographic features.",
            },
        }

        if source_row_id is not None:
            try:
                prediction_document["source_row_id"] = int(source_row_id)
            except (ValueError, TypeError):
                pass

        return prediction_document

    def predict_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process a list of student records individually, isolating failures.

        Args:
            records: List of raw student documents.

        Returns:
            Tuple of:
              - List of successful prediction documents.
              - List of error records containing {'student_id': ..., 'error': ...}.
        """
        successful: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for record in records:
            sid = str(record.get("_id", f"row_{record.get('source_row_id', 'unknown')}"))
            try:
                pred_doc = self.predict_single(record, student_id=sid)
                successful.append(pred_doc)
            except Exception as exc:
                errors.append(
                    {
                        "student_id": sid,
                        "source_row_id": record.get("source_row_id"),
                        "error": str(exc),
                    }
                )

        return successful, errors
