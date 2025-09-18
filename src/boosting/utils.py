import os
import glob
import json
import joblib
from typing import Tuple, Any, Dict

MODELS_DIR_NAME = "models"
MODEL_PREFIX = "boosting_model_"
METADATA_PREFIX = "boosting_metadata_"


def load_latest_model(
    project_root: str | None = None,
) -> Tuple[Any, Dict[str, Any], float]:
    """Carrega o modelo .joblib mais recente e metadata associada.

    Retorna (model, metadata_dict, used_threshold).
    Levanta FileNotFoundError se nada encontrado.
    """
    if project_root is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
    models_dir = os.path.join(project_root, MODELS_DIR_NAME)
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Diretório de modelos não encontrado: {models_dir}")

    model_paths = sorted(glob.glob(os.path.join(models_dir, f"{MODEL_PREFIX}*.joblib")))
    if not model_paths:
        raise FileNotFoundError("Nenhum modelo encontrado.")
    latest_model = model_paths[-1]

    # infer timestamp token
    ts = os.path.basename(latest_model).replace(MODEL_PREFIX, "").replace(".joblib", "")
    metadata_path = os.path.join(models_dir, f"{METADATA_PREFIX}{ts}.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    model = joblib.load(latest_model)
    threshold = metadata.get("used_threshold", 0.5)
    return model, metadata, threshold


__all__ = ["load_latest_model"]
