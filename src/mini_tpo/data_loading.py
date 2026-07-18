from pathlib import Path

import pandas as pd
import yaml

from mini_tpo.paths import CONFIG_PATH, PROJECT_ROOT


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_raw_data(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    csv_path = PROJECT_ROOT / cfg["project"]["raw_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontro el CSV raw en {csv_path}. Copia el historico a data/raw/base_mini_tpo.csv."
        )
    return pd.read_csv(csv_path)


def read_clean_full(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    return pd.read_parquet(PROJECT_ROOT / cfg["project"]["interim_full"])


def read_modeling_data(config: dict | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    return pd.read_parquet(PROJECT_ROOT / cfg["project"]["processed_modeling"])

