from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_utils import DATE_COLUMN, load_config, read_table, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create chronological source and TTA test splits.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    panel_path = Path(data_cfg["panel_path"])
    if not panel_path.exists():
        raise FileNotFoundError(f"panel does not exist: {panel_path}")
    panel = read_table(panel_path)
    panel[DATE_COLUMN] = pd.to_datetime(panel[DATE_COLUMN])

    source = panel[(panel[DATE_COLUMN] >= data_cfg["source_start"]) & (panel[DATE_COLUMN] <= data_cfg["source_end"])]
    test = panel[(panel[DATE_COLUMN] >= data_cfg["test_start"]) & (panel[DATE_COLUMN] <= data_cfg["test_end"])]

    processed_dir = panel_path.parent
    source_path = processed_dir / "source_train_2015_2019.parquet"
    test_path = processed_dir / "tta_test_2020_2024.parquet"
    write_table(source, source_path)
    write_table(test, test_path)
    print(f"wrote source split: {source_path} ({len(source):,} rows)")
    print(f"wrote TTA split: {test_path} ({len(test):,} rows)")


if __name__ == "__main__":
    main()
