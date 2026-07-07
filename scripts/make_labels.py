from __future__ import annotations

import argparse

from data_utils import label_metadata, make_labels, read_table, write_json, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create offline-only future-return labels.")
    parser.add_argument("--input", required=True, help="Input panel or price table.")
    parser.add_argument("--output", default="data/intermediate/labels_daily.parquet")
    parser.add_argument("--metadata-output", default="data/processed/label_metadata.json")
    args = parser.parse_args()

    frame = read_table(args.input)
    labels = make_labels(frame)
    write_table(labels, args.output)
    write_json(args.metadata_output, label_metadata())
    print(f"wrote labels: {args.output} ({len(labels):,} rows)")


if __name__ == "__main__":
    main()
