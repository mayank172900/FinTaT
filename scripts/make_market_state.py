from __future__ import annotations

import argparse

from data_utils import compute_market_state, read_table, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create causal daily market-state vectors.")
    parser.add_argument("--panel", required=True, help="Feature panel or price panel.")
    parser.add_argument("--output", default="data/intermediate/market_state_daily.parquet")
    args = parser.parse_args()

    panel = read_table(args.panel)
    state = compute_market_state(panel)
    write_table(state, args.output)
    print(f"wrote market state: {args.output} ({len(state):,} dates)")


if __name__ == "__main__":
    main()
