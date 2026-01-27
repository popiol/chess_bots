#!/usr/bin/env python3
"""Convert a Parquet shard to CSV in the project's training format.

Usage:
  python scripts/parquet_to_csv.py \
      --input data/train-00000-of-00017.parquet \
      --output data/tactic_evals_part.csv \
      --target 1000000

The script:
- Preserves all rows with the standard starting FEN.
- Uses reservoir sampling to pick the remaining rows up to `target`.
- Writes CSV rows with header: FEN,Evaluation,Move

Expected Parquet schema (reads these columns if present):
- fen (string)
- line (string) - space separated moves, we take the first token
- cp (numeric) engine centipawn score (optional)
- mate (int|null) mate distance (optional)

Evaluation string format in CSV:
- If `mate` is present: "#<signed_int>" e.g. "#+2" or "#-1"
- Otherwise uses `cp` as integer string (e.g. "311").

"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pyarrow.parquet as pq
except Exception:
    print(
        "pyarrow is required to run this script. Install with: pip install pyarrow",
        file=sys.stderr,
    )
    raise

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def row_from_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a record (pyarrow pydict row) into our CSV fields or None if invalid."""
    fen = rec.get("fen")
    if fen is None:
        return None
    fen = str(fen)

    line = rec.get("line")
    if line is None:
        return None
    line = str(line).strip()
    if not line:
        return None

    # pick first move token
    first_tok = line.split()[0]
    if len(first_tok) < 4:
        return None
    move = first_tok[:4]

    mate = rec.get("mate")
    cp = rec.get("cp")

    if mate is not None:
        # ensure integer
        try:
            m = int(mate)
        except Exception:
            return None
        eval_str = f"#{m:+d}"
    elif cp is not None:
        # cp may be float or int
        try:
            cpi = int(cp)
        except Exception:
            # try cast to float then int
            try:
                cpi = int(float(cp))
            except Exception:
                return None
        eval_str = str(cpi)
    else:
        # No evaluation info: skip
        return None

    return {"fen": fen, "eval": eval_str, "move": move}


def iterate_parquet_records(path: Path, columns: Optional[Iterable[str]] = None):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=columns)
        pyd = table.to_pydict()
        # pyd is dict of column -> list
        cols = list(pyd.keys())
        n = len(pyd[cols[0]]) if cols else 0
        for i in range(n):
            rec = {c: pyd[c][i] for c in cols}
            yield rec


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input Parquet file")
    p.add_argument("--output", required=True, help="Output CSV file path")
    p.add_argument("--target", type=int, default=1_000_000, help="Total rows to select")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = p.parse_args(argv)

    inp = Path(args.input)
    out = Path(args.output)
    target = int(args.target)
    seed = int(args.seed)

    if not inp.exists():
        print(f"Input file not found: {inp}", file=sys.stderr)
        return 2

    random.seed(seed)

    # Single sequential pass: write the first `target` valid records.
    print(
        f"Writing up to {target} valid rows from {inp} to {out} (first-pass streaming)"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FEN", "Evaluation", "Move"])

        for rec in iterate_parquet_records(inp, columns=["fen", "line", "cp", "mate"]):
            row = row_from_record(rec)
            if row is None:
                continue
            w.writerow([row["fen"], row["eval"], row["move"]])
            written += 1
            if written >= target:
                break

    print(f"Done. Wrote {written} rows to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
