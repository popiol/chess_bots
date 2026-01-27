#!/usr/bin/env python3
"""Convert `data/positions.csv` (lichess-style) to our training CSV format.

Input format (CSV header):
fen,playing,score,mate,depth,game_id,date,time,white,black,white_result,black_result,white_elo,black_elo,opening,time_control,termination

Output format (CSV header):
FEN,Evaluation,Move

Rules:
- Use `mate` if present -> format as "#<signed_int>" (e.g. "#+2" or "#-1").
- Otherwise use `score` (as integer string).
- `playing` is the move string; we take the first token and its first 4 chars (drop promotion suffix).
- Skip rows missing fen or evaluation or move.

Usage:
    python scripts/positions_to_csv.py --input data/positions.csv --output data/tactic_evals_from_positions.csv --limit 1000000

"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def normalize_evaluation(score: Optional[str], mate: Optional[str]) -> Optional[str]:
    if mate is not None and str(mate).strip() != "":
        try:
            m = int(mate)
        except Exception:
            return None
        return f"#{m:+d}"
    if score is not None and str(score).strip() != "":
        s = str(score).strip()
        # try parse as float/int then return int string
        try:
            si = int(float(s))
        except Exception:
            return None
        return str(si)
    return None


def normalize_move(playing: Optional[str]) -> Optional[str]:
    if not playing:
        return None
    token = str(playing).strip().split()[0]
    if len(token) < 4:
        return None
    return token[:4]


def convert(input_path: Path, output_path: Path, limit: Optional[int] = None) -> int:
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8", newline="") as inf,
        output_path.open("w", encoding="utf-8", newline="") as outf,
    ):
        reader = csv.DictReader(inf)
        writer = csv.writer(outf)
        writer.writerow(["FEN", "Evaluation", "Move"])  # header

        for row in reader:
            fen = row.get("fen")
            playing = row.get("playing") or row.get("move")
            score = row.get("score") or row.get("cp")
            mate = row.get("mate")

            if not fen:
                continue
            eval_str = normalize_evaluation(score, mate)
            if eval_str is None:
                continue
            move = normalize_move(playing)
            if move is None:
                continue

            writer.writerow([fen, eval_str, move])
            written += 1
            if limit is not None and written >= limit:
                break

    return written


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input positions CSV file")
    p.add_argument("--output", required=True, help="Output CSV file path")
    p.add_argument("--limit", type=int, default=None, help="Maximum rows to write")
    args = p.parse_args(argv)

    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        print(f"Input file not found: {inp}", file=sys.stderr)
        return 2

    n = convert(inp, out, limit=args.limit)
    print(f"Wrote {n} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
