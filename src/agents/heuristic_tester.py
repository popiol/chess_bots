from __future__ import annotations

import argparse
from typing import Any, Dict

import chess

from .heuristic_evaluator import HeuristicEvaluator


class HeuristicTester:
    """Utility to collect and display all heuristic metrics for a position."""

    def __init__(self) -> None:
        self.evaluator = HeuristicEvaluator()

    def metrics_for_fen(self, fen: str) -> Dict[str, Any]:
        board = chess.Board(fen)
        is_white = bool(board.turn)
        out: Dict[str, Any] = {}

        # individual metric helpers
        out["material"] = self.evaluator._material_eval(board, is_white)
        out["mobility"] = self.evaluator._mobility_eval(board, is_white)
        out["safe_mobility"] = self.evaluator._safe_mobility_eval(board, is_white)
        our_pe, opp_pe = self.evaluator._piece_exposed_eval(board, is_white)
        out["our_piece_exposed"] = our_pe
        out["opp_piece_exposed"] = opp_pe
        mate_against_us = self.evaluator._mate_in_one_eval(board, is_white)
        out["mate_against_us"] = mate_against_us
        out["king_safety"] = self.evaluator._king_safety_eval(board, is_white)
        out["castling"] = self.evaluator._castling_bonus(board, is_white)
        out["check"] = self.evaluator._check_eval(board, is_white)
        our_att, opp_att = self.evaluator._profitable_attack_eval(board, is_white)
        out["our_profitable_attack"] = our_att
        out["opp_profitable_attack"] = opp_att
        out["center_control"] = self.evaluator._center_control_eval(board, is_white)
        out["undeveloped"] = self.evaluator._undeveloped_pieces_eval(board, is_white)
        out["doubled_pawns"] = self.evaluator._doubled_pawns_eval(board, is_white)
        out["isolated_pawns"] = self.evaluator._isolated_pawns_eval(board, is_white)
        out["passed_pawns"] = self.evaluator._passed_pawns_eval(board, is_white)
        out["hanging"] = self.evaluator._hanging_pieces_eval(board, is_white)
        out["bishop_pair"] = self.evaluator._bishop_pair_eval(board, is_white)
        out["rook_open_file"] = self.evaluator._rook_open_file_eval(board, is_white)
        out["endgame_king_activity"] = self.evaluator._endgame_king_activity_eval(
            board, is_white
        )
        out["bishop_activity"] = self.evaluator._bishop_activity_eval(board, is_white)
        out["knight_forks"] = self.evaluator._knight_forks_eval(board, is_white)
        out["outpost_knight"] = self.evaluator._outpost_knight_eval(board, is_white)
        out["space_advantage"] = self.evaluator._space_advantage_eval(board, is_white)
        out["backward_pawns"] = self.evaluator._backward_pawns_eval(board, is_white)
        out["squares_attacked"] = self.evaluator._squares_attacked_eval(board, is_white)
        out["discovered_attacks"] = self.evaluator._discovered_attacks_eval(
            board, is_white
        )
        out["pins"] = self.evaluator._pins_eval(board, is_white)

        # overall evaluation + decisiveness
        eval_val, decisive = self.evaluator.evaluate_position(board, is_white)
        out["eval"] = eval_val
        out["decisive"] = decisive

        return out

    def display(self, fen: str) -> None:
        board = chess.Board(fen)
        is_white = bool(board.turn)
        metrics = self.metrics_for_fen(fen)
        print(f"FEN: {fen}")
        side = "White" if is_white else "Black"
        print(f"Perspective: {side}")

        # Print overall eval and decisiveness prominently
        eval_val = metrics.pop("eval", None)
        decisive = metrics.pop("decisive", None)
        if isinstance(eval_val, float):
            print(f"\nEvaluation: {eval_val:.4f}")
        else:
            print(f"\nEvaluation: {eval_val}")
        if isinstance(decisive, float):
            print(f"Decisive: {decisive:.4f}")
        else:
            print(f"Decisive: {decisive}")

        print("\nMetrics:")
        # stable ordering for remaining metrics
        for k in sorted(metrics.keys()):
            print(
                f" - {k}: {metrics[k]:.4f}"
                if isinstance(metrics[k], float)
                else f" - {k}: {metrics[k]}"
            )

    def display_moves(self, fen: str) -> None:
        """Iterate legal moves and print mover-centric evaluation for each."""
        board = chess.Board(fen)
        mover_is_white = bool(board.turn)

        rows = []
        for mv in board.legal_moves:
            b2 = board.copy()
            try:
                b2.push(mv)
            except Exception:
                continue
            # Evaluate from opponent perspective then invert so positive is good for mover
            eval_val, decisive = self.evaluator.evaluate_position(
                b2, not mover_is_white
            )
            eval_val = -eval_val
            rows.append((mv.uci(), eval_val, decisive))

        # Sort by evaluation descending (best for mover first)
        rows.sort(key=lambda r: r[1], reverse=True)

        print(f"FEN: {fen}")
        side = "White" if mover_is_white else "Black"
        print(f"Mover: {side} (listing moves good for mover first)")
        for uci, ev, dec in rows:
            print(f" - {uci}: eval={ev:.4f} decisive={dec:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Display heuristic metrics for a FEN position"
    )
    p.add_argument(
        "--fen",
        default="rkbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="FEN string of the position",
    )
    p.add_argument(
        "--mode",
        choices=("metrics", "moves"),
        default="metrics",
        help='Mode to display: "metrics" shows metric breakdown, "moves" lists move evaluations',
    )

    args = p.parse_args()

    tester = HeuristicTester()
    if args.mode == "metrics":
        tester.display(args.fen)
    else:
        tester.display_moves(args.fen)


if __name__ == "__main__":
    main()
