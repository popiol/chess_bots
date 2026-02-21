from __future__ import annotations

import logging
import random
import time
from typing import List

import chess
import numpy as np
from stockfish import Stockfish

from src.agents.trainable_agent import PredictionResult, TrainableAgent

logger = logging.getLogger(__name__)


# Module-level shared Stockfish instance (lazy-initialized).
_SHARED_STOCKFISH: Stockfish | None = None


def _get_shared_stockfish() -> Stockfish:
    global _SHARED_STOCKFISH
    if _SHARED_STOCKFISH is None:
        _SHARED_STOCKFISH = Stockfish(
            path="stockfish", parameters={"Hash": 8, "Threads": 1}
        )
        _SHARED_STOCKFISH.set_depth(1)
    return _SHARED_STOCKFISH


def shutdown_shared_stockfish() -> None:
    """Terminate and drop the shared Stockfish instance so memory is released."""
    global _SHARED_STOCKFISH
    if _SHARED_STOCKFISH is None:
        return
    try:
        _SHARED_STOCKFISH.send_quit_command()
    except Exception:
        logger.exception(
            "Exception while calling send_quit_command on shared Stockfish"
        )
        try:
            proc = _SHARED_STOCKFISH._stockfish
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
        except Exception:
            logger.exception(
                "Exception while terminating internal Stockfish subprocess"
            )
    finally:
        _SHARED_STOCKFISH = None


class StockfishAgent(TrainableAgent):
    """Agent that queries Stockfish to produce move candidates."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sf = _get_shared_stockfish()

    def _predict(self, fen: str, our_squares: List[str]) -> List[PredictionResult]:
        assert our_squares, "Our squares must be provided"

        start = time.time()
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        candidates = []

        eval_count = min(4, len(legal_moves))
        top_uci: set[str] = set()
        self._sf.set_fen_position(fen)
        top = self._sf.get_top_moves(eval_count)
        for item in top:
            assert isinstance(item["Move"], str)
            top_uci.add(item["Move"])

        for move in legal_moves:
            uci = move.uci()
            if uci in top_uci:
                # Assign random evaluation to top moves
                eval_val = random.gauss(0.2, 0.2)
                decisive = random.random()
            else:
                eval_val = 0.0
                decisive = 0.0

            candidates.append(
                {
                    "from_sq": uci[0:2],
                    "to_sq": uci[2:4],
                    "evaluation": eval_val,
                    "decisive": decisive,
                }
            )

        # Sort moves by evaluation (descending)
        candidates.sort(key=lambda x: x["evaluation"], reverse=True)

        results: List[PredictionResult] = []
        for c in candidates[: self.prediction_count]:
            results.append(
                PredictionResult(
                    from_sq=c["from_sq"],
                    to_sq=c["to_sq"],
                    evaluation=c["evaluation"],
                    decisive=c["decisive"],
                )
            )

        duration = time.time() - start
        logger.info(
            "StockfishAgent._predict: moves_considered=%d samples=%d moves_returned=%d time=%.3fs",
            len(candidates),
            eval_count,
            len(results),
            duration,
        )

        return results

    def _convert_stockfish_eval(self, raw: int | None) -> float:
        """Convert Stockfish evaluation to -1..1 scale.

        - If raw is None, return 0.0
        - If raw is centipawns (int), map via logarithmic scale clamped to [-1,1]
        - Large mate values are mapped to +/-1.0
        """
        if raw is None:
            return 0.0

        # Logarithmic scaling used to reduce impact of large values
        max_cp = 1000.0
        val = np.sign(raw) * (np.log1p(abs(raw)) / np.log1p(max_cp))
        return float(np.clip(val, -1.0, 1.0))
