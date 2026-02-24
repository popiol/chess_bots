from __future__ import annotations

import logging
from typing import List

import chess

from src.agents.heuristic_agent import HeuristicAgent
from src.agents.stockfish_agent import StockfishAgent
from src.agents.trainable_agent import PredictionResult, TrainableAgent

logger = logging.getLogger(__name__)


class ComboAgent(TrainableAgent):
    """Agent that alternates between heuristic and Stockfish predictions.

    Uses heuristic predictions on every odd move (1st, 3rd, ...) and Stockfish
    predictions on every even move (2nd, 4th, ...).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # delegate agents (lazy-init)
        self._heuristic_agent: HeuristicAgent | None = None
        self._stockfish_agent: StockfishAgent | None = None

    def _predict(self, fen: str, our_squares: List[str]) -> List[PredictionResult]:
        board = chess.Board(fen)
        ply = len(board.move_stack)
        # If ply is even -> it's move 1,3,5... (heuristic); if odd -> stockfish
        use_heuristic = (ply % 2) == 0

        # Lazy init delegate agents and keep prediction_count in sync
        if use_heuristic:
            if self._heuristic_agent is None:
                self._heuristic_agent = HeuristicAgent(
                    self.username,
                    self.password,
                    self.email,
                    self.classpath,
                    self._chess_client,
                )
            return self._heuristic_agent._predict(fen, our_squares)
        else:
            if self._stockfish_agent is None:
                self._stockfish_agent = StockfishAgent(
                    self.username,
                    self.password,
                    self.email,
                    self.classpath,
                    self._chess_client,
                )
            return self._stockfish_agent._predict(fen, our_squares)
