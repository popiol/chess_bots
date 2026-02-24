from __future__ import annotations

import logging
import random
from typing import List

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
        # preference ratio in [0,1] used for any probabilistic selection
        self._stockfish_ratio: float | None = None

    def snapshot_state(self) -> dict:
        state = super().snapshot_state()
        state["stockfish_ratio"] = self._stockfish_ratio
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        if "stockfish_ratio" in state:
            self._stockfish_ratio = state["stockfish_ratio"]
        else:
            self._stockfish_ratio = random.random()

    def _predict(self, fen: str, our_squares: List[str]) -> List[PredictionResult]:
        assert self._stockfish_ratio is not None

        use_stockfish = random.random() < self._stockfish_ratio

        if use_stockfish:
            if self._stockfish_agent is None:
                self._stockfish_agent = StockfishAgent(
                    self.username,
                    self.password,
                    self.email,
                    self.classpath,
                    self._chess_client,
                )
            logger.debug(
                "ComboAgent: selected Stockfish (ratio=%.3f)",
                self._stockfish_ratio,
                extra={"username": self.username},
            )
            return self._stockfish_agent._predict(fen, our_squares)

        # Use heuristic agent
        if self._heuristic_agent is None:
            self._heuristic_agent = HeuristicAgent(
                self.username,
                self.password,
                self.email,
                self.classpath,
                self._chess_client,
            )
        logger.debug(
            "ComboAgent: selected Heuristic (ratio=%.3f)",
            self._stockfish_ratio,
            extra={"username": self.username},
        )
        return self._heuristic_agent._predict(fen, our_squares)
