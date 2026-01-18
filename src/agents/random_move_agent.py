from __future__ import annotations

import logging
import random

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from src.agents.customizable_agent import CustomizableAgent

logger = logging.getLogger(__name__)

# All possible squares on a chess board
FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"]
ALL_SQUARES = [f + r for f in FILES for r in RANKS]


class RandomMoveAgent(CustomizableAgent):
    def _step_playing(self) -> None:
        if self._web_client.is_postgame_visible():
            logger.info(
                "Postgame visible, ending session",
                extra={"username": self.username},
            )
            self._stage = "done"
            return

        # Make random moves
        try:
            from_square = random.choice(ALL_SQUARES)
            to_square = random.choice(ALL_SQUARES)
            if from_square != to_square:
                logger.info(
                    "Making move %s -> %s",
                    from_square,
                    to_square,
                    extra={"username": self.username},
                )
                self._web_client.make_move(from_square, to_square)
        except PlaywrightTimeoutError:
            logger.warning(
                "Move failed (likely illegal), trying again",
                extra={"username": self.username},
            )
