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


class PlayableAgent(CustomizableAgent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fen_before_move: str | None = None
        self._last_from_square: str | None = None
        self._last_to_square: str | None = None
        self._player_color: str | None = None

    def _step_playing(self) -> None:
        if self._web_client.is_postgame_visible():
            logger.info(
                "Postgame visible, ending session",
                extra={"username": self.username},
            )
            self._stage = "done"
            return

        # Determine player color once
        if self._player_color is None:
            self._player_color = self._web_client.get_player_color()
            if self._player_color:
                logger.info(
                    "Playing as %s",
                    self._player_color,
                    extra={"username": self.username},
                )

        # Get current position
        try:
            current_fen = self._web_client.get_current_fen()
        except Exception:
            logger.warning(
                "Failed to get FEN, trying again",
                extra={"username": self.username},
            )
            return

        # If we have a pending move verification, check if FEN changed
        if self._fen_before_move is not None:
            if current_fen != self._fen_before_move:
                logger.info(
                    "Move successful %s -> %s",
                    self._last_from_square,
                    self._last_to_square,
                    extra={"username": self.username},
                )
            else:
                logger.info(
                    "Move failed (FEN unchanged) %s -> %s",
                    self._last_from_square,
                    self._last_to_square,
                    extra={"username": self.username},
                )
            self._fen_before_move = None

        # Check if it's our turn
        if not self._web_client.is_current_user_turn():
            return

        # Get squares with our pieces
        our_squares = self._get_our_piece_squares(current_fen)
        if not our_squares:
            logger.warning(
                "No pieces found for our color", extra={"username": self.username}
            )
            return

        # Make random move
        try:
            from_square = random.choice(our_squares)
            to_square = random.choice(ALL_SQUARES)
            if from_square != to_square:
                logger.info(
                    "Attempting move %s -> %s",
                    from_square,
                    to_square,
                    extra={"username": self.username},
                )
                self._fen_before_move = current_fen
                self._last_from_square = from_square
                self._last_to_square = to_square
                self._web_client.make_move(from_square, to_square)
        except PlaywrightTimeoutError:
            logger.warning(
                "Move timeout (likely illegal), trying again",
                extra={"username": self.username},
            )
            self._fen_before_move = None

    def _get_our_piece_squares(self, fen: str) -> list[str]:
        """Extract squares that have our pieces based on FEN.

        Args:
            fen: FEN string representing current position

        Returns:
            List of square coordinates with our pieces (e.g. ['e2', 'g1'])
        """
        if self._player_color is None:
            return []

        # Parse FEN to get board position (first part before space)
        board_part = fen.split()[0]
        rows = board_part.split("/")

        our_squares = []

        # Iterate through ranks (8 to 1)
        for rank_idx, row in enumerate(rows):
            rank = 8 - rank_idx  # rank 8, 7, 6, ..., 1
            file_idx = 0  # 0=a, 1=b, ..., 7=h

            for char in row:
                if char.isdigit():
                    # Empty squares
                    file_idx += int(char)
                else:
                    # Piece found
                    is_white_piece = char.isupper()

                    # Check if this piece is ours
                    if (self._player_color == "white" and is_white_piece) or (
                        self._player_color == "black" and not is_white_piece
                    ):
                        file = FILES[file_idx]
                        square = f"{file}{rank}"
                        our_squares.append(square)

                    file_idx += 1

        return our_squares
