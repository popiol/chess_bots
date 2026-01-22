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
        self._resign_threshold: float = -0.99
        self._draw_threshold: float = 0.01
        self._last_decisive: float | None = None
        self._time_remaining: int | None = None
        self._expected_total_moves: int = 20  # Expected moves per player
        self._allocated_time: int | None = None
        self._moves_made: int = 0  # Counter for moves we've made

    def _step_playing(self) -> None:
        if self._web_client.is_postgame_visible():
            result = self._web_client.get_game_result()
            reason = self._web_client.get_game_reason()

            # Convert result to score: 1 (win), 0 (draw), -1 (loss)
            score = None
            if result and self._player_color:
                if result == "Draw":
                    score = 0
                elif (result == "White wins" and self._player_color == "white") or (
                    result == "Black wins" and self._player_color == "black"
                ):
                    score = 1
                else:
                    score = -1

            logger.info(
                "Postgame visible: %s by %s (score: %s), ending session",
                result,
                reason,
                score,
                extra={"username": self.username},
            )
            self._on_game_end(score, reason)
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
                # Increment our move counter
                self._moves_made += 1
            else:
                logger.info(
                    "Move failed (FEN unchanged) %s -> %s",
                    self._last_from_square,
                    self._last_to_square,
                    extra={"username": self.username},
                )
            self._fen_before_move = None

        # Check if we should offer draw based on last move's decisive value (on opponent's turn)
        if not self._web_client.is_current_user_turn():
            if (
                self._last_decisive is not None
                and self._last_decisive < self._draw_threshold
            ):
                logger.info(
                    "Position decisive %.2f below threshold %.2f, offering draw on opponent's turn",
                    self._last_decisive,
                    self._draw_threshold,
                    extra={"username": self.username},
                )
                self._decision = "offer_draw"
                self._last_decisive = None
            return

        # It's our turn - get time remaining
        self._time_remaining = self._web_client.get_time_remaining()
        if self._time_remaining:
            # Calculate time allocation based on moves made so far
            # Keep a buffer to avoid over-allocating time near the end
            expected_moves_remaining = max(
                11, self._expected_total_moves - self._moves_made
            )
            self._allocated_time = round(
                self._time_remaining / expected_moves_remaining
            )

            logger.debug(
                "Time remaining: %d seconds, move %d, allocated: %d seconds",
                self._time_remaining,
                self._moves_made + 1,
                self._allocated_time,
                extra={"username": self.username},
            )
        else:
            self._allocated_time = None

        # Decide which move to make
        move = self._decide_move(current_fen)
        if move is None:
            return

        from_square, to_square, evaluation, decisive = move

        # Check if position is too bad and we should resign
        if evaluation < self._resign_threshold:
            logger.info(
                "Position evaluation %.2f below threshold %.2f, resigning",
                evaluation,
                self._resign_threshold,
                extra={"username": self.username},
            )
            self._decision = "resign"
            return

        # Check if opponent offered draw and we should accept based on decisive value
        if (
            decisive < self._draw_threshold
            and self._web_client.is_accept_draw_visible()
        ):
            logger.info(
                "Position decisive %.2f below threshold %.2f and draw offered, accepting draw",
                decisive,
                self._draw_threshold,
                extra={"username": self.username},
            )
            self._decision = "accept_draw"
            return

        # Store decisive value to potentially offer draw on opponent's turn
        self._last_decisive = decisive

        # Make move
        try:
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

    def _decide_move(self, current_fen: str) -> tuple[str, str, float, float] | None:
        """Decide which move to make.

        Args:
            current_fen: Current FEN position

        Returns:
            Tuple of (from_square, to_square, evaluation, decisive) or None if no move chosen.
            - evaluation: float from -1 (bad for us) to 1 (good for us)
            - decisive: float from 0 (drawish) to 1 (sharp/tactical, likely decisive result)
        """
        # Get squares with our pieces
        our_squares = self._get_our_piece_squares(current_fen)
        if not our_squares:
            logger.warning(
                "No pieces found for our color", extra={"username": self.username}
            )
            return None

        from_square = random.choice(our_squares)
        to_square = random.choice(ALL_SQUARES)
        if from_square == to_square:
            return None

        # Random evaluation between -1 and 1
        evaluation = random.uniform(-1.0, 1.0)

        # Random decisive result chances between 0 and 1
        decisive = random.uniform(0.0, 1.0)

        return (from_square, to_square, evaluation, decisive)

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

    def _on_game_end(self, score: int | None, reason: str | None) -> None:
        """Called when the game ends (postgame panel visible).

        Args:
            score: Game outcome - 1 (win), 0 (draw), -1 (loss), or None if unknown
            reason: Termination reason in lowercase with underscores:
                "checkmate", "timeout", "resignation", "stalemate",
                "insufficient_material", "threefold_repetition",
                "fifty_move_rule", or "agreement"
        """
        pass
