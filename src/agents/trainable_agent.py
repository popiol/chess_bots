from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import numpy as np

from src.agents.playable_agent import PlayableAgent

logger = logging.getLogger(__name__)

# Piece type encoding (independent of color)
PIECE_TYPE_TO_INDEX = {
    "P": 0,  # Pawn
    "N": 1,  # Knight
    "B": 2,  # Bishop
    "R": 3,  # Rook
    "Q": 4,  # Queen
    "K": 5,  # King
}

# Square name to index mapping (a1=0, b1=1, ..., h8=63)
FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"]


@dataclass
class PredictionResult:
    """Result of a move prediction."""

    from_idx: int  # Source square index (0-63)
    to_idx: int  # Destination square index (0-63)
    evaluation: float  # Position evaluation (-1 to 1)
    decisive: float  # Decisiveness of the move (0 to 1)


@dataclass
class AnalysisNode:
    """Node in the position analysis tree.

    Represents a chess position with predictions and potential continuations.
    """

    features: np.ndarray
    predictions: list[PredictionResult]
    parent: AnalysisNode | None = None
    parent_move: PredictionResult | None = (
        None  # Move from parent that led to this node
    )
    children: dict[tuple[int, int], AnalysisNode] = field(default_factory=dict)
    # children maps (from_square_idx, to_square_idx) -> resulting position node


@dataclass
class Decision:
    """Record of a decision made by the agent."""

    features: np.ndarray
    move: PredictionResult


class TrainableAgent(PlayableAgent):
    """Agent that delegates move decisions to a customizable model.

    The model can be any decision-making system: hardcoded rules, neural networks,
    decision trees, or any other approach. Subclasses should implement _predict().

    This agent tracks all features and decisions for training or analysis.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Track features (encoded FENs) and corresponding decisions
        self._decision_history: list[Decision] = []
        # Current analysis tree root for multi-step position analysis
        self._analysis: AnalysisNode | None = None
        # BFS queue for tree expansion
        self._expansion_queue: list[AnalysisNode] = []
        # Aggression level: -1.0 (defensive/cautious) to 1.0 (aggressive/risky)
        self._aggression: float = 0.0
        # Number of moves to return from _predict
        self.prediction_count: int = 2

    def snapshot_state(self) -> dict:
        state = super().snapshot_state()
        state["aggression"] = self._aggression
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        if "aggression" in state:
            self._aggression = float(state["aggression"])
        else:
            self._aggression = random.uniform(-1.0, 1.0)

    def _decide_move(self, current_fen: str) -> tuple[str, str, float, float] | None:
        """Decide which move to make using model prediction.

        Args:
            current_fen: Current FEN position

        Returns:
            Tuple of (from_square, to_square, evaluation, decisive) or None if no move chosen
        """

        logger.info(
            "Deciding move for FEN: %s", current_fen, extra={"username": self.username}
        )

        # Initialize analysis if not present
        if self._analysis is None:
            features = self._encode_fen(current_fen)
            our_squares = self._extract_our_squares_from_features(features)
            assert our_squares, "No pieces found for our color"
            moves = self._predict(features, our_squares)
            assert moves, "Predict returned no moves"
            self._analysis = AnalysisNode(features=features, predictions=moves)
            self._expansion_queue = [self._analysis]
            logger.debug(
                "Initialized analysis with %d predictions",
                len(moves),
                extra={"username": self.username},
            )

        # If we have less than 3 seconds, return best move immediately
        if self._allocated_time is None or self._allocated_time < 3:
            best_move = max(
                self._analysis.predictions,
                key=lambda m: m.evaluation + self._aggression * m.decisive,
            )
            from_square_idx = best_move.from_idx
            to_square_idx = best_move.to_idx
            evaluation = best_move.evaluation
            decisive = best_move.decisive

            # Track features and best move
            self._decision_history.append(
                Decision(
                    features=self._analysis.features.copy(),
                    move=best_move,
                )
            )

            # Decode square indices back to square names
            from_square = self._index_to_square(from_square_idx)
            to_square = self._index_to_square(to_square_idx)

            # Clear analysis for next position
            self._analysis = None
            self._expansion_queue = []

            logger.debug(
                "Decided move %s->%s (eval: %.2f, decisive: %.2f)",
                from_square,
                to_square,
                evaluation,
                decisive,
                extra={"username": self.username},
            )

            return (from_square, to_square, evaluation, decisive)

        # We have at least 3 seconds - expand the analysis tree
        self._expand_analysis()
        return None

    def _expand_analysis(self) -> None:
        """Expand the analysis tree by one node using BFS.

        Called when we have enough time to do deeper position analysis.
        Processes one node per tick.
        """

        logger.debug(
            "Expanding analysis tree, queue size: %d",
            len(self._expansion_queue),
            extra={"username": self.username},
        )

        if not self._expansion_queue:
            logger.debug(
                "Expansion queue empty, analysis complete",
                extra={"username": self.username},
            )
            return

        # Look at the next node to expand (don't pop yet)
        node = self._expansion_queue[0]

        # Find a move that hasn't been expanded yet
        for move in node.predictions:
            from_idx = move.from_idx
            to_idx = move.to_idx
            move_key = (from_idx, to_idx)

            if move_key in node.children:
                continue

            # This move hasn't been expanded - create child node
            from_square = self._index_to_square(from_idx)
            to_square = self._index_to_square(to_idx)

            # Apply move to features to get new position
            new_features = self._apply_move_to_features(node.features, from_idx, to_idx)
            if new_features is None:
                # Invalid move or not implemented, create node with no predictions
                child_node = AnalysisNode(
                    features=node.features,
                    predictions=[],
                    parent=node,
                    parent_move=move,
                )
                node.children[move_key] = child_node
                # Don't add to expansion queue since there's nothing to expand
                logger.debug(
                    "Move %s->%s invalid, created empty node",
                    from_square,
                    to_square,
                    extra={"username": self.username},
                )
                return

            # Predict for the new position
            # Extract our_squares from features (channels 0-5 contain our pieces)
            new_our_squares = self._extract_our_squares_from_features(new_features)
            new_predictions = self._predict(new_features, new_our_squares)

            # Create child node
            child_node = AnalysisNode(
                features=new_features,
                predictions=new_predictions,
                parent=node,
                parent_move=move,
            )
            node.children[move_key] = child_node

            # Add child to expansion queue for future expansion
            self._expansion_queue.append(child_node)

            # Update parent evaluations up the tree
            self._propagate_evaluation(child_node)

            logger.debug(
                "Expanded move %s->%s, %d predictions, queue: %d",
                from_square,
                to_square,
                len(new_predictions),
                len(self._expansion_queue),
                extra={"username": self.username},
            )

            # Only expand one move per tick, keep node in queue
            return

        # If we get here, all moves for this node have been expanded
        # Remove it from the queue and continue on next tick
        self._expansion_queue.pop(0)
        logger.debug(
            "Node fully expanded, removed from queue, remaining: %d",
            len(self._expansion_queue),
            extra={"username": self.username},
        )

    def _propagate_evaluation(self, node: AnalysisNode) -> None:
        """Propagate evaluation up the tree using minimax.

        After a node gets predictions, update parent nodes with the average child evaluation.
        Uses negamax: parent_eval = -avg_child_eval.

        Args:
            node: The node whose predictions should be propagated to its parents
        """
        current = node
        while current.parent is not None:
            # Calculate average evaluation and decisive from current node's predictions
            if not current.predictions:
                # No predictions, can't propagate
                break

            avg_eval = sum(m.evaluation for m in current.predictions) / len(
                current.predictions
            )
            avg_decisive = sum(m.decisive for m in current.predictions) / len(
                current.predictions
            )

            # Update parent's move that led to current node
            parent = current.parent
            if current.parent_move is None:
                break

            logger.debug(
                "Updating parent move eval: %.2f -> %.2f, decisive: %.2f -> %.2f",
                current.parent_move.evaluation,
                -avg_eval,
                current.parent_move.decisive,
                avg_decisive,
                extra={"username": self.username},
            )

            # Update evaluation (negated) and decisive directly
            current.parent_move.evaluation = -avg_eval
            current.parent_move.decisive = avg_decisive

            # Move up the tree
            current = parent

    def _apply_move_to_features(
        self, features: np.ndarray, from_idx: int, to_idx: int
    ) -> np.ndarray | None:
        """Apply a move to position features and return the resulting features.

        Args:
            features: Current position features (768-dim array)
            from_idx: Source square index (0-63)
            to_idx: Destination square index (0-63)

        Returns:
            New features array after applying the move, or None if move is invalid
        """
        # Copy and reshape to 8x8x12
        board = features.reshape(8, 8, 12).copy()

        # Calculate square positions
        from_rank = from_idx // 8
        from_file = from_idx % 8
        to_rank = to_idx // 8
        to_file = to_idx % 8

        # Check if there's a piece at the source square (in our pieces channels 0-5)
        if not board[from_rank, from_file, 0:6].any():
            # No piece at source square, invalid move
            return None

        # Move the piece: copy our piece from source to destination
        board[to_rank, to_file, 0:6] = board[from_rank, from_file, 0:6]
        # Clear the source square for our pieces
        board[from_rank, from_file, 0:6] = 0
        # Clear any captured opponent piece at destination
        board[to_rank, to_file, 6:12] = 0

        # Switch our/opponent pieces (next move is by opponent)
        # Swap channels 0-5 with channels 6-11
        new_board = board.copy()
        new_board[:, :, 0:6] = board[:, :, 6:12]
        new_board[:, :, 6:12] = board[:, :, 0:6]

        # Flatten back to 768 dimensions
        return new_board.flatten()

    def _predict(
        self, features: np.ndarray, our_squares: list[str]
    ) -> list[PredictionResult]:
        """Predict candidate moves and evaluations using the model.

        Args:
            features: Encoded board features (768-dimensional array from _encode_fen)
            our_squares: List of square names where our pieces are located (e.g., ['e2', 'g1'])

        Returns:
            List of PredictionResult objects containing:
            - from_idx: 0-63 index of source square
            - to_idx: 0-63 index of destination square
            - evaluation: -1 (losing) to 1 (winning)
            - decisive: 0 (unclear) to 1 (decisive)

            The best move (highest evaluation) will be selected by _decide_move.
        """
        assert our_squares, "our_squares cannot be empty"

        # Convert our_squares to indices
        our_square_indices = [self._square_to_index(square) for square in our_squares]

        moves = []
        for _ in range(self.prediction_count):
            # Use valid piece square
            from_square_idx = random.choice(our_square_indices)

            to_square_idx = random.randint(0, 63)
            evaluation = random.uniform(-1.0, 1.0)
            decisive = random.uniform(0.0, 1.0)
            moves.append(
                PredictionResult(
                    from_idx=from_square_idx,
                    to_idx=to_square_idx,
                    evaluation=evaluation,
                    decisive=decisive,
                )
            )

        logger.debug(
            "Predicted %d moves", len(moves), extra={"username": self.username}
        )

        return moves

    def _encode_fen(self, fen: str) -> np.ndarray:
        """Utility method to encode FEN string as numerical features.

        This is provided as a convenience for neural network implementations.
        Models using hardcoded rules don't need to call this method.

        Args:
            fen: FEN string representing current position

        Returns:
            NumPy array of 768 features: 8x8x12 board
            - Indices 0-5: our pieces (pawn, knight, bishop, rook, queen, king)
            - Indices 6-11: opponent pieces (pawn, knight, bishop, rook, queen, king)
        """
        # Parse FEN components
        parts = fen.split()
        if len(parts) < 2:
            raise ValueError(
                f"Invalid FEN string provided (missing active color): {fen}"
            )

        board_part = parts[0]

        # Board representation: 8x8x12 (6 our piece types + 6 opponent piece types)
        board = np.zeros((8, 8, 12), dtype=np.float32)

        # Determine which pieces are ours based on active color in FEN
        # The second field in FEN is active color ('w' or 'b')
        active_color = parts[1]
        our_pieces_are_uppercase = active_color == "w"

        rows = board_part.split("/")
        for rank_idx, row in enumerate(rows):
            file_idx = 0
            for char in row:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    piece_type = char.upper()
                    if piece_type in PIECE_TYPE_TO_INDEX:
                        piece_type_idx = PIECE_TYPE_TO_INDEX[piece_type]

                        # Determine if this is our piece or opponent's piece
                        is_uppercase = char.isupper()
                        is_our_piece = is_uppercase == our_pieces_are_uppercase

                        # Our pieces: indices 0-5, opponent pieces: indices 6-11
                        if is_our_piece:
                            piece_idx = piece_type_idx
                        else:
                            piece_idx = piece_type_idx + 6

                        # FEN rows are ordered from rank 8 down to rank 1. Store so that
                        # board[0] corresponds to rank1 (a1-h1) to match other helpers.
                        board[7 - rank_idx, file_idx, piece_idx] = 1.0
                    file_idx += 1

        # Flatten board
        return board.flatten()

    def _index_to_square(self, index: int) -> str:
        """Convert square index (0-63) to square name (a1-h8).

        Args:
            index: Square index from 0 (a1) to 63 (h8)

        Returns:
            Square name like 'e4'
        """
        file_idx = index % 8
        rank_idx = index // 8
        return f"{FILES[file_idx]}{RANKS[rank_idx]}"

    def _square_to_index(self, square: str) -> int:
        """Convert square name (a1-h8) to square index (0-63).

        Args:
            square: Square name like 'e4'

        Returns:
            Square index from 0 (a1) to 63 (h8)
        """
        file = square[0]
        rank = square[1]
        file_idx = FILES.index(file)
        rank_idx = RANKS.index(rank)
        return rank_idx * 8 + file_idx

    def _extract_our_squares_from_features(self, features: np.ndarray) -> list[str]:
        """Extract squares with our pieces from features array.

        Args:
            features: Encoded board features (768-dimensional array)

        Returns:
            List of square names where our pieces are located (e.g., ['e2', 'g1'])
        """
        board = features.reshape(8, 8, 12)
        our_squares = []

        for rank_idx in range(8):
            for file_idx in range(8):
                # Check if there's any piece in our channels (0-5)
                if board[rank_idx, file_idx, 0:6].any():
                    square = f"{FILES[file_idx]}{RANKS[rank_idx]}"
                    our_squares.append(square)

        return our_squares
