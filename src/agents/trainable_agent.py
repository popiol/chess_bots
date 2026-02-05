from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

import chess

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

# NOTE: index<->square helpers have been moved to NeuralNetworkAgent.


@dataclass
class PredictionResult:
    """Result of a move prediction."""

    from_sq: str  # Source square name like 'e2'
    to_sq: str  # Destination square name like 'e4'
    evaluation: float  # Position evaluation (-1 to 1)
    decisive: float  # Decisiveness of the move (0 to 1)


@dataclass
class AnalysisNode:
    """Node in the position analysis tree.

    Represents a chess position (by FEN) with predictions and potential continuations.
    """

    fen: str
    predictions: list[PredictionResult]
    parent: AnalysisNode | None = None
    parent_move: PredictionResult | None = (
        None  # Move from parent that led to this node
    )
    children: dict[tuple[str, str], AnalysisNode] = field(default_factory=dict)
    # children maps (from_square, to_square) -> resulting position node


@dataclass
class Decision:
    """Record of a decision made by the agent."""

    fen: str
    from_sq: str
    to_sq: str


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
        # Remember the FEN after we made our last decided move
        self._last_fen_after_our_move: str | None = None
        # Opponent move history as list of Decision objects (fen, from_sq, to_sq)
        self._opponent_move_history: list[Decision] = []

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

        logger.debug(
            "Deciding move for FEN: %s", current_fen, extra={"username": self.username}
        )

        # If we previously recorded the FEN after our move, compare it to the
        # current FEN to detect the opponent's move (if any) and store it.
        if self._last_fen_after_our_move is not None:
            if current_fen != self._last_fen_after_our_move:
                prev_board = chess.Board(self._last_fen_after_our_move)
                found = False
                for mv in prev_board.legal_moves:
                    nb = prev_board.copy()
                    nb.push(mv)
                    if nb.fen() == current_fen:
                        from_sq = chess.square_name(mv.from_square)
                        to_sq = chess.square_name(mv.to_square)
                        # Record opponent move as a Decision with fen = position before their move
                        new_decision = Decision(
                            fen=self._last_fen_after_our_move,
                            from_sq=from_sq,
                            to_sq=to_sq,
                        )

                        # Avoid appending the same move twice in a row
                        last = (
                            self._opponent_move_history[-1]
                            if self._opponent_move_history
                            else None
                        )
                        if (
                            last is None
                            or last.fen != new_decision.fen
                            or last.from_sq != new_decision.from_sq
                            or last.to_sq != new_decision.to_sq
                        ):
                            self._opponent_move_history.append(new_decision)
                            logger.info(
                                "Detected opponent move %s->%s",
                                from_sq,
                                to_sq,
                                extra={"username": self.username},
                            )
                        else:
                            logger.debug(
                                "Duplicate opponent move ignored %s->%s",
                                from_sq,
                                to_sq,
                                extra={"username": self.username},
                            )
                        found = True
                        break
                if not found:
                    logger.warning(
                        "Could not detect opponent move from recorded FEN -> current FEN",
                        extra={"username": self.username},
                    )

        # Initialize analysis if not present
        if self._analysis is None:
            our_squares = self._extract_our_squares_from_fen(current_fen)
            assert our_squares, "No pieces found for our color"
            moves = self._predict(current_fen, our_squares)
            assert moves, "Predict returned no moves"
            self._analysis = AnalysisNode(fen=current_fen, predictions=moves)
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
            from_square = best_move.from_sq
            to_square = best_move.to_sq
            evaluation = best_move.evaluation
            decisive = best_move.decisive

            # Track FEN and best move
            self._decision_history.append(
                Decision(fen=self._analysis.fen, from_sq=from_square, to_sq=to_square)
            )

            # Remember the position AFTER our decided move so we can detect the
            # opponent's reply on the next call to _decide_move.
            new_fen = self._apply_move_to_fen(
                self._analysis.fen, from_square, to_square
            )
            self._last_fen_after_our_move = new_fen

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
        """Expand the analysis tree by one node using BFS."""

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
            from_sq = move.from_sq
            to_sq = move.to_sq
            move_key = (from_sq, to_sq)

            if move_key in node.children:
                continue

            # This move hasn't been expanded - create child node
            from_square = from_sq
            to_square = to_sq

            # Apply move to FEN to get new position. Let exceptions propagate on invalid moves.
            new_fen = self._apply_move_to_fen(node.fen, from_sq, to_sq)

            # Predict for the new position
            new_our_squares = self._extract_our_squares_from_fen(new_fen)
            new_predictions = self._predict(new_fen, new_our_squares)

            # Create child node
            child_node = AnalysisNode(
                fen=new_fen,
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

        # Nothing to propagate if no predictions
        if not node.predictions:
            return

        current = node
        while current.parent is not None:
            # Calculate average evaluation and decisive from current node's predictions

            avg_eval = sum(m.evaluation for m in current.predictions) / len(
                current.predictions
            )
            avg_decisive = sum(m.decisive for m in current.predictions) / len(
                current.predictions
            )

            # Update parent's move that led to current node
            assert current.parent_move is not None

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
            current = current.parent

    def _apply_move_to_fen(self, fen: str, from_sq: str, to_sq: str) -> str:
        """Apply a move (given as square-name strings) to a FEN and return the new FEN."""

        board = chess.Board(fen)
        move = chess.Move.from_uci(f"{from_sq}{to_sq}")
        board.push(move)
        return board.fen()

    def _predict(self, fen: str, our_squares: list[str]) -> list[PredictionResult]:
        """Fallback predictor used by simpler TrainableAgent subclasses.

        Args:
            fen: FEN string for the current position (may be unused by simple agents)
            our_squares: List of square names where our pieces are located

        Returns:
            List[PredictionResult] with randomized moves (simple heuristic).
        """
        assert our_squares, "our_squares cannot be empty"

        moves = []
        for _ in range(self.prediction_count):
            from_square = random.choice(our_squares)
            to_square = chess.square_name(random.randint(0, 63))
            evaluation = random.uniform(-1.0, 1.0)
            decisive = random.uniform(0.0, 1.0)
            moves.append(
                PredictionResult(
                    from_sq=from_square,
                    to_sq=to_square,
                    evaluation=evaluation,
                    decisive=decisive,
                )
            )

        logger.debug(
            "Predicted %d moves", len(moves), extra={"username": self.username}
        )

        return moves

    def _extract_our_squares_from_fen(self, fen: str) -> list[str]:
        """Extract squares with our pieces from a FEN string.

        Args:
            fen: FEN string

        Returns:
            List of square names where our pieces are located (e.g., ['e2', 'g1'])
        """
        board = chess.Board(fen)

        our_squares: list[str] = []
        for sq, piece in board.piece_map().items():
            if piece.color == board.turn:
                our_squares.append(chess.square_name(sq))

        return our_squares
