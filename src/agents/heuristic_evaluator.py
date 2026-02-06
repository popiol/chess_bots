from __future__ import annotations

import chess
import numpy as np

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


class HeuristicEvaluator:
    """Evaluates chess positions using various heuristic metrics.

    Provides methods to evaluate material balance, mobility, king safety,
    and other positional factors for a given chess position.
    """

    def __init__(self):
        """Initialize the evaluator with default weights for each metric."""
        self.profitable_weight = 0.26
        self.material_weight = 0.38
        self.mobility_weight = 0.04
        self.king_weight = 0.04
        self.castling_weight = 0.04
        self.center_weight = 0.04
        self.undeveloped_weight = 0.04
        self.passed_weight = 0.04
        self.doubled_weight = 0.04
        self.isolated_weight = 0.04
        self.check_weight = 0.04

    def evaluate_position(
        self, board: chess.Board, is_white: bool
    ) -> tuple[float, float]:
        """Evaluate a chess position from the perspective of the given side.

        Args:
            board: The chess board to evaluate
            is_white: True if evaluating from white's perspective, False for black

        Returns:
            A tuple of (evaluation, decisive) where:
            - evaluation: float in [-1, 1] representing position value
            - decisive: float in [0, 1] representing how decisive the position is
        """
        # Calculate all individual metrics
        material_eval = self._material_eval(board, is_white)
        mobility_eval = self._mobility_eval(board, is_white)
        king_eval = self._king_safety_eval(board, is_white)
        castling_eval = self._castling_bonus(board, is_white)
        check_eval = self._check_eval(board, is_white)
        profitable_attack_eval = self._profitable_attack_eval(board, is_white)
        center_eval = self._center_control_eval(board, is_white)
        undeveloped_eval = self._undeveloped_pieces_eval(board, is_white)
        doubled_eval = self._doubled_pawns_eval(board, is_white)
        isolated_eval = self._isolated_pawns_eval(board, is_white)
        passed_eval = self._passed_pawns_eval(board, is_white)

        # Weighted sum for overall evaluation
        eval_val = (
            self.material_weight * material_eval
            + self.mobility_weight * mobility_eval
            + self.king_weight * king_eval
            + self.castling_weight * castling_eval
            + self.check_weight * check_eval
            + self.profitable_weight * profitable_attack_eval
            + self.center_weight * center_eval
            + self.undeveloped_weight * undeveloped_eval
            + self.doubled_weight * doubled_eval
            + self.isolated_weight * isolated_eval
            + self.passed_weight * passed_eval
        )
        eval_val = float(np.clip(eval_val, -1.0, 1.0))

        # Decisive ratio: weighted sum of absolute metric values
        decisive_ratio = (
            self.material_weight * abs(material_eval)
            + self.mobility_weight * abs(mobility_eval)
            + self.king_weight * abs(king_eval)
            + self.castling_weight * abs(castling_eval)
            + self.check_weight * abs(check_eval)
            + self.profitable_weight * abs(profitable_attack_eval)
            + self.center_weight * abs(center_eval)
            + self.undeveloped_weight * abs(undeveloped_eval)
            + self.doubled_weight * abs(doubled_eval)
            + self.isolated_weight * abs(isolated_eval)
            + self.passed_weight * abs(passed_eval)
        )
        decisive = float(np.clip(decisive_ratio, 0.0, 1.0))

        return eval_val, decisive

    def _material_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Compute normalized material evaluation for the side to move.

        Returns a float in roughly [-1, 1].
        """
        white_total = 0
        black_total = 0
        for _, piece in board_after.piece_map().items():
            val = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color:
                white_total += val
            else:
                black_total += val

        if is_white:
            our_total = white_total
            opp_total = black_total
        else:
            our_total = black_total
            opp_total = white_total

        max_material = 39.0
        diff = our_total - opp_total
        if diff == 0:
            return 0.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_material))
        return float(np.clip(scaled, -1.0, 1.0))

    def _mobility_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Compute mobility evaluation (normalized difference in legal moves)."""
        b_white = board_after.copy()
        b_white.turn = True
        white_moves = b_white.legal_moves.count()

        b_black = board_after.copy()
        b_black.turn = False
        black_moves = b_black.legal_moves.count()

        if is_white:
            our_moves = white_moves
            opp_moves = black_moves
        else:
            our_moves = black_moves
            opp_moves = white_moves

        return float(np.clip((our_moves - opp_moves) / 50.0, -1.0, 1.0))

    def _castling_bonus(self, board_after: chess.Board, is_white: bool) -> float:
        """Return a small positive bonus if the side to move is castled.

        Detects castling by checking king square (g1/c1 for white, g8/c8 for black).
        """
        # Detect castling by king square presence on castled squares (g1/c1 for white, g8/c8 for black)
        try:
            wking = board_after.king(chess.WHITE)
            bking = board_after.king(chess.BLACK)
        except Exception:
            wking = None
            bking = None

        white_castled = wking in (chess.G1, chess.C1)
        black_castled = bking in (chess.G8, chess.C8)

        # Map castled True -> 1, False -> -1
        white_score = 1.0 if white_castled else -1.0
        black_score = 1.0 if black_castled else -1.0

        our = white_score if is_white else black_score
        opp = black_score if is_white else white_score

        # Normalize difference to [-1,1]
        return float(np.clip((our - opp) / 2.0, -1.0, 1.0))

    def _king_safety_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Estimate king safety: penalize kings off the back rank or missing pawn shield.

        Returns a small score in roughly [-0.3, 0.2] where positive is safer.
        """
        # Compute a normalized safety score for each side in [-1, 1], then
        # return the agent-perspective difference (our - opp) / 2 clipped to [-1,1].

        def side_score(color: bool) -> float:
            king_sq = board_after.king(color)
            if king_sq is None:
                return -1.0

            # Back rank component: +1 if on back rank, -1 otherwise
            rank = chess.square_rank(king_sq) + 1
            if color == chess.WHITE:
                on_back = rank == 1
            else:
                on_back = rank == 8
            back_comp = 1.0 if on_back else -1.0

            # Pawn shield: examine up to three squares in front of the king
            file = chess.square_file(king_sq)
            forward = 1 if color == chess.WHITE else -1
            shield_count = 0
            for df in (-1, 0, 1):
                f = file + df
                if 0 <= f <= 7:
                    r = chess.square_rank(king_sq) + forward
                    if 0 <= r <= 7:
                        sq = chess.square(f, r)
                        piece = board_after.piece_at(sq)
                        if (
                            piece is not None
                            and piece.piece_type == chess.PAWN
                            and piece.color == color
                        ):
                            shield_count += 1

            # Shield component normalized to [-1,1] (0 pawns -> -1, 3 pawns -> +1)
            shield_comp = (shield_count / 3.0) * 2.0 - 1.0

            # Combine components equally and clamp
            raw = 0.5 * back_comp + 0.5 * shield_comp
            return float(np.clip(raw, -1.0, 1.0))

        white_score = side_score(chess.WHITE)
        black_score = side_score(chess.BLACK)

        # Agent perspective: our_score - opp_score, normalize to [-1,1]
        our = white_score if is_white else black_score
        opp = black_score if is_white else white_score
        diff = (our - opp) / 2.0
        return float(np.clip(diff, -1.0, 1.0))

    def _check_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Evaluate checks: +1 if opponent is in check, -1 if our king is in check, 0 otherwise.

        Checks both colors and returns agent-perspective value in [-1,1].
        """
        # After the move, board_after.turn is the side to move (the opponent).
        opp_in_check = bool(board_after.is_check())

        # To determine whether our king is in check, flip turn and ask is_check().
        tmp = board_after.copy()
        tmp.turn = not tmp.turn
        our_in_check = bool(tmp.is_check())

        if opp_in_check and not our_in_check:
            return 1.0
        if our_in_check and not opp_in_check:
            return -1.0
        return 0.0

    def _profitable_attack_eval(
        self, board_after: chess.Board, is_white: bool
    ) -> float:
        """Evaluate material exchange on attacked squares.

        For every square occupied by a piece, if it is attacked by the opponent,
        estimate the potential material change after a series of captures.
        We consider min(attackers, defenders) exchanges, prioritizing weaker pieces.

        Returns agent-perspective normalized value in [-1,1].
        """
        white_gain = 0.0
        black_gain = 0.0

        for sq, piece in board_after.piece_map().items():
            # Who is being attacked?
            defender_color = piece.color
            attacker_color = not defender_color

            attackers = list(board_after.attackers(attacker_color, sq))
            if not attackers:
                continue

            # Get values of attacking pieces, sorted weakest first
            attacker_values = sorted(
                [
                    PIECE_VALUES.get(board_after.piece_at(a).piece_type, 0)  # type: ignore
                    for a in attackers
                ]
            )

            # Get defenders (friendly pieces protecting the square)
            defenders = list(board_after.attackers(defender_color, sq))
            defender_values = sorted(
                [
                    PIECE_VALUES.get(board_after.piece_at(d).piece_type, 0)  # type: ignore
                    for d in defenders
                ]
            )

            k = min(len(attackers), len(defenders))

            # Cost for attacker: Sum of k weakest attackers
            attacker_loss = sum(attacker_values[:k])

            # Cost for defender: Target + Sum of k-1 weakest defenders
            target_val = PIECE_VALUES.get(piece.piece_type, 0)
            defender_loss = target_val + sum(defender_values[: (k - 1)])

            net_gain = defender_loss - attacker_loss
            if net_gain > 0:
                if attacker_color == chess.WHITE:
                    white_gain += net_gain
                else:
                    black_gain += net_gain

        our_gain = white_gain if is_white else black_gain
        opp_gain = black_gain if is_white else white_gain

        diff = our_gain - opp_gain
        if diff == 0:
            return 0.0

        # Scale: A queen trade is 9, so gains can be large.
        max_gain = 15.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_gain))
        return float(np.clip(scaled, -1.0, 1.0))

    def _center_control_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Evaluate control of central squares d4,e4,d5,e5.

        Returns agent-perspective normalized control in [-1,1].
        """
        centers = [chess.D4, chess.E4, chess.D5, chess.E5]
        white_control = 0
        black_control = 0
        for sq in centers:
            white_control += len(board_after.attackers(chess.WHITE, sq))
            black_control += len(board_after.attackers(chess.BLACK, sq))

        our_ctrl = white_control if is_white else black_control
        opp_ctrl = black_control if is_white else white_control

        diff = our_ctrl - opp_ctrl
        if diff == 0:
            return 0.0
        max_ctrl = 12.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_ctrl))
        return float(np.clip(scaled, -1.0, 1.0))

    def _undeveloped_pieces_eval(
        self, board_after: chess.Board, is_white: bool
    ) -> float:
        """Penalize minor pieces (knights and bishops) that remain on their initial squares.

        Returns agent-perspective normalized value in [-1,1]: positive if opponent has
        more undeveloped minors than the agent (good), negative if agent has more
        undeveloped minors (bad).
        """
        # Initial squares for minor pieces
        white_initial = {chess.B1, chess.G1, chess.C1, chess.F1}
        black_initial = {chess.B8, chess.G8, chess.C8, chess.F8}

        white_undeveloped = 0
        black_undeveloped = 0
        for sq, piece in board_after.piece_map().items():
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                if piece.color == chess.WHITE and sq in white_initial:
                    white_undeveloped += 1
                if piece.color == chess.BLACK and sq in black_initial:
                    black_undeveloped += 1

        our_undeveloped = white_undeveloped if is_white else black_undeveloped
        opp_undeveloped = black_undeveloped if is_white else white_undeveloped

        # Normalize by maximum possible minor undeveloped pieces (4)
        norm = 4.0
        diff = (opp_undeveloped - our_undeveloped) / norm
        return float(np.clip(diff, -1.0, 1.0))

    def _doubled_pawns_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Penalize doubled pawns: count files with multiple pawns for each side.

        Returns agent-perspective normalized value in [-1,1]: positive if
        opponent has more doubled pawns than agent (good), negative if agent has
        more doubled pawns (bad).
        """
        # Count pawns per file for each color
        white_files = [0] * 8
        black_files = [0] * 8
        for sq, piece in board_after.piece_map().items():
            if piece.piece_type == chess.PAWN:
                file = chess.square_file(sq)
                if piece.color == chess.WHITE:
                    white_files[file] += 1
                else:
                    black_files[file] += 1

        white_doubled = sum(max(0, c - 1) for c in white_files)
        black_doubled = sum(max(0, c - 1) for c in black_files)

        our_doubled = white_doubled if is_white else black_doubled
        opp_doubled = black_doubled if is_white else white_doubled

        diff = opp_doubled - our_doubled
        if diff == 0:
            return 0.0
        max_doubled = 7.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_doubled))
        return float(np.clip(scaled, -1.0, 1.0))

    def _isolated_pawns_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Penalize isolated pawns: count pawns with no friendly pawns on adjacent files.

        Returns agent-perspective normalized value in [-1,1]: positive if opponent has
        more isolated pawns than agent (good), negative if agent has more isolated pawns (bad).
        """
        # Count pawns per file for each color
        white_files = [0] * 8
        black_files = [0] * 8
        for sq, piece in board_after.piece_map().items():
            if piece.piece_type == chess.PAWN:
                f = chess.square_file(sq)
                if piece.color == chess.WHITE:
                    white_files[f] += 1
                else:
                    black_files[f] += 1

        def count_isolated(files: list[int]) -> int:
            isolated = 0
            for i, c in enumerate(files):
                if c == 0:
                    continue
                left = files[i - 1] if i - 1 >= 0 else 0
                right = files[i + 1] if i + 1 <= 7 else 0
                if left == 0 and right == 0:
                    isolated += c
            return isolated

        white_isolated = count_isolated(white_files)
        black_isolated = count_isolated(black_files)

        our_iso = white_isolated if is_white else black_isolated
        opp_iso = black_isolated if is_white else white_isolated

        diff = opp_iso - our_iso
        if diff == 0:
            return 0.0
        max_iso = 8.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_iso))
        return float(np.clip(scaled, -1.0, 1.0))

    def _passed_pawns_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Reward passed pawns: count pawns that have no opposing pawn on same or adjacent files ahead of them.

        Returns agent-perspective normalized value in [-1,1]: positive if agent has
        more passed pawns than opponent.
        """

        def is_passed(sq: int, color: bool) -> bool:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            # For white, opponent pawns ahead have higher ranks; for black, lower ranks
            if color == chess.WHITE:
                ranks = range(rank + 1, 8)
                opp_color = chess.BLACK
            else:
                ranks = range(rank - 1, -1, -1)
                opp_color = chess.WHITE

            for f in (file - 1, file, file + 1):
                if f < 0 or f > 7:
                    continue
                for r in ranks:
                    sq2 = chess.square(f, r)
                    piece = board_after.piece_at(sq2)
                    if (
                        piece is not None
                        and piece.piece_type == chess.PAWN
                        and piece.color == opp_color
                    ):
                        return False
            return True

        white_passed = 0
        black_passed = 0
        for sq, piece in board_after.piece_map().items():
            if piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE and is_passed(sq, chess.WHITE):
                    white_passed += 1
                if piece.color == chess.BLACK and is_passed(sq, chess.BLACK):
                    black_passed += 1

        our_passed = white_passed if is_white else black_passed
        opp_passed = black_passed if is_white else white_passed

        diff = our_passed - opp_passed
        if diff == 0:
            return 0.0
        max_passed = 8.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_passed))
        return float(np.clip(scaled, -1.0, 1.0))
