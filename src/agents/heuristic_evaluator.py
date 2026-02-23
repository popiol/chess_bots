from __future__ import annotations

import logging

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


logger = logging.getLogger(__name__)


class HeuristicEvaluator:
    """Evaluates chess positions using various heuristic metrics.

    Provides methods to evaluate material balance, mobility, king safety,
    and other positional factors for a given chess position.
    """

    def __init__(self):
        """Initialize the evaluator with default weights for each metric."""
        # Separate weights for profitable attack gains (our side and opponent)
        self.material_weight = 0.4
        self.our_attack_weight = 0.3
        self.opp_piece_exposed_weight = 0.2
        self.opp_attack_weight = 0.04
        self.our_piece_exposed_weight = 0.04
        self.mate_in_one_weight = 0.8
        self.position_mate_weight = 1.0
        self.pawn_promotion_weight = 0.03
        self.mobility_weight = 0.04
        self.safe_mobility_weight = 0.04
        self.king_weight = 0.04
        self.castling_weight = 0.04
        self.center_weight = 0.04
        self.undeveloped_weight = 0.04
        self.passed_weight = 0.04
        self.doubled_weight = 0.04
        self.isolated_weight = 0.04
        self.check_weight = 0.04
        self.hanging_weight = 0.03
        self.bishop_pair_weight = 0.03
        self.rook_open_file_weight = 0.03
        self.endgame_king_activity_weight = 0.03
        self.bishop_activity_weight = 0.03
        self.fork_weight = 0.03
        self.outpost_knight_weight = 0.03
        self.space_advantage_weight = 0.03
        self.backward_pawns_weight = 0.03
        self.squares_attacked_weight = 0.03
        self.discovered_attacks_weight = 0.03
        self.pins_weight = 0.03

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
        safe_mobility_eval = self._safe_mobility_eval(board, is_white)
        piece_exposed_our, piece_exposed_opp = self._piece_exposed_eval(board, is_white)
        mate_in_one = self._mate_in_one_eval(board, is_white)
        pawn_promo_eval = self._pawn_promotion_progress_eval(board, is_white)
        position_mate = self._position_mate_eval(board, is_white)
        king_eval = self._king_safety_eval(board, is_white)
        castling_eval = self._castling_bonus(board, is_white)
        check_eval = self._check_eval(board, is_white)
        our_attack, opp_attack = self._profitable_attack_eval(board, is_white)
        center_eval = self._center_control_eval(board, is_white)
        undeveloped_eval = self._undeveloped_pieces_eval(board, is_white)
        doubled_eval = self._doubled_pawns_eval(board, is_white)
        isolated_eval = self._isolated_pawns_eval(board, is_white)
        passed_eval = self._passed_pawns_eval(board, is_white)
        hanging_eval = self._hanging_pieces_eval(board, is_white)
        bishop_pair_eval = self._bishop_pair_eval(board, is_white)
        rook_open_file_eval = self._rook_open_file_eval(board, is_white)
        endgame_king_activity_eval = self._endgame_king_activity_eval(board, is_white)
        bishop_activity_eval = self._bishop_activity_eval(board, is_white)
        forks_eval = self._forks_eval(board, is_white)
        outpost_knight_eval = self._outpost_knight_eval(board, is_white)
        space_advantage_eval = self._space_advantage_eval(board, is_white)
        backward_pawns_eval = self._backward_pawns_eval(board, is_white)
        squares_attacked_eval = self._squares_attacked_eval(board, is_white)
        discovered_attacks_eval = self._discovered_attacks_eval(board, is_white)
        pins_eval = self._pins_eval(board, is_white)

        # Weighted sum for overall evaluation
        eval_val = (
            self.material_weight * material_eval
            + self.mobility_weight * mobility_eval
            + self.safe_mobility_weight * safe_mobility_eval
            + self.opp_piece_exposed_weight * piece_exposed_opp
            - self.our_piece_exposed_weight * piece_exposed_our
            + self.mate_in_one_weight * mate_in_one
            + self.pawn_promotion_weight * pawn_promo_eval
            + self.position_mate_weight * position_mate
            + self.king_weight * king_eval
            + self.castling_weight * castling_eval
            + self.check_weight * check_eval
            + self.our_attack_weight * our_attack
            - self.opp_attack_weight * opp_attack
            + self.center_weight * center_eval
            + self.undeveloped_weight * undeveloped_eval
            + self.doubled_weight * doubled_eval
            + self.isolated_weight * isolated_eval
            + self.passed_weight * passed_eval
            + self.hanging_weight * hanging_eval
            + self.bishop_pair_weight * bishop_pair_eval
            + self.rook_open_file_weight * rook_open_file_eval
            + self.endgame_king_activity_weight * endgame_king_activity_eval
            + self.bishop_activity_weight * bishop_activity_eval
            + self.fork_weight * forks_eval
            + self.outpost_knight_weight * outpost_knight_eval
            + self.space_advantage_weight * space_advantage_eval
            + self.backward_pawns_weight * backward_pawns_eval
            + self.squares_attacked_weight * squares_attacked_eval
            + self.discovered_attacks_weight * discovered_attacks_eval
            + self.pins_weight * pins_eval
        )
        eval_val = float(np.clip(eval_val, -1.0, 1.0))

        # Decisive ratio: weighted sum of absolute metric values
        decisive_ratio = (
            self.material_weight * abs(material_eval)
            + self.mobility_weight * abs(mobility_eval)
            + self.safe_mobility_weight * abs(safe_mobility_eval)
            + self.our_piece_exposed_weight * abs(piece_exposed_our)
            + self.opp_piece_exposed_weight * abs(piece_exposed_opp)
            + self.mate_in_one_weight * abs(mate_in_one)
            + self.pawn_promotion_weight * abs(pawn_promo_eval)
            + self.position_mate_weight * abs(position_mate)
            + self.king_weight * abs(king_eval)
            + self.castling_weight * abs(castling_eval)
            + self.check_weight * abs(check_eval)
            + self.our_attack_weight * abs(our_attack)
            + self.opp_attack_weight * abs(opp_attack)
            + self.center_weight * abs(center_eval)
            + self.undeveloped_weight * abs(undeveloped_eval)
            + self.doubled_weight * abs(doubled_eval)
            + self.isolated_weight * abs(isolated_eval)
            + self.passed_weight * abs(passed_eval)
            + self.hanging_weight * abs(hanging_eval)
            + self.bishop_pair_weight * abs(bishop_pair_eval)
            + self.rook_open_file_weight * abs(rook_open_file_eval)
            + self.endgame_king_activity_weight * abs(endgame_king_activity_eval)
            + self.bishop_activity_weight * abs(bishop_activity_eval)
            + self.fork_weight * abs(forks_eval)
            + self.outpost_knight_weight * abs(outpost_knight_eval)
            + self.space_advantage_weight * abs(space_advantage_eval)
            + self.backward_pawns_weight * abs(backward_pawns_eval)
            + self.squares_attacked_weight * abs(squares_attacked_eval)
            + self.discovered_attacks_weight * abs(discovered_attacks_eval)
            + self.pins_weight * abs(pins_eval)
        )
        decisive = float(np.clip(decisive_ratio, 0.0, 1.0))

        return eval_val, decisive

    def evaluate_fast(self, board: chess.Board, is_white: bool) -> tuple[float, float]:
        """Fast evaluation using only the most important metrics.

        This is a lightweight approximation of `evaluate_position` that
        computes material, mobility, safe mobility, piece exposures,
        king safety and profitable-attack estimates. It returns the
        same (evaluation, decisive) tuple but is cheaper to compute.
        """
        # core metrics only
        our_attack, _ = self._profitable_attack_eval(board, is_white)
        material_eval = self._material_eval(board, is_white)
        mobility_eval = self._mobility_eval(board, is_white)
        castling_eval = self._castling_bonus(board, is_white)
        check_eval = self._check_eval(board, is_white)
        center_eval = self._center_control_eval(board, is_white)
        mate_in_one = self._mate_in_one_eval(board, is_white)
        position_mate = self._position_mate_eval(board, is_white)

        eval_val = (
            self.material_weight * material_eval * 0.5
            + self.our_attack_weight * our_attack
            + self.mobility_weight * mobility_eval
            + self.castling_weight * castling_eval
            + self.check_weight * check_eval
            + self.center_weight * center_eval
            + self.mate_in_one_weight * mate_in_one
            + self.position_mate_weight * position_mate
        )
        eval_val = float(np.clip(eval_val, -1.0, 1.0))

        # Decisive ratio based on same subset
        decisive_ratio = (
            self.material_weight * abs(material_eval) * 0.5
            + self.our_attack_weight * abs(our_attack)
            + self.mobility_weight * abs(mobility_eval)
            + self.castling_weight * abs(castling_eval)
            + self.check_weight * abs(check_eval)
            + self.center_weight * abs(center_eval)
            + self.mate_in_one_weight * abs(mate_in_one)
            + self.position_mate_weight * abs(position_mate)
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

    def _safe_mobility_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Count legal moves that result in the moved piece not being attacked.

        For each legal move available to a side, simulate the move and check
        whether the piece that ended up on the destination square is attacked
        by the opponent. Returns the normalized difference (our - opp)
        clipped to [-1,1], using a similar scale to mobility.
        """

        def safe_moves_for(color: bool) -> int:
            b = board_after.copy()
            b.turn = color
            safe = 0
            for mv in b.legal_moves:
                b2 = b.copy()
                try:
                    b2.push(mv)
                except Exception:
                    continue
                dest = mv.to_square
                piece = b2.piece_at(dest)
                if piece is None:
                    continue
                if piece.color != color:
                    continue
                if not b2.is_attacked_by(not color, dest):
                    safe += 1
            return safe

        white_safe = safe_moves_for(chess.WHITE)
        black_safe = safe_moves_for(chess.BLACK)
        our = white_safe if is_white else black_safe
        opp = black_safe if is_white else white_safe
        diff = our - opp
        if diff == 0:
            return 0.0
        return float(np.clip((diff) / 50.0, -1.0, 1.0))

    def _forks_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Evaluate knight forks: sum value of top-two attacked opponent pieces per knight.

        For each knight of a side, if it attacks two or more opponent pieces,
        add the sum of the top two piece values attacked. Return a normalized
        difference (our - opp) in [-1,1].
        """

        def forks_for(color: bool) -> float:
            total = 0.0
            for sq in board_after.piece_map().keys():
                piece = board_after.piece_at(sq)
                if piece is None or piece.color != color:
                    continue
                if piece.piece_type == chess.KING:
                    continue
                attacked = list(board_after.attacks(sq))
                target_vals = []
                for t in attacked:
                    p = board_after.piece_at(t)
                    if p is None:
                        continue
                    if p.color == color:
                        continue
                    # mark king as None so we can assign it the strongest value later
                    if p.piece_type == chess.KING:
                        target_vals.append(None)
                    else:
                        target_vals.append(PIECE_VALUES.get(p.piece_type, 0))
                if len(target_vals) < 2:
                    continue
                # If king is present, set it to the strongest attacked value
                if None in target_vals:
                    non_king_vals = [v for v in target_vals if v is not None]
                    max_val = max(non_king_vals)
                    total += max_val
                else:
                    target_vals.sort(reverse=True)
                    total += target_vals[1]
            return total

        white_score = forks_for(chess.WHITE)
        black_score = forks_for(chess.BLACK)
        our = white_score if is_white else black_score
        opp = black_score if is_white else white_score
        diff = our - opp
        if diff == 0:
            return 0.0
        max_fork = 9.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_fork))
        return float(np.clip(scaled, -1.0, 1.0))

    def _piece_exposed_eval(
        self, board_after: chess.Board, is_white: bool
    ) -> tuple[float, float]:
        """Detect whether pieces can be taken (hanging or favorable trade).

        For every piece on the board, consider whether opponent attackers exist
        and whether the attackers are at least as 'cheap' as the cheapest
        defender (or there are no defenders). If so, count the potential loss
        in material terms. The metric returns (opponent_exposed_loss - our_exposed_loss)
        normalized to [-1,1] (positive is good for the evaluator).
        """

        def exposed_points(color: bool) -> int:
            opp = not color
            loss = 0
            # Iterate over all pieces of the given color
            for sq, piece in board_after.piece_map().items():
                if piece.color != color:
                    continue
                attackers = list(board_after.attackers(opp, sq))
                if not attackers:
                    continue
                _loss = PIECE_VALUES.get(piece.piece_type, 0)
                attacker_vals = []
                for a in attackers:
                    pa = board_after.piece_at(a)
                    attacker_vals.append(
                        PIECE_VALUES.get(pa.piece_type, 0) if pa is not None else 0
                    )
                defenders = list(board_after.attackers(color, sq))
                if attacker_vals and defenders:
                    _loss -= min(attacker_vals)
                loss += max(0, _loss)
            return loss

        white_loss = exposed_points(chess.WHITE)
        black_loss = exposed_points(chess.BLACK)

        our_loss = white_loss if is_white else black_loss
        opp_loss = black_loss if is_white else white_loss

        # Scale both losses independently into [0,1] using queen value as baseline
        max_loss = 9.0
        scaled_our = float(np.clip(our_loss / max_loss, 0.0, 1.0))
        scaled_opp = float(np.clip(opp_loss / max_loss, 0.0, 1.0))

        return scaled_our, scaled_opp

    def _mate_in_one_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Return 1.0 if the opponent (not is_white) has a mate-in-one, else 0.0.

        This metric is intentionally single-valued: a mate-in-one existing for
        the opponent is always bad for the evaluated side.
        """
        for mv in board_after.legal_moves:
            b2 = board_after.copy()
            b2.push(mv)
            if b2.is_checkmate():
                return 1.0
        return 0.0

    def _pawn_promotion_progress_eval(
        self, board_after: chess.Board, is_white: bool
    ) -> float:
        """Compute pawn promotion progress for each side and return agent-perspective value.

        For each pawn we compute a progress in [0,1] where 0 is on the home rank
        and 1 is on the 7th rank (ready to promote). Sum progress for each side
        (max 8) and return (our_progress - opp_progress) / 8 clipped to [-1,1].
        Positive values are good for the evaluated side.
        """
        white_progress = 0.0
        black_progress = 0.0
        for sq, piece in board_after.piece_map().items():
            if piece.piece_type != chess.PAWN:
                continue
            rank = chess.square_rank(sq)  # 0..7
            if piece.color == chess.WHITE:
                # progress 0 at rank 0, 1 at rank 7
                white_progress += rank / 7.0
            else:
                # for black, progress increases as rank decreases
                black_progress += (7 - rank) / 7.0

        our = white_progress if is_white else black_progress
        opp = black_progress if is_white else white_progress
        diff = our - opp
        if diff == 0:
            return 0.0
        # normalize by max pawns (8)
        return float(np.clip(diff / 8.0, -1.0, 1.0))

    def _position_mate_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Return +1.0 if the evaluated side has already checkmated the opponent,
        -1.0 if the evaluated side is checkmated, else 0.0.

        This detects terminal checkmate positions and expresses them from the
        evaluator's perspective (positive is good for the evaluated side).
        """
        return float(board_after.is_checkmate())

    def _castling_bonus(self, board_after: chess.Board, is_white: bool) -> float:
        """Return a small positive bonus if the side to move is castled.

        Detects castling by checking king square (g1/c1 for white, g8/c8 for black).
        """
        # Detect castling by king square presence on castled squares.
        # King-side (g1/g8) counts as 1.0, queen-side (c1/c8) counts as 0.5, else -1.0
        wking = board_after.king(chess.WHITE)
        bking = board_after.king(chess.BLACK)

        if wking == chess.G1:
            white_score = 1.0
        elif wking == chess.C1:
            white_score = 0.5
        else:
            white_score = 0.0

        if bking == chess.G8:
            black_score = 1.0
        elif bking == chess.C8:
            black_score = 0.5
        else:
            black_score = 0.0

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
        our_color = chess.WHITE if is_white else chess.BLACK
        if board_after.turn == our_color:
            our_in_check = bool(board_after.is_check())
            opp_in_check = False
        else:
            opp_in_check = bool(board_after.is_check())
            our_in_check = False

        if opp_in_check and not our_in_check:
            return 1.0
        if our_in_check and not opp_in_check:
            return -1.0
        return 0.0

    def _profitable_attack_eval(
        self, board_after: chess.Board, is_white: bool
    ) -> tuple[float, float]:
        """Evaluate material exchange on attacked squares.

        For every square occupied by a piece, if it is attacked by the opponent,
        estimate the potential material change after a series of captures.
        We consider min(attackers, defenders) exchanges, prioritizing weaker pieces.

        Returns:
            Tuple of (scaled_our_gain, scaled_opp_gain) both in [0,1].
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

            attackers_lost = min(len(attackers), len(defenders))
            defenders_lost = min(len(attackers), len(defenders) + 1)

            # Cost for attacker: Sum of k weakest attackers
            attacker_loss = sum(attacker_values[:attackers_lost])

            # Cost for defender: Target + Sum of k-1 weakest defenders
            target_val = PIECE_VALUES.get(piece.piece_type, 0)
            defender_loss = target_val + sum(defender_values[: defenders_lost - 1])

            net_gain = defender_loss - attacker_loss
            if net_gain > 0:
                if attacker_color == chess.WHITE:
                    white_gain += net_gain
                else:
                    black_gain += net_gain

        our_gain = white_gain if is_white else black_gain
        opp_gain = black_gain if is_white else white_gain

        # Scale both gains independently to [0,1]
        max_gain = 15.0
        scaled_our = float(np.clip(np.log1p(our_gain) / np.log1p(max_gain), 0.0, 1.0))
        scaled_opp = float(np.clip(np.log1p(opp_gain) / np.log1p(max_gain), 0.0, 1.0))

        return scaled_our, scaled_opp

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

    def _hanging_pieces_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Evaluate hanging pieces (attacked and undefended non-king pieces).

        Returns agent-perspective value in [-1,1]: positive if opponent has
        more hanging material than agent.
        """
        white_hanging = 0.0
        black_hanging = 0.0

        for sq, piece in board_after.piece_map().items():
            if piece.piece_type == chess.KING:
                continue

            attackers = board_after.attackers(not piece.color, sq)
            defenders = board_after.attackers(piece.color, sq)
            if attackers and not defenders:
                val = float(PIECE_VALUES.get(piece.piece_type, 0))
                if piece.color == chess.WHITE:
                    white_hanging += val
                else:
                    black_hanging += val

        our_hanging = white_hanging if is_white else black_hanging
        opp_hanging = black_hanging if is_white else white_hanging

        diff = opp_hanging - our_hanging
        if diff == 0:
            return 0.0
        max_hanging = 20.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_hanging))
        return float(np.clip(scaled, -1.0, 1.0))

    def _bishop_pair_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Reward bishop pair advantage.

        Returns in [-1,1], positive if our side has bishop pair and opponent
        does not, negative in the reverse case.
        """
        white_bishops = len(board_after.pieces(chess.BISHOP, chess.WHITE))
        black_bishops = len(board_after.pieces(chess.BISHOP, chess.BLACK))

        our_pair = 1.0 if ((white_bishops if is_white else black_bishops) >= 2) else 0.0
        opp_pair = 1.0 if ((black_bishops if is_white else white_bishops) >= 2) else 0.0
        return float(np.clip(our_pair - opp_pair, -1.0, 1.0))

    def _rook_open_file_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Reward rooks on open and semi-open files.

        Open file: no pawns on file (+1.0)
        Semi-open file: no friendly pawn but at least one enemy pawn (+0.5)
        """

        def side_rook_score(color: bool) -> float:
            score = 0.0
            rooks = board_after.pieces(chess.ROOK, color)
            for sq in rooks:
                f = chess.square_file(sq)
                has_our_pawn = False
                has_opp_pawn = False
                for r in range(8):
                    p = board_after.piece_at(chess.square(f, r))
                    if p is None or p.piece_type != chess.PAWN:
                        continue
                    if p.color == color:
                        has_our_pawn = True
                    else:
                        has_opp_pawn = True
                if not has_our_pawn and not has_opp_pawn:
                    score += 1.0
                elif not has_our_pawn and has_opp_pawn:
                    score += 0.5
            return score

        white_score = side_rook_score(chess.WHITE)
        black_score = side_rook_score(chess.BLACK)
        our = white_score if is_white else black_score
        opp = black_score if is_white else white_score

        # Max practical absolute difference is around 2.0
        return float(np.clip((our - opp) / 2.0, -1.0, 1.0))

    def _endgame_king_activity_eval(
        self, board_after: chess.Board, is_white: bool
    ) -> float:
        """Reward active king in endgames; neutral outside endgame.

        Endgame is detected by total non-pawn material (excluding kings)
        being sufficiently low.
        """

        def non_pawn_material(color: bool) -> int:
            total = 0
            for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                total += PIECE_VALUES[piece_type] * len(
                    board_after.pieces(piece_type, color)
                )
            return total

        total_non_pawn = non_pawn_material(chess.WHITE) + non_pawn_material(chess.BLACK)
        if total_non_pawn > 24:
            return 0.0

        centers = (chess.D4, chess.E4, chess.D5, chess.E5)

        def king_activity(color: bool) -> float:
            sq = board_after.king(color)
            if sq is None:
                return -1.0
            d = min(chess.square_distance(sq, c) for c in centers)
            # distance 0 -> 1.0, distance 7 -> 0.0
            return float(np.clip(1.0 - (d / 7.0), 0.0, 1.0))

        white_act = king_activity(chess.WHITE)
        black_act = king_activity(chess.BLACK)
        our = white_act if is_white else black_act
        opp = black_act if is_white else white_act
        return float(np.clip(our - opp, -1.0, 1.0))

    def _bishop_activity_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Evaluate bishop activity: squares attacked, weighted toward center/extended center.

        Returns agent-perspective normalized value in [-1,1].
        """
        extended_center = {
            chess.C3,
            chess.D3,
            chess.E3,
            chess.F3,
            chess.C4,
            chess.D4,
            chess.E4,
            chess.F4,
            chess.C5,
            chess.D5,
            chess.E5,
            chess.F5,
            chess.C6,
            chess.D6,
            chess.E6,
            chess.F6,
        }

        def side_bishop_activity(color: bool) -> float:
            score = 0.0
            for sq in board_after.pieces(chess.BISHOP, color):
                attacks = board_after.attacks(sq)
                for target in attacks:
                    score += 2.0 if target in extended_center else 1.0
            return score

        white_score = side_bishop_activity(chess.WHITE)
        black_score = side_bishop_activity(chess.BLACK)
        our = white_score if is_white else black_score
        opp = black_score if is_white else white_score
        diff = our - opp
        if diff == 0:
            return 0.0
        max_activity = 30.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_activity))
        return float(np.clip(scaled, -1.0, 1.0))

    def _outpost_knight_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Evaluate outpost knights: knights on ranks 4-6 supported by a friendly pawn
        and not attackable by an enemy pawn.

        Returns agent-perspective normalized value in [-1,1].
        """

        def count_outposts(color: bool) -> int:
            count = 0
            opp_color = not color
            forward = 1 if color == chess.WHITE else -1
            good_ranks = (
                range(3, 6) if color == chess.WHITE else range(2, 5)
            )  # ranks 4-6 (0-indexed 3-5)
            for sq in board_after.pieces(chess.KNIGHT, color):
                rank = chess.square_rank(sq)
                if rank not in good_ranks:
                    continue
                file = chess.square_file(sq)
                # Check if supported by friendly pawn
                supported = False
                for df in (-1, 1):
                    f = file + df
                    r = rank - forward
                    if 0 <= f <= 7 and 0 <= r <= 7:
                        p = board_after.piece_at(chess.square(f, r))
                        if p and p.piece_type == chess.PAWN and p.color == color:
                            supported = True
                            break
                if not supported:
                    continue
                # Check no enemy pawn can attack this square
                can_be_attacked = False
                for df in (-1, 1):
                    f = file + df
                    if f < 0 or f > 7:
                        continue
                    # Check enemy pawns on adjacent files that could advance to attack
                    for r in (
                        range(rank + forward, 8)
                        if forward == 1
                        else range(rank + forward, -1, -1)
                    ):
                        if r < 0 or r > 7:
                            break
                        p = board_after.piece_at(chess.square(f, r))
                        if p and p.piece_type == chess.PAWN and p.color == opp_color:
                            can_be_attacked = True
                            break
                if not can_be_attacked:
                    count += 1
            return count

        white_outposts = count_outposts(chess.WHITE)
        black_outposts = count_outposts(chess.BLACK)
        our = white_outposts if is_white else black_outposts
        opp = black_outposts if is_white else white_outposts
        diff = our - opp
        return float(np.clip(diff / 2.0, -1.0, 1.0))

    def _space_advantage_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Evaluate space advantage: count squares controlled in opponent's half.

        Returns agent-perspective normalized value in [-1,1].
        """

        def controlled_in_opp_half(color: bool) -> int:
            count = 0
            # Opponent's half: ranks 5-8 for white (indices 4-7), ranks 1-4 for black (indices 0-3)
            opp_ranks = range(4, 8) if color == chess.WHITE else range(0, 4)
            for sq in chess.SQUARES:
                if chess.square_rank(sq) not in opp_ranks:
                    continue
                if board_after.is_attacked_by(color, sq):
                    count += 1
            return count

        white_space = controlled_in_opp_half(chess.WHITE)
        black_space = controlled_in_opp_half(chess.BLACK)
        our = white_space if is_white else black_space
        opp = black_space if is_white else white_space
        diff = our - opp
        if diff == 0:
            return 0.0
        max_space = 20.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_space))
        return float(np.clip(scaled, -1.0, 1.0))

    def _backward_pawns_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Penalize backward pawns: pawns that cannot safely advance because the
        stop square is controlled by an enemy pawn and no friendly pawn on an
        adjacent file can support the advance.

        Returns agent-perspective normalized value in [-1,1]: positive if
        opponent has more backward pawns.
        """

        def count_backward(color: bool) -> int:
            count = 0
            opp_color = not color
            forward = 1 if color == chess.WHITE else -1
            pawn_files = [0] * 8
            for sq in board_after.pieces(chess.PAWN, color):
                pawn_files[chess.square_file(sq)] = 1

            for sq in board_after.pieces(chess.PAWN, color):
                rank = chess.square_rank(sq)
                file = chess.square_file(sq)
                stop_rank = rank + forward
                if stop_rank < 0 or stop_rank > 7:
                    continue
                stop_sq = chess.square(file, stop_rank)
                # Check if stop square is controlled by enemy pawn
                if not board_after.is_attacked_by(opp_color, stop_sq):
                    continue
                # Check no friendly pawn on adjacent files at same or behind rank
                has_support = False
                for df in (-1, 1):
                    f = file + df
                    if f < 0 or f > 7:
                        continue
                    # Check friendly pawns on adjacent file at same rank or behind
                    check_ranks = (
                        range(rank, -1, -1) if color == chess.WHITE else range(rank, 8)
                    )
                    for r in check_ranks:
                        p = board_after.piece_at(chess.square(f, r))
                        if p and p.piece_type == chess.PAWN and p.color == color:
                            has_support = True
                            break
                    if has_support:
                        break

                if not has_support:
                    count += 1
            return count

        white_backward = count_backward(chess.WHITE)
        black_backward = count_backward(chess.BLACK)
        our = white_backward if is_white else black_backward
        opp = black_backward if is_white else white_backward
        diff = opp - our
        if diff == 0:
            return 0.0
        max_backward = 4.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_backward))
        return float(np.clip(scaled, -1.0, 1.0))

    def _squares_attacked_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Count unique squares attacked by each side (excluding king squares).

        Returns agent-perspective normalized value in [-1,1]: positive if
        our side attacks more squares than opponent.
        """
        white_attacked = 0
        black_attacked = 0
        for sq in chess.SQUARES:
            if board_after.is_attacked_by(chess.WHITE, sq):
                white_attacked += 1
            if board_after.is_attacked_by(chess.BLACK, sq):
                black_attacked += 1

        our = white_attacked if is_white else black_attacked
        opp = black_attacked if is_white else white_attacked
        diff = our - opp
        if diff == 0:
            return 0.0
        max_diff = 30.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_diff))
        return float(np.clip(scaled, -1.0, 1.0))

    def _discovered_attacks_eval(
        self, board_after: chess.Board, is_white: bool
    ) -> float:
        """Evaluate discovered attack potential.

        A discovered attack exists when a sliding piece (bishop, rook, queen)
        is blocked by a single friendly piece, and moving that blocker would
        reveal an attack on a higher-value enemy piece.

        Returns agent-perspective normalized value in [-1,1].
        """
        _BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        _ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def _ray_dirs(piece_type: int) -> list:
            if piece_type == chess.BISHOP:
                return _BISHOP_DIRS
            if piece_type == chess.ROOK:
                return _ROOK_DIRS
            return _BISHOP_DIRS + _ROOK_DIRS  # queen

        def count_discovered(color: bool) -> float:
            score = 0.0
            opp = not color
            for sq in (
                board_after.pieces(chess.BISHOP, color)
                | board_after.pieces(chess.ROOK, color)
                | board_after.pieces(chess.QUEEN, color)
            ):
                piece = board_after.piece_at(sq)
                if piece is None:
                    continue
                for df, dr in _ray_dirs(piece.piece_type):
                    f = chess.square_file(sq) + df
                    r = chess.square_rank(sq) + dr
                    blocker = None
                    # Walk along the ray
                    while 0 <= f <= 7 and 0 <= r <= 7:
                        target_sq = chess.square(f, r)
                        occupant = board_after.piece_at(target_sq)
                        if occupant is not None:
                            if blocker is None:
                                # First piece on the ray
                                if occupant.color == color:
                                    blocker = occupant
                                else:
                                    break  # enemy piece directly visible, not discovered
                            else:
                                # Second piece on the ray behind blocker
                                if occupant.color == opp:
                                    target_val = PIECE_VALUES.get(
                                        occupant.piece_type, 0
                                    )
                                    slider_val = PIECE_VALUES.get(piece.piece_type, 0)
                                    if target_val > slider_val:
                                        score += float(target_val - slider_val)
                                break
                        f += df
                        r += dr
            return score

        white_disc = count_discovered(chess.WHITE)
        black_disc = count_discovered(chess.BLACK)
        our = white_disc if is_white else black_disc
        opp = black_disc if is_white else white_disc
        diff = our - opp
        if diff == 0:
            return 0.0
        max_disc = 15.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_disc))
        return float(np.clip(scaled, -1.0, 1.0))

    def _pins_eval(self, board_after: chess.Board, is_white: bool) -> float:
        """Evaluate pins: count pinned pieces weighted by value.

        A piece is pinned if it is between a friendly king and an enemy
        sliding attacker along a line.

        Returns agent-perspective normalized value in [-1,1]: positive if
        opponent has more pinned material.
        """

        def pinned_value(color: bool) -> float:
            """Sum of piece values for all pinned pieces of the given color."""
            king_sq = board_after.king(color)
            if king_sq is None:
                return 0.0
            total = 0.0
            # python-chess pin() returns the pin mask for a square;
            # a piece is pinned if its pin mask is not BB_ALL (i.e. it's restricted).
            for sq in (
                board_after.pieces(chess.PAWN, color)
                | board_after.pieces(chess.KNIGHT, color)
                | board_after.pieces(chess.BISHOP, color)
                | board_after.pieces(chess.ROOK, color)
                | board_after.pieces(chess.QUEEN, color)
            ):
                if board_after.is_pinned(color, sq):
                    piece = board_after.piece_at(sq)
                    if piece is not None:
                        total += float(PIECE_VALUES.get(piece.piece_type, 0))
            return total

        white_pinned = pinned_value(chess.WHITE)
        black_pinned = pinned_value(chess.BLACK)
        our_pinned = white_pinned if is_white else black_pinned
        opp_pinned = black_pinned if is_white else white_pinned
        diff = opp_pinned - our_pinned
        if diff == 0:
            return 0.0
        max_pin = 15.0
        scaled = np.sign(diff) * (np.log1p(abs(diff)) / np.log1p(max_pin))
        return float(np.clip(scaled, -1.0, 1.0))
