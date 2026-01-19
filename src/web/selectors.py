from dataclasses import dataclass


@dataclass(frozen=True)
class AuthSelectors:
    open_signup: str
    open_signin: str
    email: str
    username: str
    password: str
    confirm_password: str
    submit_signup: str
    submit_signin: str
    signout: str


@dataclass(frozen=True)
class GameSelectors:
    play_as_guest: str
    play_now: str
    cancel_matchmaking: str
    rated_toggle: str
    time_control_option_by_index: str


@dataclass(frozen=True)
class GlobalSelectors:
    header_logo: str
    theme_toggle: str
    profile_link: str
    sign_in: str
    sign_up: str
    sign_out: str
    footer_feedback: str
    footer_terms: str
    footer_privacy: str


@dataclass(frozen=True)
class FeedbackSelectors:
    dialog_root: str
    category_button_by_value: str
    details_textarea: str
    submit: str
    cancel: str
    success_state: str
    success_close: str


@dataclass(frozen=True)
class GamePageSelectors:
    chess_board: str
    game_fen: str
    active_player: str
    square_by_coord: str
    resign: str
    resign_confirm: str
    resign_cancel: str
    offer_draw: str
    accept_draw: str
    postgame_panel: str
    postgame_analyze: str
    postgame_new_game: str
    move_list_selectable: str
    move_list_selected: str


@dataclass(frozen=True)
class AnalysisSelectors:
    move_list: str
    variation_list: str
    move_by_index: str
    flip_board: str
    copy_pgn: str
    copy_pgn_success: str
    back_to_game: str
    eval_chart: str
    fen_text: str
    copy_fen: str
    copy_fen_success: str


@dataclass(frozen=True)
class DocsSelectors:
    document_container: str
    document_content: str


@dataclass(frozen=True)
class SiteSelectors:
    global_nav: GlobalSelectors
    auth: AuthSelectors
    game: GameSelectors
    feedback: FeedbackSelectors
    game_page: GamePageSelectors
    analysis: AnalysisSelectors
    docs: DocsSelectors
    post_login_ready: str
    guest_ready: str
    play_ready: str

    def time_control_option(self, index: int) -> str:
        return f"css=[data-testid='time-control-{index}']"

    def square(self, coord: str) -> str:
        """Get selector for a chess square (e.g. 'e4', 'a1')."""
        return f"css=[data-square='{coord}']"

    def feedback_category(self, value: str) -> str:
        return f"css=[data-testid='feedback-category-{value}']"

    def analysis_move_by_index(self, index: int) -> str:
        return f"css=[data-move-index='{index}']"


def site_selectors() -> SiteSelectors:
    # Based on docs/bot_selectors.md for playbullet.gg.
    return SiteSelectors(
        global_nav=GlobalSelectors(
            header_logo="css=[data-testid='header-logo-link']",
            theme_toggle="css=[data-testid='theme-toggle']",
            profile_link="css=[data-testid='header-profile-link']",
            sign_in="css=[data-testid='header-signin']",
            sign_up="css=[data-testid='header-signup']",
            sign_out="css=[data-testid='header-signout']",
            footer_feedback="css=[data-testid='footer-feedback']",
            footer_terms="css=[data-testid='footer-terms']",
            footer_privacy="css=[data-testid='footer-privacy']",
        ),
        auth=AuthSelectors(
            open_signup="css=[data-testid='header-signup']",
            open_signin="css=[data-testid='header-signin']",
            email="label=Email",
            username="label=Username",
            password="label=Password",
            confirm_password="label=Confirm password",
            submit_signup="css=form button",
            submit_signin="css=form button",
            signout="css=[data-testid='header-signout']",
        ),
        game=GameSelectors(
            play_as_guest="css=[data-testid='play-guest']",
            play_now="css=[data-testid='play-now']",
            cancel_matchmaking="css=[data-testid='cancel-matchmaking']",
            rated_toggle="css=[data-testid='rated-toggle']",
            time_control_option_by_index="css=[data-testid='time-control-<index>']",
        ),
        feedback=FeedbackSelectors(
            dialog_root="css=[data-testid='feedback-dialog']",
            category_button_by_value="css=[data-testid='feedback-category-<value>']",
            details_textarea="css=[data-testid='feedback-details']",
            submit="css=[data-testid='feedback-submit']",
            cancel="css=[data-testid='feedback-cancel']",
            success_state="css=[data-testid='feedback-success']",
            success_close="css=[data-testid='feedback-close']",
        ),
        game_page=GamePageSelectors(
            chess_board="css=[data-testid='chess-board']",
            game_fen="css=[data-testid='game-fen']",
            active_player="css=.player-info--active",
            square_by_coord="css=[data-square='<coord>']",
            resign="css=[data-testid='resign']",
            resign_confirm="css=[data-testid='resign-confirm']",
            resign_cancel="css=[data-testid='resign-cancel']",
            offer_draw="css=[data-testid='offer-draw']",
            accept_draw="css=[data-testid='accept-draw']",
            postgame_panel="css=[data-testid='postgame-panel']",
            postgame_analyze="css=[data-testid='postgame-analyze']",
            postgame_new_game="css=[data-testid='postgame-new-game']",
            move_list_selectable="css=.selectable",
            move_list_selected="css=.move-list-selected",
        ),
        analysis=AnalysisSelectors(
            move_list="css=[data-testid='analysis-move-list']",
            variation_list="css=[data-testid='analysis-variation-list']",
            move_by_index="css=[data-move-index='<index>']",
            flip_board="css=[data-testid='analysis-flip-board']",
            copy_pgn="css=[data-testid='analysis-copy-pgn']",
            copy_pgn_success="css=[data-testid='analysis-copy-pgn-success']",
            back_to_game="css=[data-testid='analysis-back-to-game']",
            eval_chart="css=[data-testid='analysis-eval-chart']",
            fen_text="css=[data-testid='analysis-fen']",
            copy_fen="css=[data-testid='analysis-copy-fen']",
            copy_fen_success="css=[data-testid='analysis-copy-fen-success']",
        ),
        docs=DocsSelectors(
            document_container="css=.doc-container",
            document_content="css=.doc-content",
        ),
        post_login_ready="css=[data-testid='header-profile-link']",
        guest_ready="css=[data-testid='chess-board']",
        play_ready="css=[data-testid='chess-board']",
    )
