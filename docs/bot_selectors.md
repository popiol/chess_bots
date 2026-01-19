# UI Selectors for Automation

Use `data-testid` where available. When `data-testid` is not present, use the provided CSS selectors or data attributes.

## Global (all pages)
- Header logo: `[data-testid="header-logo-link"]`
- Theme toggle: `[data-testid="theme-toggle"]`
- Signed-in profile link: `[data-testid="header-profile-link"]`
- Sign in: `[data-testid="header-signin"]`
- Sign up: `[data-testid="header-signup"]`
- Sign out: `[data-testid="header-signout"]`
- Footer feedback: `[data-testid="footer-feedback"]`
- Footer terms: `[data-testid="footer-terms"]`
- Footer privacy: `[data-testid="footer-privacy"]`

## Feedback dialog
- Dialog root: `[data-testid="feedback-dialog"]`
- Category buttons: `[data-testid="feedback-category-<value>"]`
  - Values: `general_feedback`, `report_a_bug`, `suggest_improvement`, `comment_on_page`, `report_abuse`, `request_contact`
- Details textarea: `[data-testid="feedback-details"]`
- Submit: `[data-testid="feedback-submit"]`
- Cancel: `[data-testid="feedback-cancel"]`
- Success state: `[data-testid="feedback-success"]`
- Success close: `[data-testid="feedback-close"]`

## Home page
- Play now: `[data-testid="play-now"]`
- Play as guest: `[data-testid="play-guest"]`
- Cancel matchmaking: `[data-testid="cancel-matchmaking"]`
- Rated toggle: `[data-testid="rated-toggle"]`
- Time control option by index: `[data-testid="time-control-<index>"]`

## Game page / Replay page
- Chess board: `[data-testid="chess-board"]`
- Current FEN (hidden): `[data-testid="game-fen"]`
- Squares: `[data-square="<file><rank>"]` (e.g. `[data-square="e4"]`)
- Active player panel: `.player-info--active`
- Active clock: `.player-info-clock--active`
- Current user (live game): the bottom PlayerInfo is the current user.
- Resign: `[data-testid="resign"]`
- Resign confirm: `[data-testid="resign-confirm"]`
- Resign cancel: `[data-testid="resign-cancel"]`
- Offer draw: `[data-testid="offer-draw"]`
- Accept draw: `[data-testid="accept-draw"]`
- Post‑game panel: `[data-testid="postgame-panel"]`
- Post‑game analyze: `[data-testid="postgame-analyze"]`
- Post‑game new game: `[data-testid="postgame-new-game"]`

Move list cells (replay navigation):
- Selectable cells: `.selectable`
- Selected move: `.move-list-selected`

## Analysis page
- Move list container: `[data-testid="analysis-move-list"]`
- Variation list container: `[data-testid="analysis-variation-list"]`
- Move/variation buttons use data index: `[data-move-index="<index>"]`
- Flip board: `[data-testid="analysis-flip-board"]`
- Copy PGN: `[data-testid="analysis-copy-pgn"]`
- Copy PGN success: `[data-testid="analysis-copy-pgn-success"]`
- Back to game: `[data-testid="analysis-back-to-game"]`
- Evaluation chart: `[data-testid="analysis-eval-chart"]`
- FEN text: `[data-testid="analysis-fen"]`
- Copy FEN: `[data-testid="analysis-copy-fen"]`
- Copy FEN success: `[data-testid="analysis-copy-fen-success"]`

## Login / Signup
- Use the input labels to locate fields (MUI `TextField` with `label`).
- Primary action buttons are the only large `Button` within the form card.

## Terms / Privacy
- Document container: `.doc-container`
- Document content: `.doc-content`
