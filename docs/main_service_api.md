# Main Service API (v1)

This document describes the REST API exposed by the Main Service for matchmaking, game
orchestration, history, ratings, and profiles. Paths are prefixed by the configured base path
(e.g., `/api/v1`). Unless noted, responses are JSON and require
`Authorization: Bearer <access_token>`.

---

## Authentication

- User-facing endpoints require an access token issued by the Identity Service.
- Internal game-result callback currently has no auth guard; it is intended to be called only by
  the Game Service.
- Error responses follow FastAPI defaults with `detail` strings.

---

## Matchmaking

### Join Queue
- `POST /matchmaking/join`
- Body:
  - `time_control_initial: int` (seconds, 1-3600)
  - `time_control_increment: int` (seconds, 0-60)
  - `mode: string` (`casual` | `rated`)
  - `auth_mode: string` (`account` | `guest`)
- Auth:
  - `account`: required
  - `guest`: uses `guest_id` cookie (created if missing)
- Returns 202 with `{ "message": "Joined matchmaking queue", "user_id": "..." }`

### Leave Queue
- `DELETE /matchmaking/leave`
- Auth: required
- Returns 204; 404 if not in queue.

### Poll for Match
- `POST /matchmaking/find`
- Body: same as Join Queue request (must include `auth_mode`)
- Auth:
  - `account`: required
  - `guest`: requires `guest_id` cookie
- Returns 200:
  - If waiting: `{ "matched": false }`
  - If matched: `{ "matched": true, "game_id", "websocket_url", "game_token", "my_color", "opponent_id", ... }`
  - Note: `websocket_url` is a path (e.g., `/ws/game/v1/<game_id>`). Clients should prepend
    their current scheme/host.

---

## Game Orchestration

### Create Game
- `POST /games/create`
- Auth: required (caller must be one of the users)
- Body (`GameCreationRequest`):
  - `player_white_id: UUID`
  - `player_black_id: UUID`
  - `time_control_initial: int` (seconds)
  - `time_control_increment: int` (seconds)
  - `starting_fen: string|null`
  - `mode: string` (`casual` | `rated`)
- Returns 200 (`GameCreationResponse`): `{ game_id, status, websocket_url }`
  - Note: `websocket_url` is a path (e.g., `/ws/game/v1/<game_id>`). Clients should prepend
    their current scheme/host.

### Submit Game Result (internal callback)
- `POST /games/{game_id}/result`
- Auth: none (assumed internal call from Game Service)
- Path params:
  - `game_id: UUID`
- Body (`GameResultSubmission`):
  - `game_id: UUID`
  - `result: string` (`1-0` | `0-1` | `1/2-1/2` | `*`)
  - `termination_reason: string`
  - `move_list: list[string]`
  - `move_clocks: list[MoveClockEntry]|null`
  - `final_fen: string`
  - `duration_seconds: float`
  - `white_time_remaining: float`
  - `black_time_remaining: float`
- Returns 204 on success; 400 if `game_id` mismatch.

### Get Game Record
- `GET /games/{game_id}`
- Auth: none
- Path params:
  - `game_id: UUID`
- Returns 200 (`GameRecordResponse`) with players, usernames, mode, time control, outcome,
  move list, and rating info; 404 if not found.
-
### Spectate (issue token)
- `GET /games/{game_id}/spectate`
- Auth: not required
- Path params:
  - `game_id: UUID`
- Returns 200 JSON: `{ "websocket_url": "<path>", "game_token": "<jwt>" }`
- Notes:
  - Issues a short-lived spectator game token (same TTL as player game tokens) and a
    websocket path clients can use to connect and watch the live game.
  - Spectating is only allowed for games with status `ACTIVE`.
  - The server treats spectators as read-only: action messages (moves, resign, offers)
    sent by spectator clients are rejected and authoritative game state is returned.
  - Consider rate-limiting this endpoint to reduce abuse or token scraping.

### Game Replay (positions)
- `GET /games/{game_id}/replay`
- Auth: none
- Path params:
  - `game_id: UUID`
- Returns 200 (`GameReplayResponse`) with move-by-move positions and per-move clocks; 404 if not found.

### Game PGN
- `GET /games/{game_id}/pgn`
- Auth: none
- Path params:
  - `game_id: UUID`
- Returns 200 text/PGN content; 404 if not found.

---

## History & Listings

### Query Games
- `GET /games`
- Auth: required
- Query filters:
  - `player_id: UUID|null`
  - `opponent_id: UUID|null`
  - `mode: GameMode|null`
  - `outcome: GameOutcome|null`
  - `status: GameStatus|null`
  - `rating_category: string|null` (`bullet` | `blitz` | `rapid` | `classical`)
  - `result_for_me: string|null` (`win` | `loss` | `draw`)
  - `played_color: string|null` (`white` | `black`)
  - `min_rating: int|null`
  - `max_rating: int|null`
  - `date_from: string|null` (ISO datetime)
  - `date_to: string|null` (ISO datetime)
  - `sort_by: string` (`created_at` | `rating` | `move_count`, default `created_at`)
  - `sort_order: string` (`asc` | `desc`, default `desc`)
  - `limit: int` (1-100, default 20)
  - `offset: int` (default 0)
- Returns 200 (`GameHistoryResponse`) with `games`, `total`, `limit`, `offset`.

### Recent Active Games (public)
- `GET /games/active`
- Auth: not required
- Query:
  - `limit: int` (1-100, default 20)
  - `offset: int` (default 0)
- Returns 200 (`GameHistoryResponse`) with a paginated list of currently active games (status `ACTIVE`).
- Notes:
  - This is a public endpoint intended for listing recently started/ongoing games.
  - Responses are short-lived cached (small TTL) to reduce load on the History service.

## Ratings

### Current Rating
- `GET /users/{user_id}/rating`
- Auth: required
- Path params:
  - `user_id: UUID`
- Returns 200 (`PlayerRatingResponse`) with per-category rating fields, totals, W/L/D, peak, RD, last game date.
- If no rating exists and caller is the same user, a default rating is created; otherwise 404.

### Rating History
- `GET /users/{user_id}/rating/history`
- Auth: not required
- Path params:
  - `user_id: UUID`
- Query:
  - `days: int` (1-3650, default 30)
  - `rating_category: string|null` (`bullet` | `blitz` | `rapid` | `classical`)
- Response: `200 RatingHistoryResponse` with `history` entries ordered newest first.

### User Stats
- `GET /users/{user_id}/stats`
- Auth: required
- Path params:
  - `user_id: UUID`
- Query:
  - `rating_category: string` (`bullet` | `blitz` | `rapid` | `classical`)
- Response: `200 UserStatsResponse`
  - Week/month windows use trailing 7/30 days based on game end time (or creation time if missing).

### Categories & Defaults
- The service infers which rating category to use from the game's time control using
  `time_control_category(initial_seconds, increment_seconds)`.
- Current thresholds (as implemented in `src/main_service/rating.py`):
  - `bullet`: `initial_seconds <= 120`
  - `blitz`: `initial_seconds <= 300`
  - `rapid`: `initial_seconds <= 1800`
  - `classical`: otherwise
- New players get a canonical default rating when created via the rating service. The canonical
  default value is exposed via `DEFAULT_RATING` in the code.

---

## Profiles & Preferences

### Get My Profile
- `GET /profile`
- Auth: required
- Response: `200 UserProfileResponse` (includes `rating_bullet`, `rating_blitz`, `rating_rapid` when available)

### Update My Profile
- `PUT /profile`
- Auth: required
- Body (`UserProfileUpdate`):
  - `display_name: string|null`
  - `bio: string|null`
  - `country: string|null` (2-letter code)
  - `avatar_url: string|null`
  - `profile_visibility: string|null` (`public` | `friends` | `private`)
  - `show_rating: bool|null`
  - `show_game_history: bool|null`
  - `allow_challenges: bool|null`
- Response: `200 UserProfileResponse`

### Get My Preferences
- `GET /preferences`
- Auth: required
- Response: `200 UserPreferencesResponse`

### Update My Preferences
- `PUT /preferences`
- Auth: required
- Body (`UserPreferencesUpdate`):
  - `theme: string|null` (`light` | `dark` | `auto`)
  - `board_theme: string|null`
  - `piece_set: string|null`
  - `show_coordinates: bool|null`
  - `show_legal_moves: bool|null`
  - `auto_queen: bool|null`
  - `premove_enabled: bool|null`
  - `sound_enabled: bool|null`
  - `move_confirmation: bool|null`
  - `email_notifications: bool|null`
  - `game_invites: bool|null`
  - `game_results: bool|null`
- Response: `200 UserPreferencesResponse`

### Public Profile
- `GET /users/{user_id}/public`
- Auth: not required
- Path params:
  - `user_id: UUID`
- Response: `200 UserPublicProfile` (fields redacted according to `profile_visibility`;
  includes rating summary only if `show_rating` is enabled)

---

## Feedback

### Submit Feedback
- `POST /feedback`
- Auth: not required
- Body (`FeedbackCreateRequest`):
  - `category: string` (1-50)
  - `content: string` (1-2000)
- Response: `201 FeedbackCreateResponse` with `{ feedback_id: int }`

---

## Health

- `GET /health`
- Auth: not required
- Response: `{ "status": "healthy", "service": "main-service", "version": "<ver>" }`

---

## Notes

- All UUIDs are lowercase canonical strings.
- Time controls are expressed in seconds.
- Move lists and clock details are validated server-side; invalid submissions return 400 with a
  `detail` message.
- For rated games, ratings are updated during result submission; matchmaking uses current
  per-category rating when joining queue.

---

## Models (summaries)

- `GameCreationRequest`: `player_white_id: UUID`, `player_black_id: UUID`,
  `time_control_initial: int`, `time_control_increment: int`, `mode: GameMode`,
  `starting_fen: string|null`
- `GameCreationResponse`: `game_id: UUID`, `status: GameStatus`, `websocket_url: string`
  - `websocket_url` is a path (not a full URL).
- `GameResultSubmission`: `game_id: UUID`, `result: string`, `termination_reason: string`,
  `move_list: list[string]`, `final_fen: string`, `duration_seconds: float`,
  `white_time_remaining: float`, `black_time_remaining: float`
- `GameRecordResponse`: `game_id: UUID`, `player_white_id: UUID`, `player_black_id: UUID`,
  `white_username: string|null`, `black_username: string|null`, `time_control_initial: int`,
  `time_control_increment: int`, `rating_category: string`, `starting_fen: string`,
  `mode: GameMode`, `status: GameStatus`, `outcome: GameOutcome|null`,
  `termination_reason: string|null`, `move_list: string|null`, `final_fen: string|null`,
  `move_count: int`, `duration_seconds: float|null`, `white_time_remaining: float|null`,
  `black_time_remaining: float|null`, `created_at: datetime`, `started_at: datetime|null`,
  `ended_at: datetime|null`, `white_rating_before: int|null`, `black_rating_before: int|null`,
  `white_rating_after: int|null`, `black_rating_after: int|null`
- `GameReplayResponse`: `game_id: UUID`, `player_white_id: UUID`, `player_black_id: UUID`,
  `mode: GameMode`, `time_control_initial: int`, `time_control_increment: int`, `starting_fen: string`,
  `moves: list[GameReplayMove]`, `outcome: GameOutcome|null`, `termination_reason: string|null`
- `GameReplayMove`: `move_number: int`, `white_move: string|null`, `black_move: string|null`,
  `fen_after_white: string|null`, `fen_after_black: string|null`,
  `white_clock: ClockSnapshot|null`, `black_clock: ClockSnapshot|null`
- `ClockSnapshot`: `white_time_remaining: float`, `black_time_remaining: float`
- `GameHistoryResponse`: `games: list[GameRecordResponse]`, `total: int`, `limit: int`, `offset: int`
- `PlayerRatingResponse`: `user_id: UUID`, `rating_bullet: int`, `rating_blitz: int`,
  `rating_rapid: int`, `rd: int|null`, `volatility: float|null`, `games_played: int`,
  `games_won: int`, `games_lost: int`, `games_drawn: int`, `peak_rating: int`,
  `peak_rating_date: datetime|null`, `last_game_at: datetime|null`, `updated_at: datetime`
- `RatingHistoryResponse`: `user_id: UUID`, `history: list[RatingHistoryPoint]`, `total_games: int`
- `RatingHistoryPoint`: `game_id: UUID`, `rating_before: int`, `rating_after: int`,
  `rd_before: int`, `rd_after: int`, `opponent_id: UUID`, `opponent_rating: int`,
  `game_outcome: string`, `rating_category: string|null`, `recorded_at: datetime`
- `RatingTrendPoint`: `recorded_at: datetime`, `rating: int`
- `StreakStats`: `current_type: string|null`, `current_count: int`, `longest_win: int`,
  `longest_loss: int`, `longest_draw: int`
- `UserStatsResponse`: `user_id: UUID`, `rating_category: string`, `games_played: int`,
  `wins: int`, `draws: int`, `losses: int`, `streaks: StreakStats`,
  `rating_trend: list[RatingTrendPoint]`, `peak_rating: int|null`, `lowest_rating: int|null`,
  `games_played_week: int`, `games_played_month: int`, `average_games_per_day: float`,
  `most_games_in_one_day: int`, `avg_time_per_move_seconds: float|null`,
  `avg_game_length_moves: float|null`, `avg_game_duration_seconds: float|null`,
  `avg_opponent_rating: float|null`, `best_wins: list[BestWinEntry]`
- `BestWinEntry`: `game_id: UUID`, `opponent_id: UUID`, `opponent_username: string|null`,
  `opponent_rating_before: int`, `created_at: datetime`
- `UserProfileResponse`: `user_id: UUID`, `display_name: string|null`, `bio: string|null`,
  `country: string|null`, `avatar_url: string|null`, `profile_visibility: string`,
  `show_rating: bool`, `show_game_history: bool`, `allow_challenges: bool`,
  `rating_bullet: int|null`, `rating_blitz: int|null`, `rating_rapid: int|null`,
  `games_played: int|null`, `games_won: int|null`, `games_lost: int|null`, `games_drawn: int|null`,
  `created_at: datetime`, `updated_at: datetime`
- `UserProfileUpdate`: `display_name: string|null`, `bio: string|null`, `country: string|null`,
  `avatar_url: string|null`, `profile_visibility: string|null`, `show_rating: bool|null`,
  `show_game_history: bool|null`, `allow_challenges: bool|null`
- `UserPreferencesResponse`: `user_id: UUID`, `theme: string`, `board_theme: string`,
  `piece_set: string`, `show_coordinates: bool`, `show_legal_moves: bool`, `auto_queen: bool`,
  `premove_enabled: bool`, `sound_enabled: bool`, `move_confirmation: bool`,
  `email_notifications: bool`, `game_invites: bool`, `game_results: bool`,
  `created_at: datetime`, `updated_at: datetime`
- `UserPreferencesUpdate`: `theme: string|null`, `board_theme: string|null`,
  `piece_set: string|null`, `show_coordinates: bool|null`, `show_legal_moves: bool|null`,
  `auto_queen: bool|null`, `premove_enabled: bool|null`, `sound_enabled: bool|null`,
  `move_confirmation: bool|null`, `email_notifications: bool|null`, `game_invites: bool|null`,
  `game_results: bool|null`
- `UserPublicProfile`: `user_id: UUID`, `username: string`, `display_name: string|null`,
  `bio: string|null`, `country: string|null`, `avatar_url: string|null`,
  `account_state: AccountState`, `created_at: datetime`, `rating_bullet: int|null`,
  `rating_blitz: int|null`, `rating_rapid: int|null`, `games_played: int|null`,
  `games_won: int|null`, `games_lost: int|null`, `games_drawn: int|null`
- `FeedbackCreateRequest`: `category: string`, `content: string`
- `FeedbackCreateResponse`: `feedback_id: int`
