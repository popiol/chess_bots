# WebSocket Message Contract - v1

This document defines the authoritative WebSocket protocol between frontend and backend for live chess games.

This WebSocket API is part of the data plane and is implemented by the Single-Game Service.

It is used exclusively for real-time game execution.
All non-real-time operations (game creation, matchmaking, history, analysis, tournaments, etc.) are handled via control-plane REST APIs exposed by the Main Application Service.

The contract is intentionally conservative, explicit, and versioned. Breaking changes require a new version.

---

## Design Principles

- Server is authoritative
- Client is optimistic but correctable
- Messages are small, explicit, and typed
- No implicit state transitions
- No UI-driven messages

If a message exists only to drive UI convenience, it does not belong here.

---

## Service Ownership and Hosting

- This WebSocket API runs with the Single-Game Service
- It is not hosted by the Main Application (orchestration) service
- Each WebSocket connection corresponds to one live game

Typical flow:

1. Client uses REST API (control plane) to create or join a game
2. Orchestration service returns:
   - `game_id`
   - WebSocket endpoint (`ws_url`)
   - Short-lived access token
3. Client connects directly to the game service WebSocket

The orchestration service is not in the live message path.

---

## Connection Lifecycle

### WebSocket Connection

Live games use WebSocket connections for real-time communication:

```javascript
// Connect to a game WebSocket endpoint provided by orchestration
const ws = new WebSocket(
  'wss://{game-service-host}/ws/game/v1/{game_id}?game_token={game_token}'
);

// Receive state updates
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'state_update') {
    // Update UI with authoritative state
    console.log(message.data);
  } else if (message.type === 'game_over') {
    // Game has ended
    console.log(message.data);
  }
};

// Send a move
ws.send(JSON.stringify({
  type: 'move',
  data: {
    from_square: 'e2',
    to_square: 'e4',
    promotion: null
  }
}));

// Resign
ws.send(JSON.stringify({ type: 'resign', data: {} }));

// Offer draw
ws.send(JSON.stringify({ type: 'offer_draw', data: {} }));

// Accept draw
ws.send(JSON.stringify({ type: 'accept_draw', data: {} }));

// Keep-alive ping
ws.send(JSON.stringify({ type: 'ping', data: {} }));
```

### Important

The WebSocket layer is server-authoritative. Client-side optimistic updates must always be reconciled with server state.

---

## Envelope Format (All Messages)

Every message MUST follow this structure:

```json
{
  "type": "string",
  "data": {}
}
```

- `type: string` (message identifier)
- `data: object` (message-specific content)

---

## Client -> Server Messages

### 1. `move`

Sent when player makes a move.

```json
{
  "type": "move",
  "data": {
    "from_square": "e2",
    "to_square": "e4",
    "promotion": "q"
  }
}
```

- `from_square: string` (algebraic square, e.g., `e2`)
- `to_square: string` (algebraic square, e.g., `e4`)
- `promotion: string|null` (`q`, `r`, `b`, `n`, or null)

Rules:

- Server validates legality
- Invalid moves result in an `error` message

---

### 2. `resign`

Resign from the game.

```json
{
  "type": "resign",
  "data": {}
}
```

---

### 3. `offer_draw`

Offer a draw to the opponent.

```json
{
  "type": "offer_draw",
  "data": {}
}
```

---

### 4. `accept_draw`

Accept a pending draw offer.

```json
{
  "type": "accept_draw",
  "data": {}
}
```

---

### 5. `ping`

Keep-alive ping. Server responds with `pong`.

```json
{
  "type": "ping",
  "data": {}
}
```

---

## Server -> Client Messages

### 1. `state_update`

Sent on connection and after successful moves while game is ongoing (`result == "*"`).

```json
{
  "type": "state_update",
  "data": {
    "game_id": "550e8400-e29b-41d4-a716-446655440000",
    "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "move_list": ["e2e4"],
    "current_turn": "black",
    "move_number": 1,
    "result": "*",
    "termination_reason": null,
    "clocks": {
      "white_seconds": 598,
      "black_seconds": 600,
      "last_update": "2025-12-20T12:00:00.000000"
    },
    "first_move_time_remaining": 9.5,
    "ended_at": null
  }
}
```

Fields:
- `game_id: UUID`
- `fen: string`
- `move_list: list[string]` (UCI moves)
- `current_turn: string` (`white` | `black`)
- `move_number: int`
- `result: string` (`*` | `1-0` | `0-1` | `1/2-1/2`)
- `termination_reason: string|null`
- `clocks: object`
  - `white_seconds: float`
  - `black_seconds: float`
  - `last_update: datetime`
- `ended_at: datetime|null`
- `first_move_time_remaining: float|null` (seconds remaining for the first move; present only before the first move)

Rules:

- This message is the source of truth for game state
- Sent automatically on connect
- Broadcast to all participants after each move

---

### 2. `game_over`

Sent when the game ends. Same payload structure as `state_update` but with `result != "*"`.

```json
{
  "type": "game_over",
  "data": {
    "game_id": "550e8400-e29b-41d4-a716-446655440000",
    "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
    "move_list": ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
    "current_turn": "black",
    "move_number": 4,
    "result": "1-0",
    "termination_reason": "checkmate",
    "clocks": {
      "white_seconds": 595,
      "black_seconds": 597,
      "last_update": "2025-12-20T12:05:00.000000"
    },
    "ended_at": "2025-12-20T12:05:00.000000"
  }
}
```

- `result: string` (`1-0` | `0-1` | `1/2-1/2`)
- `termination_reason: string|null` (`checkmate`, `resignation`, `timeout`, `draw`, or null)
- `ended_at: datetime`

---

### 3. `draw_offered`

Notifies that a player has offered a draw.

```json
{
  "type": "draw_offered",
  "data": {
    "by": "white"
  }
}
```

- `by: string` (`white` | `black`)

---

### 4. `error`

Sent when a client action fails or validation error occurs.

```json
{
  "type": "error",
  "data": {
    "message": "Invalid move format: ..."
  }
}
```

- `message: string`

Rules:

- Errors are recoverable; connection remains open
- Invalid token or unauthorized access closes the connection

---

### 5. `pong`

Response to `ping`.

```json
{
  "type": "pong",
  "data": {}
}
```

---

## Connection Behavior

- On connect, server validates `game_token` JWT
- Token must contain:
  - `game_id: string` (UUID)
  - `user_id: string` (UUID)
  - `player_white_id: string` (UUID)
  - `player_black_id: string` (UUID)
  - `time_control_initial: int` (seconds)
  - `time_control_increment: int` (seconds)
  - `rated: bool`
  - `mode: string` (`casual` | `rated`)
  - `starting_fen: string`
- Server verifies `game_id` in token matches URL parameter
- Server verifies `user_id` is either `player_white_id` or `player_black_id`
- Server creates game instance from token data if it does not already exist
- Initial `state_update` sent only to connecting client
- Subsequent state updates broadcast to all participants of the game

---

## Anti-Cheat Boundary

- Client never sends:
  - FEN
  - Clock values
  - Game status

- Server never trusts client timing

All critical state lives server-side.

---

## Versioning

This document defines v1.

Breaking changes require:

- New message types, or
- New endpoint path (`/ws/game/v2/...`)

---

## Final Principle

If frontend and backend disagree, the server is always right.

This contract prioritizes correctness, simplicity, and long-term stability over cleverness.
