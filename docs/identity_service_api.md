# Identity Service API (v1)

Authentication endpoints only. All routes are under the identity base path (for example, `/api/v1/auth`). Unless noted, responses are JSON.

---

## Authentication

- Access tokens are JWTs returned by `/login` and must be sent via `Authorization: Bearer <token>`.
- Token payload includes `user_id`, `username`, and `account_state`.
- Expiration is returned as `expires_in` (seconds) in the login response.

---

## Endpoints

### Register
- `POST /register`
- Body (`UserRegistration`):
  - `username: string` (3-50)
  - `email: string` (email format)
  - `password: string` (8-128; must include upper, lower, digit)
- Auth: not required
- Responses:
  - `201 UserResponse` `{ user_id, username, email, account_state, created_at, last_login_at }`
  - `400` on validation errors (duplicate username/email, weak password, invalid format)

### Login
- `POST /login`
- Form-encoded (OAuth2 Password):
  - `username: string`
  - `password: string`
- Auth: not required
- Responses:
  - `200 TokenResponse` `{ access_token, token_type: "bearer", expires_in, user: UserResponse }`
  - `401` invalid credentials
  - `403` account issues (e.g., locked/disabled)

### Token Refresh
- `POST /refresh`
- Auth: not required (uses server-side refresh session)
- Behavior:
  - Browser clients: prefer the HttpOnly cookie named `refresh_session_id` (set by `/login`).
  - Non-browser clients: may POST JSON body `{ "session_id": "<uuid>" }` as a fallback.
  - On success the endpoint returns a new access token. If the server rotates the refresh session
    (to extend the session lifetime) a new `refresh_session_id` will be returned in the JSON body
    and an HttpOnly cookie `refresh_session_id` will be set for browser clients.
  - Rotation is conservative and only performed when the server decides the existing session
    is close to expiry.
- Responses:
  - `200` `{ access_token, refresh_session_id|null, token_type: "bearer", expires_in, refresh_expires_in }`
  - `400` when neither cookie nor `session_id` provided
  - `401` when the session is invalid or expired

### Me
- `GET /me`
- Auth: required
- Response: `200 UserResponse`

### Logout (invalidate sessions)
- `POST /logout`
- Auth: required
- Response: `204`

### Batch User Lookup
- `POST /users/batch`
- Auth: not required
- Body (`UserBatchRequest`):
  - `user_ids: list[UUID]`
- Response: `200 UserBatchResponse` with `users: list[UserResponse]`.

### Health
- `GET /health`
- Auth: not required
- Response: `{ status: "healthy", service: "identity-service", version: string }`

---

## Models (summaries)

- `UserResponse`: `user_id: UUID`, `username: string`, `email: string`, `account_state: AccountState`,
  `created_at: datetime`, `last_login_at: datetime|null`
- `TokenResponse`: `access_token: string`, `token_type: string`, `expires_in: int`,
  `refresh_session_id: UUID|null`, `refresh_expires_in: int`, `user: UserResponse`
- `UserBatchRequest`: `user_ids: list[UUID]`
- `UserBatchResponse`: `users: list[UserResponse]`

---

## Notes

- All UUIDs are canonical lowercase strings.
- Error format uses FastAPI default `{ "detail": "..." }`.
