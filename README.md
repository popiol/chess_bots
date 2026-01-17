# chess_bots

Agents that use a chess web app like humans: register, sign in, join games, play, analyze, and review stats.

## Goals
- Build reliable, human-like web automation for common chess site flows.
- Support end-to-end play loops, from account creation to post-game analysis.
- Keep behavior consistent, observable, and easy to debug.
- Support multiple agent implementations with different decision approaches.

## Scope (current)
- Account lifecycle: register, sign in, sign out.
- Game lifecycle: join/host, play moves, handle resign/draw, post-game analysis.
- Stats and history: fetch recent games, view stats, verify results.

## Non-goals (for now)
- Beating top-tier engines.
- Evading site anti-bot systems or bypassing ToS.
- Large-scale automation across many accounts.

## How it works (high level)
- Web driver sessions simulate user actions in the browser.
- Agents maintain state and goals for a full game loop.
- Logs and artifacts capture decisions, timings, and page state.
- Agent implementations can range from rule-based play to ML-driven policies.

## Status
Early development.

## Next ideas
- Define target site selectors and page flows.
- Add a minimal play loop with a simple move policy.
- Capture structured game artifacts (PGN, analysis outputs, stats snapshots).
