# gbrain-retrieval-reflex — ambient brain context on every turn

A Hermes Agent plugin that puts the **top ranked pages from a GBrain brain** in
front of the model before it answers, without the agent deciding to look them
up.

On `pre_llm_call` it:

1. Rebuilds a short multi-turn window of the recent conversation (prior
   injected blocks are stripped out, so the plugin never feeds itself).
2. Calls GBrain's `volunteer_context` tool over HTTP MCP — this resolves named
   entities, so a turn that mentions a person or project surfaces that page.
3. Also runs a hybrid topical `query`, so a turn with no nameable entity still
   gets its subject matter.
4. Merges, de-duplicates and ranks: entity hits first (precision), then query
   hits by score, capped at `GBRAIN_RETRIEVAL_REFLEX_MAX_POINTERS`.
5. Injects `## Brain pages (ambient push)` with short previews into the turn's
   context. A text byte cap bounds the injected payload.

If GBrain is down, slow, or has no bearer token, the hook returns `None`
silently. It never blocks a turn and never raises into the agent loop.

## Requirements

- A **GBrain** instance reachable over **HTTP MCP**, with a bearer token.
  There is no bundled brain: this is the client half.
- Optional: a `.gbrain-resolve.sock` for the resolve IPC fast path. Current
  `gbrain serve --http` does not bind it (stdio serve only), so the HTTP path
  is the normal one.

## Install

```bash
hermes plugins install <owner>/hermes-gbrain-retrieval-reflex
hermes plugins enable gbrain-retrieval-reflex
```

Manual install: copy this directory into
`~/.hermes/plugins/gbrain-retrieval-reflex` and add it to `plugins.enabled`.

## Configuration

Nothing is user-specific in the code: state, tokens and endpoints all come
from the environment.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GBRAIN_MCP_URL` | `http://127.0.0.1:3131/mcp` | HTTP MCP endpoint of the brain. |
| `GBRAIN_TOKEN` | – | Bearer token. Usually injected by Hermes from `$HERMES_HOME/.env`. |
| `GBRAIN_HOME` | – | GBrain state directory itself (the one holding `hermes-mcp.token` / `brain.pglite`). When unset, `$HOME/.gbrain` and `$HERMES_HOME/.gbrain` are tried, in that order. |
| `GBRAIN_RESOLVE_SOCKET` | – | Explicit resolve IPC socket path; otherwise derived from the state directories. |
| `GBRAIN_RESOLVE_IPC_TIMEOUT_S` | `0.25` | Budget for the optional IPC fast path. |
| `GBRAIN_VOLUNTEER_HTTP_TIMEOUT_S` | `8.0` | Per-call HTTP timeout. |
| `GBRAIN_RETRIEVAL_REFLEX_MAX_POINTERS` | `5` | Maximum pages injected per turn. |
| `GBRAIN_RETRIEVAL_REFLEX_MIN_CONF` | `0.6` | Minimum strength for an entity hit to be kept. |
| `GBRAIN_RETRIEVAL_REFLEX_HISTORY_TURNS` | `6` | Turns of context sent to `volunteer_context`. |
| `GBRAIN_RETRIEVAL_REFLEX_SYNOPSIS_MAX` | `400` | Characters of page preview per pointer. |
| `GBRAIN_RETRIEVAL_REFLEX_MAX_CONTEXT_BYTES` | `3200` | Byte cap for the whole injected block. |
| `GBRAIN_RETRIEVAL_AUDIT` | – | Write one JSON line per retrieval here (diagnostics only). |

## Design notes

- **Never shells out.** No `gbrain` CLI process, no second writer, no direct
  PGLite access — the plugin is an MCP client and nothing else.
- **Trivial turns are skipped** (`ok`, `thanks`, …) so a one-word reply does
  not pay for a retrieval round-trip.
- **Soft failure everywhere**: connect errors, timeouts, malformed SSE, and
  missing credentials all degrade to "no context injected" — logged at debug
  or info level, never surfaced as an error to the model.
- **Bounded cost**: one volunteer call plus one query call per non-trivial
  turn, both with explicit timeouts.

## Tests

```bash
python -m pytest tests/ -q
```

Stock CPython — no GBrain, no network, no sockets. The suite covers state-path
resolution (`GBRAIN_HOME` precedence, de-duplication), the no-baked-in-paths
invariant, message normalization, injected-block stripping, window building,
merge/rank/dedupe and capping, the trivial-message gate, and fail-soft
behaviour when the endpoint errors or the token is absent.

## License

MIT — see [LICENSE](LICENSE).
