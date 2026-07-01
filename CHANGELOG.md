# Changelog

All notable changes to this integration.

## [1.3.0] - 2026-07-01

### Added

- **Token usage sensors** per config entry: cumulative prompt, completion, total, reasoning, and API request counters, plus last-request total with dedicated prompt/completion sensors (reasoning remains on the last-request attributes when the API reports it). **Reset usage** button on the device to zero all counters manually.
- **Reauthentication** when your API key expires or is rejected — fix credentials in Home Assistant settings without removing and re-adding the integration.
- **Reconfigure** (integration card ⋮ → Reconfigure) to change your API key or base URL while keeping Assist and model options unchanged.
- **More translations** for the `generate_content` service, invalid config entry errors, reasoning effort labels, reconfigure, options field hints, and the debug notification (EN, DE, FR, ZH).
- **DeepSeek brand icons** in the integrations list and config flow (Home Assistant 2026.3+).
- **Configurable max tool iterations** (1–20, default 10) in integration options for the Assist tool loop.
- **`generate_content` per-call overrides**: optional `chat_model`, `temperature`, `thinking_enabled`, `max_tokens`, and `response_format: json_object`. Unset fields use the config entry options.

### Improved

- **More reliable device control**: Home Assistant tools with invalid schemas are skipped instead of being sent with empty parameters, which previously caused opaque API errors.
- **Faster option changes**: updates to prompt, model, temperature, thinking, and similar options apply immediately in Assist without reloading the whole integration. Reload only runs for connection settings (base URL, API key).
- **Clearer configuration UX**: the gear icon opens Assist and model options directly; API key and base URL are changed only via Reconfigure (not mixed into options). API keys use a password field in setup, reauth, and reconfigure.
- **Options form**: all fields stay visible (reasoning, reasoning effort, temperature, top_p); one OK saves everything — no form reload when toggling reasoning.
- **Quicker setup and reauth**: credentials are verified via `models.list` instead of a chat completion, so setup does not consume tokens.
- **`generate_content` with reasoning**: temperature and top_p are no longer sent when reasoning is enabled (consistent with Assist).
- **Simplified the streaming tool loop** to match the stock Ollama/OpenAI conversation integrations: one API round per `async_add_delta_content_stream`, the assistant role carried on each round's first delta, and tool execution triggered by the stream ending. Removed the earlier Assist-UI workarounds — the single continuous stream with explicit tool-boundary deltas, the zero-width-space "priming" between rounds, and the final combined role+content push — which were attempts to work around the frontend bug above. They added complexity and polluted the streamed thinking/answer text without ever fixing the display.

### Fixed

- **Reasoning off on DeepSeek V4**: the API defaults to thinking enabled when `extra_body` is omitted; the integration now sends `thinking: disabled` explicitly and no longer shows the “Details” reasoning block in Assist when reasoning is turned off.
- **Reasoning on**: `reasoning_effort` (e.g. low, high, xhigh) is sent correctly with `thinking: enabled`; temperature and top_p are omitted as required by the API.
- **Reasoning off on other endpoints**: DeepSeek-specific `extra_body` is only sent for DeepSeek model IDs, so custom OpenAI-compatible gateways are not sent thinking fields unless the model id indicates DeepSeek.
- **Assist chat now shows the final answer after tool calls.** With multi-step tool use (for example several `GetLiveContext` lookups followed by an action such as turning a light off), the Assist chat could show the preamble text and the tool calls but drop the final answer, even though the model replied correctly and the action was carried out. Root cause is a Home Assistant **frontend** bug ([frontend #52753](https://github.com/home-assistant/frontend/pull/52753)): expanding the "Details" / thinking section while the response is still streaming replaced the chat bubble with a new object and detached it from the streaming processor, so the streamed answer *and* the authoritative `intent-end` text were written to an orphaned object and never rendered. This is fixed in the frontend that ships with **Home Assistant 2026.7** (frontend `20260624.0`+). On 2026.6 the workaround is to not open the thinking details until the answer has finished streaming.