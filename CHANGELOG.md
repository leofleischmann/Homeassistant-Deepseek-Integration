# Changelog

All notable changes to this integration.

## [1.3.0] - 2026-07-01

### Added

- **Vision in Assist**: image attachments on `UserContent` (Assist chat, AI Task) are sent to DeepSeek V4 as `image_url` parts; shared encoding with `generate_content` via `vision.py`. Legacy `deepseek-reasoner` is rejected when images are attached.
- **Allow vision option**: toggle in integration options (default on). When off, image attachments and `generate_content` filenames are rejected. When on and Home Assistant supports it, the conversation entity advertises `SUPPORT_ATTACHMENTS` for the Assist attach UI.
- **Token usage monitoring** on the integration device (per config entry), updated after Assist and `generate_content` calls (`run_debug` does not count):
  - Cumulative sensors (persist across restarts): `prompt_tokens`, `completion_tokens`, `total_tokens`, `reasoning_tokens`, `api_requests`
  - Last-request sensors: `last_request_tokens` (with prompt, completion, reasoning, source, and request count as attributes), `last_request_prompt_tokens`, `last_request_completion_tokens`
  - **Reset usage** button to zero all counters manually (replaces the earlier auto-reset on Reconfigure)
  - `generate_content` service responses include a `usage` token breakdown when the API reports it
- **Context management**: optional shortening of large Home Assistant tool result JSON before API calls (default 12 000 chars per tool result; set **Max tool result size** to `0` to disable) and optional **Max conversation rounds** for Assist history (default `0` = unlimited). Older complete user turns are dropped before each API call; the system prompt and the current round including tool chains stay intact.
- **Reauthentication** when your API key expires or is rejected: fix credentials in Home Assistant settings without removing and re-adding the integration.
- **Reconfigure** (integration card ⋮ -> Reconfigure, or shortcut in the options menu) to change your API key or base URL while keeping Assist and model options unchanged.
- **`generate_content` per-call overrides**: optional `chat_model`, `temperature`, `thinking_enabled`, `max_tokens`, and `response_format: json_object`. Unset fields use the config entry options.
- **`generate_content` reasoning in response**: when reasoning is enabled for the call, the service response includes a `reasoning` field with the model's thinking text alongside `text` and `usage`.
- **Configurable max tool iterations** (1–20, default 10) in integration options for the Assist tool loop.
- **DeepSeek brand icons** in the integrations list and config flow (Home Assistant 2026.3+).
- **More translations** for the `generate_content` service, invalid config entry errors, reasoning effort labels, reconfigure, options field hints, sensor names, and the debug notification (EN, DE, FR, ZH).

### Improved

- **Simplified streaming tool loop** aligned with the stock Ollama/OpenAI conversation integrations: one API round per `async_add_delta_content_stream`, assistant role on each round's first delta, tool execution when the stream ends.
- **More reliable device control**: Home Assistant tools with invalid schemas are skipped instead of being sent with empty parameters, which previously caused opaque API errors.
- **Faster option changes**: updates to prompt, model, temperature, thinking, and similar options apply immediately in Assist without reloading the whole integration. Reload only runs for connection settings (base URL, API key).
- **Clearer configuration UX**: the gear icon opens Assist and model options directly; API key and base URL are changed only via Reconfigure (not mixed into options). API keys use a password field in setup, reauth, and reconfigure.
- **Options form**: all fields stay visible (reasoning, reasoning effort, temperature, top_p); one OK saves everything with no form reload when toggling reasoning.
- **Quicker setup and reauth**: credentials are verified via `models.list` instead of a chat completion, so setup does not consume tokens.
- **`generate_content` with reasoning**: temperature and top_p are no longer sent when reasoning is enabled (consistent with Assist).
- **Shared vision encoding** for Assist attachments and `generate_content` filenames (`vision.py`).

### Fixed

- **Assist chat shows the final answer after tool calls.** With multi-step tool use (for example several `GetLiveContext` lookups followed by an action such as turning a light off), the Assist chat could show the preamble text and the tool calls but drop the final answer, even though the model replied correctly and the action was carried out. Fixed in the Home Assistant frontend that ships with **2026.7** ([frontend #52753](https://github.com/home-assistant/frontend/pull/52753)). On 2026.6, wait until streaming finishes before opening the thinking details.
- **Reasoning off on DeepSeek V4**: the API defaults to thinking enabled when `extra_body` is omitted; the integration now sends `thinking: disabled` explicitly and no longer shows the “Details” reasoning block in Assist when reasoning is turned off.
- **Reasoning on**: `reasoning_effort` (e.g. low, high, xhigh) is sent correctly with `thinking: enabled`; temperature and top_p are omitted as required by the API.
- **Reasoning off on other endpoints**: DeepSeek-specific `extra_body` is only sent for DeepSeek model IDs, so custom OpenAI-compatible gateways are not sent thinking fields unless the model id indicates DeepSeek.
- **Vision attachments** use the resolved image MIME type instead of always sending `image/jpeg`.
- **`generate_content` with filenames**: fixed image encoding (`async_add_executor_job` `strict` argument). The service now fails with a clear error when no image could be read instead of calling the API with text only (which caused hallucinated descriptions).
