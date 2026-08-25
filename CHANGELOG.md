# Changelog

All notable changes to this integration.

## [1.7.0] - 2026-08-26

### Fixed
- **The legacy model ids stopped working.** DeepSeek retired `deepseek-chat` and `deepseek-reasoner` on 2026-07-24, but the model picker still offered them — an entry left on one failed every single request with nothing but an API error to go on. They are gone from the picker; an entry still set to one is moved to `deepseek-v4-flash` on startup and the change is reported under **Settings → Repairs** rather than happening silently. Entries pointing at a custom base URL are never touched, because a gateway may still route those ids somewhere. Typing a retired id back into the model field is refused by the form, and a retired per-call `chat_model` override on `generate_content` is mapped with a warning.
- **Images were blocked on the official API although it now accepts them.** DeepSeek serves `deepseek-v4-flash-vision-exp`, which takes the same base64 `image_url` parts the integration already sends — but every image was refused whenever the base URL was `api.deepseek.com`, on the assumption that endpoint is text-only. The check now looks at the **model**: on the official API only `deepseek-v4-flash-vision-exp` is accepted and the other models are refused with a message naming it, while a custom base URL is never second-guessed — a gateway may route any model id to any backend, so the request goes through and the API answers for itself. Camera snapshots through Assist, AI Task and `generate_content` no longer need a third-party gateway.
- **Home Assistant was told attachments are supported on models that reject them.** `SUPPORT_ATTACHMENTS` was advertised on the strength of the *Allow vision* option alone, so on a text-only model the attachment button was offered and every use of it ended in an error. The conversation and AI Task entities now require the option *and* a capable model.
- **A stalled API call could hang a voice pipeline for ten minutes.** The client was built without a timeout, so the OpenAI SDK defaults of 600 s and two retries applied to Assist. New **Request timeout** option (default 60 s, range 5–600): on a streamed turn it bounds the gap between two chunks, so a long answer is never cut off while a stalled endpoint fails fast. `generate_content` is not streamed and keeps waiting for at least 300 s, because there the read timeout has to cover the whole generation. Retries are down from two to one.
- **The Brave web search tool had no timeout either.** It runs inside the Assist tool loop, so a search that never came back hung the whole voice turn. Bounded at 10 s.
- The setup form no longer clears every field when the API key or the model is rejected — what you typed comes back with the error.
- **Two shipped files still described the old image behaviour.** The error text for a rejected image claimed the official API is text-only, and the bundled vision sample automation asked `deepseek-v4-flash` to look at a picture — which the new model check refuses before the request is even sent. Both corrected, along with the sample-automation README.
- **Max tokens accepted values the API always rejects.** The ceiling was 1 000 000, which is the size of the *context window*; V4 models generate at most 384 000 tokens. That is now the limit in the option, in the `generate_content` action schema and in the clamp.

### Added
- `deepseek-v4-flash-vision-exp` in the model picker, and an **Images** section in the README covering what each endpoint accepts.
- The `run_debug` report records the request timeout, next to the existing latency profile.

### Removed
- Compatibility code for Home Assistant versions below the declared minimum of 2026.1: the `async_update_reload_and_abort` fallback and the two “legacy core” branches around it (`async_update_and_abort` has been core since 2025.11), the `ai_task` import guard (this integration sets up an AI Task platform, so a core without that component never gets far enough for the guard to help), and the `runtime_data` / `model_dump` attribute probes.

## [1.6.0] - 2026-08-10

### Added
- **The model now knows who is speaking.** Home Assistant only ever resolved `user_name` for a system prompt, and voice satellites run the pipeline without a user account — so in exactly the households this matters for, `{{ user_name }}` rendered the literal string `None`. The prompt now gets the full speaker identity as real Jinja variables: `user_id`, `user_name`, `user_is_admin`, `person_entity_id`, `person_name`, `person_state`, `device_id`, `device_name`, `user_area`, `user_floor`. Unknown values are empty strings, so `{% if user_name %}` works and nothing renders as `None`. Values are injected as `{% set %}` statements ahead of the configured prompt, so Home Assistant still does the single render pass and a user name can never inject Jinja.
- **"Tell the model who is speaking"** option, **off by default**: appends a short speaker block — name, presence, and the area of the satellite that was spoken to — without needing a custom prompt. It is appended last, after the exposed-entity list, so the large speaker-independent part of the system prompt stays byte-identical between household members and DeepSeek's prefix cache keeps hitting. Left off, nothing about the speaker reaches the API and the system prompt is byte-identical to before this release.
- Documentation for the system prompt template variables in the README and in the Configure dialog, where none existed before.

### Fixed
- **`generate_content` sent the configured system prompt unrendered.** It talks to the API without a chat log, so nothing expanded its template and a prompt containing `{{ ... }}` reached the model as literal text. It now renders the same variables as Assist; a template that fails to render falls back to the unrendered prompt instead of failing the service call.
- A caller's `extra_system_prompt` (e.g. from `conversation.process`) is no longer lost on follow-up turns now that the speaker block shares that slot.

## [1.5.0] - 2026-08-07

### Improved
- **HTTP/2 to the API when the endpoint supports it.** The OpenAI SDK ends a streamed completion by breaking out of the SSE iterator at `[DONE]` and closing the response without draining it. Over HTTP/1.1 httpx cannot return such a connection to the pool, so **every** API round opened a new TCP+TLS connection — a full handshake on each tool-calling round trip, which is costly on long-haul or proxied routes. The integration now asks Home Assistant for its HTTP/2-capable shared client; ALPN negotiates per endpoint, so an API without HTTP/2 transparently keeps using HTTP/1.1. Requires `h2` (added to the manifest) and a core that supports `alpn_protocols`; otherwise the previous HTTP/1.1 client is used.

### Added
- **Prompt cache tokens** from the API `usage` object: new cumulative `cache_hit_tokens` sensor, plus `cache_hit_tokens`, `cache_miss_tokens` and `cache_hit_rate` attributes on `last_request_tokens` and in the `generate_content` service response. Reads DeepSeek's `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` and the OpenAI-style `prompt_tokens_details.cached_tokens` used by some gateways.
- **Warning when an endpoint ignores `thinking: disabled`.** DeepSeek V4 reasons by default when the field is absent, so a gateway that drops unknown `extra_body` keys leaves reasoning on: the tokens are generated, billed and waited for, while the integration discards the text. This was only visible at DEBUG level; it now logs one warning per config entry pointing at the `reasoning_tokens` sensor.
- **Latency profiling in `run_debug`.** The report now records the negotiated HTTP version, the installed `h2` version, per-stream time-to-first-chunk, mean inter-chunk gap and a `looks_buffered_by_proxy` flag (a proxy that buffers the reply defeats streaming TTS), token usage per stream test, and a `latency_profile` block that separates connection-setup cost from provider-side inference by timing three back-to-back requests against one after a 16 s idle.

### Fixed
- Nested `completion_tokens_details` / `prompt_tokens_details` are now read from gateways that return them as plain objects rather than dicts or pydantic models; `reasoning_tokens` was silently reported as 0 for those.

## [1.4.0] - 2026-07-24

### Added
- **AI Task entity** for `ai_task.generate_data` in automations (plain text or structured JSON via `structure:`). The HA `structure` schema is converted to JSON Schema and injected into the prompt; official DeepSeek uses `json_object`, custom gateways may use `json_schema`.
- Same **Configure system prompt** and **Home Assistant LLM APIs** as Assist (`llm_api` on the action overrides).
- **Usage sensors** track AI Task calls (`source=ai_task`).
- **Vision attachments** on AI Task when Allow vision is on and the base URL supports multimodal chat.
- Sample automation: `sample_automations/deepseek_ai_task_demo.yaml`.
### Improved
- Shared streaming chat loop (`async_handle_chat_log`) for Assist and AI Task.

## [1.3.2] - 2026-07-21

### Added
- **Optional Brave Search web tool**: set a Brave Search API key on setup or Reconfigure; a `Web Search (Brave)` LLM API is registered and can be enabled under Configure → Home Assistant API alongside Assist. Assist-only (`generate_content` stays without tools).

### Fixed
- **Config entry reload**: Reconfigure/Reauth use `async_update_and_abort`; the conversation update listener schedules reload when `entry.data` changes. Avoids the Home Assistant warning that this pattern breaks in 2026.12.

## [1.3.1] - 2026-07-02

### Fixed
- **HACS installation**: release zip no longer creates a nested `deepseek_conversation/deepseek_conversation/` folder; files extract correctly into `custom_components/deepseek_conversation/`.
- **Recovery from 1.3.0**: if the integration failed to load after installing 1.3.0, update to 1.3.1 via HACS and restart Home Assistant. That restores the correct file layout. Optionally delete `custom_components/deepseek_conversation/` first and reinstall 1.3.1 for a clean folder (an leftover nested subfolder is harmless but may remain after update).

## [1.3.0] - 2026-07-01

### Added

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
- **Vision** (Assist attachments and `generate_content` filenames via `vision.py`): OpenAI-style `image_url` parts; **Allow vision** option (default on) gates input and advertises `SUPPORT_ATTACHMENTS` when Home Assistant supports it; legacy `deepseek-reasoner` is rejected when images are attached; resolved image MIME types; fails with a clear error when no file could be read or when the base URL is the official `api.deepseek.com` endpoint (text-only). Actual image analysis requires a custom OpenAI-compatible base URL with multimodal chat.
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

### Fixed

- **Reasoning off on DeepSeek V4**: the API defaults to thinking enabled when `extra_body` is omitted; the integration now sends `thinking: disabled` explicitly and no longer shows the “Details” reasoning block in Assist when reasoning is turned off.
- **Reasoning on**: `reasoning_effort` (e.g. low, high, xhigh) is sent correctly with `thinking: enabled`; temperature and top_p are omitted as required by the API.
- **`generate_content` with reasoning enabled**: temperature and top_p are no longer sent (consistent with Assist).
- **Reasoning off on other endpoints**: DeepSeek-specific `extra_body` is only sent for DeepSeek model IDs, so custom OpenAI-compatible gateways are not sent thinking fields unless the model id indicates DeepSeek.
- **Assist chat shows the final answer after tool calls.** With multi-step tool use (for example several `GetLiveContext` lookups followed by an action such as turning a light off), the Assist chat could show the preamble text and the tool calls but drop the final answer, even though the model replied correctly and the action was carried out. Fixed in the Home Assistant frontend that ships with **2026.7** ([frontend #52753](https://github.com/home-assistant/frontend/pull/52753)). On 2026.6, wait until streaming finishes before opening the thinking details.
- **Home Assistant 2026.7**: allow `voluptuous-openapi` 0.4.x (Core ships 0.4.0; the previous `<0.4` pin blocked setup).
