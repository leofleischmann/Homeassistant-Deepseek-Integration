# Changelog

All notable changes to this integration.

## [1.8.0] - 2026-08-26

### Added
- **Several agents on one API key.** An entry used to be a single agent, and its settings lived in the entry's own options. Each agent is now a subentry with its own name, prompt, model and tools — so a fast **V4 Flash** agent can answer voice while a **V4 Pro** agent with full tool access handles automations, and an AI Task entity runs on a third prompt entirely. Add them under the integration card: **Add conversation agent** or **Add AI task entity**. Each agent gets its own device, and each can be reconfigured on its own row.
- Adding an agent is a two-step form: name, prompt, Home Assistant APIs and model, then a second step for reply length, reasoning, timeouts and context handling that only opens when **Recommended settings** is switched off. An agent left on the recommended settings stores only what you actually chose, so later changes to a default reach it.

### Changed
- **The second step is grouped and much shorter to read.** Its settings now sit in four collapsible groups — *Reply*, *Tools*, *Conversation* (Assist only) and *Limits and input* — with only the first one open. Every description is a single line; the reasoning behind a setting moved to the README, where there is room for it.
- **One switch fewer.** *Context management* did nothing except force the two limits under it to zero, which is what zero already meant in either field. It is gone; an agent that had it switched off keeps its behaviour as explicit zeros.
- **The conversation history cap is no longer offered to AI Task agents.** An AI Task chat log is a single turn, so the setting never did anything there.
- **The gear icon is gone.** Agent settings belong to the agents now; the entry itself only holds the credentials, which are still changed through **⋮ → Reconfigure**.
- `deepseek_conversation.generate_content` and `run_debug` address a config entry rather than a specific agent, so they follow the entry's **first conversation agent**. After the upgrade that is the agent carrying your previous settings, so nothing about these actions changes.
- Editing an agent reloads the entry. Settings used to be applied in place, which is no longer possible now that an entity is built from a subentry.
- The token counters and the **Reset usage** button stay on the config entry's own device: usage is billed per API key and has to keep adding up across every agent sharing it.

### Fixed
- Nothing to do on upgrade. Your existing settings become the first conversation agent, an AI Task agent is created with the same ones, and both entities keep their entity id — `conversation.deepseek` still answers, and voice pipelines and automations pointing at it are untouched. The Assist-only settings (markdown stripping, naming the speaker) are not carried into the AI Task agent, where they never did anything.

## [1.7.0] - 2026-08-26

### Fixed
- **Retired model ids.** DeepSeek stopped serving `deepseek-chat` and `deepseek-reasoner` on 2026-07-24, so an entry left on one failed every request. They are gone from the model picker, an affected entry is switched to `deepseek-v4-flash` on startup, and the change is reported under **Settings → Repairs**. Entries on a custom base URL keep their id, because a gateway may still serve it.
- **Images on the official API.** `deepseek-v4-flash-vision-exp` accepts image input, but images were refused whenever the base URL was `api.deepseek.com`. Support now depends on the model: on the official API only the vision model is accepted, a custom base URL is never gated. Camera snapshots no longer need a third-party gateway.
- **Attachment support was advertised on models that reject images.** It now needs the *Allow vision* option *and* a capable model.
- **No request timeout.** The OpenAI SDK defaults (600 s, two retries) applied to Assist, so one unresponsive endpoint could block a voice pipeline for ten minutes. New **Request timeout** option, default 60 s: while streaming it bounds the gap between two chunks, so long answers are never cut off. `generate_content` is not streamed and waits at least 300 s. Retries drop from two to one, and the Brave web search tool is bounded at 10 s.
- **Max tokens** could be set up to 1 000 000, which is the context window rather than the output limit. The ceiling is now 384 000, the most V4 generates.
- **Token counters lost every failed turn.** Usage was recorded only after a whole turn succeeded, so an API error part-way through a tool loop, or hitting the iteration cap, discarded the tokens of the rounds that had already been billed. They are counted now whatever the outcome.
- **Tool calls vanished on gateways that omit `type`.** The opening chunk of a streamed tool call had to carry `id`, `type` and the function name; several OpenAI-compatible gateways leave `type` out, and the call was dropped with nothing but a log line — the model asked to switch a light and nothing happened. `type` now defaults to `function`.
- **`response_format: json_object` failed when the prompt never mentioned JSON.** DeepSeek requires the word in the prompt; `generate_content` adds it when it is missing, instead of returning an API error or an empty reply.
- **Usage sensors could fail a request during startup.** An API call finishing between the sensors being bound and Home Assistant adding them raised out of the state write. Writes now wait until the entity exists.
- **Markdown stripping came too late to help voice.** The option cleaned up the finished answer, but Home Assistant had already forwarded every chunk to the UI and to text-to-speech, so the asterisks were read out anyway. Formatting is now removed from the stream itself, holding text back only until a construct can no longer reach across the cut — 25 to 37 characters on real replies, well inside a sentence. AI Task output is untouched, so structured JSON cannot be damaged.
- The setup form keeps what you typed when the API key or the model is rejected.

### Added
- `deepseek-v4-flash-vision-exp` in the model picker, and an **Images** section in the README.
- The `run_debug` report records the request timeout.

### Removed
- Compatibility code for Home Assistant versions below the declared minimum of 2026.1: the `async_update_reload_and_abort` fallback, the `ai_task` import guard, and the `runtime_data` / `model_dump` attribute probes.

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
