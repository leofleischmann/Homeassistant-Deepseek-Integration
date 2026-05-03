![Release](https://img.shields.io/github/v/release/leofleischmann/Homeassistant-Deepseek-Integration?label=Version)
![Downloads](https://img.shields.io/github/downloads/leofleischmann/Homeassistant-Deepseek-Integration/total?label=Downloads)

# Home Assistant — DeepSeek Conversation

Custom integration that uses the [DeepSeek API](https://api-docs.deepseek.com/) (OpenAI-compatible endpoint) as a **conversation agent** for Home Assistant Assist. It is derived from the ideas and flow of the core [`openai_conversation`](https://www.home-assistant.io/integrations/openai_conversation) integration, but targets DeepSeek’s base URL, models, and thinking/reasoning parameters.

> **Note:** This is a **community custom component**. It is not maintained by the Home Assistant core team. Test updates on a non-production instance when possible.

## Features

- **Assist / conversation agent** — text conversations with streaming responses.
- **DeepSeek V4 by default** — `deepseek-v4-flash` (fast, cost-efficient) and `deepseek-v4-pro` selectable in the UI; optional **custom model id** for future API names.
- **Reasoning (thinking)** — toggle maps to DeepSeek’s `extra_body` thinking switch; **reasoning effort** is sent only when reasoning is enabled.
- **Sampling params** — `temperature` and `top_p` are sent when reasoning is **off**; they are **omitted when reasoning is on**, matching DeepSeek’s thinking-mode behaviour.
- **Home Assistant tools** — if you pick an LLM API in options (e.g. Assist), tools are passed to the model when supported.
- **`generate_content` service** — scripted calls with prompt and optional file hints (see `services.yaml`).
- **Translations** — UI strings for English, German, French, and Chinese (Simplified).

## Requirements

- **Home Assistant** 2026.1 or newer recommended (uses `ChatLog.async_provide_llm_data` and current conversation APIs).
- A **DeepSeek API key** from [DeepSeek](https://platform.deepseek.com/) (or your compatible gateway).

## Models (DeepSeek V4 and legacy)

| Model id | Role |
|----------|------|
| **`deepseek-v4-flash`** | Default — optimised for speed and cost; good fit for Assist and home automation. |
| **`deepseek-v4-pro`** | Stronger reasoning and harder tasks; higher cost/latency than Flash. |

DeepSeek documents that legacy aliases **`deepseek-chat`** and **`deepseek-reasoner`** still route to V4-style behaviour but are scheduled for **removal after 2026-07-24**. This integration lists them under **Legacy** in the model selector so existing configurations keep working until you migrate.

Reasoning is **not** a separate “reasoner-only” product line anymore: on V4 you turn **reasoning on or off** with the integration’s **“Enable reasoning”** option (API thinking parameter), independent of Flash vs Pro.

## Installation

### HACS

1. Install [HACS](https://hacs.xyz/).
2. **HACS → Integrations → ⋮ → Custom repositories**
3. Add repository `https://github.com/leofleischmann/Homeassistant-Deepseek-Integration`, category **Integration**.
4. Install **DeepSeek Conversation** and **restart Home Assistant**.

### Manual

Copy the folder `custom_components/deepseek_conversation/` into your Home Assistant configuration directory (next to `configuration.yaml`), then restart.

## Configuration

1. **Settings → Devices & services → Add integration**
2. Search **DeepSeek Conversation**
3. Enter your **API key** (and optionally change **API base URL** or **model** on first step).
4. Open **Configure** on the integration entry for full options:

| Option | Meaning |
|--------|--------|
| **API base URL** | Default `https://api.deepseek.com/v1` unless you use a proxy. |
| **System prompt** | Instructions for the model; supports Jinja2 like core LLM prompts. |
| **Home Assistant API** | Which registered LLM API supplies tools (`none` = no tools). |
| **Model** | Dropdown: V4 Flash / V4 Pro / legacy ids; **custom value** allowed for other ids. |
| **Max tokens** | Cap on completion length (default 1500). |
| **Temperature / Top P** | Used when **reasoning is disabled** only. |
| **Enable reasoning** | Sends DeepSeek thinking (`enabled` / `disabled`) and optional reasoning effort. |
| **Reasoning effort** | e.g. low … xhigh; only applied when reasoning is enabled. |

After changing the base URL, the integration reloads so the OpenAI client picks up the new endpoint.

## Integration icon (optional)

From **Home Assistant 2026.3** onward, you can ship brand images next to the integration:

- `custom_components/deepseek_conversation/brand/icon.png` (256×256 PNG)
- Optional: `icon@2x.png`, `logo.png`, dark variants — see [Home Assistant Brands](https://github.com/home-assistant/brands) image rules.

## Debug suite (service)

The integration **registers** the service **`deepseek_conversation.run_debug`** when Home Assistant loads the custom component (same as other services). **Nothing is executed automatically** — the debug suite runs **only** when you call that service (e.g. *Developer tools → Services*).

- Runs **extended API diagnostics** (environment, HTTP, optional `models.list`, LLM APIs, entity registry, many non-stream/stream checks, edge cases, parallel pings).
- Writes **`/config/deepseek_conversation_debug_report.txt`** (full text including a **filtered** excerpt of `home-assistant.log`).
- Shows a **persistent notification** with a short summary.

**How to run:** *Developer tools → Services* → `deepseek_conversation.run_debug` → choose your config entry → **Perform action**.  
For a **one-click** flow, create a **script** (see `examples/run_deepseek_debug_script.yaml`) and add a **dashboard button** that calls `script.your_script_name`.

This runs **many** short API calls (still small `max_tokens` per call, but several minutes possible on slow links). The service response includes `environment`, `summary`, flattened `tests`, and `report_path`; the **report file** ends with a **JSON appendix** for copy/paste analysis.

## Technical notes

- **Conversation API** — Uses `async_provide_llm_data` with `user_input.as_llm_context` (no deprecated `async_update_llm_data` wrapper).
- **Message history** — `reasoning_content` in outbound messages follows DeepSeek guidance: stripped for **`deepseek-reasoner`**-style ids; for other models with reasoning on, reasoning is attached mainly where **tool calls** require continuity. Adjust if DeepSeek changes rules.
- **Dependencies** — Declared in `manifest.json` (`openai`, `voluptuous-openapi`, etc.).

## Disclaimer

This integration is provided **as is**, without warranty. API behaviour, model names, and quotas are controlled by DeepSeek; monitor their [API documentation](https://api-docs.deepseek.com/) for changes.

## Contributing

Issues and pull requests are welcome in this repository.
