![Release](https://img.shields.io/github/v/release/leofleischmann/Homeassistant-Deepseek-Integration?label=Version)
![Release downloads](https://img.shields.io/github/downloads/leofleischmann/Homeassistant-Deepseek-Integration/deepseek_conversation.zip?label=Release%20downloads)

# Home Assistant DeepSeek Integration (DeepSeek Conversation)

Custom **Home Assistant DeepSeek integration** for **Assist** (voice and chat). Connect [DeepSeek](https://api-docs.deepseek.com/) as a conversation agent with optional smart home tools, reasoning, and automations. OpenAI-compatible API; custom base URL supported.

Install via **HACS** (`deepseek_conversation`). Community project — not part of Home Assistant Core.

**Requires:** Home Assistant 2026.1+, DeepSeek API key.

## What it does

Use DeepSeek **V4 Flash** (default) or **V4 Pro** as the brain behind Assist: streaming replies, optional extended reasoning, and optional Home Assistant tool calls (lights, context lookups, and more when an LLM API is enabled in options).

| Area | What you get |
|------|----------------|
| **Assist** | Pick the agent in your voice assistant settings; same config for voice and text chat |
| **Tools** | Expose selected Home Assistant LLM APIs to the model (configurable tool loop, 1–20 iterations). Optional Brave Search web tool when a Brave API key is set |
| **Reasoning** | Toggle thinking on/off and set effort; temperature and top_p apply only when thinking is off |
| **Context** | Optional trimming of large tool results and limit on Assist history rounds (helps with GetLiveContext-heavy chats) |
| **Automations** | `ai_task.generate_data` (recommended, same prompt/tools as Assist), `conversation.process`, or `deepseek_conversation.generate_content` |
| **Usage** | Token sensors per config entry, last-request breakdown, manual **Reset usage** on the device |
| **Credentials** | Reauth when the key is rejected; **Reconfigure** for API key, base URL, or optional Brave Search key without losing options |

`generate_content` returns `text`, optional `reasoning`, and `usage` tokens. Per-call overrides: model, temperature, thinking, max_tokens, JSON mode.

Legacy model ids (`deepseek-chat`, `deepseek-reasoner`) map to V4 until 2026-07-24.

## Install

[![Add to HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=leofleischmann&repository=Homeassistant-Deepseek-Integration&category=integration)

**HACS:** Integrations → search *DeepSeek Conversation* or add this repository → install → restart Home Assistant.

Release download badge counts GitHub `deepseek_conversation.zip` assets, not the number inside HACS.

**Manual:** copy `custom_components/deepseek_conversation/` into `/config` → restart.

## Setup

1. **Settings → Devices & services → Add integration → DeepSeek Conversation**
2. Enter API key (optional: custom base URL, model, Brave Search API key)
3. Open **Configure** (gear): system prompt, model, reasoning, tools, context limits. If you set a Brave key, also select **Web Search (Brave)** under Home Assistant API
4. Assign the agent to your Assist pipeline / voice assistant

Change API key, base URL, or Brave Search key via the integration card **⋮ → Reconfigure** (not the gear).

## Automations

### AI Task entity (recommended)

Pick the integration's AI Task entity in the visual automation editor or use `ai_task.generate_data`. It uses the same **system prompt** and **Home Assistant API** tools (Configure → gear) as Assist; per-call `llm_api` on the action overrides the configured APIs.

Plain text:

```yaml
action: ai_task.generate_data
data:
  task_name: weather_summary
  instructions: >-
    Today's forecast: {{ states('weather.home') }}.
    Summarise it in one short sentence.
  entity_id: ai_task.deepseek_conversation
response_variable: result
# result.data holds the generated text
```

Structured (JSON) output — invalid JSON fails the step with a clear error:

```yaml
action: ai_task.generate_data
data:
  task_name: forecast_extract
  instructions: "From {{ states('weather.home') }}, produce structured data."
  entity_id: ai_task.deepseek_conversation
  structure:
    summary:
      selector:
        text:
    high_c:
      selector:
        number:
response_variable: result
# result.data.summary, result.data.high_c, …
```

Replace `entity_id` with your AI Task entity (integration device → AI Task). With **Allow vision** and a custom multimodal base URL, you can attach images via the action's attachments field.

### Other paths

```yaml
# Like Assist: natural language, tools, integration options
action: conversation.process
data:
  agent_id: conversation.deepseek
  text: "Turn off the living room lights."

# Legacy: direct prompt → text (+ usage, optional reasoning)
action: deepseek_conversation.generate_content
data:
  config_entry: <your config entry id>
  prompt: "Summarise today's weather in one sentence."
response_variable: deepseek
```

Sample automations: [`sample_automations/`](sample_automations/).

## Debug

`deepseek_conversation.run_debug` writes `/config/deepseek_conversation_debug_report.txt`. Many API calls — use manually only. Does not update usage sensors.

## Links

- [DeepSeek API docs](https://api-docs.deepseek.com/)
- [Issues & contributions](https://github.com/leofleischmann/Homeassistant-Deepseek-Integration/issues)
