![Release](https://img.shields.io/github/v/release/leofleischmann/Homeassistant-Deepseek-Integration?label=Version)
![Release downloads](https://img.shields.io/github/downloads/leofleischmann/Homeassistant-Deepseek-Integration/deepseek_conversation.zip?label=Release%20downloads)

# DeepSeek Conversation

Connect [DeepSeek](https://api-docs.deepseek.com/) to Home Assistant as a **conversation agent** for Assist (voice and chat). The integration speaks the OpenAI-compatible Chat Completions API, so you can also point it at a compatible proxy via a custom base URL.

Community project — not part of Home Assistant Core.

**Requires:** Home Assistant 2026.1+, DeepSeek API key.

## What it does

Use DeepSeek **V4 Flash** (default) or **V4 Pro** as the brain behind Assist: streaming replies, optional extended reasoning, and optional Home Assistant tool calls (lights, context lookups, and more when an LLM API is enabled in options).

| Area | What you get |
|------|----------------|
| **Assist** | Pick the agent in your voice assistant settings; same config for voice and text chat |
| **Tools** | Expose selected Home Assistant LLM APIs to the model (configurable tool loop, 1–20 iterations) |
| **Reasoning** | Toggle thinking on/off and set effort; temperature and top_p apply only when thinking is off |
| **Context** | Optional trimming of large tool results and limit on Assist history rounds (helps with GetLiveContext-heavy chats) |
| **Automations** | `conversation.process` like Assist, or `deepseek_conversation.generate_content` for direct prompt → text |
| **Usage** | Token sensors per config entry, last-request breakdown, manual **Reset usage** on the device |
| **Credentials** | Reauth when the key is rejected; **Reconfigure** for API key or base URL without losing options |

`generate_content` returns `text`, optional `reasoning`, and `usage` tokens. Per-call overrides: model, temperature, thinking, max_tokens, JSON mode.

Legacy model ids (`deepseek-chat`, `deepseek-reasoner`) map to V4 until 2026-07-24.

## Install

[![Add to HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=leofleischmann&repository=Homeassistant-Deepseek-Integration&category=integration)

HACS → install → restart Home Assistant.

Release download badge counts GitHub `deepseek_conversation.zip` assets, not the number inside HACS.

**Manual:** copy `custom_components/deepseek_conversation/` into `/config` → restart.

## Setup

1. **Settings → Devices & services → Add integration → DeepSeek Conversation**
2. Enter API key (optional: custom base URL, model)
3. Open **Configure** (gear): system prompt, model, reasoning, tools, context limits
4. Assign the agent to your Assist pipeline / voice assistant

Change API key or base URL via the integration card **⋮ → Reconfigure** (not the gear).

## Automations

```yaml
# Like Assist: natural language, tools, integration options
action: conversation.process
data:
  agent_id: conversation.deepseek
  text: "Turn off the living room lights."

# Scripted call: prompt in, text (+ usage, optional reasoning) out
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
