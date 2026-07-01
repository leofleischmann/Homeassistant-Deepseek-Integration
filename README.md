![Release](https://img.shields.io/github/v/release/leofleischmann/Homeassistant-Deepseek-Integration?label=Version)

# DeepSeek Conversation

Home Assistant integration for [DeepSeek](https://api-docs.deepseek.com/) as an **Assist conversation agent** (OpenAI-compatible API).

Community project — not part of Home Assistant Core.

## What you get

- **Assist** (voice and text) with streaming, optional **Home Assistant tools**, and **reasoning** toggle
- **Automations** via `conversation.process` (same agent as Assist) or service `deepseek_conversation.generate_content`
- **Token sensors** per config entry (cumulative + last request); updated on Assist and `generate_content` calls
- Models **V4 Flash** (default) and **V4 Pro**; legacy ids until 2026-07-24

**Requires:** Home Assistant 2026.1+, DeepSeek API key.

## Install

**HACS:** Custom repository `leofleischmann/Homeassistant-Deepseek-Integration` (Integration) → install → restart.

**Manual:** Copy `custom_components/deepseek_conversation/` into your config folder → restart.

## Setup

1. **Settings → Devices & services → Add integration → DeepSeek Conversation**
2. API key (optional: base URL, model)
3. **Configure** (gear) for system prompt, model, reasoning, tools, etc. First screen: **Assist & model settings** or **API key & base URL**.

**Change API key:** integration **⋮ → Reconfigure**, or **Configure → API key & base URL**.

When **reasoning is on**, the API uses reasoning effort (temperature/top_p are ignored). When **off**, temperature and top_p apply. All option fields stay visible; one OK saves everything.

## Automations

| Action | Use when |
|--------|----------|
| [`conversation.process`](https://www.home-assistant.io/integrations/conversation/) | Natural language + tools like Assist; uses your integration options and system prompt |
| `deepseek_conversation.generate_content` | Simple prompt → text response; returns `text` and `usage` (tokens) |

Example (`generate_content`):

```yaml
action: deepseek_conversation.generate_content
data:
  config_entry: <your config entry id>
  prompt: "Summarise the weather for today in one sentence."
response_variable: deepseek
```

Token sensors on the integration device update after each API call (`assist` or `generate_content`). **`run_debug`** does not count toward usage sensors.

## Debug

Service `deepseek_conversation.run_debug` — extended API diagnostics, writes `/config/deepseek_conversation_debug_report.txt`. Many API calls; run manually only.

## Links

- [DeepSeek API docs](https://api-docs.deepseek.com/)
- [Issues & contributions](https://github.com/leofleischmann/Homeassistant-Deepseek-Integration/issues)
