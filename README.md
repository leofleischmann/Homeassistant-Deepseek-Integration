![Release](https://img.shields.io/github/v/release/leofleischmann/Homeassistant-Deepseek-Integration?label=Version)
![Release downloads](https://img.shields.io/github/downloads/leofleischmann/Homeassistant-Deepseek-Integration/deepseek_conversation.zip?label=Release%20downloads)

# Home Assistant DeepSeek Integration (DeepSeek Conversation)

Custom **Home Assistant DeepSeek integration** for **Assist** (voice and chat). Connect [DeepSeek](https://api-docs.deepseek.com/) as a conversation agent with optional smart home tools, reasoning, and automations. OpenAI-compatible API; custom base URL supported.

Install via **HACS** (`deepseek_conversation`). Community project — not part of Home Assistant Core.

**Requires:** Home Assistant 2026.1+, DeepSeek API key.

## What it does

Use DeepSeek **V4 Flash** (default), **V4 Pro** or **V4 Flash Vision** as the brain behind Assist: streaming replies, optional extended reasoning, and optional Home Assistant tool calls (lights, context lookups, and more when an LLM API is enabled). One API key can carry several agents, each with its own prompt, model and tools.

| Area | What you get |
|------|----------------|
| **Agents** | One API key, as many agents as you like — a fast one for voice, a capable one for automations, an AI Task entity on its own prompt ([details](#agents)) |
| **Assist** | Pick an agent in your voice assistant settings; same config for voice and text chat |
| **Who is speaking** | Optional: pass the Home Assistant user and room into the prompt so replies can be personalised ([details](#who-is-speaking)) |
| **Tools** | Expose selected Home Assistant LLM APIs to the model (configurable tool loop, 1–20 iterations). Optional Brave Search web tool when a Brave API key is set |
| **Reasoning** | Toggle thinking on/off and set effort; temperature and top_p apply only when thinking is off |
| **Images** | Select `deepseek-v4-flash-vision-exp` to send camera snapshots and attachments to the official API ([details](#images)) |
| **Context** | Optional trimming of large tool results and limit on Assist history rounds (helps with GetLiveContext-heavy chats) |
| **Timeouts** | Configurable request timeout (default 60 s) so a stalled endpoint cannot hang a voice pipeline |
| **Automations** | `ai_task.generate_data` (recommended, same prompt/tools as Assist), `conversation.process`, or `deepseek_conversation.generate_content` |
| **Usage** | Token sensors per config entry, last-request breakdown, manual **Reset usage** on the device |
| **Credentials** | Reauth when the key is rejected; **Reconfigure** for API key, base URL, or optional Brave Search key without touching your agents |

`generate_content` returns `text`, optional `reasoning`, and `usage` tokens. Per-call overrides: model, temperature, thinking, max_tokens, JSON mode.

## Install

[![Add to HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=leofleischmann&repository=Homeassistant-Deepseek-Integration&category=integration)

**HACS:** Integrations → search *DeepSeek Conversation* or add this repository → install → restart Home Assistant.

Release download badge counts GitHub `deepseek_conversation.zip` assets, not the number inside HACS.

**Manual:** copy `custom_components/deepseek_conversation/` into `/config` → restart.

## Setup

1. **Settings → Devices & services → Add integration → DeepSeek Conversation**
2. Enter API key (optional: custom base URL, model, Brave Search API key)
3. Setup creates one conversation agent and one AI Task entity. Open an agent's row to set its prompt, model and tools. If you set a Brave key, also select **Web Search (Brave)** under Home Assistant API
4. Assign the conversation agent to your Assist pipeline / voice assistant

Change API key, base URL, or Brave Search key via the integration card **⋮ → Reconfigure**.

## Agents

The config entry holds the credentials. Everything else belongs to an **agent**, and one entry can carry any number of them:

- **Add conversation agent** — answers in Assist. Give the voice pipeline a V4 Flash agent for speed and point automations at a V4 Pro agent with wider tool access, both on the same key.
- **Add AI task entity** — runs `ai_task.generate_data`, with its own prompt and model. It starts without access to your home; add a Home Assistant API if the task needs one.

Adding an agent asks for a name, prompt, Home Assistant APIs and model. Turn **Recommended settings** off and a second step opens, grouped so you only unfold what you came for:

| Group | Settings |
|-------|----------|
| **Reply** | Maximum reply length, temperature, top P, reasoning and its effort |
| **Tools** | Tool rounds per answer, size limit for tool results |
| **Conversation** *(Assist only)* | Remove formatting (on by default), tell the model who is speaking, how much history to send |
| **Limits and input** | Request timeout, allow images |

Anything left untouched follows the recommended default, so a later change to a default reaches the agent. Setting a limit to `0` turns it off: tool results stay whole, history is unlimited.

Each agent appears as its own device and can be reconfigured on its own row.

Token counters and the **Reset usage** button stay on the entry's device — usage is billed per API key, so it adds up across every agent that shares it.

## Who is speaking

Off by default. An agent's *Tell the model who is speaking* setting appends the speaker's name, presence and the room of the voice satellite to the system prompt. Nothing is sent when nobody is identified.

For your own wording, the system prompt is a Jinja template with `user_name`, `user_id`, `user_is_admin`, `person_entity_id`, `person_name`, `person_state`, `device_id`, `device_name`, `user_area`, `user_floor` (plus HA's `ha_name` and `llm_context`). Unknown values are empty strings, so branch with `{% if user_name %}`:

```jinja
{% if user_name %}You are speaking with {{ user_name }}.{% endif %}
{% if user_area %}They are in {{ user_area }}; "my room" means {{ user_area }}.{% endif %}
```

Voice satellites usually identify no user (Home Assistant runs those pipelines without an account) — `user_area` still works. Automations never carry a user either, so `conversation.process` and `ai_task.generate_data` from an automation see empty values; the variables stay defined, so one prompt works everywhere.

## Images

Image input needs two things: **Allow images** on (default) and a model that
accepts images.

- **Official API** (`api.deepseek.com`): select `deepseek-v4-flash-vision-exp`
  as the agent's model. The other DeepSeek models are text-only and reject
  images.
- **Custom base URL**: never gated. A gateway may route any model id to any
  backend, so every request is passed through and the API decides.

Images are sent as base64 `image_url` parts (JPEG, PNG, GIF, WebP). The
attachment button is offered to Home Assistant only while both the option and
the model allow images.

Attach images through Assist, the AI Task action's attachments field, or
`filenames` on `deepseek_conversation.generate_content`.

## Automations

### AI Task entity (recommended)

Pick an AI Task entity in the visual automation editor or use `ai_task.generate_data`. It runs on its own agent's **prompt**, **model** and **Home Assistant API** tools; per-call `llm_api` on the action overrides them.

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

Structured (JSON) output — the `structure` fields define the JSON shape (injected into the prompt; official DeepSeek uses `json_object` mode). Invalid JSON fails the step with a clear error:

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

Replace `entity_id` with your AI Task entity (integration device → AI Task). With **Allow images** and a vision-capable model you can attach images via the action's attachments field — see [Images](#images).

### Other paths

```yaml
# Like Assist: natural language, tools, the agent's own settings
action: conversation.process
data:
  agent_id: conversation.deepseek
  text: "Turn off the living room lights."

# Direct prompt → text (+ usage, optional reasoning). `agent` picks which
# agent answers, with its prompt, model and tools.
action: deepseek_conversation.generate_content
data:
  agent: conversation.deepseek
  prompt: "Summarise today's weather in one sentence."
response_variable: deepseek
```

Both `generate_content` and `run_debug` take either an `agent` or a `config_entry`. Naming the agent is the precise one: a single entity id says which credentials to use *and* which prompt and model to answer with. A bare `config_entry` follows that entry's first agent, which is what these actions did before an entry could hold several.

Sample automations: [`sample_automations/`](sample_automations/).

## Debug

`deepseek_conversation.run_debug` writes `/config/deepseek_conversation_debug_report.txt`. Many API calls — use manually only. Does not update usage sensors.

## Links

- [DeepSeek API docs](https://api-docs.deepseek.com/)
- [Issues & contributions](https://github.com/leofleischmann/Homeassistant-Deepseek-Integration/issues)
