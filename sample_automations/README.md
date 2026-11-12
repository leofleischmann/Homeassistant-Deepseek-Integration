# Sample automations

Copy a YAML file into **Automations → Create automation → Edit in YAML**, save, then trigger via **Developer tools → Events**.

**Requirement:** one loaded `deepseek_conversation` config entry (no IDs to edit; entry and agent are resolved automatically).

| Event | File |
|-------|------|
| `deepseek_integration_demo` | `deepseek_integration_demo.yaml` |
| `deepseek_ai_task_demo` | `deepseek_ai_task_demo.yaml` (plain + structured `ai_task.generate_data`) |
| `deepseek_vision_demo` | `deepseek_vision_demo.yaml` (needs a vision-capable model — the call sets `deepseek-flash` — and a readable image, see below) |
| `deepseek_web_search_demo` | `deepseek_web_search_demo.yaml` (structured `ai_task.generate_data` from live web results; set **Web search** on the AI Task agent first) |

**Vision demo, image path.** Home Assistant only reads files from allowed folders, and `custom_components` is not one of them. Either point `image_path` at a picture in `/config/www` — allowed by default, no restart needed — or keep the bundled `brand/icon.png` and allow its folder once in `configuration.yaml`, then restart:

```yaml
homeassistant:
  allowlist_external_dirs:
    - /config/custom_components/deepseek_conversation
```

Results: **persistent notification** + logbook. Token sensors update after each API call.

`run_debug`: see `examples/run_deepseek_debug_script.yaml`.
