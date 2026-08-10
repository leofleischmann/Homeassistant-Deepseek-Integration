# Sample automations

Copy a YAML file into **Automations → Create automation → Edit in YAML**, save, then trigger via **Developer tools → Events**.

**Requirement:** one loaded `deepseek_conversation` config entry (no IDs to edit; entry and agent are resolved automatically).

| Event | File |
|-------|------|
| `deepseek_integration_demo` | `deepseek_integration_demo.yaml` |
| `deepseek_ai_task_demo` | `deepseek_ai_task_demo.yaml` (plain + structured `ai_task.generate_data`) |
| `deepseek_vision_demo` | `deepseek_vision_demo.yaml` (uses bundled `brand/icon.png`; **custom base URL with vision only**, not official `api.deepseek.com`) |

Vision demo reads `/config/custom_components/deepseek_conversation/brand/icon.png`. If Home Assistant blocks the path, add to `configuration.yaml`:

```yaml
homeassistant:
  allowlist_external_dirs:
    - /config/custom_components/deepseek_conversation
```

Results: **persistent notification** + logbook. Token sensors update after each API call.

`run_debug`: see `examples/run_deepseek_debug_script.yaml`.
