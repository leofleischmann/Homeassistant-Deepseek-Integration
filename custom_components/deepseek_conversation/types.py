"""Shared type aliases for the DeepSeek Conversation integration."""

from __future__ import annotations

from typing import TypeAlias

import openai
from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]

DeepSeekConfigEntry: TypeAlias = ConfigEntry[openai.AsyncClient]
