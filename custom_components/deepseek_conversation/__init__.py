"""The DeepSeek Conversation integration."""

from __future__ import annotations

import base64
from mimetypes import guess_file_type
from pathlib import Path
import logging # Use logging instead of LOGGER directly if needed elsewhere

import openai
# Removed OpenAI specific response/image types not directly used or replaced
# from openai.types.images_response import ImagesResponse
# from openai.types.responses import (...)
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry  # pyright: ignore[reportMissingImports]
from homeassistant.const import CONF_API_KEY, Platform  # pyright: ignore[reportMissingImports]
from homeassistant.core import (  # pyright: ignore[reportMissingImports]
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (  # pyright: ignore[reportMissingImports]
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.httpx_client import get_async_client  # pyright: ignore[reportMissingImports]
from homeassistant.helpers.typing import ConfigType  # pyright: ignore[reportMissingImports]
from typing import TypeAlias  # pyright: ignore[reportMissingImports]

# Updated imports from const.py
from .const import (
    CONF_CHAT_MODEL,
    CONF_FILENAMES,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_BASE_URL,
    DOMAIN, # Use the updated domain
    LOGGER, # Keep using the logger from const
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    DEEPSEEK_API_BASE_URL, # Use the base URL constant
)

# Removed SERVICE_GENERATE_IMAGE
SERVICE_GENERATE_CONTENT = "generate_content"

# Only conversation platform remains
PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# Define type alias using the updated domain if needed, or keep generic
DeepSeekConfigEntry: TypeAlias = ConfigEntry[openai.AsyncClient]


def encode_file(file_path: str) -> tuple[str, str]:
    """Return base64 version of file contents."""
    mime_type, _ = guess_file_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as image_file:
        return (mime_type, base64.b64encode(image_file.read()).decode("utf-8"))


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up DeepSeek Conversation."""

    # Removed render_image service definition

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        """Send a prompt to DeepSeek and return the response."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        model: str = entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
        client: openai.AsyncClient = entry.runtime_data

        # --- Adapt message format for chat.completions.create ---
        # Start with a system message if a general prompt/instruction is set in options
        messages = []
        if system_prompt := entry.options.get(CONF_PROMPT):
             messages.append({"role": "system", "content": system_prompt})

        # Handle user prompt and potential files (basic text for now)
        user_content = [{"type": "text", "text": call.data[CONF_PROMPT]}]

        # File handling - NOTE: DeepSeek API might require different format or not support
        # files via OpenAI library directly. This part might need adjustment based on
        # DeepSeek's specific multimodal capabilities and API structure.
        # Keeping basic structure, assuming text/image URL format might work.
        async def append_files_to_content() -> None:
            for filename in call.data.get(CONF_FILENAMES, []):
                if not hass.config.is_allowed_path(filename):
                    LOGGER.warning(
                        "Cannot read %s, no access to path; "
                        "`allowlist_external_dirs` may need to be adjusted in "
                        "`configuration.yaml`", filename
                    )
                    continue # Skip this file
                if not Path(filename).exists():
                    LOGGER.warning("%s does not exist", filename)
                    continue # Skip this file

                try:
                    mime_type, base64_file = await hass.async_add_executor_job(
                        encode_file, filename
                    )
                    if "image/" in mime_type:
                        # Using standard image URL format for messages
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_file}",
                                    # "detail": "auto" # Detail might not be supported
                                }
                            }
                        )
                    # Add handling for other file types if DeepSeek supports them
                    # elif "application/pdf" in mime_type:
                    #    ...
                    else:
                        LOGGER.warning(
                            "Skipping file %s: Unsupported file type %s for DeepSeek via this integration.",
                            filename, mime_type
                        )
                except Exception as e:
                     LOGGER.error("Error processing file %s: %s", filename, e)


        if CONF_FILENAMES in call.data:
            await append_files_to_content()

        messages.append({"role": "user", "content": user_content})
        # --- End of message format adaptation ---

        try:
            # --- Switched to client.chat.completions.create ---
            model_args = {
                "model": model,
                "messages": messages,
                "max_tokens": entry.options.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                "top_p": entry.options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                "temperature": entry.options.get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
                # 'user': call.context.user_id, # Optional: Check if DeepSeek uses this
                "stream": False, # Service call expects a single response
            }

            # Removed OpenAI specific 'reasoning' and 'store' args
            # if model.startswith("o"): ...

            response = await client.chat.completions.create(**model_args)
            # --- End of API call change ---

            # Extract response text
            # Assuming response structure is similar to OpenAI's chat completion
            response_text = response.choices[0].message.content

        except openai.OpenAIError as err:
            LOGGER.error("Error generating content with DeepSeek: %s", err)
            raise HomeAssistantError(f"Error generating content: {err}") from err
        except Exception as err: # Catch potential file errors or other issues
            LOGGER.error("Unexpected error during content generation: %s", err)
            raise HomeAssistantError(f"Error generating content: {err}") from err

        return {"text": response_text or ""} # Ensure text is not None

    # Register the generate_content service
    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN, # Use updated domain
                    }
                ),
                vol.Required(CONF_PROMPT): cv.string,
                # Keep filenames optional, but functionality depends on DeepSeek support
                vol.Optional(CONF_FILENAMES, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    # Removed registration for generate_image service

    return True


async def async_setup_entry(hass: HomeAssistant, entry: DeepSeekConfigEntry) -> bool:
    """Set up DeepSeek Conversation from a config entry."""
    # --- Initialize client with DeepSeek base URL from config entry ---
    base_url = entry.data.get(CONF_BASE_URL, DEEPSEEK_API_BASE_URL)
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        base_url=base_url, # Use base URL from config entry or default
        http_client=get_async_client(hass),
    )
    # --- End of client initialization change ---

    # Removed platform_headers cache call, may not be relevant/needed

    # --- Validate API key using a test chat completion ---
    try:
        await client.with_options(timeout=10.0).chat.completions.create(
            model=entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
        )
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid DeepSeek API key: %s", err)
        return False
    except openai.OpenAIError as err:
        # Log the specific error for better debugging
        LOGGER.error("Failed to connect to DeepSeek API: %s", err)
        raise ConfigEntryNotReady(f"Failed to connect to DeepSeek API: {err}") from err
    # --- End of validation change ---

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload DeepSeek."""
    # Close the client if the library supports it (optional)
    # client: openai.AsyncClient = entry.runtime_data
    # await client.close()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

