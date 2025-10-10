"""Constants for the DeepSeek Conversation integration."""

import logging

# Changed domain to reflect DeepSeek
DOMAIN = "deepseek_conversation"
LOGGER: logging.Logger = logging.getLogger(__package__)

# Configuration keys
CONF_CHAT_MODEL = "chat_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_PROMPT = "prompt" # Keep prompt for system message/instructions
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_API_KEY = "api_key" # Already defined in homeassistant.const, but useful here
CONF_BASE_URL = "base_url" # Added for clarity, though set internally

# Service related (Image generation removed)
CONF_FILENAMES = "filenames" # Kept for potential future file support if DeepSeek adds it

# Default values
# Changed recommended model to DeepSeek's chat model
RECOMMENDED_CHAT_MODEL = "deepseek-chat"
# Adjusted default tokens, temperature, top_p if needed, keeping OpenAI's for now
RECOMMENDED_MAX_TOKENS = 150
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0

# DeepSeek API endpoint
DEEPSEEK_API_BASE_URL = "https://api.deepseek.com/v1"

# Removed OpenAI specific constants
# CONF_REASONING_EFFORT = "reasoning_effort"
# CONF_RECOMMENDED = "recommended" # No longer using OpenAI recommended toggle
# CONF_WEB_SEARCH = "web_search"
# CONF_WEB_SEARCH_USER_LOCATION = "user_location"
# CONF_WEB_SEARCH_CONTEXT_SIZE = "search_context_size"
# CONF_WEB_SEARCH_CITY = "city"
# CONF_WEB_SEARCH_REGION = "region"
# CONF_WEB_SEARCH_COUNTRY = "country"
# CONF_WEB_SEARCH_TIMEZONE = "timezone"
# RECOMMENDED_REASONING_EFFORT = "low"
# RECOMMENDED_WEB_SEARCH = False
# RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE = "medium"
# RECOMMENDED_WEB_SEARCH_USER_LOCATION = False
# UNSUPPORTED_MODELS = [...] # Removed OpenAI unsupported models list

