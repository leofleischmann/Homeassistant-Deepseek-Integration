"""Web tools an agent can be given: looking things up on the public web.

Three layers, and the imports only ever point down this list:

``types.py``, ``selection.py``, ``brave.py``
    The core. What a search result and a failed search are, which providers a
    config entry can offer, and the providers themselves. No Home Assistant,
    no SDK - the HTTP client is handed in - so the stubbed suite runs it.

``factory.py``, ``search_tool.py``
    The Home Assistant side. One builds a provider from a config entry, the
    other is the ``llm.Tool`` the model calls. The tool knows the
    ``SearchProvider`` protocol and never a concrete provider.

``agent_tools.py`` (one level up)
    Puts the tool in front of one agent, next to whatever Home Assistant APIs
    that agent selected.

This file stays empty on purpose: the stubbed suite imports the core modules
one by one, and anything imported here would drag Home Assistant in with them.

Affects / influenced by:
- ``agent_tools.py`` is the only caller of ``factory`` and ``search_tool``.
- ``config_flow.py`` and ``debug.py`` read ``selection`` to know what an entry
  can offer.
- ``const.py`` holds the provider ids, because they are stored in subentries.
"""
