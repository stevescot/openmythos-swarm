#!/usr/bin/env python3
"""E.ON Next integration for Home Assistant.

Auth model
----------
E.ON Next migrated to Auth0 Universal Login. Headless username/password login is
no longer possible (Cloudflare Turnstile), so this integration bootstraps from a
refresh token obtained once via a browser PKCE login and stored at
``/config/eon_auth0_refresh.json``. Refresh tokens rotate on every use; the
integration persists the rotated token after each successful refresh.

IMPORTANT: presenting a spent (already-rotated) refresh token revokes the whole
token family. All rotation is therefore serialised behind an asyncio lock with a
cooldown after any rejection. See ``EonNext.__login_with_refresh_token``.

No credentials are stored in this code. The refresh token lives only in the
config file, which is git-ignored.
"""

import logging

from .eonnext import EonNext

_LOGGER = logging.getLogger(__name__)

DOMAIN = "eon_next"
PLATFORMS = ["sensor"]


async def async_setup(hass, config):
    """Legacy YAML setup is not supported; a config entry is required."""
    return True


async def async_setup_entry(hass, entry):
    """Set up the integration from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    api = EonNext()
    # login_with_username_and_password ignores the password (Auth0 + captcha) and
    # drives off the persisted refresh token. We pass empty strings so nothing
    # secret is ever held here.
    success = await api.login_with_username_and_password("", "")

    if not success:
        _LOGGER.error(
            "E.ON Next: could not authenticate from the stored refresh token. "
            "Re-run the browser PKCE bootstrap and write the refresh token to "
            "/config/eon_auth0_refresh.json, then reload this integration."
        )
        return False

    hass.data[DOMAIN][entry.entry_id] = api
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass, entry):
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
