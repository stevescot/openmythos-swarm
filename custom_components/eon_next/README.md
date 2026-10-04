# E.ON Next — Home Assistant integration

Reads your E.ON Next account (meter readings, tariff, EV smart-charging
schedule, saving sessions, billing) through the E.ON "Kraken" GraphQL API, and —
importantly — exposes the **real half-hourly cheap-rate schedule** so automations
can tell whether *now* is inside a cheap window instead of guessing from a clock.

## Why this version exists

E.ON Next migrated login to **Auth0 Universal Login** with a Cloudflare Turnstile
captcha. Headless username/password login is no longer possible, and the old
Kraken `login` mutation is gone. This integration therefore authenticates with a
**refresh token** that you mint once in a browser via the standard OAuth2 PKCE
flow, and rotates automatically thereafter.

### Auth0 refresh-token rotation (read this before debugging)

Auth0 refresh tokens **rotate on every use**. Presenting an already-used
(refreshed or spent) token **revokes the entire token family**, which is what
causes the "Unknown or invalid refresh token" / `invalid_grant` errors and makes
every tariff sensor go `unavailable`.

To survive that, all token rotation in `eonnext.py` is:
- serialised behind an `asyncio.Lock` (so concurrent sensor updates can never
  present the same spent token twice), and
- followed by a **5-minute cooldown** after any rejection, so a bad token can't
  hammer Auth0 and keep the family revoked.

The current (rotated) refresh token is persisted back to the bootstrap file after
every successful refresh, so the file always holds the freshest token.

## Install

1. Copy `custom_components/eon_next/` into your Home Assistant `config/custom_components/`.
2. Bootstrap a refresh token (below).
3. Add a manual config entry for the `eon_next` domain (no UI config flow — the
   integration reads credentials from the refresh-token file, not from config).
4. Restart / reload Home Assistant.

## Bootstrapping the refresh token (one-time, in a browser)

You need the Auth0 **client id** used by the E.ON Next web app (already set in
`eonnext.py` as `AUTH0_CLIENT_ID`) and a redirect URI registered for that app
(`https://www.eonnext.com/`).

1. Generate a PKCE verifier/challenge and state:

   ```bash
   python3 - <<'PY'
   import base64, hashlib, os
   v = base64.urlsafe_b64encode(os.urandom(64)).rstrip(b"=").decode()
   c = base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
   print("verifier =", v)
   print("challenge =", c)
   PY
   ```

2. Open the Auth0 authorise URL in a browser and log in with your E.ON Next
   account:

   ```
   https://auth.eonnext.com/authorize
     ?response_type=code
     &client_id=<AUTH0_CLIENT_ID>
     &redirect_uri=https://www.eonnext.com/
     &scope=openid profile email offline_access
     &audience=eonnext
     &state=<state>
     &code_challenge=<challenge>
     &code_challenge_method=S256
   ```

3. After login the browser redirects to `https://www.eonnext.com/?code=...`.
   Copy the `code` from the address bar.

4. Exchange the code for tokens (run from a machine that can reach Auth0):

   ```bash
   curl -s -X POST https://auth.eonnext.com/oauth/token \
     -H 'content-type: application/x-www-form-urlencoded' \
     -d 'grant_type=authorization_code' \
     -d 'client_id=<AUTH0_CLIENT_ID>' \
     -d 'code=<THE_CODE>' \
     -d 'redirect_uri=https://www.eonnext.com/' \
     -d 'code_verifier=<THE_VERIFIER>'
   ```

5. Take the `refresh_token` from the response and write it to
   `<config>/eon_auth0_refresh.json`:

   ```json
   { "refresh_token": "v1...Mtjw..." }
   ```

6. Reload the integration. It will refresh from here and keep the file updated.

> The integration uses the **id_token** as the Kraken `Bearer` credential. The
> SPA `access_token` is rejected by Kraken with `KT-CT-1143 "not a valid
> credential"`.

## Sensors

| Sensor | Meaning |
|---|---|
| `sensor.account_unit_rate` | Current unit rate (GBP/kWh), matched to the live half-hour slot |
| `binary_sensor.cheap_electricity_window_active` | **`on` when the current slot is cheap** (rate ≤ threshold) |
| `sensor.cheap_windows_today` | Count + list of today's cheap windows |
| `sensor.next_cheap_window_start` | Timestamp of the next cheap window |
| `sensor.account_tariff_name` | Active tariff display name / code |
| `sensor.account_standing_charge` | Daily standing charge |
| `sensor.<serial>_electricity_kwh` / `_gas_*` | Latest meter readings |
| `sensor.<charger>_smart_charging_schedule` | EV dispatch schedule + next start/end |
| `sensor.account_saving_sessions` | Saving sessions |
| `sensor.account_billing_history` | Recent bills |

The cheap-window sensors read `HalfHourlyTariff.unitRates` (plus
`unitRateUplifts`) for a window spanning yesterday→+2 days, collapse adjacent
cheap slots into contiguous windows, and compare against a threshold
(default **12 p/kWh**, `DEFAULT_CHEAP_THRESHOLD_P`). This handles both the
"wide block" shape (one cheap + one peak range) and the full 48-slot shape.

### Example automation

```yaml
automation:
  - trigger:
      - platform: state
        entity_id: binary_sensor.cheap_electricity_window_active
        to: "on"
    action:
      - service: switch.turn_on
        target: { entity_id: switch.car_charger }
```

## Troubleshooting

- **All tariff sensors `unavailable`** → the refresh token is revoked/expired.
  Re-run the browser PKCE bootstrap and replace
  `<config>/eon_auth0_refresh.json`, then reload.
- **`KT-CT-1143 not a valid credential`** → you presented the SPA access_token;
  use the `id_token`.
- **`invalid_grant / Unknown or invalid refresh token`** → the token was already
  rotated (or reused). Bootstrap a fresh one. The lock + cooldown in this
  integration prevent this from recurring under normal concurrent polling.

## Security

This repository contains **no** account credentials, account numbers, meter
serials, or vehicle identifiers. The refresh token lives only in
`eon_auth0_refresh.json` inside your Home Assistant config directory, which is
not part of this integration and must never be committed.
