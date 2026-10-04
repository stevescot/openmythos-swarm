#!/usr/bin/env python3
"""Core E.ON Next API client (Kraken GraphQL) with Auth0 refresh-token auth.

This module is credential-free. The Auth0 refresh token is read from / written to
``/config/eon_auth0_refresh.json`` at runtime and never embedded here.
"""

import asyncio
import datetime
import json
import logging

import aiohttp

_LOGGER = logging.getLogger(__name__)

# Auth0 (E.ON Next Universal Login) - replaces the deprecated Kraken password
# mutation. These are public client identifiers, not secrets.
AUTH0_TOKEN_URL = "https://auth.eonnext.com/oauth/token"
AUTH0_CLIENT_ID = "OrFeFacHUoXK2afczePYMLCRMXpwRxzW"
AUTH0_AUDIENCE = "eonnext"
AUTH0_SCOPE = "openid profile email offline_access"

# Bootstrap file written once via a browser PKCE login; refreshed thereafter.
AUTH0_REFRESH_FILE = "/config/eon_auth0_refresh.json"

KRAKEN_URL = "https://api.eonnext-kraken.energy/v1/graphql/"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

METER_TYPE_GAS = "gas"
METER_TYPE_ELECTRIC = "electricity"
METER_TYPE_EV = "ev"
METER_TYPE_UNKNOWN = "unknown"

# Cooldown (seconds) after a rejected refresh so we don't hammer Auth0 with a
# spent token and keep the family revoked.
AUTH_COOLDOWN_SECONDS = 300

# A rate at or below this (pence/kWh) is treated as "cheap". Overridable per call.
DEFAULT_CHEAP_THRESHOLD_P = 12.0


class EonNext:

    def __init__(self):
        self.username = ""
        self.password = ""
        self.__reset_authentation()
        self.__reset_accounts()
        # Auth0 refresh tokens rotate on every use, and presenting a spent token
        # revokes the entire token family. Several sensors refresh concurrently, so
        # all token rotation must be serialised behind this lock.
        self._refresh_lock = asyncio.Lock()
        # After a rejected refresh, stop hammering Auth0 until the cooldown ends.
        self._auth_blocked_until = 0

    def _json_contains_key_chain(self, data, key_chain):
        for key in key_chain:
            if data is None:
                return False
            if key in data:
                data = data[key]
            else:
                return False
        return True

    def __current_timestamp(self):
        now = datetime.datetime.now()
        return int(datetime.datetime.timestamp(now))

    def __reset_authentation(self):
        self.auth = {
            "issued": None,
            "token": {"token": None, "expires": None},
            "refresh": {"token": None, "expires": None},
        }

    def __decode_jwt(self, token):
        import base64
        part = token.split(".")[1]
        part += "=" * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part))

    async def __store_auth0(self, data):
        # Auth0 token response: access_token, id_token, refresh_token, expires_in.
        # Kraken accepts the id_token as a `Bearer` credential (verified).
        id_token = data["id_token"]
        payload = self.__decode_jwt(id_token)
        iat = int(payload.get("iat", self.__current_timestamp()))
        exp = int(payload.get("exp", iat + 3600))
        # Auth0 refresh tokens rotate on every use; assume a generous lifetime and
        # persist the rotated token after each refresh.
        refresh_exp = iat + 30 * 86400
        self.auth = {
            "issued": iat,
            "token": {"token": id_token, "expires": exp},
            "refresh": {"token": data.get("refresh_token"), "expires": refresh_exp},
        }
        await self.__persist_refresh()

    async def __persist_refresh(self):
        try:
            if self.auth["refresh"]["token"]:
                await asyncio.to_thread(
                    lambda: json.dump(
                        {"refresh_token": self.auth["refresh"]["token"]},
                        open(AUTH0_REFRESH_FILE, "w"),
                    )
                )
        except Exception as e:
            _LOGGER.warning("Could not persist E.ON refresh token: %s", e)

    async def __load_refresh(self):
        # Prefer the persisted file: it is updated on every successful refresh and
        # therefore always holds the freshest (non-rotated) token.
        def _read():
            try:
                with open(AUTH0_REFRESH_FILE) as f:
                    return json.load(f).get("refresh_token")
            except Exception:
                return None
        tok = await asyncio.to_thread(_read)
        if tok:
            return tok
        return self.auth["refresh"]["token"]

    def __auth_token_is_valid(self):
        if self.auth["token"]["token"] is None:
            return False
        if self.auth["token"]["expires"] <= self.__current_timestamp():
            return False
        return True

    def __refresh_token_is_valid(self):
        if self.auth["refresh"]["token"] is None:
            return False
        if self.auth["refresh"]["expires"] <= self.__current_timestamp():
            return False
        return True

    async def __auth_token(self):
        if not self.__auth_token_is_valid():
            await self.__ensure_token()
        if not self.__auth_token_is_valid():
            raise Exception("Unable to authenticate")
        return self.auth["token"]["token"]

    async def __ensure_token(self):
        # Cooldown after a rejected refresh so we don't hammer Auth0 with a spent
        # token (which keeps the family revoked and spams the log).
        if self.__current_timestamp() < self._auth_blocked_until:
            return
        if self.__refresh_token_is_valid():
            if await self.__login_with_refresh_token():
                return
        self._auth_blocked_until = self.__current_timestamp() + AUTH_COOLDOWN_SECONDS

    async def _graphql_post(self, operation, query, variables=None, authenticated=True):
        variables = variables or {}
        use_headers = {"User-Agent": USER_AGENT}
        if authenticated:
            use_headers["authorization"] = "Bearer " + await self.__auth_token()

        payload = {"operationName": operation, "variables": variables, "query": query}
        _LOGGER.debug("GraphQL Payload: %s", payload)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                KRAKEN_URL, json=payload, headers=use_headers
            ) as response:
                try:
                    json_data = await response.json()
                    _LOGGER.debug("GraphQL Response for %s: %s", operation, json_data)
                    return json_data
                except Exception as e:
                    text = await response.text()
                    _LOGGER.error(
                        "Failed to parse JSON response. Status: %s. Body: %s",
                        response.status, text,
                    )
                    raise e

    async def login_with_username_and_password(self, username="", password="", initialise=True):
        # E.ON Next migrated to Auth0 Universal Login; the Kraken password
        # mutation is gone and headless password login is blocked by Cloudflare
        # Turnstile. We bootstrap from a refresh token obtained once via a
        # browser PKCE login.
        self.username = username
        self.password = password

        rt = await self.__load_refresh()
        if rt:
            self.auth["refresh"]["token"] = rt
            return await self.__login_with_refresh_token(initialise)

        _LOGGER.error(
            "E.ON Next: no stored Auth0 refresh token. Headless password login is "
            "no longer supported (Auth0 + captcha). Bootstrap the refresh token "
            "via a browser PKCE login and write it to " + AUTH0_REFRESH_FILE
        )
        self.__reset_authentation()
        return False

    async def login_with_refresh_token(self, token):
        self.auth["refresh"]["token"] = token
        # Persist so the file (which takes priority) holds this token.
        await self.__persist_refresh()
        return await self.__login_with_refresh_token(True)

    async def __login_with_refresh_token(self, initialise=False):
        # Serialise every rotation. Re-check inside the lock: if another task
        # already rotated while we waited, reuse its token rather than presenting
        # the spent refresh token again (which revokes the whole family).
        async with self._refresh_lock:
            if self.__auth_token_is_valid():
                if initialise:
                    await self.__init_accounts()
                return True
            # Another task may have failed and engaged the cooldown while we
            # queued behind the lock - don't hammer Auth0 with the spent token.
            if self.__current_timestamp() < self._auth_blocked_until:
                return False
            return await self.__do_refresh(initialise)

    async def __do_refresh(self, initialise=False):
        rt = await self.__load_refresh()
        if not rt:
            self.__reset_authentation()
            return False

        form = {
            "grant_type": "refresh_token",
            "client_id": AUTH0_CLIENT_ID,
            "refresh_token": rt,
            "scope": AUTH0_SCOPE,
            "audience": AUTH0_AUDIENCE,
        }
        headers = {"User-Agent": USER_AGENT}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(AUTH0_TOKEN_URL, data=form, headers=headers) as response:
                    data = await response.json()
        except Exception as e:
            _LOGGER.error("E.ON Next Auth0 refresh failed: %s", e)
            self.__reset_authentation()
            self._auth_blocked_until = self.__current_timestamp() + AUTH_COOLDOWN_SECONDS
            return False

        if "id_token" in data and "refresh_token" in data:
            # Refresh tokens rotate; persist the new one immediately.
            await self.__store_auth0(data)
            self._auth_blocked_until = 0
            if initialise:
                await self.__init_accounts()
            return True

        _LOGGER.error(
            "E.ON Next Auth0 refresh rejected: %s %s",
            data.get("error"), data.get("error_description"),
        )
        self.__reset_authentation()
        # Engage cooldown while still holding the lock so tasks queued behind
        # us see it and don't present the spent token again.
        self._auth_blocked_until = self.__current_timestamp() + AUTH_COOLDOWN_SECONDS
        return False

    def __reset_accounts(self):
        self.accounts = []

    async def __get_account_numbers(self):
        result = await self._graphql_post(
            "headerGetLoggedInUser",
            "query headerGetLoggedInUser {\n  viewer {\n    accounts {\n      ... on AccountType {\n        applications(first: 1) {\n          edges {\n            node {\n              isMigrated\n              migrationSource\n              __typename\n            }\n            __typename\n          }\n          __typename\n        }\n        balance\n        id\n        number\n        __typename\n      }\n      __typename\n    }\n    id\n    preferredName\n    __typename\n  }\n}\n",
        )

        if not self._json_contains_key_chain(result, ["data", "viewer", "accounts"]):
            raise Exception("Unable to load energy accounts")

        return [a["number"] for a in result["data"]["viewer"]["accounts"]]

    async def __init_accounts(self):
        if len(self.accounts) == 0:
            for account_number in await self.__get_account_numbers():
                account = EnergyAccount(self, account_number)
                await account._load_meters()
                await account._load_ev_chargers()
                await account._load_tariff_data()
                await account._load_saving_sessions()
                await account._load_billing_data()
                self.accounts.append(account)


class EnergyAccount:

    def __init__(self, api, account_number):
        self.api = api
        self.account_number = account_number
        self.ev_chargers = []
        self.tariff_data = None
        # Full half-hourly rate schedule for the current + next day, sourced from
        # the new E.ON interface's `tariffs` operation. Each entry:
        #   {"validFrom": iso, "validTo": iso, "value": pence_per_kwh}
        self.rate_schedule = []
        # Uplifts (e.g. wholesale pass-through) applied on top of unit rates.
        self.rate_uplifts = []
        self.saving_sessions = []
        self.billing_data = []
        self.postcode = ""

    async def _load_tariff_data(self):
        """Load active tariff/agreement details for this account.

        Requests the half-hourly unit-rate schedule for a window spanning today
        and tomorrow so the whole day's cheap/expensive blocks are available,
        matching what the new E.ON web interface shows.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        day_start = (now - datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = day_start + datetime.timedelta(days=3)
        date_from = day_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_to = day_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        result = await self.api._graphql_post(
            "getAccountAgreements",
            "query getAccountAgreements($accountNumber: String!, $from: DateTime, $to: DateTime) { properties(accountNumber: $accountNumber) { electricityMeterPoints { mpan agreements { id validFrom validTo tariff { __typename ... on TariffType { displayName fullName tariffCode } ... on StandardTariff { unitRate standingCharge } ... on PrepayTariff { unitRate standingCharge } ... on HalfHourlyTariff { unitRates(from: $from, to: $to) { value validFrom validTo rateType } standingCharge } } unitRateUplifts { unitRateUplift validFrom validTo } } } } }",
            {
                "accountNumber": self.account_number,
                "from": date_from,
                "to": date_to,
            },
        )

        self.tariff_data = []
        self.rate_schedule = []
        self.rate_uplifts = []
        if self.api._json_contains_key_chain(result, ["data", "properties"]):
            for prop in result["data"]["properties"]:
                if "electricityMeterPoints" not in prop:
                    continue
                for point in prop["electricityMeterPoints"]:
                    mpan = point.get("mpan")
                    for agreement in point.get("agreements", []):
                        # Inject meterPoint info for sensor compatibility
                        agreement["meterPoint"] = {"mpan": mpan}
                        self.tariff_data.append(agreement)

                        tariff = agreement.get("tariff") or {}
                        for rate in tariff.get("unitRates") or []:
                            if rate.get("validFrom") and rate.get("validTo"):
                                self.rate_schedule.append({
                                    "validFrom": rate["validFrom"],
                                    "validTo": rate["validTo"],
                                    "value": rate.get("value"),
                                    "rateType": rate.get("rateType"),
                                    "mpan": mpan,
                                })
                        for up in agreement.get("unitRateUplifts") or []:
                            if up.get("validFrom") and up.get("validTo"):
                                self.rate_uplifts.append({
                                    "unitRateUplift": up.get("unitRateUplift"),
                                    "validFrom": up["validFrom"],
                                    "validTo": up["validTo"],
                                    "mpan": mpan,
                                })

        # Sort the schedule chronologically so window computation is stable.
        self.rate_schedule.sort(key=lambda r: r["validFrom"])

    # ------------------------------------------------------------------
    # Cheap-window helpers (the "active window all day" logic)
    # ------------------------------------------------------------------
    def _uplift_at(self, moment_iso, mpan):
        """Return the uplift (p/kWh) that applies at a given instant, if any."""
        total = 0.0
        for u in self.rate_uplifts:
            if u.get("mpan") not in (mpan, None):
                continue
            if u["validFrom"] <= moment_iso < u["validTo"]:
                try:
                    total += float(u.get("unitRateUplift") or 0)
                except (TypeError, ValueError):
                    pass
        return total

    def get_rate_schedule(self):
        return self.rate_schedule

    def get_cheap_windows(self, threshold_p=DEFAULT_CHEAP_THRESHOLD_P):
        """Collapse the rate schedule into contiguous cheap windows.

        A slot is cheap when its effective rate (unit rate + uplift) is at or
        below ``threshold_p`` pence/kWh. Returns a list of
        ``{"start": iso, "end": iso, "rate_p": float}`` merged across adjacent
        cheap slots, so a whole-day cheap period is reported as one window.
        """
        windows = []
        for r in self.rate_schedule:
            try:
                base = float(r.get("value") or 0)
            except (TypeError, ValueError):
                continue
            eff = base + self._uplift_at(r["validFrom"], r.get("mpan"))
            if eff <= threshold_p:
                if windows and windows[-1]["end"] == r["validFrom"]:
                    # Merge with the previous contiguous cheap window.
                    windows[-1]["end"] = r["validTo"]
                    windows[-1]["rate_p"] = max(windows[-1]["rate_p"], round(eff, 4))
                else:
                    windows.append({
                        "start": r["validFrom"],
                        "end": r["validTo"],
                        "rate_p": round(eff, 4),
                    })
        return windows

    def current_rate_p(self):
        """Effective rate (p/kWh) for the slot containing now, or None."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for r in self.rate_schedule:
            if r["validFrom"] <= now_iso < r["validTo"]:
                base = float(r.get("value") or 0)
                return round(base + self._uplift_at(now_iso, r.get("mpan")), 4)
        return None

    def is_cheap_now(self, threshold_p=DEFAULT_CHEAP_THRESHOLD_P):
        rate = self.current_rate_p()
        if rate is None:
            return None
        return rate <= threshold_p

    def get_charging_windows(self):
        """Return the EV charging windows E.ON has assigned.

        These are the ``flexPlannedDispatches`` slots E.ON scheduled for the
        car's charge. They are the authoritative "cheap window" for charging:
        if the car is scheduled to charge in a slot, that slot is E.ON's chosen
        cheap period. Exposing them lets you confirm the integration retrieved
        the correct charging schedule (e.g. an all-day window).

        Returns a list of ``{"start": iso, "end": iso, "type": ...,
        "energy_added_kwh": ...}`` merged across chargers and adjacent slots.
        """
        raw = []
        for charger in self.ev_chargers:
            for d in (charger.schedule or []):
                if d.get("start") and d.get("end"):
                    raw.append({
                        "start": d["start"],
                        "end": d["end"],
                        "type": d.get("type"),
                        "energy_added_kwh": d.get("energyAddedKwh"),
                    })
        raw.sort(key=lambda w: w["start"])

        merged = []
        for w in raw:
            if merged and merged[-1]["end"] == w["start"]:
                merged[-1]["end"] = w["end"]
                if w.get("energy_added_kwh"):
                    merged[-1]["energy_added_kwh"] = round(
                        (merged[-1].get("energy_added_kwh") or 0)
                        + w["energy_added_kwh"], 3
                    )
            else:
                merged.append(dict(w))
        return merged

    def is_charging_window_now(self):
        """True if now falls inside an E.ON-assigned EV charging window.

        Returns None if no charging schedule is available (unknown), so callers
        can distinguish "not in a window" from "no schedule to judge".
        """
        windows = self.get_charging_windows()
        if not windows:
            return None
        now_iso = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for w in windows:
            if w["start"] <= now_iso < w["end"]:
                return True
        return False

    async def _load_saving_sessions(self):
        """Load saving session data (similar to Octopus Saving Sessions)."""
        result = await self.api._graphql_post(
            "getSavingSessions",
            "query getSavingSessions($postcode: String!) { appSessions(postcode: $postcode) { edges { node { id startedAt __typename } } } }",
            {"postcode": self.postcode},
        )

        if self.api._json_contains_key_chain(result, ["data", "appSessions", "edges"]):
            self.saving_sessions = [e["node"] for e in result["data"]["appSessions"]["edges"]]
        else:
            self.saving_sessions = []

    async def _load_billing_data(self):
        """Load billing history for the account."""
        result = await self.api._graphql_post(
            "getAccountBilling",
            "query getAccountBilling($accountNumber: String!) { account(accountNumber: $accountNumber) { bills(first: 10) { edges { node { id billDate dueDate amount status } } } } }",
            {"accountNumber": self.account_number},
        )

        if self.api._json_contains_key_chain(result, ["data", "account", "bills", "edges"]):
            self.billing_data = [e["node"] for e in result["data"]["account"]["bills"]["edges"]]
        else:
            self.billing_data = []

    async def _load_ev_chargers(self):
        result = await self.api._graphql_post(
            "getAccountDevices",
            "query getAccountDevices($accountNumber: String!) {\n  devices(accountNumber: $accountNumber) {\n    id\n    provider\n    deviceType\n    status {\n      current\n    }\n    __typename\n    ... on SmartFlexVehicle {\n      make\n      model\n    }\n    ... on SmartFlexChargePoint {\n      make\n      model\n    }\n  }\n}\n",
            {"accountNumber": self.account_number},
        )

        if self.api._json_contains_key_chain(result, ["data", "devices"]) is True:
            seen_names = set()
            for device in result["data"]["devices"]:
                # Treat live vehicles and charge points as SmartCharging entities.
                if device.get("status", {}).get("current") == "LIVE":
                    name = f"{device.get('make', 'Unknown')} {device.get('model', 'Device')}"
                    if name in seen_names:
                        _LOGGER.warning("Skipping duplicate EV device: %s", name)
                        continue
                    seen_names.add(name)
                    charger = SmartCharging(self, device["id"], name)
                    self.ev_chargers.append(charger)

    async def _load_meters(self):
        result = await self.api._graphql_post(
            "getAccountMeterSelector",
            "query getAccountMeterSelector($accountNumber: String!, $showInactive: Boolean!) {\n  properties(accountNumber: $accountNumber) {\n    ...MeterSelectorPropertyFields\n    __typename\n  }\n}\n\nfragment MeterSelectorPropertyFields on PropertyType {\n  __typename\n  electricityMeterPoints {\n    ...MeterSelectorElectricityMeterPointFields\n    __typename\n  }\n  gasMeterPoints {\n    ...MeterSelectorGasMeterPointFields\n    __typename\n  }\n  id\n  postcode\n}\n\nfragment MeterSelectorElectricityMeterPointFields on ElectricityMeterPointType {\n  __typename\n  id\n  meters(includeInactive: $showInactive) {\n    ...MeterSelectorElectricityMeterFields\n    __typename\n  }\n}\n\nfragment MeterSelectorElectricityMeterFields on ElectricityMeterType {\n  __typename\n  activeTo\n  id\n  registers {\n    id\n    name\n    __typename\n  }\n  serialNumber\n}\n\nfragment MeterSelectorGasMeterPointFields on GasMeterPointType {\n  __typename\n  id\n  meters(includeInactive: $showInactive) {\n    ...MeterSelectorGasMeterFields\n    __typename\n  }\n}\n\nfragment MeterSelectorGasMeterFields on GasMeterType {\n  __typename\n  activeTo\n  id\n  registers {\n    id\n    name\n    __typename\n  }\n  serialNumber\n}\n",
            {"accountNumber": self.account_number, "showInactive": False},
        )

        if not self.api._json_contains_key_chain(result, ["data", "properties"]):
            raise Exception("Unable to load energy meters for account " + self.account_number)

        self.meters = []
        for prop in result["data"]["properties"]:
            self.postcode = prop.get("postcode")

            for electricity_point in prop["electricityMeterPoints"]:
                for meter_config in electricity_point["meters"]:
                    meter = ElectricityMeter(self, meter_config["id"], meter_config["serialNumber"])
                    self.meters.append(meter)

            for gas_point in prop["gasMeterPoints"]:
                for meter_config in gas_point["meters"]:
                    meter = GasMeter(self, meter_config["id"], meter_config["serialNumber"])
                    self.meters.append(meter)


class EnergyMeter:

    def __init__(self, account, meter_id, serial):
        self.account = account
        self.api = account.api
        self.last_updated = None
        self.type = METER_TYPE_UNKNOWN
        self.meter_id = meter_id
        self.serial = serial
        self.latest_reading = None
        self.latest_reading_date = None

    def get_type(self):
        return self.type

    def get_serial(self):
        return self.serial

    def _should_update(self):
        if self.last_updated is None:
            return True
        now = datetime.datetime.now()
        if now.strftime("%d") != self.last_updated.strftime("%d"):
            if now.hour >= 7:
                return True
        return False

    def _convert_datetime_str_to_date(self, datetime_str):
        date_chunks = str(datetime_str.split("T")[0]).split("-")
        return datetime.date(int(date_chunks[0]), int(date_chunks[1]), int(date_chunks[2]))

    async def _update(self):
        pass

    async def update(self):
        if self._should_update() is True:
            await self._update()

    async def has_reading(self):
        await self.update()
        return self.latest_reading is not None

    async def get_latest_reading(self):
        await self.update()
        return self.latest_reading

    async def get_latest_reading_date(self):
        await self.update()
        return self.latest_reading_date


class ElectricityMeter(EnergyMeter):

    def __init__(self, account, meter_id, serial):
        super().__init__(account, meter_id, serial)
        self.type = METER_TYPE_ELECTRIC

    async def _update(self):
        result = await self.api._graphql_post(
            "meterReadingsHistoryTableElectricityReadings",
            "query meterReadingsHistoryTableElectricityReadings($accountNumber: String!, $cursor: String, $meterId: String!) {\n  readings: electricityMeterReadings(\n    accountNumber: $accountNumber\n    after: $cursor\n    first: 12\n    meterId: $meterId\n  ) {\n    edges {\n      ...MeterReadingsHistoryTableElectricityMeterReadingConnectionTypeEdge\n      __typename\n    }\n    pageInfo {\n      endCursor\n      hasNextPage\n      __typename\n    }\n    __typename\n  }\n}\n\nfragment MeterReadingsHistoryTableElectricityMeterReadingConnectionTypeEdge on ElectricityMeterReadingConnectionTypeEdge {\n  node {\n    id\n    readAt\n    readingSource\n    registers {\n      name\n      value\n      __typename\n    }\n    source\n    __typename\n  }\n  __typename\n}\n",
            {"accountNumber": self.account.account_number, "cursor": "", "meterId": self.meter_id},
        )

        if not self.api._json_contains_key_chain(result, ["data", "readings"]):
            raise Exception("Unable to load readings for meter " + self.serial)

        readings = result["data"]["readings"]["edges"]
        if len(readings) > 0:
            self.latest_reading = round(float(readings[0]["node"]["registers"][0]["value"]))
            self.latest_reading_date = self._convert_datetime_str_to_date(readings[0]["node"]["readAt"])
            self.last_updated = datetime.datetime.now()


class GasMeter(EnergyMeter):

    def __init__(self, account, meter_id, serial):
        super().__init__(account, meter_id, serial)
        self.type = METER_TYPE_GAS

    async def _update(self):
        result = await self.api._graphql_post(
            "meterReadingsHistoryTableGasReadings",
            "query meterReadingsHistoryTableGasReadings($accountNumber: String!, $cursor: String, $meterId: String!) {\n  readings: gasMeterReadings(\n    accountNumber: $accountNumber\n    after: $cursor\n    first: 12\n    meterId: $meterId\n  ) {\n    edges {\n      ...MeterReadingsHistoryTableGasMeterReadingConnectionTypeEdge\n      __typename\n    }\n    pageInfo {\n      endCursor\n      hasNextPage\n      __typename\n    }\n    __typename\n  }\n}\n\nfragment MeterReadingsHistoryTableGasMeterReadingConnectionTypeEdge on GasMeterReadingConnectionTypeEdge {\n  node {\n    id\n    readAt\n    readingSource\n    registers {\n      name\n      value\n      __typename\n    }\n    source\n    __typename\n  }\n  __typename\n}\n",
            {"accountNumber": self.account.account_number, "cursor": "", "meterId": self.meter_id},
        )

        if not self.api._json_contains_key_chain(result, ["data", "readings"]):
            raise Exception("Unable to load readings for meter " + self.serial)

        readings = result["data"]["readings"]["edges"]
        if len(readings) > 0:
            self.latest_reading = round(float(readings[0]["node"]["registers"][0]["value"]))
            self.latest_reading_date = self._convert_datetime_str_to_date(readings[0]["node"]["readAt"])
            self.last_updated = datetime.datetime.now()

    async def get_latest_reading_kwh(self):
        m3 = await self.get_latest_reading()
        gas_caloric_value = 38
        kwh = m3 * 1.02264
        kwh = kwh * gas_caloric_value
        kwh = kwh / 3.6
        return round(kwh)


class SmartCharging(EnergyMeter):

    def __init__(self, account, meter_id, serial):
        super().__init__(account, meter_id, serial)
        self.type = METER_TYPE_EV
        self.schedule = None
        self.failed_attempts = 0

    def _should_update(self):
        if self.last_updated is None:
            return True
        now = datetime.datetime.now()
        # Retry sooner if we had failures (exponential backoff: 1-5 min).
        retry_minutes = min(1 + self.failed_attempts, 5)
        return (now - self.last_updated) >= datetime.timedelta(minutes=retry_minutes)

    async def _update(self):
        try:
            result = await self.api._graphql_post(
                "getSmartChargingSchedule",
                "query getSmartChargingSchedule($deviceId: String!) {\n  flexPlannedDispatches(deviceId: $deviceId) {\n    start\n    end\n    type\n    energyAddedKwh\n  }\n}\n",
                {"deviceId": self.meter_id},
            )

            if self.api._json_contains_key_chain(result, ["data", "flexPlannedDispatches"]):
                new_schedule = result["data"].get("flexPlannedDispatches")
                if new_schedule is None:
                    _LOGGER.warning(
                        "SmartCharging API returned null flexPlannedDispatches for %s; "
                        "keeping previous schedule", self.serial,
                    )
                    self.failed_attempts += 1
                else:
                    self.schedule = new_schedule
                    self.failed_attempts = 0
            else:
                _LOGGER.warning(
                    "SmartCharging API response missing flexPlannedDispatches for %s; "
                    "keeping previous schedule", self.serial,
                )
                self.failed_attempts += 1
        except Exception as e:
            _LOGGER.error("SmartCharging._update() failed for %s: %s; keeping previous schedule", self.serial, e)
            self.failed_attempts += 1
        finally:
            # Always update timestamp to prevent API hammering on failure.
            self.last_updated = datetime.datetime.now()

    async def get_schedule(self):
        await self.update()
        return self.schedule
