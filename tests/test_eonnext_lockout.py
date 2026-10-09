#!/usr/bin/env python3
"""Regression test for the E.ON Next permanent-auth-lockout bug.

A single transient refresh failure (e.g. a DNS timeout) used to clear the
in-memory refresh token via ``__reset_authentation()``. ``__refresh_token_is_valid``
only consulted that in-memory field, so every later poll raised
``Unable to authenticate`` forever - only a HA restart recovered, even though the
token on disk was still valid.

Run with: python tests/test_eonnext_lockout.py [path/to/eonnext.py]
Stubs aiohttp, so no network is touched.
"""
import asyncio
import json
import os
import sys
import tempfile
import types
from pathlib import Path

# --- stub aiohttp before importing the module -------------------------------
CALLS = {"token": 0, "graphql": 0}
FAIL_NEXT = {"token": 0}          # number of upcoming token calls to raise on
REJECT = {"on": False}            # if set, token endpoint returns an error body
TOKEN_FILE = os.path.join(tempfile.mkdtemp(), "eon_auth0_refresh.json")


def _jwt(iat, exp):
    import base64

    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{b64({'alg':'RS256'})}.{b64({'iat': iat, 'exp': exp})}.sig"


class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status = data, status

    async def json(self):
        return self._d

    async def text(self):
        return json.dumps(self._d)


class _Ctx:
    def __init__(self, r):
        self._r = r

    async def __aenter__(self):
        return self._r

    async def __aexit__(self, *a):
        return False


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def post(self, url, **kw):
        if "oauth/token" in url:
            CALLS["token"] += 1
            if FAIL_NEXT["token"] > 0:
                FAIL_NEXT["token"] -= 1
                raise OSError("Simulated DNS resolution failure")
            if REJECT["on"]:
                return _Ctx(_Resp(
                    {"error": "invalid_grant",
                     "error_description": "Unknown or invalid refresh token"}, 403))
            now = 1_800_000_000
            return _Ctx(_Resp({
                "id_token": _jwt(now, now + 3600),
                "refresh_token": "rotated-%d" % CALLS["token"],
            }))
        CALLS["graphql"] += 1
        return _Ctx(_Resp({"data": {"viewer": {"accounts": []}}}))


aiohttp_stub = types.ModuleType("aiohttp")
aiohttp_stub.ClientSession = _Session
sys.modules["aiohttp"] = aiohttp_stub

# --- import the module under test -------------------------------------------
DEFAULT = Path(__file__).parent.parent / "custom_components" / "eon_next" / "eonnext.py"
VARIANT = next(
    (a for a in sys.argv[1:]
     if os.path.isfile(a) and os.path.basename(a) == "eonnext.py"),
    str(DEFAULT),
)
sys.path.insert(0, os.path.dirname(os.path.abspath(VARIANT)))
import eonnext  # noqa: E402

eonnext.AUTH0_REFRESH_FILE = TOKEN_FILE


def write_token(tok):
    with open(TOKEN_FILE, "w") as f:
        json.dump({"refresh_token": tok}, f)


def reset():
    CALLS.update(token=0, graphql=0)
    FAIL_NEXT["token"] = 0
    REJECT["on"] = False


def expire_access(api):
    """Force the next poll to need a fresh access token."""
    api.auth["token"] = {"token": None, "expires": None}
    api._auth_blocked_until = 0


async def poll(api, n=1):
    for _ in range(n):
        try:
            await api._graphql_post("op", "query op {}", authenticated=True)
            return True
        except Exception:
            await asyncio.sleep(0)
    return False


async def main():
    ok = True
    write_token("bootstrapped-token")

    # ---- Scenario 1: transient network failure must self-recover -----------
    reset()
    FAIL_NEXT["token"] = 1
    api = eonnext.EonNext()
    await api.login_with_username_and_password("u", "p")
    first = CALLS["token"]
    expire_access(api)
    recovered = await poll(api)
    print(f"1 transient: initial_calls={first} "
          f"recovered_without_restart={recovered} token_calls={CALLS['token']}")
    ok &= recovered

    # ---- Scenario 2: many transient failures in a row ----------------------
    reset()
    api2 = eonnext.EonNext()
    await api2.login_with_username_and_password("u", "p")
    for _ in range(5):
        FAIL_NEXT["token"] = 1
        expire_access(api2)
        await poll(api2)
    expire_access(api2)
    survived = await poll(api2)
    print(f"2 repeated transient failures -> still working={survived}")
    ok &= survived

    # ---- Scenario 3: rejected token must NOT be re-presented ---------------
    reset()
    REJECT["on"] = True
    api3 = eonnext.EonNext()
    await api3.login_with_username_and_password("u", "p")
    calls_after_reject = CALLS["token"]
    for _ in range(10):
        expire_access(api3)
        await poll(api3)
    spent = CALLS["token"] - calls_after_reject
    print(f"3 rejected token: extra presentations over 10 polls={spent} (want 0)")
    ok &= spent == 0

    # ---- Scenario 4: PKCE re-login recovers WITHOUT a restart --------------
    write_token("fresh-pkce-token")
    expire_access(api3)
    REJECT["on"] = False
    recovered = await poll(api3)
    print(f"4 recovery after writing new token file, no restart={recovered}")
    ok &= recovered

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def test_lockout_scenarios():
    """pytest entry point for the scenarios in main()."""
    assert asyncio.run(main()) == 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
