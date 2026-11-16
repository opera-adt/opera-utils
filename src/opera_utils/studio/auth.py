"""Earthdata Login for the studio.

Credentials come from, in order:

1. a sign-in made in the UI during this session (kept in memory only);
2. a sign-in the user asked the studio to remember (``<config>/netrc``, mode 0600);
3. the user's own ``~/.netrc`` (``urs.earthdata.nasa.gov`` entry);
4. ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``.

opera-utils reads ``~/.netrc`` before the environment, and the CSLC path reads
only ``~/.netrc``. So when (1) or (2) is active, every job subprocess gets a
private ``HOME`` holding a ``.netrc`` with those credentials. The user's own
``~/.netrc`` is never written.
"""

from __future__ import annotations

import netrc
import os
import stat
import threading
from dataclasses import dataclass
from pathlib import Path

import requests

URS_HOST = "urs.earthdata.nasa.gov"
# Lists the user's tokens; answers 401 for a wrong username or password.
VALIDATE_URL = f"https://{URS_HOST}/api/users/tokens"


class AuthError(ValueError):
    """Earthdata rejected the credentials, or they could not be checked."""


@dataclass
class Credentials:
    """An Earthdata username and password, and where they came from."""

    username: str
    password: str
    source: str  # "session" | "saved" | "netrc" | "env"


def _netrc_entry(path: Path) -> tuple[str, str] | None:
    try:
        auth = netrc.netrc(str(path)).authenticators(URS_HOST)
    except (FileNotFoundError, netrc.NetrcParseError, PermissionError):
        return None
    if auth and auth[0] and auth[2]:
        return auth[0], auth[2]
    return None


def netrc_text(username: str, password: str) -> str:
    """Return a ``.netrc`` body for Earthdata (URS and the hosts redirecting to it)."""
    if any(c.isspace() for c in username + password):
        msg = "Earthdata usernames and passwords cannot contain spaces for .netrc"
        raise AuthError(msg)
    return f"machine {URS_HOST} login {username} password {password}\n"


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with 0600 from the start, so the password is never world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    Path(path).chmod(stat.S_IRUSR | stat.S_IWUSR)


def validate(username: str, password: str, timeout: float = 20) -> None:
    """Raise AuthError unless Earthdata accepts ``username`` / ``password``."""
    if not username or not password:
        msg = "Enter both a username and a password."
        raise AuthError(msg)
    try:
        r = requests.get(VALIDATE_URL, auth=(username, password), timeout=timeout)
    except requests.RequestException as e:
        msg = f"Could not reach Earthdata Login: {e}"
        raise AuthError(msg) from e
    if r.status_code in (401, 403):
        msg = "Earthdata Login rejected this username or password."
        raise AuthError(msg)
    if r.status_code >= 400:
        msg = f"Earthdata Login answered {r.status_code}; try again later."
        raise AuthError(msg)


class AuthStore:
    """The active Earthdata credentials, and the job environment that carries them."""

    def __init__(self, config_dir: Path, home: Path | None = None):
        """Save credentials under ``config_dir``; look for ``.netrc`` in ``home``."""
        self.config_dir = Path(config_dir)
        self.saved_path = self.config_dir / "netrc"
        self.home = Path(home) if home is not None else Path.home()
        self._session: Credentials | None = None
        self._lock = threading.Lock()

    # -- reading -----------------------------------------------------------
    def current(self) -> Credentials | None:
        with self._lock:
            if self._session is not None:
                return self._session
        saved = _netrc_entry(self.saved_path)
        if saved:
            return Credentials(saved[0], saved[1], "saved")
        own = _netrc_entry(self.home / ".netrc")
        if own:
            return Credentials(own[0], own[1], "netrc")
        user = os.environ.get("EARTHDATA_USERNAME", "")
        pw = os.environ.get("EARTHDATA_PASSWORD", "")
        if user and pw:
            return Credentials(user, pw, "env")
        return None

    def status(self) -> dict:
        c = self.current()
        labels = {
            "session": "Signed in for this session",
            "saved": "Remembered by OPERA Studio",
            "netrc": "From ~/.netrc",
            "env": "From EARTHDATA_* variables",
        }
        if c is None:
            return {"signed_in": False, "source": None, "username": None, "label": None}
        return {
            "signed_in": True,
            "source": c.source,
            "username": c.username,
            "label": labels[c.source],
            "can_sign_out": c.source in ("session", "saved"),
        }

    # -- changing ----------------------------------------------------------
    def sign_in(
        self, username: str, password: str, remember: bool = False, check: bool = True
    ) -> dict:
        username = username.strip()
        if check:
            validate(username, password)
        text = netrc_text(username, password)
        with self._lock:
            self._session = Credentials(username, password, "session")
        if remember:
            _write_private(self.saved_path, text)
        return self.status()

    def sign_out(self) -> dict:
        with self._lock:
            self._session = None
        self.saved_path.unlink(missing_ok=True)
        return self.status()

    # -- jobs ---------------------------------------------------------------
    def job_env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Environment for a job subprocess.

        With UI credentials, ``HOME`` points at a private folder whose ``.netrc``
        holds them (caches stay where they were through ``XDG_CACHE_HOME``).
        """
        env = dict(os.environ if base is None else base)
        c = self.current()
        if c is None or c.source in ("netrc", "env"):
            return env  # opera-utils finds these on its own
        job_home = self.config_dir / "job-home"
        netrc_path = job_home / ".netrc"
        _write_private(netrc_path, netrc_text(c.username, c.password))
        real_home = env.get("HOME", str(self.home))
        env.setdefault("XDG_CACHE_HOME", str(Path(real_home) / ".cache"))
        env["HOME"] = str(job_home)
        env["NETRC"] = str(netrc_path)  # aiohttp (TROPO) honours NETRC
        env["GDAL_HTTP_NETRC"] = "YES"
        env["GDAL_HTTP_NETRC_FILE"] = str(netrc_path)
        env["EARTHDATA_USERNAME"] = c.username
        env["EARTHDATA_PASSWORD"] = c.password
        return env
