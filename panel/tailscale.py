"""
Tailscale lets a phone reach the panel from any network. The phone talks straight
to the panel's own port over the tailnet, where it still has to sign in and still
counts as another device. (Not `tailscale serve`: it proxies from localhost, which
the panel would take for this PC.)
"""
import json
import os
import shutil
import subprocess
import time

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# Seconds connect() gives Tailscale to come up, and to be signed in to on a first run.
CONNECT_WAIT = 30
SIGN_IN = "Sign in to Tailscale from its tray icon."
NOT_CONNECTED = "Tailscale isn't connected."


def program(name):
    """A Tailscale program, from the PATH or where the Windows installer puts it."""
    found = shutil.which(name)
    if found:
        return found
    path = os.path.join(os.getenv("ProgramFiles", ""), "Tailscale", f"{name}.exe")
    return path if os.path.exists(path) else None


def status():
    """`tailscale status --json`, or None if it's not installed or its service isn't running."""
    cli = program("tailscale")
    if not cli:
        return None
    try:
        out = subprocess.run([cli, "status", "--json"], capture_output=True, text=True, encoding="utf-8",
                             timeout=5, creationflags=NO_WINDOW).stdout
        return json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def reach(port):
    """(the panel's address on the tailnet, what's in the way), both None if Tailscale isn't installed."""
    if not program("tailscale"):
        return None, None
    state = status() or {}
    backend = state.get("BackendState")
    ipv4 = next((ip for ip in state.get("TailscaleIPs") or [] if "." in ip), None)
    if backend == "Running" and ipv4:
        return f"http://{ipv4}:{port}", None
    return None, SIGN_IN if backend in ("NeedsLogin", "NeedsMachineAuth") else NOT_CONNECTED


def start_tray():
    """On Windows the connection only lasts while the tray app is open, and a user can quit it."""
    app = program("tailscale-ipn")
    if os.name != "nt" or not app:
        return
    running = subprocess.run(["tasklist", "/FI", "IMAGENAME eq tailscale-ipn.exe", "/NH"],
                             capture_output=True, text=True, creationflags=NO_WINDOW).stdout
    if "tailscale-ipn.exe" not in running.lower():
        # Detached: it stays up, and closable from its icon, after Shut down stops the server.
        subprocess.Popen([app], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)


def connect(port):
    """Starts Tailscale if it's installed and waits for it to come up. Returns the same as reach()."""
    cli = program("tailscale")
    if not cli:
        return None, None
    start_tray()
    asked = False
    deadline = time.time() + CONNECT_WAIT
    while True:
        url, problem = reach(port)
        if url or time.time() > deadline:
            return url, problem
        if not asked and (status() or {}).get("BackendState") == "Stopped":
            asked = True
            # No flags on purpose: a bare `up` just reconnects, while any flag makes it
            # compare every setting against the current ones and refuse if they differ.
            try:
                subprocess.run([cli, "up"], capture_output=True, timeout=CONNECT_WAIT, creationflags=NO_WINDOW)
            except (OSError, subprocess.SubprocessError):
                pass
        time.sleep(1)
