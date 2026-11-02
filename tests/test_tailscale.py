"""
Starting Tailscale with the panel, against a fake CLI: signing in to a real tailnet
isn't something a test can do. The panel's side uses the real routes.
"""
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fakes  # noqa: F401  (keeps the real .env out of the tests)

from panel import app as _both_pipelines  # noqa: F401
from panel import server as webapp
from panel import tailscale

RUNNING = {"BackendState": "Running", "TailscaleIPs": ["100.64.0.7", "fd7a:115c:a1e0::7"]}


class FakeCli:
    """Answers `status --json` with the current state; a bare `up` moves Stopped to Running."""

    def __init__(self, state, tray_open=True):
        self.state = state
        self.tray_open = tray_open
        self.ups = []

    def run(self, argv, **kwargs):
        if argv[0] == "tasklist":
            return subprocess.CompletedProcess(argv, 0, "tailscale-ipn.exe  14188 Console" if self.tray_open else "INFO: No tasks")
        if argv[1:] == ["status", "--json"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.state))
        self.ups.append(argv[1:])
        self.state = RUNNING
        return subprocess.CompletedProcess(argv, 0, "")


class TailscaleTest(unittest.TestCase):
    def use(self, cli, installed=True):
        for patcher in (mock.patch.object(tailscale, "program", lambda name: f"C:/ts/{name}.exe" if installed else None),
                        mock.patch.object(tailscale.subprocess, "run", cli.run),
                        mock.patch.object(tailscale.time, "sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)
        popen = mock.patch.object(tailscale.subprocess, "Popen").start()
        self.addCleanup(mock.patch.stopall)
        return popen

    def test_not_installed_does_nothing(self):
        cli = FakeCli(RUNNING)
        popen = self.use(cli, installed=False)
        self.assertEqual(tailscale.connect(8800), (None, None))
        self.assertEqual(tailscale.reach(8800), (None, None))
        popen.assert_not_called()

    def test_connected_gives_the_ipv4_address(self):
        cli = FakeCli(RUNNING)
        self.use(cli)
        self.assertEqual(tailscale.connect(8800), ("http://100.64.0.7:8800", None))
        self.assertEqual(cli.ups, [], "already connected, so nothing to ask for")

    def test_disconnected_reconnects_with_a_bare_up(self):
        cli = FakeCli({"BackendState": "Stopped", "TailscaleIPs": []})
        self.use(cli)
        self.assertEqual(tailscale.connect(8800), ("http://100.64.0.7:8800", None))
        self.assertEqual(cli.ups, [["up"]])

    def test_signed_out_says_so_without_hanging_on_up(self):
        cli = FakeCli({"BackendState": "NeedsLogin", "TailscaleIPs": []})
        self.use(cli)
        with mock.patch.object(tailscale, "CONNECT_WAIT", 0):
            self.assertEqual(tailscale.connect(8800), (None, tailscale.SIGN_IN))
        self.assertEqual(cli.ups, [], "`up` would block on signing in")

    def test_service_down_says_not_connected(self):
        cli = FakeCli(RUNNING)
        self.use(cli)
        with mock.patch.object(tailscale, "status", return_value=None):
            self.assertEqual(tailscale.reach(8800), (None, tailscale.NOT_CONNECTED))

    @unittest.skipUnless(os.name == "nt", "the tray app is Windows-only")
    def test_tray_starts_only_when_closed(self):
        popen = self.use(FakeCli(RUNNING, tray_open=True))
        tailscale.start_tray()
        popen.assert_not_called()
        popen = self.use(FakeCli(RUNNING, tray_open=False))
        tailscale.start_tray()
        popen.assert_called_once()
        self.assertEqual(popen.call_args.args[0], ["C:/ts/tailscale-ipn.exe"])


class PanelOverTailscaleTest(unittest.TestCase):
    TAILNET_PHONE = {"REMOTE_ADDR": "100.64.0.9"}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"APP_PASSWORD": "open sesame", "COMFYUI_URL": "http://127.0.0.1:9"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = webapp.app.test_client()

    def sign_in(self):
        with self.client.session_transaction() as session:
            session["authed"] = True

    def test_panel_shows_the_address(self):
        self.sign_in()
        with mock.patch.object(webapp.tailscale, "reach", return_value=("http://100.64.0.7:8800", None)) as reach:
            page = self.client.get("/", base_url="http://localhost:8800").data.decode()
        reach.assert_called_once_with("8800")
        self.assertIn("<code>http://100.64.0.7:8800</code>", page)

    def test_panel_says_whats_in_the_way(self):
        self.sign_in()
        with mock.patch.object(webapp.tailscale, "reach", return_value=(None, tailscale.SIGN_IN)):
            self.assertIn(tailscale.SIGN_IN, self.client.get("/").data.decode())

    def test_panel_says_nothing_without_tailscale(self):
        self.sign_in()
        with mock.patch.object(webapp.tailscale, "reach", return_value=(None, None)):
            self.assertNotIn("Phone (anywhere", self.client.get("/").data.decode())

    def test_no_password_means_no_phone_so_it_isnt_asked(self):
        with mock.patch.dict(os.environ, {"APP_PASSWORD": ""}), mock.patch.object(webapp.tailscale, "reach") as reach:
            self.client.get("/")
        reach.assert_not_called()

    def test_a_phone_on_the_tailnet_isnt_this_pc(self):
        self.assertEqual(self.client.get("/", environ_base=self.TAILNET_PHONE).status_code, 302, "it still signs in")
        self.assertEqual(self.client.post("/launcher/refresh", data={"code_changed_at": "1"},
                                          environ_base=self.TAILNET_PHONE).status_code, 403)


class StartupTest(unittest.TestCase):
    def run_main(self, *args):
        with mock.patch.object(sys, "argv", ["webapp.py", *args]), \
                mock.patch.dict(os.environ, {"APP_PASSWORD": "open sesame"}), \
                mock.patch.object(webapp.app, "run"), \
                mock.patch.object(webapp, "ComfyMonitor"), \
                mock.patch.object(webapp.tailscale, "connect", return_value=(None, None)) as connect, \
                mock.patch.object(webapp.threading, "Thread", side_effect=lambda target, args, daemon: mock.Mock(start=lambda: target(*args))):
            webapp.main()
        return connect

    def test_listening_to_other_devices_starts_tailscale(self):
        self.run_main("--host", "0.0.0.0", "--port", "8800").assert_called_once_with(8800)

    def test_this_pc_only_leaves_it_alone(self):
        self.run_main().assert_not_called()


if __name__ == "__main__":
    unittest.main()
