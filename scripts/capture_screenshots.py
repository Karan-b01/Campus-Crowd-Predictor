"""Automated screenshot capture script using headless Microsoft Edge via CDP."""

from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
EDGE_EXE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def get_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_for_server(port: int, timeout_sec: int = 35) -> bool:
    """Poll the Streamlit server until it responds with HTTP 200."""
    url = f"http://127.0.0.1:{port}/_stcore/health"
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthChecker"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def get_edge_websocket_url(dbg_port: int, timeout_sec: int = 15) -> str:
    """Retrieve the WebSocket debugger URL from Edge."""
    url = f"http://127.0.0.1:{dbg_port}/json"
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                tabs = json.loads(resp.read())
                for tab in tabs:
                    if tab.get("type") == "page" and "webSocketDebuggerUrl" in tab:
                        return str(tab["webSocketDebuggerUrl"])
                if tabs and "webSocketDebuggerUrl" in tabs[0]:
                    return str(tabs[0]["webSocketDebuggerUrl"])
        except Exception:
            time.sleep(0.4)
    raise RuntimeError(f"Edge debugger on port {dbg_port} failed to respond.")


def capture() -> None:
    """Launch Streamlit and Edge CDP to capture high-fidelity screenshots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not EDGE_EXE.is_file():
        raise FileNotFoundError(f"Edge executable not found at {EDGE_EXE}")

    port = get_free_port()
    dbg_port = get_free_port()
    print(f"Streamlit port: {port} | Edge CDP port: {dbg_port}")

    log_path = PROJECT_ROOT / ".tmp" / "streamlit_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    cmd = [
        str(PYTHON_EXE),
        "-u",
        "-m",
        "streamlit",
        "run",
        "app/app.py",
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    print("Starting Streamlit server...")
    server_process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    edge_cmd = [
        str(EDGE_EXE),
        "--headless=new",
        f"--remote-debugging-port={dbg_port}",
        "--hide-scrollbars",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    print("Starting Edge in CDP mode...")
    edge_process = subprocess.Popen(
        edge_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    pages = [
        (
            "forecast_desktop",
            f"http://127.0.0.1:{port}/?page=forecast",
            1440,
            1150,
            False,
            "app_predict_desktop.png",
        ),
        (
            "forecast_mobile",
            f"http://127.0.0.1:{port}/?page=forecast",
            390,
            844,
            True,
            "app_predict_mobile.png",
        ),
        (
            "holiday_desktop",
            f"http://127.0.0.1:{port}/?page=forecast&holiday=1",
            1440,
            1150,
            False,
            "app_predict_holiday_desktop.png",
        ),
        (
            "sunday_desktop",
            f"http://127.0.0.1:{port}/?page=forecast&sunday=1",
            1440,
            1150,
            False,
            "app_predict_sunday_desktop.png",
        ),
        (
            "offhours_desktop",
            f"http://127.0.0.1:{port}/?page=forecast&offhours=1",
            1440,
            1150,
            False,
            "app_predict_offhours_desktop.png",
        ),
        (
            "planner_desktop",
            f"http://127.0.0.1:{port}/?page=planner",
            1440,
            1150,
            False,
            "app_planner_desktop.png",
        ),
        (
            "planner_mobile",
            f"http://127.0.0.1:{port}/?page=planner",
            390,
            844,
            True,
            "app_planner_mobile.png",
        ),
        (
            "insights_desktop",
            f"http://127.0.0.1:{port}/?page=insights",
            1440,
            1150,
            False,
            "app_insights_desktop.png",
        ),
        (
            "insights_mobile",
            f"http://127.0.0.1:{port}/?page=insights",
            390,
            844,
            True,
            "app_insights_mobile.png",
        ),
        (
            "about_desktop",
            f"http://127.0.0.1:{port}/?page=about",
            1440,
            1150,
            False,
            "app_about_desktop.png",
        ),
        (
            "about_mobile",
            f"http://127.0.0.1:{port}/?page=about",
            390,
            844,
            True,
            "app_about_mobile.png",
        ),
    ]

    try:
        print("Waiting for Streamlit...")
        if not wait_for_server(port):
            log_file.close()
            with log_path.open(encoding="utf-8") as f:
                logs = f.read()
            print("Streamlit startup logs:\n", logs)
            raise RuntimeError("Streamlit failed to start.")

        print("Connecting to Edge CDP...")
        ws_url = get_edge_websocket_url(dbg_port)
        print(f"Debugger connected: {ws_url}")

        with connect(ws_url, max_size=20 * 1024 * 1024) as ws:
            msg_id = 1

            # Warm-up navigation to compile app scripts & cache models
            print("Warming up Streamlit server...")
            ws.send(
                json.dumps(
                    {
                        "id": msg_id,
                        "method": "Page.navigate",
                        "params": {"url": f"http://127.0.0.1:{port}/?page=forecast"},
                    }
                )
            )
            msg_id += 1
            _ = ws.recv()
            time.sleep(7)

            for label, url, width, height, is_mobile, filename in pages:
                print(f"Capturing {label} -> {filename}...")

                # 1. Set Viewport metrics
                ws.send(
                    json.dumps(
                        {
                            "id": msg_id,
                            "method": "Emulation.setDeviceMetricsOverride",
                            "params": {
                                "width": width,
                                "height": height,
                                "deviceScaleFactor": 1,
                                "mobile": is_mobile,
                            },
                        }
                    )
                )
                msg_id += 1
                _ = ws.recv()

                # 2. Navigate to URL
                ws.send(
                    json.dumps(
                        {
                            "id": msg_id,
                            "method": "Page.navigate",
                            "params": {"url": url},
                        }
                    )
                )
                msg_id += 1
                _ = ws.recv()

                # 3. Allow Streamlit React + WebSocket + Plotly charts to render fully
                time.sleep(4.5)

                # 4. Capture Screenshot
                ws.send(
                    json.dumps(
                        {
                            "id": msg_id,
                            "method": "Page.captureScreenshot",
                            "params": {"format": "png"},
                        }
                    )
                )
                msg_id += 1
                resp = json.loads(ws.recv())
                data_b64 = resp["result"]["data"]
                img_bytes = base64.b64decode(data_b64)

                out_path = FIGURES_DIR / filename
                with out_path.open("wb") as img_file:
                    img_file.write(img_bytes)

                print(f"  Successfully wrote {filename} ({len(img_bytes)} bytes)")

    finally:
        print("Cleaning up processes...")
        edge_process.terminate()
        server_process.terminate()
        try:
            edge_process.wait(timeout=3)
            server_process.wait(timeout=3)
        except Exception:
            edge_process.kill()
            server_process.kill()
        log_file.close()
        print("Cleanup completed.")


if __name__ == "__main__":
    capture()
