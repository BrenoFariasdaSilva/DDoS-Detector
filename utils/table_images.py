"""Provision Playwright and render result tables without a desktop/browser dependency.

All browser downloads belong to the active Python environment. Installation and
health checks are shared by Make, direct CLI entry points and lazy exporters.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any

from filelock import FileLock

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = Path(__file__).resolve()
_LOCK = threading.RLock()
_backend: tuple[int, str] | None = None
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _log(message: str) -> None:
    print(f"[TABLE IMAGES] {message}", flush=True)


def browser_directory() -> Path:
    """Keep browser revisions in the venv, separate from its pip packages."""
    return Path(sys.prefix).resolve() / "playwright-browsers"


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_directory())
    env["PIP_USER"] = "0"
    env.setdefault("PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT", "30000")
    return env


def _run(command: list[str], timeout: float) -> None:
    """Bound external operations; preserve the useful tail of failure output."""
    try:
        result = subprocess.run(
            command, env=_environment(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{command[1:]}: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"{command[1:]} exited {result.returncode}: {result.stdout[-3000:]}")


def _package_requirement() -> str:
    for line in (_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("playwright=="):
            return line.strip()
    raise RuntimeError("Playwright must be pinned in requirements.txt")


def _ensure_package() -> None:
    # Never use a global pip or silently mutate a system Python installation.
    try:
        _run([sys.executable, "-c", "from playwright.sync_api import sync_playwright"], 30)
        return
    except RuntimeError:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError("Playwright is unavailable; use the project's virtual environment")
    _log("Installing/repairing Playwright in the active virtual environment")
    _run([sys.executable, "-m", "pip", "install", "--no-user", "--force-reinstall",
          _package_requirement()], 300)


def _install_chromium() -> None:
    _log(f"Installing Chromium into {browser_directory()}")
    _run([sys.executable, "-m", "playwright", "install", "chromium"], 600)


def _install_linux_dependencies() -> None:
    if not sys.platform.startswith("linux"):
        return
    command = [sys.executable, "-m", "playwright", "install-deps", "chromium"]
    if os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if not sudo:
            raise RuntimeError("Linux browser libraries require administrator access; sudo unavailable")
        # Never block detached jobs waiting for a sudo password.
        command = [sudo, "-n", "--", *command]
    _log("Installing Chromium system libraries (non-interactive)")
    _run(command, 600)


def _verify_png(path: Path) -> None:
    from PIL import Image

    with path.open("rb") as stream:
        if stream.read(8) != _PNG_SIGNATURE:
            raise ValueError(f"Not a PNG: {path}")
    with Image.open(path) as image:
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Empty PNG: {path}")
        image.verify()


def _browser_render(html: Path, output: Path, timeout_ms: int, backend: str) -> None:
    """Executed in a child interpreter, including when caller has an asyncio loop."""
    from playwright.sync_api import sync_playwright

    options: dict[str, Any] = {"headless": True, "timeout": timeout_ms}
    if backend.startswith("channel:"):
        options["channel"] = backend.split(":", 1)[1]
    elif backend.startswith("executable:"):
        options["executable_path"] = backend.split(":", 1)[1]
    elif backend != "managed":
        raise ValueError(f"Unknown browser backend: {backend}")
    with sync_playwright() as playwright:
        with playwright.chromium.launch(**options) as browser:
            page = browser.new_page(device_scale_factor=1, java_script_enabled=False)
            page.set_default_timeout(timeout_ms)
            # Results are local data; never fetch remote resources or execute scripts.
            page.route("**/*", lambda route: route.abort())
            page.set_content(html.read_text(encoding="utf-8"), wait_until="load", timeout=timeout_ms)
            table = page.locator("table").first
            table.screenshot(path=str(output), timeout=timeout_ms, animations="disabled")
    _verify_png(output)


def _browser_subprocess(html: Path, output: Path, timeout_ms: int, backend: str) -> None:
    _run([sys.executable, str(_SCRIPT), "--render", str(html), str(output),
          "--timeout-ms", str(timeout_ms), "--backend", backend],
         max(30, timeout_ms / 1000 * 3 + 15))
    _verify_png(output)


def _probe(backend: str) -> None:
    """Check actual Chromium launch and PNG screenshot, not merely importability."""
    with tempfile.TemporaryDirectory(prefix="ddos-table-probe-") as directory:
        html = Path(directory) / "table.html"
        png = Path(directory) / "table.png"
        html.write_text("<table><tr><td>Chromium ready</td></tr></table>", encoding="utf-8")
        _browser_subprocess(html, png, 15000, backend)


def _system_backends():
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "msedge"):
        executable = shutil.which(name)
        if executable:
            yield f"executable:{executable}"
    yield "channel:chrome"
    yield "channel:msedge"


def _prepare_browser() -> str:
    _ensure_package()
    try:
        _probe("managed")
        return "managed"
    except (OSError, RuntimeError, ValueError) as exc:
        _log(f"Managed Chromium needs setup: {exc}")
    try:
        _install_chromium()
        try:
            _probe("managed")
        except (OSError, RuntimeError, ValueError):
            if not sys.platform.startswith("linux"):
                raise
            _install_linux_dependencies()
            _probe("managed")
        return "managed"
    except (OSError, RuntimeError, ValueError) as exc:
        _log(f"Managed Chromium unavailable: {exc}")
    for backend in _system_backends():
        try:
            _probe(backend)
            return backend
        except (OSError, RuntimeError, ValueError):
            continue
    return "native"


def _native_render(html: Path, output: Path) -> None:
    """Browser-free PNG fallback: preserve displayed text and zebra rows.

    Use the Styler HTML (not reconstructed numeric values), including its headers,
    index, formatting, and spans. CSS-specific decoration is not guaranteed.
    """
    from bs4 import BeautifulSoup
    from PIL import Image, ImageDraw, ImageFont

    table = BeautifulSoup(html.read_text(encoding="utf-8"), "html.parser").find("table")
    if table is None:
        raise ValueError("No table in rendered Styler HTML")
    # Expand row/column spans without duplicating the original displayed label.
    occupied: dict[tuple[int, int], str] = {}
    headers: set[int] = set()
    rows = table.find_all("tr")
    for r, row in enumerate(rows):
        if row.find_parent("thead") is not None:
            headers.add(r)
        col = 0
        for cell in row.find_all(["th", "td"], recursive=False):
            while (r, col) in occupied:
                col += 1
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            for rr in range(r, r + rowspan):
                for cc in range(col, col + colspan):
                    occupied[rr, cc] = ""
            occupied[r, col] = cell.get_text(" ", strip=False).replace("\xa0", " ")
            col += colspan
    if not occupied:
        raise ValueError("No cells in rendered Styler HTML")
    height = max(r for r, _ in occupied) + 1
    width = max(c for _, c in occupied) + 1
    # Matplotlib ships DejaVu Sans even on minimal/headless systems.
    import matplotlib
    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 14)
    pad = 8
    line_height = 20
    max_cell = 560

    def wrap(text: str) -> list[str]:
        lines = []
        for paragraph in text.splitlines() or [""]:
            current = ""
            for word in paragraph.split(" "):
                candidate = f"{current} {word}" if current else word
                if font.getlength(candidate) <= max_cell:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                current = ""
                for char in word:
                    if current and font.getlength(current + char) > max_cell:
                        lines.append(current)
                        current = ""
                    current += char
            lines.append(current)
        return lines or [""]

    cells = [[wrap(occupied.get((r, c), "")) for c in range(width)] for r in range(height)]
    widths = [max(48, int(max(font.getlength(line) for row in cells for line in row[c]) + 2 * pad + 1))
              for c in range(width)]
    heights = [max(len(lines) for lines in row) * line_height + 2 * pad for row in cells]
    image = Image.new("RGB", (sum(widths) + 1, sum(heights) + 1), "white")
    draw = ImageDraw.Draw(image)
    y = 0
    for r, row in enumerate(cells):
        x = 0
        fill = "#dbe3ed" if r in headers else ("#f1f1f1" if r % 2 else "white")
        for c, lines in enumerate(row):
            draw.rectangle((x, y, x + widths[c], y + heights[r]), fill=fill, outline="#d0d0d0")
            for i, line in enumerate(lines):
                draw.text((x + pad, y + pad + i * line_height), line, fill="#202020", font=font)
            x += widths[c]
        y += heights[r]
    image.save(output, format="PNG")
    image.close()
    _verify_png(output)


def _probe_native() -> None:
    with tempfile.TemporaryDirectory(prefix="ddos-native-probe-") as directory:
        html = Path(directory) / "table.html"
        png = Path(directory) / "table.png"
        html.write_text("<table><thead><tr><th>Ready</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>", encoding="utf-8")
        _native_render(html, png)


def ensure_table_images() -> str:
    """Set up once per process; serialize browser installs across concurrent jobs."""
    global _backend
    with _LOCK:
        if _backend is not None and _backend[0] == os.getpid():
            return _backend[1]
        backend = "native"
        try:
            directory = browser_directory()
            directory.mkdir(parents=True, exist_ok=True)
            with FileLock(str(directory / ".ddos-install.lock"), timeout=1500):
                backend = _prepare_browser()
        except Exception as exc:
            _log(f"Browser setup unavailable: {exc}")
        # Verify fallback as well, so missing rendering dependencies fail at startup.
        if backend == "native":
            _probe_native()
            _log("WARNING: using verified browser-free PNG rendering; CSS styling may differ")
        else:
            _log(f"Verified Chromium screenshot ({backend})")
        _backend = (os.getpid(), backend)
        return backend


def preflight_table_images() -> None:
    """CLI preflight: leave help/version invocations side-effect free."""
    if not any(arg in {"-h", "--help", "--version"} for arg in sys.argv[1:]):
        ensure_table_images()


def render_table_image(styled_df: Any, output_path: str | Path, *, timeout_ms: int = 30000) -> str:
    """Write a verified PNG atomically. Fall back if Chromium later fails."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    backend = ensure_table_images()
    if not hasattr(styled_df, "to_html"):
        raise TypeError("Expected a pandas DataFrame or Styler")
    # No silent pandas Styler truncation, and preserve the supplied CSS/formatters.
    import pandas as pd
    if isinstance(styled_df, pd.DataFrame):
        styled_df = styled_df.style
    data = styled_df.data
    with pd.option_context("styler.render.max_elements", max(1, int(data.size))):
        markup = styled_df.to_html(max_rows=len(data), max_columns=len(data.columns))
    markup = ("<!doctype html><html><head><meta charset='utf-8'>"
              "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:\">"
              "<style>body{margin:0;background:white;font:14px Arial,sans-serif}"
              "table{border-collapse:collapse;background:white;width:max-content}"
              "th,td{padding:6px 9px;text-align:right;white-space:pre-wrap;overflow-wrap:anywhere;max-width:650px}"
              "thead th{background:#e5e7eb;border-bottom:1px solid #aaa}</style></head><body>"
              + markup + "</body></html>")
    with tempfile.TemporaryDirectory(prefix=".ddos-table-", dir=target.parent) as directory:
        html = Path(directory) / "table.html"
        png = Path(directory) / "table.png"
        html.write_text(markup, encoding="utf-8")
        if backend != "native":
            try:
                _browser_subprocess(html, png, timeout_ms, backend)
            except (OSError, RuntimeError, ValueError) as exc:
                _log(f"WARNING: Chromium export failed for {target.name}; using browser-free renderer: {exc}")
                backend = "native"
        if backend == "native":
            _native_render(html, png)
        _verify_png(png)
        os.replace(png, target)
    _log(f"Saved {target} ({backend})")
    return backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and verify DDoS-Detector table-image rendering")
    parser.add_argument("--render", nargs=2, metavar=("HTML", "PNG"), help=argparse.SUPPRESS)
    parser.add_argument("--timeout-ms", type=int, default=30000, help=argparse.SUPPRESS)
    parser.add_argument("--backend", default="managed", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.render:
        _browser_render(Path(args.render[0]), Path(args.render[1]), args.timeout_ms, args.backend)
    else:
        ensure_table_images()


if __name__ == "__main__":
    main()
