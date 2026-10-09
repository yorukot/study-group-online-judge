"""Screenshot the website's actual leaderboard for native Slack uploads."""

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from playwright.sync_api import expect, sync_playwright


def render_standings(board, entries, directory: Path) -> str:
    """Render the built React UI against the notification's frozen snapshot."""
    dist = Path(
        os.getenv("JUDGE_UI_DIST", str(Path(__file__).resolve().parents[2] / "ui/dist"))
    ).resolve()
    if not (dist / "index.html").is_file():
        raise RuntimeError("Build the leaderboard UI with npm run build --prefix ui")
    snapshot = {
        "source": "wandb",
        "updated_at": None,
        "error": None,
        "labs": [{**board, "entries": entries}],
    }

    def serve(route):
        path = unquote(urlparse(route.request.url).path)
        if path == "/api/leaderboard":
            route.fulfill(content_type="application/json", body=json.dumps(snapshot))
            return
        asset = (dist / (path.lstrip("/") or "index.html")).resolve()
        if not asset.is_relative_to(dist) or not asset.is_file():
            route.fulfill(status=404)
            return
        route.fulfill(path=str(asset))

    with (
        sync_playwright() as playwright,
        playwright.chromium.launch(
            executable_path=os.getenv("JUDGE_CHROMIUM_EXECUTABLE"),
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        ) as browser,
    ):
        page = browser.new_page(
            viewport={"width": 1280, "height": 3000},
            device_scale_factor=2,
            locale="en-US",
            timezone_id="Asia/Taipei",
        )
        page.route("http://leaderboard.invalid/**", serve)
        page.goto("http://leaderboard.invalid/", wait_until="networkidle")
        expect(page.locator(".board tbody tr")).to_have_count(len(entries))
        page.evaluate("document.fonts.ready")
        page.locator(".board").scroll_into_view_if_needed()
        page.wait_for_function(
            "Array.from(document.querySelectorAll('.board img')).every(img => img.complete)",
            timeout=15000,
        )
        content = page.locator(".board").screenshot(animations="disabled")
    filename = f"{hashlib.sha256(content).hexdigest()}.png"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists():
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    return filename
