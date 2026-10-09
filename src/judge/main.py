import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from judge.database import migrate_database
from judge.execution_logging import configure_logging
from judge.leaderboard import LeaderboardService
from judge.routers import health, leaderboard, submissions
from judge.ssh import RemoteConfig


@asynccontextmanager
async def lifespan(judge_app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()
    configure_logging()

    api_token = os.environ.get("JUDGE_API_TOKEN")
    if not api_token:
        raise RuntimeError("JUDGE_API_TOKEN must be configured")

    database_path = Path(os.environ.get("JUDGE_DATABASE_PATH", "data/judge.db"))
    migrate_database(database_path)
    judge_app.state.api_token = api_token
    judge_app.state.remote_config = RemoteConfig.from_environment()
    judge_app.state.database_path = database_path
    judge_app.state.leaderboard = LeaderboardService(database_path)
    interval = max(10, float(os.getenv("JUDGE_LEADERBOARD_REFRESH_SECONDS", "60")))

    async def refresh_leaderboard():
        while True:
            try:
                await asyncio.to_thread(judge_app.state.leaderboard.refresh)
            except OSError:
                logging.getLogger("uvicorn.error.judge").exception(
                    "Unable to persist leaderboard snapshot"
                )
            await asyncio.sleep(interval)

    poller = asyncio.create_task(refresh_leaderboard())

    try:
        yield
    finally:
        poller.cancel()
        with suppress(asyncio.CancelledError):
            await poller


app = FastAPI(lifespan=lifespan)
app.include_router(health.router)
app.include_router(submissions.router)

app.include_router(leaderboard.router)

# Register API routes before the optional frontend catch-all.
ui_dist = Path(
    os.getenv("JUDGE_UI_DIST", str(Path(__file__).resolve().parents[2] / "ui" / "dist"))
)
if ui_dist.is_dir():
    app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")
