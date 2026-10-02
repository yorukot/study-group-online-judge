import asyncio
import logging
import os
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from judge.agent_channel import AgentChannel
from judge.database import migrate_database, prune_stale_sub_judges
from judge.leaderboard import LeaderboardService
from judge.routers import agents, health, leaderboard, submissions


@asynccontextmanager
async def lifespan(judge_app: FastAPI) -> AsyncIterator[None]:
    load_dotenv()

    api_token = os.environ.get("JUDGE_API_TOKEN")
    if not api_token:
        raise RuntimeError("JUDGE_API_TOKEN must be configured")

    database_path = Path(os.environ.get("JUDGE_DATABASE_PATH", "data/judge.db"))
    migrate_database(database_path)
    judge_app.state.api_token = api_token
    judge_app.state.agent_token = os.environ.get("JUDGE_AGENT_TOKEN") or None
    judge_app.state.judge_revision = os.environ.get("JUDGE_REVISION") or None
    judge_app.state.agent_channel = AgentChannel()
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

    async def expire_sub_judges():
        while True:
            try:
                expired = await asyncio.to_thread(prune_stale_sub_judges, database_path)
                for judge_id in expired:
                    logging.getLogger("uvicorn.error.judge").warning(
                        "Deregistered sub-judge %s after five minutes without a heartbeat",
                        judge_id,
                    )
            except OSError, sqlite3.Error:
                logging.getLogger("uvicorn.error.judge").exception(
                    "Unable to expire stale sub-judges"
                )
            await asyncio.sleep(15)

    agent_reaper = asyncio.create_task(expire_sub_judges())
    try:
        yield
    finally:
        poller.cancel()
        agent_reaper.cancel()
        with suppress(asyncio.CancelledError):
            await poller
        with suppress(asyncio.CancelledError):
            await agent_reaper


app = FastAPI(lifespan=lifespan)
app.include_router(agents.router)
app.include_router(health.router)
app.include_router(submissions.router)

app.include_router(leaderboard.router)

# Register API routes before the optional frontend catch-all.
ui_dist = Path(
    os.getenv("JUDGE_UI_DIST", str(Path(__file__).resolve().parents[2] / "ui" / "dist"))
)
if ui_dist.is_dir():
    app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")
