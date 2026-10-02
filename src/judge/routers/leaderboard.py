from fastapi import APIRouter, Request

router = APIRouter(tags=["Leaderboard"])


@router.get("/api/leaderboard")
def leaderboard(request: Request):
    """Public, sanitized standings. Never expose judge or W&B credentials."""
    return request.app.state.leaderboard.snapshot
