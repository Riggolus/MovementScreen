"""FastAPI web server for MovementScreen."""
from __future__ import annotations

import asyncio
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from movementscreen.auth import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from movementscreen.capture.video_pipeline import (
    PipelineConfig,
    collect_frames,
    iter_frames_from_file,
)
from movementscreen.config import settings
from movementscreen.database import (
    close_pool,
    create_user,
    get_assessment_detail,
    get_pool,
    get_progress,
    get_user_by_email,
    get_user_by_id,
    init_pool,
    list_assessments,
    save_assessment,
)
from movementscreen.pose.estimator import PoseEstimator
from movementscreen.screens.lunge import LungeScreen
from movementscreen.screens.overhead_reach import OverheadReachScreen
from movementscreen.screens.squat import SquatScreen


# ── Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool(settings.database_url)
    yield
    await close_pool()


app = FastAPI(title="MovementScreen API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ── Auth dependencies ─────────────────────────────────────
_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


async def get_current_user(token: Optional[str] = Depends(_oauth2)) -> dict:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")
    user_id = decode_access_token(token)
    user = await get_user_by_id(get_pool(), user_id)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return user


async def get_optional_user(token: Optional[str] = Depends(_oauth2)) -> Optional[dict]:
    if not token:
        return None
    try:
        user_id = decode_access_token(token)
        return await get_user_by_id(get_pool(), user_id)
    except HTTPException:
        return None


# ── Request bodies ────────────────────────────────────────
class RegisterBody(BaseModel):
    email: str
    name: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


class RefreshBody(BaseModel):
    refresh_token: str


# ── Static ────────────────────────────────────────────────
@app.get("/")
async def index():
    return FileResponse(str(_STATIC / "index.html"))


# ── Auth routes ───────────────────────────────────────────
@app.post("/auth/register")
async def register(body: RegisterBody):
    pool = get_pool()
    if await get_user_by_email(pool, body.email):
        raise HTTPException(400, detail="Email already registered.")
    user = await create_user(pool, body.email, body.name, hash_password(body.password))
    uid = str(user["id"])
    return {
        "user": {"id": uid, "email": user["email"], "name": user["name"]},
        "access_token": create_access_token(uid),
        "refresh_token": create_refresh_token(uid),
    }


@app.post("/auth/login")
async def login(body: LoginBody):
    pool = get_pool()
    user = await get_user_by_email(pool, body.email)
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, detail="Invalid email or password.")
    uid = str(user["id"])
    return {
        "user": {"id": uid, "email": user["email"], "name": user["name"]},
        "access_token": create_access_token(uid),
        "refresh_token": create_refresh_token(uid),
    }


@app.post("/auth/refresh")
async def refresh_token(body: RefreshBody):
    user_id = decode_refresh_token(body.refresh_token)
    user = await get_user_by_id(get_pool(), user_id)
    if not user:
        raise HTTPException(401, detail="User not found.")
    return {"access_token": create_access_token(user_id)}


@app.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return {"id": str(user["id"]), "email": user["email"], "name": user["name"]}


# ── Assessment routes ─────────────────────────────────────
@app.get("/assessments")
async def list_user_assessments(
    page: int = 1,
    user: dict = Depends(get_current_user),
):
    page_size = 20
    rows = await list_assessments(
        get_pool(), str(user["id"]),
        limit=page_size, offset=(page - 1) * page_size,
    )
    return {"assessments": [_serialise(r) for r in rows]}


@app.get("/assessments/{assessment_id}")
async def get_assessment(
    assessment_id: str,
    user: dict = Depends(get_current_user),
):
    detail = await get_assessment_detail(get_pool(), str(user["id"]), assessment_id)
    if not detail:
        raise HTTPException(404, detail="Assessment not found.")
    return _serialise_detail(detail)


@app.get("/progress")
async def progress(user: dict = Depends(get_current_user)):
    return await get_progress(get_pool(), str(user["id"]))


# ── Analysis route ────────────────────────────────────────
def _result_to_json(result, camera_angle: str = "anterior") -> dict:
    return {
        "screen_name": result.screen_name,
        "frame_count": result.frame_count,
        "camera_angle": camera_angle,
        "worst_severity": result.compensation_report.worst_severity.value,
        "has_findings": result.compensation_report.has_findings,
        "findings": [
            {
                "name": f.name,
                "severity": f.severity.value,
                "description": f.description,
                "metric_value": round(f.metric_value, 1) if f.metric_value is not None else None,
                "metric_label": f.metric_label,
            }
            for f in result.compensation_report.findings
        ],
        "stats": [
            {
                "name": s.name,
                "min": round(s.min, 1) if s.min is not None else None,
                "max": round(s.max, 1) if s.max is not None else None,
                "mean": round(s.mean, 1) if s.mean is not None else None,
            }
            for s in result.stats.values()
            if s.mean is not None
        ],
    }


@app.post("/analyse")
async def analyse(
    video: UploadFile = File(...),
    screen: str = Form("squat"),
    lead_side: str = Form("left"),
    camera_angle: str = Form("anterior"),
    model_complexity: int = Form(1),
    user: Optional[dict] = Depends(get_optional_user),
):
    if screen not in ("squat", "lunge", "overhead"):
        raise HTTPException(400, detail="Invalid screen type.")

    suffix = Path(video.filename or "recording.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = Path(tmp.name)

    try:
        if screen == "squat":
            movement_screen = SquatScreen()
        elif screen == "lunge":
            movement_screen = LungeScreen(lead_side=lead_side)
        else:
            movement_screen = OverheadReachScreen()

        config = PipelineConfig(skip_frames=1, draw_landmarks=False, show_preview=False)

        def _run():
            with PoseEstimator(model_complexity=model_complexity) as estimator:
                frames = collect_frames(
                    iter_frames_from_file(tmp_path, estimator, config=config)
                )
            return None if not frames else movement_screen.run(frames)

        result = await asyncio.get_event_loop().run_in_executor(None, _run)

        if result is None:
            raise HTTPException(
                422,
                detail="No pose detected. Make sure your full body is visible and well-lit.",
            )

        result_json = _result_to_json(result, camera_angle)

        if user:
            assessment_id = await save_assessment(
                get_pool(), str(user["id"]), result_json, screen, camera_angle, lead_side,
            )
            result_json["assessment_id"] = assessment_id
            result_json["saved"] = True
        else:
            result_json["saved"] = False

        return JSONResponse(result_json)

    finally:
        tmp_path.unlink(missing_ok=True)


# ── Serialisation helpers ─────────────────────────────────
def _serialise(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "screen_type": row["screen_type"],
        "camera_angle": row["camera_angle"],
        "lead_side": row["lead_side"],
        "frame_count": row["frame_count"],
        "worst_severity": row["worst_severity"],
        "has_findings": row["has_findings"],
        "recorded_at": row["recorded_at"].isoformat(),
    }


def _serialise_detail(row: dict) -> dict:
    base = _serialise(row)
    base["findings"] = [
        {
            "name": f["name"],
            "severity": f["severity"],
            "description": f["description"],
            "metric_value": float(f["metric_value"]) if f["metric_value"] is not None else None,
            "metric_label": f["metric_label"],
        }
        for f in row.get("findings", [])
    ]
    base["stats"] = [
        {
            "name": s["name"],
            "min": float(s["min_value"]) if s["min_value"] is not None else None,
            "max": float(s["max_value"]) if s["max_value"] is not None else None,
            "mean": float(s["mean_value"]) if s["mean_value"] is not None else None,
        }
        for s in row.get("stats", [])
    ]
    return base
