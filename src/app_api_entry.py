from __future__ import annotations

from collections import deque
import logging
import os
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .climate_pipeline.feedback import FeedbackStore
from .climate_pipeline.inference import (
    CropSuitabilityInferenceService,
    InferenceValidationError,
)
from .climate_pipeline.llm_guide import (
    GroqGuideClient,
    LlmGuideNotConfiguredError,
    LlmGuideUpstreamError,
)
from .env_loader import load_project_env
from .climate_pipeline.utils import ensure_parent_dir

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
load_project_env(root_dir=ROOT_DIR)

DEFAULT_ARTIFACT_CANDIDATES = [
    ROOT_DIR / "artifacts" / "data_new_training",
    ROOT_DIR / "artifacts" / "training",
    ROOT_DIR / "artifacts" / "demo_training",
]
DEFAULT_FEEDBACK_DIR = ROOT_DIR / "artifacts" / "feedback_store"
SUPPORTED_LANGUAGES = [
    "English",
    "Hindi",
    "Marathi",
    "Kannada",
    "Telugu",
    "Tamil",
]
DEFAULT_FEEDBACK_RATE_LIMIT_COUNT = 12
DEFAULT_FEEDBACK_RATE_LIMIT_WINDOW_SECONDS = 900
DEFAULT_LLM_GUIDE_RATE_LIMIT_COUNT = 8
DEFAULT_LLM_GUIDE_RATE_LIMIT_WINDOW_SECONDS = 600


class PredictRequest(BaseModel):
    region: str = Field(..., min_length=1)
    state: str | None = None
    target_time: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    irrigation_index: float | None = None
    rotation_score: float | None = None
    fertility_class: str | None = None
    geo_confidence: float | None = None
    data_confidence: float | None = None
    farmer_explanation: str | None = None


class SimulateRequest(PredictRequest):
    scenario_names: list[str] | None = None


class ScenarioExplainRequest(PredictRequest):
    scenario_name: str = Field(..., min_length=1)


class FeedbackRequest(BaseModel):
    request_id: str | None = None
    region: str = Field(..., min_length=1)
    state: str | None = None
    preferred_language: str | None = None
    selected_crop: str | None = None
    actual_crop: str | None = None
    outcome_label: str | None = None
    helpfulness_rating: int | None = None
    clarity_rating: int | None = None
    consent_for_training: bool = False
    comment: str | None = None
    input_snapshot: dict[str, Any] | None = None
    prediction_snapshot: dict[str, Any] | None = None


class LlmGuideRequest(BaseModel):
    prediction: dict[str, Any] = Field(default_factory=dict)
    input_snapshot: dict[str, Any] | None = None
    preferred_language: str | None = None
    user_question: str | None = Field(default=None, max_length=320)
    region: str | None = None
    state: str | None = None


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._lock = Lock()
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        current_time = time()
        cutoff = current_time - self.window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(current_time)
            return True


class ApiMetricsTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self.total_requests = 0
        self.error_requests = 0
        self.prediction_requests = 0
        self.simulation_requests = 0
        self.feedback_requests = 0
        self.context_requests = 0
        self.catalog_requests = 0
        self.health_requests = 0
        self.total_latency_ms = 0.0

    def record(self, path: str, latency_ms: float, status_code: int) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += float(latency_ms)
            if status_code >= 400:
                self.error_requests += 1
            if path == "/predict":
                self.prediction_requests += 1
            elif path == "/simulate":
                self.simulation_requests += 1
            elif path == "/feedback":
                self.feedback_requests += 1
            elif path == "/context":
                self.context_requests += 1
            elif path == "/catalog":
                self.catalog_requests += 1
            elif path in {"/health", "/sanity", "/metrics"}:
                self.health_requests += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average_latency = (
                self.total_latency_ms / self.total_requests if self.total_requests else 0.0
            )
            error_rate = self.error_requests / self.total_requests if self.total_requests else 0.0
            return {
                "total_requests": self.total_requests,
                "error_requests": self.error_requests,
                "prediction_requests": self.prediction_requests,
                "simulation_requests": self.simulation_requests,
                "feedback_requests": self.feedback_requests,
                "context_requests": self.context_requests,
                "catalog_requests": self.catalog_requests,
                "health_requests": self.health_requests,
                "avg_latency_ms": round(float(average_latency), 2),
                "error_rate": round(float(error_rate), 4),
            }


def resolve_default_artifact_dir() -> Path | None:
    env_path = os.getenv("MODEL_ARTIFACT_DIR")
    if env_path:
        return Path(env_path).resolve()
    for candidate in DEFAULT_ARTIFACT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def configure_api_logger(root_dir: Path) -> logging.Logger:
    logger = logging.getLogger("climate_pipeline.api")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = root_dir / "logs" / "api.log"
    ensure_parent_dir(log_path)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


@lru_cache(maxsize=1)
def get_service() -> CropSuitabilityInferenceService:
    artifact_dir = resolve_default_artifact_dir()
    if artifact_dir is None:
        return CropSuitabilityInferenceService.from_artifact_dir(root_dir=ROOT_DIR)
    return CropSuitabilityInferenceService.from_artifact_dir(
        artifact_dir=artifact_dir,
        root_dir=ROOT_DIR,
    )


@lru_cache(maxsize=1)
def get_feedback_store() -> FeedbackStore:
    signing_secret = os.getenv("FEEDBACK_SIGNING_SECRET")
    return FeedbackStore(storage_dir=DEFAULT_FEEDBACK_DIR, signing_secret=signing_secret)


@lru_cache(maxsize=1)
def get_llm_guide_client() -> GroqGuideClient:
    return GroqGuideClient.from_env()


def build_guidance_scope() -> dict[str, Any]:
    return {
        "target_type": "historical_district_crop_pattern_guidance",
        "live_weather_enabled": False,
        "year_specific_context": False,
        "current_context_source": "historical_same_month_climatology",
        "recommended_use": "Use this as a shortlist and validate with local agronomy advice before planting.",
        "product_note": "The model ranks crops from district-season training patterns plus field inputs. It is not a guaranteed agronomic outcome model.",
    }


def choose_farmer_action(prediction: dict[str, Any]) -> str:
    recommendations = prediction.get("recommendations", [])
    if not recommendations:
        return "Collect a few more field details before deciding."

    top_crop = str(recommendations[0].get("crop", "the top crop")).strip()
    confidence = float(prediction.get("confidence", 0.0) or 0.0)
    warning_count = len(prediction.get("warnings", []))
    if confidence >= 0.8 and warning_count == 0:
        return f"Use {top_crop} as a strong shortlist, then cross-check irrigation, seed access, and local field advice."
    if confidence >= 0.6:
        return f"Shortlist {top_crop}, then validate sowing window and management needs before committing."
    return f"Treat {top_crop} as a starting point only and verify with local field guidance before planting."


def build_farmer_message(prediction: dict[str, Any]) -> str:
    recommendations = prediction.get("recommendations", [])
    if not recommendations:
        return "The app could not rank crops from the current inputs."
    top_crop = str(recommendations[0].get("crop", "Unknown")).strip().title()
    confidence = round(float(prediction.get("confidence", 0.0) or 0.0) * 100.0, 1)
    explanation = str(prediction.get("explanation", "")).strip()
    if explanation:
        return f"Pattern-based top match is {top_crop} at {confidence}% confidence. {explanation}"
    return f"Pattern-based top match is {top_crop} at {confidence}% confidence from the current district-season profile."


def model_dump(instance: BaseModel) -> dict[str, Any]:
    if hasattr(instance, "model_dump"):
        return instance.model_dump(exclude_none=True)
    return instance.dict(exclude_none=True)


def get_feedback_rate_limit_count() -> int:
    try:
        return max(1, int(os.getenv("FEEDBACK_RATE_LIMIT_COUNT", DEFAULT_FEEDBACK_RATE_LIMIT_COUNT)))
    except ValueError:
        return DEFAULT_FEEDBACK_RATE_LIMIT_COUNT


def get_feedback_rate_limit_window_seconds() -> int:
    try:
        return max(1, int(os.getenv("FEEDBACK_RATE_LIMIT_WINDOW_SECONDS", DEFAULT_FEEDBACK_RATE_LIMIT_WINDOW_SECONDS)))
    except ValueError:
        return DEFAULT_FEEDBACK_RATE_LIMIT_WINDOW_SECONDS


def get_llm_guide_rate_limit_count() -> int:
    try:
        return max(1, int(os.getenv("LLM_GUIDE_RATE_LIMIT_COUNT", DEFAULT_LLM_GUIDE_RATE_LIMIT_COUNT)))
    except ValueError:
        return DEFAULT_LLM_GUIDE_RATE_LIMIT_COUNT


def get_llm_guide_rate_limit_window_seconds() -> int:
    try:
        return max(
            1,
            int(
                os.getenv(
                    "LLM_GUIDE_RATE_LIMIT_WINDOW_SECONDS",
                    DEFAULT_LLM_GUIDE_RATE_LIMIT_WINDOW_SECONDS,
                )
            ),
        )
    except ValueError:
        return DEFAULT_LLM_GUIDE_RATE_LIMIT_WINDOW_SECONDS


def resolve_service_for_app(app: FastAPI) -> CropSuitabilityInferenceService:
    service = getattr(app.state, "service", None)
    if service is not None:
        return service

    lock = getattr(app.state, "service_lock", None)
    if lock is None:
        lock = Lock()
        app.state.service_lock = lock

    with lock:
        service = getattr(app.state, "service", None)
        if service is None:
            service = get_service()
            app.state.service = service
        return service


def resolve_feedback_store_for_app(app: FastAPI) -> FeedbackStore:
    feedback_store = getattr(app.state, "feedback_store", None)
    if feedback_store is not None:
        return feedback_store

    lock = getattr(app.state, "feedback_store_lock", None)
    if lock is None:
        lock = Lock()
        app.state.feedback_store_lock = lock

    with lock:
        feedback_store = getattr(app.state, "feedback_store", None)
        if feedback_store is None:
            feedback_store = get_feedback_store()
            app.state.feedback_store = feedback_store
        return feedback_store


def resolve_llm_guide_client_for_app(app: FastAPI) -> GroqGuideClient:
    client = getattr(app.state, "llm_guide_client", None)
    if client is not None:
        return client

    lock = getattr(app.state, "llm_guide_client_lock", None)
    if lock is None:
        lock = Lock()
        app.state.llm_guide_client_lock = lock

    with lock:
        client = getattr(app.state, "llm_guide_client", None)
        if client is None:
            client = get_llm_guide_client()
            app.state.llm_guide_client = client
        return client


def require_service(app: FastAPI) -> CropSuitabilityInferenceService:
    try:
        return resolve_service_for_app(app)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model service unavailable: {exc}") from exc


def create_app(
    service: CropSuitabilityInferenceService | None = None,
    feedback_store: FeedbackStore | None = None,
    llm_guide_client: GroqGuideClient | None = None,
    feedback_rate_limit_count: int | None = None,
    feedback_rate_limit_window_seconds: int | None = None,
    llm_guide_rate_limit_count: int | None = None,
    llm_guide_rate_limit_window_seconds: int | None = None,
) -> FastAPI:
    api_logger = configure_api_logger(ROOT_DIR)
    metrics = ApiMetricsTracker()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        try:
            resolve_service_for_app(app_instance).warmup()
        except Exception as exc:
            api_logger.warning("inference_warmup_failed error=%s", exc)
        yield

    app = FastAPI(
        title="Climate Crop Advisory API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.service_lock = Lock()
    app.state.feedback_store = feedback_store
    app.state.feedback_store_lock = Lock()
    app.state.llm_guide_client = llm_guide_client
    app.state.llm_guide_client_lock = Lock()
    app.state.metrics = metrics
    app.state.api_logger = api_logger
    app.state.guidance_scope = build_guidance_scope()
    app.state.feedback_rate_limiter = SlidingWindowRateLimiter(
        feedback_rate_limit_count or get_feedback_rate_limit_count(),
        feedback_rate_limit_window_seconds or get_feedback_rate_limit_window_seconds(),
    )
    app.state.llm_guide_rate_limiter = SlidingWindowRateLimiter(
        llm_guide_rate_limit_count or get_llm_guide_rate_limit_count(),
        llm_guide_rate_limit_window_seconds or get_llm_guide_rate_limit_window_seconds(),
    )

    @app.middleware("http")
    async def track_requests(request: Request, call_next):
        request_id = uuid4().hex
        request.state.request_id = request_id
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed_ms = (perf_counter() - started) * 1000.0
            metrics.record(request.url.path, elapsed_ms, status_code)
            api_logger.info(
                "request_complete request_id=%s path=%s status=%s latency_ms=%.2f",
                request_id,
                request.url.path,
                status_code,
                elapsed_ms,
            )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        try:
            service_instance = resolve_service_for_app(request.app)
        except Exception as exc:
            llm_support = resolve_llm_guide_client_for_app(request.app).support_metadata()
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "request_id": request.state.request_id,
                    "detail": f"Model service unavailable: {exc}",
                    "artifact_dir": str(resolve_default_artifact_dir() or ""),
                    "llm_support": llm_support,
                },
            )
        llm_support = resolve_llm_guide_client_for_app(request.app).support_metadata()
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            "model_version": service_instance.model_version,
            "mode": service_instance.mode,
            "sanity_mode": service_instance.sanity_mode,
            "artifact_dir": str(service_instance.artifact_dir),
            "llm_support": llm_support,
        }

    @app.get("/sanity")
    def sanity(request: Request) -> dict[str, Any]:
        service_instance = require_service(request.app)
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            "sanity_checks": service_instance.get_sanity_summary(),
        }

    @app.get("/metrics")
    def metrics_view(request: Request) -> dict[str, Any]:
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            **metrics.snapshot(),
        }

    @app.get("/catalog")
    def catalog(request: Request) -> dict[str, Any]:
        service_instance = require_service(request.app)
        feedback_store_instance = resolve_feedback_store_for_app(request.app)
        llm_support = resolve_llm_guide_client_for_app(request.app).support_metadata()
        catalog_payload = service_instance.get_catalog()
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            **catalog_payload,
            "feedback": feedback_store_instance.get_storage_info(),
            "llm_support": llm_support,
            "language_support": {
                "current": ["English"],
                "planned": SUPPORTED_LANGUAGES[1:],
                "human_review_required_for_training": True,
            },
            "guidance_scope": request.app.state.guidance_scope,
        }

    @app.get("/context")
    def context(
        request: Request,
        region: str | None = None,
        state: str | None = None,
        target_time: str | None = None,
    ) -> dict[str, Any]:
        service_instance = require_service(request.app)
        localized = service_instance.get_localized_context(
            region=region,
            state=state,
            target_time=target_time,
        )
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            "guidance_scope": request.app.state.guidance_scope,
            **localized,
        }

    @app.post("/predict")
    def predict(request: Request, payload: PredictRequest) -> dict[str, Any]:
        if not payload.features:
            raise HTTPException(status_code=422, detail="Provide at least one feature value.")
        started = perf_counter()
        service_instance = require_service(request.app)
        try:
            prediction = service_instance.predict(model_dump(payload))
        except InferenceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            "prediction_time_ms": elapsed_ms,
            "farmer_action": choose_farmer_action(prediction),
            "farmer_message": build_farmer_message(prediction),
            "guidance_scope": request.app.state.guidance_scope,
            **prediction,
        }

    @app.post("/simulate")
    def simulate(request: Request, payload: SimulateRequest) -> dict[str, Any]:
        if not payload.features:
            raise HTTPException(status_code=422, detail="Provide at least one feature value.")
        started = perf_counter()
        service_instance = require_service(request.app)
        try:
            simulation = service_instance.simulate_scenarios(
                model_dump(payload),
                scenario_names=payload.scenario_names,
            )
        except InferenceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            "prediction_time_ms": elapsed_ms,
            "guidance_scope": request.app.state.guidance_scope,
            **simulation,
        }

    @app.post("/scenario-explain")
    def scenario_explain(request: Request, payload: ScenarioExplainRequest) -> dict[str, Any]:
        if not payload.features:
            raise HTTPException(status_code=422, detail="Provide at least one feature value.")
        started = perf_counter()
        service_instance = require_service(request.app)
        try:
            explanation_payload = service_instance.explain_scenario(
                model_dump(payload),
                scenario_name=payload.scenario_name,
            )
        except InferenceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            "latency_ms": elapsed_ms,
            "guidance_scope": request.app.state.guidance_scope,
            **explanation_payload,
        }

    @app.post("/feedback")
    def feedback(request: Request, payload: FeedbackRequest) -> dict[str, Any]:
        client_host = request.client.host if request.client and request.client.host else "unknown"
        rate_limit_key = f"feedback:{client_host}"
        if not request.app.state.feedback_rate_limiter.allow(rate_limit_key):
            raise HTTPException(
                status_code=429,
                detail="Feedback rate limit reached for this client. Please wait before submitting again.",
            )
        feedback_store_instance = resolve_feedback_store_for_app(request.app)
        submitted = feedback_store_instance.submit(model_dump(payload))
        consent_for_training = bool(payload.consent_for_training)
        return {
            "status": "stored",
            "request_id": request.state.request_id,
            "linked_request_id": payload.request_id,
            "stored_at": submitted["submitted_at"],
            "review_status": "pending_human_review" if consent_for_training else "not_requested",
            "eligible_for_training": False,
            **submitted,
        }

    @app.post("/llm-guide")
    def llm_guide(request: Request, payload: LlmGuideRequest) -> dict[str, Any]:
        if not payload.prediction or not payload.prediction.get("recommendations"):
            raise HTTPException(
                status_code=422,
                detail="Provide a prediction payload with at least one recommendation.",
            )

        client_host = request.client.host if request.client and request.client.host else "unknown"
        rate_limit_key = f"llm-guide:{client_host}"
        if not request.app.state.llm_guide_rate_limiter.allow(rate_limit_key):
            raise HTTPException(
                status_code=429,
                detail="AI guide rate limit reached for this client. Please wait before asking again.",
            )

        llm_client = resolve_llm_guide_client_for_app(request.app)
        if hasattr(llm_client, "is_enabled") and not bool(llm_client.is_enabled()):
            raise HTTPException(
                status_code=503,
                detail="Groq AI guide is not configured. Set GROQ_API_KEY to enable it.",
            )
        started = perf_counter()
        try:
            result = llm_client.generate_answer(
                prediction=payload.prediction,
                input_snapshot=payload.input_snapshot,
                preferred_language=payload.preferred_language,
                user_question=payload.user_question,
                region=payload.region,
                state=payload.state,
            )
        except LlmGuideNotConfiguredError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LlmGuideUpstreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        elapsed_ms = round((perf_counter() - started) * 1000.0, 2)
        return {
            "status": "ok",
            "request_id": request.state.request_id,
            "latency_ms": elapsed_ms,
            **result,
        }

    return app


app = create_app()
