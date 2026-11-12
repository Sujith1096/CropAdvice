from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import joblib
import os
import json
import io
import logging
import re
import threading
import numpy as np
from PIL import Image, UnidentifiedImageError
import auth, models, schemas, database
from sqlalchemy.orm import Session
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ml",
    tags=["Machine Learning"]
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DEFAULT_FALLBACK_TREATMENTS = {
    "organic": ["Remove visibly infected plant debris and improve air circulation around foliage."],
    "chemical": ["Consult local agricultural extension guidelines before applying targeted fungicides or bactericides."],
    "fertilizer": ["Maintain balanced NPK nutrition guided by a soil test."]
}


class _GeminiError(Exception):
    """Internal error carrying a categorized Gemini failure kind."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _load_crop_model():
    path = os.path.join(BASE_DIR, "crop_model.joblib")
    try:
        return joblib.load(path)
    except Exception as exc:
        logger.warning("Failed to load crop_model.joblib: %s", exc)
        return None


def _load_json_file(filename: str, default):
    path = os.path.join(BASE_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", filename, exc)
        return default


def _load_disease_interpreter():
    """Load the TFLite/LiteRT disease detection interpreter once at startup."""
    model_path = os.path.join(BASE_DIR, "disease_model.tflite")
    if not os.path.exists(model_path):
        logger.warning("disease_model.tflite not found at %s", model_path)
        return None, None, None

    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError:
                import tflite_runtime.interpreter as tflite
                Interpreter = tflite.Interpreter

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, interpreter.get_input_details(), interpreter.get_output_details()
    except Exception as exc:
        logger.warning("Failed to load disease_model.tflite: %s", exc)
        return None, None, None


crop_model = _load_crop_model()
class_names = _load_json_file("class_names.json", [])
disease_treatments = _load_json_file("disease_treatments.json", {})
_disease_interpreter, _input_details, _output_details = _load_disease_interpreter()
_interpreter_lock = threading.Lock()


def _get_gemini_candidates() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    fallbacks_env = os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-flash-lite-latest,gemini-3.8-flash,gemini-flash-latest"
    )
    candidates = [primary] + [m.strip() for m in fallbacks_env.split(",")]
    seen = set()
    ordered = []
    for name in candidates:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _generate_with_gemini(prompt: str, json_mode: bool = False) -> str:
    """Call Gemini via google-genai with configurable model fallback and request timeout."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise _GeminiError("missing_key", "GEMINI_API_KEY is not configured.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=20000),
    )
    config = (
        types.GenerateContentConfig(response_mime_type="application/json")
        if json_mode
        else None
    )

    last_error = None
    all_rate_limited = True
    for model_name in _get_gemini_candidates():
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            if response and getattr(response, "text", None):
                return response.text
            raise RuntimeError(f"Empty response from {model_name}")
        except Exception as exc:
            last_error = exc
            err_text = str(exc).lower()
            is_quota = "429" in err_text or "quota" in err_text or "resource_exhausted" in err_text or "resourceexhausted" in err_text
            if not is_quota:
                all_rate_limited = False
            logger.warning("Gemini model '%s' failed: %s", model_name, exc)

    kind = "rate_limited" if all_rate_limited else "api_error"
    raise _GeminiError(kind, str(last_error))


def _parse_prediction_label(prediction: str) -> tuple[str, str]:
    text = str(prediction).strip()
    if "___" in text:
        crop_part, disease_part = text.split("___", 1)
        return crop_part.replace("_", " ").strip(), disease_part.replace("_", " ").strip()
    if " " in text:
        parts = text.split(" ")
        return parts[0].strip(), " ".join(parts[1:]).strip()
    return "Unknown", text


def _normalize_treatments(raw: dict) -> dict:
    return {
        "organic": list(raw.get("organic", [])) or DEFAULT_FALLBACK_TREATMENTS["organic"],
        "chemical": list(raw.get("chemical", [])) or DEFAULT_FALLBACK_TREATMENTS["chemical"],
        "fertilizer": list(raw.get("fertilizer", [])) or DEFAULT_FALLBACK_TREATMENTS["fertilizer"],
    }


def _resolve_treatments(prediction: str) -> dict:
    """Resolve treatments from the local 38-class table, or Gemini if explicitly enabled."""
    local_entry = disease_treatments.get(prediction, DEFAULT_FALLBACK_TREATMENTS)
    if os.getenv("USE_GEMINI_TREATMENTS", "false").lower() != "true":
        return _normalize_treatments(local_entry)

    prompt = (
        f"The crop disease detected is '{prediction}'. Provide concise, practical remedies. "
        'Return valid JSON with exact keys: {"organic": ["string"], "chemical": ["string"], "fertilizer": ["string"]}'
    )
    try:
        raw_text = _generate_with_gemini(prompt, json_mode=True).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return _normalize_treatments(parsed)
    except Exception as exc:
        logger.warning("Falling back to local treatment guide for '%s': %s", prediction, exc)

    return _normalize_treatments(local_entry)


@router.post("/predict-crop")
def recommend_crop(
    data: schemas.CropInput,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    if not crop_model:
        raise HTTPException(status_code=500, detail="Crop recommendation model not loaded")

    features = np.array([[
        data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall
    ]])

    try:
        proba = crop_model.predict_proba(features)[0]
        top_3_idx = np.argsort(proba)[-3:][::-1]
        classes = crop_model.classes_
        top_crops = [
            {"crop": str(classes[i]), "confidence": round(float(proba[i]) * 100, 1)}
            for i in top_3_idx
        ]
    except Exception:
        prediction = crop_model.predict(features)[0]
        top_crops = [{"crop": str(prediction), "confidence": 99.0}]

    history_item = models.CropRecommendationHistory(
        user_id=current_user.id,
        nitrogen=data.N, phosphorus=data.P, potassium=data.K,
        temperature=data.temperature, humidity=data.humidity, ph=data.ph, rainfall=data.rainfall,
        recommended_crop=top_crops[0]["crop"]
    )
    db.add(history_item)
    db.commit()
    return {"top_crops": top_crops}


@router.post("/predict-disease")
def detect_disease(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    global _disease_interpreter, _input_details, _output_details

    image_data = file.file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a PNG, JPG, or WEBP leaf photo.")

    if _disease_interpreter is None:
        with _interpreter_lock:
            if _disease_interpreter is None:
                _disease_interpreter, _input_details, _output_details = _load_disease_interpreter()

    if _disease_interpreter is None or not class_names:
        raise HTTPException(status_code=503, detail="Disease detection model is unavailable on the server.")

    try:
        resized = image.resize((224, 224))
        input_data = np.expand_dims(np.array(resized, dtype=np.float32), axis=0)

        with _interpreter_lock:
            _disease_interpreter.set_tensor(_input_details[0]["index"], input_data)
            _disease_interpreter.invoke()
            output_data = _disease_interpreter.get_tensor(_output_details[0]["index"])[0].copy()

        predicted_idx = int(np.argmax(output_data))
        confidence = round(float(output_data[predicted_idx]) * 100, 1)
        prediction = class_names[predicted_idx] if predicted_idx < len(class_names) else "Unknown"

        crop_name, disease_name = _parse_prediction_label(prediction)
        treatments = _resolve_treatments(prediction)

        history_item = models.DiseaseDetectionHistory(
            user_id=current_user.id,
            crop_name=crop_name,
            disease_name=disease_name,
            confidence=str(confidence)
        )
        db.add(history_item)
        db.commit()

        return {
            "crop": crop_name,
            "disease": disease_name,
            "confidence": confidence,
            "treatments": treatments,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in Disease ML Pipeline: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to analyze leaf image. Please try again.")


@router.post("/chat")
def chat_with_agronomist(
    chat: schemas.ChatRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    history_lines = []
    for msg in chat.history[-10:]:
        role_label = "Farmer" if msg.role == "user" else "Agronomist"
        history_lines.append(f"{role_label}: {msg.content}")

    history_block = "\n".join(history_lines)
    prompt = (
        "You are an expert AI Agronomist for CropAdvice. Give clear, practical agricultural advice.\n\n"
        + (f"Conversation so far:\n{history_block}\n\n" if history_block else "")
        + f"Farmer: {chat.message}\nAgronomist:"
    )

    try:
        reply_text = _generate_with_gemini(prompt)
        db.add(models.AssistantChatMessage(user_id=current_user.id, role="user", content=chat.message))
        db.add(models.AssistantChatMessage(user_id=current_user.id, role="assistant", content=reply_text))
        db.commit()
        return {"reply": reply_text}
    except _GeminiError as exc:
        logger.warning("Chat Gemini failure (%s): %s", exc.kind, exc)
        if exc.kind == "missing_key":
            return {"reply": "Gemini API key is not configured on the server. Please set GEMINI_API_KEY in backend environment variables."}
        if exc.kind == "rate_limited":
            return {"reply": "The free-tier Gemini API daily/minute quota is temporarily exceeded. Please wait a bit and retry, or set GEMINI_MODEL=gemini-3.5-flash-lite in your backend environment."}
        return {"reply": "The AI Agronomist is temporarily unavailable due to an upstream Gemini API error. Please check backend logs and try again."}


@router.get("/chat/history")
def get_chat_history(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    rows = (
        db.query(models.AssistantChatMessage)
        .filter(models.AssistantChatMessage.user_id == current_user.id)
        .order_by(models.AssistantChatMessage.created_at.asc(), models.AssistantChatMessage.id.asc())
        .all()
    )
    return [
        {
            "id": row.id,
            "role": "assistant" if row.role in ("assistant", "model") else "user",
            "content": row.content,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@router.delete("/chat/history")
def clear_chat_history(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db),
):
    (
        db.query(models.AssistantChatMessage)
        .filter(models.AssistantChatMessage.user_id == current_user.id)
        .delete()
    )
    db.commit()
    return {"status": "cleared"}
