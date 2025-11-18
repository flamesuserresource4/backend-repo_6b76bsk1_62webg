import os
import io
import hashlib
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageDraw
import numpy as np

from database import db, create_document, get_documents
from schemas import User, SelfAssessment, XrayScan, HistoryItem

app = FastAPI(title="Deepneumoscan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files (annotated images)
app.mount("/uploads", StaticFiles(directory="uploads", check_dir=False), name="uploads")


# ---------------------------
# Lightweight mock model (no external ML deps)
# ---------------------------
class SimpleXrayModel:
    """
    Simulated SVM→KNN fallback using only NumPy so we avoid heavy deps.
    - We compute a grayscale histogram feature and a few simple stats.
    - "SVM" is simulated by a margin-like score from a linear rule.
    - If the confidence < 0.9, we "fallback" to a KNN-like rule using
      distances to two synthetic centroids.
    """

    def __init__(self):
        self.labels = {0: "normal", 1: "pneumonia"}
        # Synthetic centroids for the fallback
        self.centroid_normal = np.concatenate([np.ones(16)*0.8, np.ones(16)*0.2]).astype(np.float32)
        self.centroid_pneu = np.concatenate([np.ones(16)*0.2, np.ones(16)*0.8]).astype(np.float32)
        self.w = np.concatenate([np.ones(16)*-0.5, np.ones(16)*0.5]).astype(np.float32)
        self.b = 0.0

    def _features(self, img: Image.Image) -> np.ndarray:
        g = img.convert("L").resize((128, 128))
        arr = np.array(g)
        # 32-bin histogram normalized, plus mean/std/skew proxy
        hist, _ = np.histogram(arr, bins=32, range=(0, 255), density=True)
        hist = hist.astype(np.float32)
        # compress to 32 dims used above
        return hist

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        f = self._features(img)
        # Simulated SVM probability via sigmoid of linear score
        score = float(np.dot(self.w, f[:32]) + self.b)
        p_pneu = self._sigmoid(score)
        svm_conf = max(p_pneu, 1 - p_pneu)
        svm_label = 1 if p_pneu >= 0.5 else 0
        if svm_conf >= 0.9:
            return {"label": self.labels[svm_label], "confidence": float(svm_conf), "model": "svm"}

        # Fallback: KNN-like by comparing distances to two centroids
        f16 = (f.reshape(2, 16).mean(axis=0)).astype(np.float32)
        d_norm = float(np.linalg.norm(f16 - self.centroid_normal))
        d_pneu = float(np.linalg.norm(f16 - self.centroid_pneu))
        if d_norm + d_pneu == 0:
            knn_conf = 0.5
            knn_label = 0
        else:
            p_norm = d_pneu / (d_norm + d_pneu)
            p_pn = d_norm / (d_norm + d_pneu)
            if p_pn >= 0.5:
                knn_label = 1
                knn_conf = p_pn
            else:
                knn_label = 0
                knn_conf = p_norm
        return {"label": self.labels[knn_label], "confidence": float(knn_conf), "model": "knn"}

    def localize(self, img: Image.Image) -> Image.Image:
        # Draw simple circle as a visual indicator (demo only)
        w, h = img.size
        overlay = img.convert("RGB").copy()
        draw = ImageDraw.Draw(overlay)
        r = min(w, h) // 6
        cx, cy = w // 2, h // 2
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 0, 0), width=4)
        return overlay


xr_model = SimpleXrayModel()


# ---------------------------
# Auth endpoints (basic demo)
# ---------------------------
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    language: str = "en"  # en or kn


class LoginRequest(BaseModel):
    email: str
    password: str


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


@app.post("/auth/signup")
def signup(req: SignupRequest):
    existing = db["user"].find_one({"email": req.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    doc = User(name=req.name, email=req.email, password_hash=hash_password(req.password), language=req.language)
    _id = create_document("user", doc)
    return {"user_id": _id, "name": req.name, "language": req.language}


@app.post("/auth/login")
def login(req: LoginRequest):
    u = db["user"].find_one({"email": req.email})
    if not u or u.get("password_hash") != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": str(u.get("_id")), "name": u.get("name"), "language": u.get("language", "en")}


# ---------------------------
# Self Assessment
# ---------------------------
class SelfAssessmentRequest(BaseModel):
    user_id: str
    language: str = "en"
    answers: Dict[str, Any]


@app.post("/self-assessment")
def self_assessment(req: SelfAssessmentRequest):
    answers = req.answers
    score = 0
    pneumonia_keys = ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue"]
    for k in pneumonia_keys:
        v = answers.get(k)
        if isinstance(v, (int, float)):
            score += float(v)
        elif isinstance(v, bool) and v:
            score += 1
    label = "pneumonia_suspected" if score >= 3 else "low_risk"
    conf = min(0.99, 0.5 + score * 0.1)

    doc = {
        "user_id": req.user_id,
        "answers": answers,
        "result_label": label,
        "result_confidence": conf,
        "language": req.language
    }
    sid = create_document("selfassessment", doc)
    create_document("historyitem", {"user_id": req.user_id, "item_type": "self_assessment", "ref_id": sid, "summary": f"Self assessment: {label} ({conf:.2f})", "language": req.language})

    messages = {
        "en": f"Self assessment indicates: {label.replace('_', ' ')} with confidence {conf:.2f}",
        "kn": f"ಸ್ವಯಂ ಮೌಲ್ಯಮಾಪನ ಫಲಿತಾಂಶ: {label.replace('_', ' ')} ವಿಶ್ವಾಸ {conf:.2f}"
    }
    return {"id": sid, "label": label, "confidence": conf, "message": messages.get(req.language, messages["en"])}


# ---------------------------
# X-ray Scan Upload & Inference
# ---------------------------
@app.post("/xray/scan")
async def xray_scan(
    user_id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    medical_condition: Optional[str] = Form(None),
    language: str = Form("en"),
    file: UploadFile = File(...),
):
    if file.content_type not in ["image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG images are supported")

    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    pred = xr_model.predict(img)
    annotated = xr_model.localize(img)

    os.makedirs("uploads/annotated", exist_ok=True)
    base_name = hashlib.md5(content).hexdigest() + ".jpg"
    raw_path = os.path.join("uploads", base_name)
    ann_path = os.path.join("uploads/annotated", base_name)
    try:
        os.makedirs("uploads", exist_ok=True)
        img.save(raw_path, format="JPEG")
        annotated.save(ann_path, format="JPEG")
    except Exception:
        # ensure dirs exist; continue even if save fails
        pass

    xdoc = {
        "user_id": user_id,
        "name": name,
        "age": age,
        "gender": gender,
        "medical_condition": medical_condition,
        "file_path": raw_path,
        "annotated_path": ann_path,
        "model_used": pred["model"],
        "predicted_label": pred["label"],
        "confidence": pred["confidence"],
        "language": language,
    }
    xid = create_document("xrayscan", xdoc)
    create_document("historyitem", {"user_id": user_id, "item_type": "xray", "ref_id": xid, "summary": f"X-ray: {pred['label']} ({pred['confidence']:.2f})", "language": language})

    messages = {
        "en": f"X-ray analysis suggests: {pred['label']} with confidence {pred['confidence']:.2f} using {pred['model'].upper()}.",
        "kn": f"ಎಕ್ಸ್-ರೇ ವಿಶ್ಲೇಷಣೆ: {pred['label']} ವಿಶ್ವಾಸ {pred['confidence']:.2f} ({pred['model'].upper()})."
    }
    return {"id": xid, "label": pred["label"], "confidence": pred["confidence"], "model": pred["model"], "annotated_image_path": ann_path, "message": messages.get(language, messages["en"])}


# ---------------------------
# Curing Assessment (trend)
# ---------------------------
class CuringAssessmentRequest(BaseModel):
    user_id: str
    language: str = "en"
    inputs: Dict[str, Any]


@app.post("/curing-assessment")
def curing_assessment(req: CuringAssessmentRequest):
    symptoms = req.inputs
    numeric_vals = [float(v) for v in symptoms.values() if isinstance(v, (int, float))]
    score = float(np.mean(numeric_vals)) if numeric_vals else 0.5

    prev = db["curingassessment"].find_one({"user_id": req.user_id}, sort=[("created_at", -1)])
    prev_score = float(prev.get("score", 0.5)) if prev else 0.5
    delta = score - prev_score
    if delta < -0.05:
        status = "better"
    elif delta > 0.05:
        status = "worse"
    else:
        status = "stable"

    doc = {
        "user_id": req.user_id,
        "inputs": symptoms,
        "status": status,
        "score_delta": float(delta),
        "score": float(score),
        "language": req.language,
    }
    _id = create_document("curingassessment", doc)
    create_document("historyitem", {"user_id": req.user_id, "item_type": "curing", "ref_id": _id, "summary": f"Curing: {status} (Δ {delta:.2f})", "language": req.language})

    messages = {
        "en": f"Your trend is {status}. Change in score: {delta:.2f}",
        "kn": f"ನಿಮ್ಮ ಪ್ರವೃತ್ತಿ {status}. ಅಂಕದ ಬದಲಾವಣೆ: {delta:.2f}"
    }
    return {"id": _id, "status": status, "delta": float(delta), "message": messages.get(req.language, messages["en"])}


# ---------------------------
# History & Delete
# ---------------------------
@app.get("/history/{user_id}")
def get_history(user_id: str):
    items = get_documents("historyitem", {"user_id": user_id})
    for it in items:
        it["_id"] = str(it.get("_id"))
    return {"items": items}


@app.delete("/history/{user_id}/{item_id}")
def delete_history_item(user_id: str, item_id: str):
    try:
        from bson import ObjectId
        db["historyitem"].delete_one({"_id": ObjectId(item_id), "user_id": user_id})
    except Exception:
        pass
    return {"deleted": True}


@app.get("/test")
def test_database():
    ok = db is not None
    return {"backend": "running", "database": ok}


@app.get("/")
def root():
    return {"name": "Deepneumoscan API"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
