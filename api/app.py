import sys
import os
import re
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. AYARLAR VE PATH YONETIMI ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxic_model_v1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ml_models = {}

# --- 2. TEMIZLIK FONKSIYONU ---
def clean_text(text):
    if not text: return ""
    text = str(text).replace("I", "ı").replace("İ", "i").lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. LIFESPAN (YUKLEME) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n[BASLATILIYOR] Model yukleniyor: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        ml_models["tokenizer"] = tokenizer
        ml_models["model"] = model
        print("[BASARILI] Model hafizaya alindi!")
    except Exception as e:
        print(f"[HATA] Model yuklenemedi: {e}")
    yield
    ml_models.clear()

app = FastAPI(title="Toxic Comment Filter API", version="2.0", lifespan=lifespan)

class CommentRequest(BaseModel):
    text: str

# --- 4. TAHMIN ENDPOINT ---
@app.post("/predict")
def predict_comment(request: CommentRequest):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model yuklenmedi")

    tokenizer = ml_models["tokenizer"]
    model = ml_models["model"]

    # Temizlik ve Tahmin
    cleaned_text = clean_text(request.text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().squeeze().numpy()

    # Sonuclari Hazirla
    labels = ["Toxic", "Profanity", "Insult", "Hate Speech"]
    thresholds = {'Toxic': 0.50, 'Profanity': 0.50, 'Insult': 0.50, 'Hate Speech': 0.50}
    
    results = {}
    is_toxic_found = False

    for i, label in enumerate(labels):
        # Tek boyutlu array kontrolu
        val = float(probs[i]) if probs.ndim == 1 else float(probs)
        score = round(val * 100, 2)
        results[label] = score
        
        if score > (thresholds[label] * 100):
            is_toxic_found = True

    # --- EN ONEMLI KISIM: ARAYUZ BUNU BEKLIYOR ---
    return {
        "is_toxic": is_toxic_found,
        "probabilities": results,  # <-- Hata burada cikiyordu, artik duzeldi.
        "original_text": request.text
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)