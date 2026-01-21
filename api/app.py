import sys
import os
import torch
import re
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. AYARLAR ---
# Proje kok dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxic_model_v1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelleri tutacak sozluk (Global degisken yerine bunu kullaniyoruz)
ml_models = {}

# --- 2. LIFESPAN (YENI MODEL YUKLEME YONTEMI) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n[BASLATILIYOR] Model yukleniyor: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        
        # Hafizaya atiyoruz
        ml_models["tokenizer"] = tokenizer
        ml_models["model"] = model
        print("[BASARILI] Model API icin hazir ve hafizada!")
    except Exception as e:
        print(f"[HATA] Model yuklenemedi! Lutfen model yolunu kontrol et.\nHata: {e}")
    
    yield # API calistigi surece burasi bekler
    
    # Kapanirken yapilacaklar
    ml_models.clear()
    print("[DURDURULUYOR] API kapatildi.")

# API Uygulamasini Baslat
app = FastAPI(title="Toxic Comment Filter API", version="2.0", lifespan=lifespan)

# --- 3. VERI MODELLERI ---
class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    original_text: str
    is_toxic: bool
    confidence_scores: dict

# --- 4. YARDIMCI FONKSIYONLAR ---
def clean_text(text):
    """Senin eski kodundaki temizleme fonksiyonu"""
    if not text: return ""
    text = str(text).replace("I", "ı").replace("İ", "i").lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 5. ENDPOINTLER ---
@app.get("/")
def root():
    return {"status": "active", "message": "Toxic Comment API v2 Calisiyor!"}

@app.post("/predict", response_model=PredictionResponse)
def predict_comment(request: CommentRequest):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model henuz yuklenmedi veya hata olustu.")

    tokenizer = ml_models["tokenizer"]
    model = ml_models["model"]

    # Temizlik
    cleaned_text = clean_text(request.text)

    # Hazirlik
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    # Tahmin
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Olasiliklar
    probs = torch.sigmoid(outputs.logits).cpu().squeeze().numpy()
    
    labels = ['Toxic', 'Profanity', 'Insult', 'Hate Speech']
    results = {}
    is_toxic_found = False

    for i, prob in enumerate(probs):
        score = round(float(prob) * 100, 2)
        results[labels[i]] = score
        
        # Eger Toxic orani %40'i gecerse tehlikeli isaretle
        if labels[i] == 'Toxic' and score > 40.0:
            is_toxic_found = True

    return {
        "original_text": request.text,
        "is_toxic": is_toxic_found,
        "confidence_scores": results
    }

# --- 6. BASLATMA KOMUTU (EN ONEMLI KISIM) ---
if __name__ == "__main__":
    print("[BILGI] Sunucu ayaga kaldiriliyor...")
    # reload=True, kodu degistirince sunucuyu otomatik yeniler (Gelistirici modu)
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)