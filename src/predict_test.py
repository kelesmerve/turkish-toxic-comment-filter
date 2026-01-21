import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import re

# Ayarlar
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "toxic_model_v1")

def turkish_lower(text):
    """Egitimdeki fonksiyonun aynisi"""
    if not text: return ""
    # I -> i donusumu kritik
    return text.replace("I", "ı").replace("İ", "i").lower()

def clean_text(text):
    """Girdiyi egitim verisine benzetir"""
    if not text: return ""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = turkish_lower(text)
    return text

def load_model():
    print(f"[BASLATILIYOR] Model yukleniyor: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        print(f"[HATA] Model yuklenemedi: {e}")
        return None, None

def predict(raw_text, tokenizer, model):
    # 1. GIRDIYI TEMIZLE (Egitimdeki gibi)
    processed_text = clean_text(raw_text)
    
    # Eger temizlik sonrasi metin bombos kalirsa uyari ver
    if len(processed_text) < 2:
        print("[UYARI] Yorum cok kisa veya anlamsiz.")
        return

    model.eval()
    
    inputs = tokenizer(
        processed_text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.sigmoid(outputs.logits).squeeze().tolist()
    
    labels = ['Toxic', 'Profanity (Kufur)', 'Insult (Hakaret)', 'Hate Speech (Nefret)']
    
    print(f"\n[GIRDI]   : '{raw_text}'")
    print(f"[ISLENEN] : '{processed_text}' (Model bunu goruyor)")
    print("-" * 40)
    
    for i, prob in enumerate(probs):
        percentage = prob * 100
        
        # --- HASSASIYET AYARI ---
        # Veri az oldugu icin model %90 emin olamaz. 
        # Esik degerini %40'a (0.4) cektik.
        threshold = 40.0 
        
        status = "[!] VAR" if percentage > threshold else "[ ] YOK"
        
        print(f"{labels[i]:<25} : %{percentage:.1f}  \t{status}")

if __name__ == "__main__":
    tokenizer, model = load_model()
    
    if model:
        print("\n*** TOXIC COMMENT FILTER TEST ARACI (Cikmak icin 'q') ***")
        print("Not: Veri az oldugu icin esik degeri %40 olarak ayarlandi.")
        while True:
            user_input = input("\n[?] Bir yorum yazin: ")
            if user_input.lower() == 'q':
                break
            
            predict(user_input, tokenizer, model)