import pandas as pd
import re
import os
import sys

# Yollari ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "youtube_raw.csv")
# Temizlenmis veriyi buraya kaydedecegiz
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "to_be_labeled.csv")

def turkish_lower(text):
    """Turkce karakter sorunu olmadan kucuk harfe cevirir"""
    if pd.isna(text): return ""
    return text.replace("I", "ı").replace("İ", "i").lower()

def clean_text(text):
    if pd.isna(text): return ""
    
    # HTML tagleri, Linkler, Mentionlar ve Hashtag isaretini temizle
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = text.replace('\n', ' ')
    
    # Fazla bosluklari sil ve kucult
    text = re.sub(r'\s+', ' ', text).strip()
    text = turkish_lower(text)
    
    return text

def main():
    print("[BASLATILIYOR] Veri temizligi baslatiliyor...")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[HATA] Ham veri dosyasi bulunamadi: {RAW_DATA_PATH}")
        return

    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"[BILGI] {len(df)} satir ham veri okundu.")
        
        # Temizligi uygula
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Cop verileri (cok kisa veya tekrar eden) sil
        df = df[df['cleaned_text'].str.len() > 3]
        df = df.drop_duplicates(subset=['cleaned_text'])
        
        # Etiketleme icin bos sutun ac (Varsayilan 0)
        df['label'] = 0 
        
        # Sadece gerekli sutunlari al
        final_df = df[['cleaned_text', 'label', 'keyword_tag']]
        
        # Kaydet
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        final_df.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8-sig')
        
        print(f"[BASARILI] Temiz veri kaydedildi: {PROCESSED_DATA_PATH}")
        print(f"[BILGI] Etiketlenecek satir sayisi: {len(final_df)}")
        print("\n[SONRAKI ADIM] Bu dosyayi Excel veya Notepad ile acip 'label' sutununu (1=Toksik, 0=Temiz) doldurun.")
        
    except Exception as e:
        print(f"[HATA] Islem sirasinda bir hata olustu: {e}")

if __name__ == "__main__":
    main()