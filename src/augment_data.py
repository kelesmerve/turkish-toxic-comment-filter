import pandas as pd
import os
import random
import sys

# Ayarlar
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "to_be_labeled.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "augmented_train_data.csv")

# --- 1. KELIME HAVUZU ---
PROFANITY_WORDS = [
    "bok", "kıç", "sürtük", "kahretsin", "defol", "lan", "ulan",
    "amk", "aq", "sik", "s.k", "oç", "piç", "yavşak", "kahpe", "göt",
    "yarrak", "yarak", "amcık", "siktir", "sg", "ananı", "bacını",
    "pcin", "o.ç", "amına", "gavat", "ibne", "kaşar", "fahişe",
    "orospu", "kaltak", "dalyarak", "sikik", "yosma", "am biti"
]

INSULT_WORDS = [
    "aptal", "salak", "gerizekalı", "geri zekalı", "mal", "dangalak", "lavuk",
    "şerefsiz", "köpek", "it", "haysiyetsiz", "yobaz", "çomar",
    "öküz", "sığır", "beyinsiz", "ahmak", "kuduz", "troll", "yalaka",
    "onursuz", "cahil", "yavşak", "davar", "koyun", "ezik", "çakal",
    "adi", "pislik", "yaratık", "kamil", "zürriyetsiz", "mankurt",
    "çapsız", "beceriksiz", "vasıfsız", "kifayetsiz", "ahmak"
]

HATE_SPEECH_WORDS = [
    "fetö", "terörist", "allah belanı", "vatan haini", "hain",
    "kâfir", "dinsiz", "yunan tohumu", "ermeni dölü", "suriyeli", 
    "ülkeyi sattınız", "idam", "katil", "sapık", "tecavüzcü", 
    "soysuz", "kanınız bozuk", "din düşmanı", "ırkçı"
]

# --- 2. URETIM LISTELERI (Patterns) ---
# Nefret kaliplari
hate_speech_patterns = [
    "ülkeyi sattınız vatan hainleri", "bunların hepsi terörist", "fetöcü bunlar asmak lazım",
    "kökünüzü kazıyacağız", "sizin gibi kafirlerin yaşamaya hakkı yok", "defolun gidin bu ülkeden",
    "arap seviciler sizi", "yahudi tohumu bunlar", "ermeni dölü", "yunan tohumu",
    "hepiniz idam edilmelisiniz", "bu millete ihanet edenler bedel ödeyecek",
    "kanınız bozuk sizin", "soysuz köpekler", "dinsiz imansız vatan hainleri",
    "suriyeliler dışarı", "mülteciler defolsun", "kürtler bölücüdür", "türk düşmanları",
    "aleviler şöyle böyledir", "sünniler şöyledir", "bu ırkı yok etmek lazım",
    "senin gibilerin soyunu kurutmak lazım", "vatan haini şerefsizler", "ajan bunlar ajan",
    "dış güçlerin maşası olmuşsunuz", "din düşmanları", "yallah arabistana", "yallah yunanistana",
    "kellelerini alacağız", "taş üstünde taş bırakmayacağız", "sizi denize dökeceğiz"
]

def contains_word(text, word_list):
    if pd.isna(text) or text == "": return 0
    text = str(text).lower()
    for word in word_list:
        if word in text: return 1
    return 0

def generate_synthetic_data():
    print("   [ISLEM] Sentetik veri uretimi basladi...")
    new_data = []

    # A) NEFRET SOYLEMI URETIMI (Eskisi gibi)
    print("     -> Nefret soylemi cogaltiliyor...")
    for _ in range(25): 
        for text in hate_speech_patterns:
            variations = [text, text.upper(), text + "!!!", "lan " + text, "hepiniz " + text]
            selected = random.choice(variations)
            new_data.append({"cleaned_text": selected, "is_toxic": 1, "is_profanity": 0, "is_insult": 1, "is_hate_speech": 1})

    # B) KÜFÜR URETIMI (YENI EKLENDI!) ⚡
    print("     -> Kufurler cogaltiliyor...")
    for word in PROFANITY_WORDS:
        # Her kelime icin 15 farkli cumle uret
        templates = [
            f"{word} git buradan", f"senin ben {word}", f"kes lan {word}", 
            f"ne {word} bir herifsin", f"tam bir {word}", f"ananı {word}",
            f"{word} ağızlı", f"siktir {word}", f"{word} seni", f"Allah'ın {word}u"
        ]
        for t in templates:
            new_data.append({"cleaned_text": t, "is_toxic": 1, "is_profanity": 1, "is_insult": 1, "is_hate_speech": 0})

    # C) HAKARET URETIMI (YENI EKLENDI!) ⚡
    print("     -> Hakaretler cogaltiliyor...")
    for word in INSULT_WORDS:
        # Her kelime icin 15 farkli cumle uret
        templates = [
            f"sen tam bir {word}sın", f"kes sesini {word}", f"bıktık senin gibi {word}lardan",
            f"ne kadar {word} birisin", f"{word} herif", f"hayatımda gördüğüm en büyük {word}",
            f"gerçekten {word} mısın", f"allahın {word}ı", f"bas git {word}", f"sus be {word}"
        ]
        for t in templates:
            new_data.append({"cleaned_text": t, "is_toxic": 1, "is_profanity": 0, "is_insult": 1, "is_hate_speech": 0})
            
    return pd.DataFrame(new_data)

def main():
    print("[BASLATILIYOR] Veri Zenginlestirme ve Dengeleme (V2 - Full Kapsam)...")

    if not os.path.exists(INPUT_PATH):
        print(f"[HATA] Dosya yok: {INPUT_PATH}. Once preprocess yapin.")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"[BILGI] Gercek Veri: {len(df)} satir")

    # Gercek veriyi etiketle
    df["is_profanity"] = df["cleaned_text"].apply(lambda x: contains_word(x, PROFANITY_WORDS))
    df["is_insult"] = df["cleaned_text"].apply(lambda x: contains_word(x, INSULT_WORDS))
    df["is_hate_speech"] = df["cleaned_text"].apply(lambda x: contains_word(x, HATE_SPEECH_WORDS))
    df["is_toxic"] = (df["is_profanity"] | df["is_insult"] | df["is_hate_speech"]).astype(int)

    # Sentetik veri ile birlestir
    synthetic_df = generate_synthetic_data()
    full_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    # Dengeleme
    toxic_df = full_df[full_df["is_toxic"] == 1]
    clean_df = full_df[full_df["is_toxic"] == 0]
    
    # Temiz veriyi toksik sayisina esitle
    n_samples = min(len(toxic_df), len(clean_df))
    final_df = pd.concat([toxic_df, clean_df.sample(n=n_samples, random_state=42)])
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[SONUC] Yeni Veri Seti Hazir: {len(final_df)} satir.")
    print(final_df[["is_toxic", "is_profanity", "is_insult", "is_hate_speech"]].sum())

if __name__ == "__main__":
    main()