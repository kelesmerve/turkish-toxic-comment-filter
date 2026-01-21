import pandas as pd
import os
import time
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from youtubesearchpython import VideosSearch

class YoutubeAutoScraper:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.raw_data_dir = os.path.join(self.base_dir, "data", "raw")
        self.output_file = os.path.join(self.raw_data_dir, "youtube_raw_aggressive.csv")
        
        self.downloader = YoutubeCommentDownloader()
        self.data = []

    def search_and_scrape(self, keyword, video_limit=5, comment_limit=200):
        print(f"[BILGI] '{keyword}' kelimesi icin video araniyor...")
        
        try:
            videos_search = VideosSearch(keyword, limit=video_limit)
            results = videos_search.result()
            
            if not results or 'result' not in results:
                print("[UYARI] Arama sonucu bulunamadi.")
                return

            videos = results['result']
            print(f"[OK] {len(videos)} adet video bulundu. Yorumlar cekiliyor...")

            for video in videos:
                video_id = video['id']
                video_title = video['title']
                self.scrape_comments(video_id, video_title, comment_limit)
                
        except Exception as e:
            print(f"[HATA] Arama sirasinda hata: {e}")

    def scrape_comments(self, video_id, title, limit):
        print(f"   -> Video Isleniyor: {title} (ID: {video_id})")
        
        try:
            # Yorumlari indir (En populerleri once ceker)
            comments = self.downloader.get_comments_from_url(
                f'https://www.youtube.com/watch?v={video_id}', 
                sort_by=SORT_BY_POPULAR
            )
            
            count = 0
            for comment in islice(comments, limit):
                text = comment['text']
                
                # Sadece 3 karakterden uzun ve link icermeyen yorumlari al
                if len(text) > 3 and "http" not in text:
                    self.data.append({
                        "text": text,
                        "source": "youtube",
                        "keyword_tag": title,
                        "video_id": video_id,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    count += 1
            
            print(f"      + {count} yorum alindi.")
            
        except Exception as e:
            print(f"      [ATLANDI] Yorum cekilemedi (Yorumlar kapali olabilir).")

    def save_to_csv(self):
        if not self.data:
            return

        os.makedirs(self.raw_data_dir, exist_ok=True)
        df = pd.DataFrame(self.data)
        
        file_exists = os.path.isfile(self.output_file)
        
        try:
            # Mod 'a' (append) ile dosyanin sonuna ekleme yapar, verin kaybolmaz
            df.to_csv(self.output_file, index=False, mode='a', header=not file_exists, encoding="utf-8-sig")
            print(f"[KAYIT] {len(df)} yeni satir eklendi -> {self.output_file}")
            self.data = [] 
        except PermissionError:
            print("[HATA] CSV dosyasi acik! Lutfen kapatin ve tekrar deneyin.")

if __name__ == "__main__":
    scraper = YoutubeAutoScraper()
    
    # AGRESIF VERI TOPLAMA LISTESI
    # Hedef: Kavga, kufur, hakaret, nefret soylemi ve argoyu dogal ortaminda yakalamak.
    # Bu liste ozellikle Turkiye'deki gerginlik noktalari uzerine kurulmustur.
    keywords = [
        # 1. SPOR VE FANATIZM (En yogun kufur kaynagi)
        "hakem hataları küfürlü tepkiler",
        "derbi sonrası kavga",
        "tribün olayları kavga",
        "futbol yorumcuları kavga",
        "amatör lig kavga",
        "maç sonu olaylar",
        "galatasaray fenerbahçe kavga",
        
        # 2. TRAFIK VE SOKAK (Dogal sinir patlamalari)
        "trafikte tekme tokat kavga",
        "trafik magandası dehşet saçtı",
        "yol verme kavgası",
        "taksici uber kavgası",
        "otobüste tartışma küfür",
        "metroda kavga",
        "metrobüs kavgası",
        
        # 3. SIYASET VE SOKAK ROPORTAJLARI (Nefret soylemi ve hakaret)
        "sokak röportajı kavga dayı",
        "sokak röportajı tartışma küfürlü",
        "mecliste yumruklu kavga",
        "canlı yayında kavga siyaset",
        "telefonda küfürleşme kaydı",
        
        # 4. OYUN VE INTERNET (Toxic gamer dili)
        "oyun oynarken çıldıran gamer",
        "twitch canlı yayın kavga",
        "lol küfürlü maç",
        "valorant toxic oyuncular",
        "cs go küfürleşme",
        "discord kavga kayıtları",
        
        # 5. REALITY VE TV (Kaos ve hakaret)
        "evlilik programı kavga",
        "gelinim mutfakta kavga",
        "survivor kavga sansürsüz",
        "yemekteyiz kavga",
        "müge anlı kavga",
        "esra erol kavga"
    ]
    
    print("--- AGRESIF VERI TOPLAMA MODU BASLATILIYOR ---")
    print(f"Toplam Kategori Sayisi: {len(keywords)}")
    print("Not: Bu islem internet hizina bagli olarak zaman alabilir.")
    
    for key in keywords:
        # Her anahtar kelimeden 5 video, her videodan 200 yorum
        # Teorik olarak: 30 kelime * 5 video * 200 yorum = 30.000 veri taramasi
        scraper.search_and_scrape(key, video_limit=5, comment_limit=200)
        
        # Her aramadan sonra kaydet ki elektrik gitse bile veri kaybolmasin
        scraper.save_to_csv()
        
        # YouTube spam algilamasin diye 2 saniye bekle
        time.sleep(2)
        
    print("--- TUM ISLEMLER TAMAMLANDI ---")