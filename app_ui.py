import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Toxic Comment Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Adresi (Localhost)
API_URL = "http://127.0.0.1:8000/predict"

# --- YAN MENU (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1680/1680012.png", width=100)
    st.title(" Toxic Guard")
    st.markdown("---")
    st.write("Bu proje, **Türkçe** metinlerdeki toksik davranışları derin öğrenme (Electra) ile tespit eder.")
    
    st.markdown("###  Tespit Kategorileri")
    st.code("1. Toxic (Genel)\n2. Profanity (Küfür)\n3. Insult (Hakaret)\n4. Hate Speech (Nefret)")
    
    st.markdown("---")
    st.caption("Geliştirici: **Merve Keleş**")
    st.caption("© 2026 - v1.0.0")

# --- ANA EKRAN ---
st.title("🇹🇷 Türkçe Toksik Yorum Filtreleme Sistemi")
st.markdown("Aşağıdaki alanları kullanarak tekli veya çoklu analiz yapabilirsiniz.")

# Sekmeler
tab1, tab2 = st.tabs([" Canlı Analiz", " Toplu Analiz (CSV)"])

# --- TAB 1: TEKLI ANALIZ ---
with tab1:
    st.subheader("Anlık Yorum Kontrolü")
    user_input = st.text_area("Analiz edilecek yorumu giriniz:", height=100, placeholder="Örn: Bu video gerçekten berbat ve sen bir aptalsın...")

    if st.button("Analiz Et", type="primary"):
        if user_input:
            try:
                with st.spinner('Yapay zeka düşünüyor...'):
                    # API'ye istek at
                    response = requests.post(API_URL, json={"text": user_input})
                    
                if response.status_code == 200:
                    result = response.json()
                    
                    # Sonuçları Gorsellestir
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Ana Karar Kutusu
                        if result['is_toxic']:
                            st.error("TOKSİK İÇERİK TESPİT EDİLDİ!")
                        else:
                            st.success("YORUM TEMİZ")
                            
                    with col2:
                        # Detayli Barlar
                        scores = result['probabilities']
                        for label, score in scores.items():
                            st.write(f"**{label}**")
                            # Renkli Progress Bar
                            bar_color = "red" if score > 50 else "green"
                            st.progress(score / 100, text=f"%{score}")
                            
                else:
                    st.error("API Bağlantı Hatası! Sunucunun açık olduğundan emin olun.")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                st.info("İpucu: Terminalde 'python api/app.py' komutunu çalıştırdınız mı?")
        else:
            st.warning("Lütfen bir metin giriniz.")

# --- TAB 2: TOPLU ANALIZ (CSV) ---
with tab2:
    st.subheader("Dosya Yükleme ve Toplu Tarama")
    st.markdown("İçinde `comment` veya `text` sütunu olan bir CSV dosyası yükleyin.")
    
    uploaded_file = st.file_uploader("CSV Dosyası Seçin", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        # Sütun seçimi
        text_column = st.selectbox("Hangi sütunu analiz edelim?", df.columns)
        
        if st.button("Dosyayı Tara"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_rows = len(df)
            # Demo icin ilk 50 satiri alalim (Cok uzun surmemesi icin)
            # Gercek kullanimda bu siniri kaldirabilirsin
            process_limit = min(total_rows, 100) 
            
            for i, text in enumerate(df[text_column][:process_limit]):
                try:
                    res = requests.post(API_URL, json={"text": str(text)}).json()
                    row_result = res['probabilities']
                    row_result['Yorum'] = text
                    row_result['Durum'] = "TOKSİK" if res['is_toxic'] else "TEMİZ"
                    results.append(row_result)
                except:
                    pass
                
                # İlerleme çubuğunu güncelle
                progress_bar.progress((i + 1) / process_limit)
                status_text.text(f"İşleniyor: {i+1}/{process_limit}")
            
            # Sonuç Tablosu
            result_df = pd.DataFrame(results)
            
            st.success("Tarama Tamamlandı!")
            
            # Grafikli Ozet
            st.subheader("Analiz Özeti")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(result_df, names='Durum', title='Temiz vs Toksik Dağılımı', color='Durum',
                             color_discrete_map={'TOKSİK':'red', 'TEMİZ':'green'})
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.dataframe(result_df)
                
            # İndirme Butonu
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Sonuçları İndir (CSV)",
                data=csv,
                file_name='analiz_sonuclari.csv',
                mime='text/csv',
            )