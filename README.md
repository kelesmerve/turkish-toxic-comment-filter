# Türkçe Toksik Yorum Filtreleme (Toxic Comment Filter)

Bu proje, Türkçe metinlerdeki (özellikle sosyal medya ve YouTube yorumları) toksik davranışları, küfürleri, hakaretleri ve nefret söylemlerini tespit eden derin öğrenme tabanlı bir NLP projesidir.

Proje; veri toplama, veri zenginleştirme, model eğitimi ve canlı API sunumu olmak üzere uçtan uca bir pipeline içerir.

## Proje Hakkında

Bu sistem, tek bir yorumu analiz ederek aşağıdaki 4 farklı kategoride olasılık değerleri üretir:

* **Toxic:** Genel rahatsız edici içerik.
* **Profanity:** Küfür ve argo kullanımı.
* **Insult:** Kişiye veya kuruma yönelik hakaret.
* **Hate Speech:** Din, ırk, etnik köken veya siyasi görüşe dayalı nefret söylemi.

## Çalışma Mantığı (Workflow)

Proje 4 ana aşamadan oluşur:

1.  **Veri Toplama (Scraping):** `src/scraper.py` ile YouTube üzerindeki tartışmalı videolardan (spor, siyaset, oyun vb.) binlerce gerçek kullanıcı yorumu toplanmıştır.
2.  **Veri İşleme ve Zenginleştirme (Preprocessing & Augmentation):**
    * Ham veriler temizlenmiştir (HTML tagleri, linkler vb.).
    * Dengesiz veri setini düzeltmek için **Sentetik Veri Üretimi** yapılmıştır. Özellikle "Nefret Söylemi" ve "Küfür" kategorileri için yapay zeka destekli veri çoğaltma (augmentation) uygulanarak modelin kör noktaları giderilmiştir.
3.  **Model Eğitimi (Fine-Tuning):**
    * Google Colab (GPU) üzerinde `dbmdz/electra-base-turkish-cased-discriminator` modeli eğitilmiştir.
    * Eğitim süreci ve grafikleri `notebooks/` klasöründeki Jupyter Notebook dosyasında mevcuttur.
4.  **Deployment (API):**
    * Eğitilen model, **FastAPI** ve **Uvicorn** kullanılarak asenkron bir REST API haline getirilmiştir.

## Kurulum

Projeyi yerel ortamınızda çalıştırmak için:

1.  Depoyu klonlayın:
    ```bash
    git clone [https://github.com/kelesmerve/turkish-toxic-comment-filter.git](https://github.com/kelesmerve/turkish-toxic-comment-filter.git)
    cd toxic-comment-filter
    ```

2.  Gereksiz kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım

### API'yi Başlatma
Modeli canlıya almak ve test etmek için terminalde şu komutu çalıştırın:

```bash
python api/app.py

```

Sunucu açıldığında tarayıcınızdan `http://127.0.0.1:8000/docs` adresine giderek Swagger arayüzü üzerinden test yapabilirsiniz.

### Model Eğitimi (Colab)

Modelin nasıl eğitildiğini incelemek için `notebooks/Model_Training_Electra.ipynb` dosyasını Google Colab üzerinde açabilirsiniz.

## Performans Sonuçları

Sentetik veri desteği ile eğitilen modelin test seti sonuçları:

| Kategori | F1-Score | Başarı Durumu |
| --- | --- | --- |
| **Hate Speech** | %96+ | Yüksek Başarı |
| **Insult** | %95+ | Yüksek Başarı |
| **Toxic** | %90+ | Yüksek Başarı |

## Kullanılan Teknolojiler

* **Dil:** Python 3.10+
* **Model:** Transformers (Hugging Face), PyTorch
* **API:** FastAPI, Uvicorn
* **Veri:** Pandas, Scikit-learn

```

```