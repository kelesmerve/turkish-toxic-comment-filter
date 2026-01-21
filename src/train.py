# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Ilerleme cubugu
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix

# Proje dizinini path'e ekle
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in os.sys.path:
    os.sys.path.append(sys_path)

try:
    from src.dataset import ToxicCommentDataset
except ImportError:
    from dataset import ToxicCommentDataset

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "toxic_model_v1")
PLOT_METRICS_PATH = os.path.join(BASE_DIR, "training_metrics.png")
PLOT_CM_PATH = os.path.join(BASE_DIR, "confusion_matrices.png")
MODEL_NAME = "dbmdz/electra-base-turkish-cased-discriminator"

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128
THRESHOLD = 0.5
LABEL_NAMES = ['Toxic', 'Profanity', 'Insult', 'Hate Speech']

def calculate_metrics(preds, labels):
    preds = torch.sigmoid(preds).cpu().numpy()
    preds = (preds > THRESHOLD).astype(int)
    labels = labels.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "preds": preds,
        "labels": labels
    }

def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title('Egitim ve Dogrulama Kaybi')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='F1 Score', color='green', marker='s')
    plt.plot(history['val_acc'], label='Accuracy', color='blue', marker='s')
    plt.title('Performans Metrikleri')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_METRICS_PATH)
    print(f"[GRAFIK] Metrik grafigi kaydedildi: {PLOT_METRICS_PATH}")

def plot_confusion_matrices(last_preds, last_labels):
    cms = multilabel_confusion_matrix(last_labels, last_preds)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, cm in enumerate(cms):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=['Negatif', 'Pozitif'], 
                    yticklabels=['Negatif', 'Pozitif'])
        axes[i].set_title(f"Confusion Matrix: {LABEL_NAMES[i]}")
    
    plt.tight_layout()
    plt.savefig(PLOT_CM_PATH)
    print(f"[GRAFIK] Confusion Matrix kaydedildi: {PLOT_CM_PATH}")

def train():
    print("[BASLATILIYOR] Ilerleme Cubuklu Egitim Modu...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BILGI] Cihaz: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if not os.path.exists(DATA_PATH):
        print(f"[HATA] Veri yok: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    
    train_df.to_csv("temp_train.csv", index=False)
    val_df.to_csv("temp_val.csv", index=False)

    train_dataset = ToxicCommentDataset("temp_train.csv", tokenizer, MAX_LEN)
    val_dataset = ToxicCommentDataset("temp_val.csv", tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=4,
        problem_type="multi_label_classification"
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": [], "val_recall": []}
    
    last_val_preds = None
    last_val_labels = None

    print(f"\n[BASLIYOR] Toplam {EPOCHS} Epoch islenecek.")
    print("--------------------------------------------------")
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        # TQDM ilerleme cubugu
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            
            # --- BURASI YENI: HER 10 ADIMDA BIR YAZDIR ---
            # tqdm.write kullanarak cubugu bozmadan ekrana yazi yaziyoruz
            if step % 10 == 0:
                tqdm.write(f"   [Epoch {epoch+1}] Adim {step}/{len(train_loader)} -> Loss: {loss.item():.4f}")
            
            progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0
        val_preds_list = []
        val_labels_list = []

        # Validation icin de kisa bir cubuk gosterelim
        print("   [Dogrulama] Validation testi yapiliyor...")
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                val_preds_list.append(outputs.logits)
                val_labels_list.append(labels)

        val_preds_tensor = torch.cat(val_preds_list)
        val_labels_tensor = torch.cat(val_labels_list)
        
        avg_val_loss = total_val_loss / len(val_loader)
        metrics = calculate_metrics(val_preds_tensor, val_labels_tensor)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(metrics['f1'])
        history['val_acc'].append(metrics['accuracy'])
        history['val_recall'].append(metrics['recall'])
        
        if epoch == EPOCHS - 1:
            last_val_preds = metrics['preds']
            last_val_labels = metrics['labels']

        print(f"\n✅ EPOCH {epoch+1} OZET RAPORU:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f} | F1 Score: {metrics['f1']:.4f}")
        print("-" * 40)

    print(f"\n[KAYIT] Model kaydediliyor: {MODEL_SAVE_PATH}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    print("[ISLEM] Grafikler ciziliyor...")
    plot_history(history)
    plot_confusion_matrices(last_val_preds, last_val_labels)

    if os.path.exists("temp_train.csv"): os.remove("temp_train.csv")
    if os.path.exists("temp_val.csv"): os.remove("temp_val.csv")
    
    print("[BITTI] Model egitimi tamamlandi.")

if __name__ == "__main__":
    train()