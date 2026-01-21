# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import sys

class ToxicCommentDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128):
        # Dosya var mi kontrolu
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Veri dosyasi bulunamadi: {data_path}")
            
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Etiket sutunlari (auto_label.py ile uretilenler)
        self.label_columns = ['is_toxic', 'is_profanity', 'is_insult', 'is_hate_speech']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 1. Metni al (String oldugundan emin ol)
        text = str(self.data.loc[index, 'cleaned_text'])
        
        # 2. Etiketleri al
        labels = self.data.loc[index, self.label_columns].values.astype(float)

        # 3. Tokenize islemi
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }