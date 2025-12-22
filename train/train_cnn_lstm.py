# Filename: train_cnn_lstm.py

import logging
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import joblib

# 하이퍼파라미터
SEQ_LEN = 10  # 시퀀스 길이 (CSV 시퀀스 길이)
PER_STEP_FEATURE = 8  # interval 당 featrue 개수
BATCH_SIZE = 32
EPOCHS = 50     
LEARNING_RATE = 1e-3

CSV_PATH = "dataset/sequence_dataset.csv"

# 맥북이라 mts
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", DEVICE)

# dataset 정의
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # 분류라 long
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# CNN-LSTM 모델 정의
class CNNLSTM(nn.Module):
    def __init__(self, per_step_feature, num_classes=3):
        super().__init__()
        
        # 1D CNN
        self.conv1 = nn.Conv1d(in_channels=per_step_feature, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        
        # FC
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len=10, feature=8)
        x = x.permute(0, 2, 1)        # (batch, 8, 10)
        x = self.pool(self.relu(self.conv1(x)))  # (batch, 32, 5)
        x = x.permute(0, 2, 1)        # (batch, 5, 32)

        x, _ = self.lstm(x)
        x = x[:, -1, :]               # last timestep
        return self.fc(x)


# CSV 데이터 로드
df = pd.read_csv(CSV_PATH)
print("CSV shape:", df.shape)

# 라벨 분리
X = df.drop(columns=['label']).values   # (N, 80)
y = df['label'].values                  # (N,)

# CSV는 이미 시퀀스 형태로 저장되어 있으므로 reshape만 수행
num_samples = len(X)                    # N = 1039
X = X.reshape(num_samples, SEQ_LEN, PER_STEP_FEATURE)  # (N, 10, 8)


print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label distribution:", np.bincount(y)) # 클래스 분포 확인

# 표준화 
scaler = StandardScaler()
X_2d = X.reshape(-1, PER_STEP_FEATURE)  # (num_samples * SEQ_LEN, PER_STEP_FEATURE)
X_2d = scaler.fit_transform(X_2d)

# scaler 저장
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/feature_scaler.pkl")
print("Scaler saved to models/feature_scaler.pkl")

# 다시 3D로 reshape
X = X_2d.reshape(num_samples, SEQ_LEN, PER_STEP_FEATURE)

# train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 학습
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
model = CNNLSTM(PER_STEP_FEATURE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# 시각화를 위해 로그 저장
train_losses = []
val_losses = []

train_macro_f1s = []
val_macro_f1s = []

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(out, dim=1)
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(y_batch.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    train_macro_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
    train_macro_f1s.append(train_macro_f1)

    # 평가
    model.eval()
    all_val_preds = []
    all_val_labels = []
    total_val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_val_loss += loss.item() * X_batch.size(0)
            
            preds = torch.argmax(out, dim=1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(y_batch.cpu().numpy())


    avg_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)
    val_macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_macro_f1s.append(val_macro_f1)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Macro F1: {train_macro_f1:.4f} | Val Macro F1: {val_macro_f1:.4f}")

    scheduler.step()

# 시각화
# loss 비교 그래프
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("fig/train_val_loss.png")
plt.show()


# Macro F1 비교 그래프
plt.figure(figsize=(8,5))
plt.plot(train_macro_f1s, label="Train Macro F1")
plt.plot(val_macro_f1s, label="Val Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Macro F1: Train vs Val")
plt.legend()
plt.grid(True)
plt.savefig("fig/train_val_macro_f1.png")
plt.show()

# Confusion Matrix Heatmap (last epoch)
cm = confusion_matrix(all_val_labels, all_val_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("fig/confusion_matrix.png") 
plt.show()

# 모델 저장
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn_lstm_model.pth")
print("Model saved to models/cnn_lstm_model.pth")
