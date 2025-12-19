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



# ----------------------------
# 1️⃣ 하이퍼파라미터
# ----------------------------
SEQ_LEN = 10  # 시퀀스 길이 (CSV 시퀀스 길이)
FEATURE_DIM = 80  # sequence 하나당 feature 개수
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

CSV_PATH = "dataset/sequence_dataset.csv"

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", DEVICE)

# ----------------------------
# 2️⃣ Dataset 정의
# ----------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # 분류라 long
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------------------------
# 3️⃣ CNN-LSTM 모델 정의
# ----------------------------
class CNNLSTM(nn.Module):
    def __init__(self, feature_dim, seq_len, num_classes=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        
        # 1D CNN
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        
        # FC
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # -> (batch, feature_dim, seq_len) for CNN1d
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # -> (batch, channels, seq_len/2)
        x = x.permute(0, 2, 1)  # -> (batch, seq_len/2, channels) for LSTM
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # 마지막 타임스텝
        out = self.fc(out)
        return out

# ----------------------------
# 4️⃣ CSV 데이터 로드
# ----------------------------
df = pd.read_csv(CSV_PATH)
print("CSV shape:", df.shape)

# 라벨 분리
y = df['label'].values
X = df.drop(columns=['label']).values
print("X shape after drop label:", X.shape)

# 특징 개수 80개 더블체크 후 출력
FEATURE_DIM = X.shape[1]
print("Detected FEATURE_DIM:", FEATURE_DIM)

# 시퀀스로 reshape: (num_samples, seq_len, feature_dim)
num_samples = len(X) // SEQ_LEN
X = X[:num_samples*SEQ_LEN].reshape(num_samples, SEQ_LEN, FEATURE_DIM)
y = y[:num_samples*SEQ_LEN:SEQ_LEN]  # 각 시퀀스 마지막 라벨만

# 표준화 - feature 값이 전부 상대값이라 필요없을듯
#scaler = StandardScaler()
#X = X.reshape(-1, FEATURE_DIM)
#X = scaler.fit_transform(X)
#X = X.reshape(num_samples, SEQ_LEN, FEATURE_DIM)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #stratify=y 는 클래스 비율이 불균형해서 제외

train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 5️⃣ 모델 학습
# ----------------------------
model = CNNLSTM(FEATURE_DIM, SEQ_LEN).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item() * X_batch.size(0)
    
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # 평가
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_val_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    avg_val_loss = total_val_loss / len(test_loader.dataset)
    val_losses.append(avg_val_loss)
    acc = correct / total
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Test Acc: {acc:.4f}")


plt.figure(figsize=(8,5))
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")  # 파일로 저장
plt.show()
# ----------------------------
# 6️⃣ 모델 저장
# ----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn_lstm_model.pth")
print("Model saved to models/cnn_lstm_model.pth")
