import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ==============================
# 1. Загрузка данных
# ==============================

df = pd.read_csv("dataset.csv")

# Укажите названия целевых колонок
target_columns = ["CTR", "Переходы"]

X = df.drop(columns=target_columns)
y = df[target_columns].values

# Если есть категориальные признаки — кодируем
X = pd.get_dummies(X, drop_first=True)

# Train / Validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабирование
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)


# ==============================
# 2. PyTorch Dataset
# ==============================

class CTRDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = CTRDataset(X_train, y_train)
test_dataset = CTRDataset(X_test, y_test)

# В будующем будут оргомные данные поэтому оставляем num_workers=2
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==============================
# 3. Модель
# ==============================

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RegressionModel(input_dim=X_train.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# 4. Обучение
# ==============================

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss/len(train_loader):.4f} "
          f"Test Loss: {val_loss/len(test_loader):.4f}")

# ==============================
# 5. Сохранение модели в .pkl
# ==============================

save_dict = {
    "model_state_dict": model.state_dict(),
    "y_scaler": y_scaler,
    "X_scaler": X_scaler,
    "input_dim": X_train.shape[1],
    "feature_columns": X.columns.tolist()
}

torch.save(save_dict, "ctr_regression_model.pth")

print("Модель сохранена как ctr_regression_model.pth")
