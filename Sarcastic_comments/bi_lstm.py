import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Load data
with open("padded_sequences.pkl", "rb") as f:
    X = pickle.load(f)

y = np.load("labels.npy")
embedding_matrix = np.load("embedding_matrix.npy")

X_tensor = torch.tensor(X).long()
y_tensor = torch.tensor(y).long()

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42)

# Dataloaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(BiLSTM, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True,
            batch_first=True
        )

        self.fc1 = nn.Linear(128 * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Last time step
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        return self.out(out)


model = BiLSTM(embedding_matrix).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
for epoch in range(1, 21):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    train_acc = correct / total
    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # Evaluation
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    test_acc = correct / total
    test_losses.append(total_loss / len(test_loader))
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch:02d} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_losses[-1]:.4f} | Test Acc: {test_acc:.4f}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(test_accuracies, label="Test Acc")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("bilstm_training_plot.png")
plt.show()
