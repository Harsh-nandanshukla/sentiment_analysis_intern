import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training started on device:", device)

# ----------------- Load Data -----------------
with open('padded_sequences.pkl', 'rb') as f:
    padded_sequences = pickle.load(f)

labels = np.load('labels.npy')
embedding_matrix = np.load('embedding_matrix.npy')

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train_tensor = torch.tensor(X_train).long()
y_train_tensor = torch.tensor(y_train).long()
X_test_tensor = torch.tensor(X_test).long()
y_test_tensor = torch.tensor(y_test).long()

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# ----------------- GRU Model -----------------
class GRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, output_dim=2, n_layers=1, bidirectional=True, dropout=0.3):
        super(GRUClassifier, self).__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.fc(self.dropout(hidden))

# ----------------- Training -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUClassifier(embedding_matrix).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(50):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        train_correct += (preds == y_batch).sum().item()
        train_total += y_batch.size(0)

    train_acc = train_correct / train_total
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # Evaluate on test data
    model.eval()
    test_loss, test_correct, test_total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)

            test_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)

    test_acc = test_correct / test_total
    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(test_acc)

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_losses[-1]:.4f} | Test Acc: {test_acc:.4f}")

# ----------------- Plotting -----------------
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
plt.savefig("gru_training_plot.png")
plt.show()
