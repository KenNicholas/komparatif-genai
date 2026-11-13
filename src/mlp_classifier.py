import torch, torch.nn as nn, torch.nn.functional as F
from src.data import get_loaders
from src.utils import set_seed, get_device

class MLPClassifier(nn.Module):
    def __init__(self, hidden1=256, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(28*28, hidden1), nn.ReLU(),
            nn.Linear(hidden1, 64), nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 10)
    def forward(self, x):
        z = self.features(x.flatten(1))
        return self.classifier(self.dropout(z)), z

def train_mlp(hidden1=256, dropout=0.0, epochs=20, lr=1e-3, batch_size=128, seed=42):
    set_seed(seed)
    device = get_device()
    train, val, test = get_loaders(batch_size, seed)
    model = MLPClassifier(hidden1, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for images, labels in train:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val:
                images = images.to(device)
                labels = labels.to(device)
                logits, _ = model(images)
                correct += (logits.argmax(1) == labels).sum().item()
        val_acc = correct / len(val.dataset)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        print(f'Epoch {epoch+1:02d} : val_acc {val_acc:.4f}')
    model.load_state_dict(best_state)
    return model, best_val
