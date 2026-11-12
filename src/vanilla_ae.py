import torch, torch.nn as nn, torch.nn.functional as F, time
from src.data import get_loaders
from src.utils import set_seed, get_device

class VanillaAE(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x.flatten(1))
        return self.decoder(z), z

def train_vanilla_ae(epochs=30, lr=1e-3, batch_size=128, seed=42):
    set_seed(seed)
    device = get_device()
    train, val, test = get_loaders(batch_size, seed)
    model = VanillaAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train': [], 'val': []}
    best_val, best_state, best_epoch = float('inf'), None, -1
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        for images, _ in train:
            images = images.to(device)
            optimizer.zero_grad()
            reconstruction, _ = model(images)
            loss = F.mse_loss(reconstruction, images.flatten(1))
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * images.size(0)
        history['train'].append(train_loss_sum / len(train.dataset))

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for images, _ in val:
                images = images.to(device)
                reconstruction, _ = model(images)
                val_loss_sum += F.mse_loss(reconstruction, images.flatten(1)).item() * images.size(0)
        history['val'].append(val_loss_sum / len(val.dataset))

        if history['val'][-1] < best_val:
            best_val = history['val'][-1]
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
            best_epoch = epoch + 1

        train_loss = history['train'][-1]
        val_loss = history['val'][-1]
        print(f'epoch {epoch+1:02d} train {train_loss:.5f} val {val_loss:.5f}')

    model.load_state_dict(best_state)
    return model, history, (time.time() - start_time) / epochs, best_epoch, best_val
