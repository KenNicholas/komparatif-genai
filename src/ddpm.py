import torch, torch.nn as nn, torch.nn.functional as F, time, math
from src.data import get_loaders
from src.utils import set_seed, get_device

class TimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.linear = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.linear(emb)

class UNetSmall(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.time_embed = TimeEmbedding(64)
        self.down1 = nn.Conv2d(1, channels, 3, padding=1)
        self.down2 = nn.Conv2d(channels, channels, 4, 2, 1)
        self.mid = nn.Conv2d(channels, channels, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(channels, channels, 4, 2, 1)
        self.out = nn.Conv2d(channels, 1, 3, padding=1)
        self.act = nn.SiLU()
    def forward(self, x, t):
        temb = self.time_embed(t)[:, :, None, None]
        h = self.act(self.down1(x))
        h = self.act(self.down2(h))
        h = self.act(self.mid(h) + temb)
        h = self.act(self.up1(h))
        return self.out(h)

def ddpm_schedule(T=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
    beta = torch.linspace(beta_start, beta_end, T, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

def train_ddpm(epochs=40, lr=2e-4, batch_size=128, seed=42, T=1000):
    set_seed(seed)
    device = get_device()
    train, val, test = get_loaders(batch_size, seed)
    model = UNetSmall(64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    beta, alpha, alpha_bar = ddpm_schedule(T, device=device)
    history = {'train_loss': [], 'val_loss': []}
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_sum = 0
        for images, _ in train:
            images = images.to(device) * 2 - 1
            t = torch.randint(0, T, (images.size(0),), device=device)
            noise = torch.randn_like(images)
            ab = alpha_bar[t][:, None, None, None]
            x_t = torch.sqrt(ab) * images + torch.sqrt(1 - ab) * noise
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            train_sum += loss.item() * images.size(0)
        history['train_loss'].append(train_sum / len(train.dataset))

        model.eval()
        val_sum = 0
        with torch.no_grad():
            for images, _ in val:
                images = images.to(device) * 2 - 1
                t = torch.randint(0, T, (images.size(0),), device=device)
                noise = torch.randn_like(images)
                ab = alpha_bar[t][:, None, None, None]
                x_t = torch.sqrt(ab) * images + torch.sqrt(1 - ab) * noise
                pred = model(x_t, t)
                val_sum += F.mse_loss(pred, noise).item() * images.size(0)
        history['val_loss'].append(val_sum / len(val.dataset))
        print(f'Epoch {epoch+1:02d} : train {history["train_loss"][-1]:.5f} , val {history["val_loss"][-1]:.5f}')

    sec_per_epoch = (time.time() - start_time) / epochs
    return model, history, sec_per_epoch, (beta, alpha, alpha_bar)
