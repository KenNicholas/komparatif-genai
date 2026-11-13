import torch, torch.nn as nn, time
from src.data import get_loaders
from src.utils import set_seed, get_device

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 28*28), nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28*28, 512), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x.flatten(1))

def train_vanilla_gan(epochs=50, lr=2e-4, batch_size=128, seed=42):
    set_seed(seed)
    device = get_device()
    train, val, test = get_loaders(batch_size, seed)
    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    history = {'loss_G': [], 'loss_D': [], 'D_real': [], 'D_fake': [], 'grad_norm_G': []}
    start_time = time.time()

    for epoch in range(epochs):
        G.train(); D.train()
        lossG_sum = lossD_sum = dreal_sum = dfake_sum = gnorm_sum = 0
        batch_count = 0
        for images, _ in train:
            images = images.to(device)
            images = images * 2 - 1
            n = images.size(0)

            opt_D.zero_grad()
            real_labels = torch.full((n, 1), 0.9, device=device)
            fake_labels = torch.zeros((n, 1), device=device)
            d_real = D(images)
            loss_real = bce(d_real, real_labels)

            z = torch.randn(n, 100, device=device)
            fake_images = G(z)
            d_fake = D(fake_images.detach())
            loss_fake = bce(d_fake, fake_labels)
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            opt_G.zero_grad()
            z = torch.randn(n, 100, device=device)
            fake_images = G(z)
            d_fake_for_G = D(fake_images)
            target = torch.ones((n, 1), device=device)
            loss_G = bce(d_fake_for_G, target)
            loss_G.backward()
            grad_norm_G = sum(p.grad.norm().item() for p in G.parameters() if p.grad is not None)
            opt_G.step()

            lossG_sum += loss_G.item()
            lossD_sum += loss_D.item()
            dreal_sum += d_real.mean().item()
            dfake_sum += d_fake.mean().item()
            gnorm_sum += grad_norm_G
            batch_count += 1

        history['loss_G'].append(lossG_sum / batch_count)
        history['loss_D'].append(lossD_sum / batch_count)
        history['D_real'].append(dreal_sum / batch_count)
        history['D_fake'].append(dfake_sum / batch_count)
        history['grad_norm_G'].append(gnorm_sum / batch_count)
        print(f'Epoch {epoch+1:02d} : lossG {history["loss_G"][-1]:.4f} , lossD {history["loss_D"][-1]:.4f} , D(x) {history["D_real"][-1]:.3f} , D(G(z)) {history["D_fake"][-1]:.3f}')

    sec_per_epoch = (time.time() - start_time) / epochs
    return G, D, history, sec_per_epoch
