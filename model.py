from blocks import Generator, Discriminator
from utils import weights_init
import torch.optim as optim
import torch.nn as nn
import torch
from utils import update_req_grad, not_equal_loss
import matplotlib.pyplot as plt
import random


class GAN:
    def __init__(self, num_epochs, device, dataloader, lr=0.0002, beta1=0.5, emb_size=100):
        self.num_epochs = num_epochs
        self.device = device
        self.dataloader = dataloader
        self.gen = Generator(emb_size=emb_size).to(device)
        self.disc = Discriminator(device).to(device)
        weights_init(self.gen)
        weights_init(self.disc)

        self.optimizerD = optim.Adam(self.disc.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))

        self.emb_size = emb_size

    def train(self):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, self.num_epochs + 1):
            gen_loss_cum = 0
            disc_loss_cum = 0
            num_batches = 0
            for i, data in enumerate(self.dataloader):
                num_batches += 1

                imgs, labels = data
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                batch = imgs.shape[0]

                # updating disc
                update_req_grad([self.disc], True)
                self.disc.zero_grad()
                disc_loss = criterion(self.disc(imgs), labels)
                z = torch.rand((batch, self.emb_size)).to(self.device)
                disc_loss += criterion(self.disc(self.gen([z, labels]).detach()),
                                       torch.full((batch,), 10,
                                                  dtype=torch.long,
                                                  device=self.device))
                disc_loss /= 4
                disc_loss.backward()
                self.optimizerD.step()
                disc_loss_cum += disc_loss.item()

                # updating generator
                update_req_grad([self.disc], False)
                self.gen.zero_grad()

                z = torch.rand((batch, self.emb_size)).to(self.device)
                output = self.gen([z, labels])
                gen_loss = criterion(self.disc(output), labels)

                gen_loss_cum += gen_loss.item()
                gen_loss.backward()
                self.optimizerG.step()

            gen_loss_avg = gen_loss_cum / num_batches
            disc_loss_avg = disc_loss_cum / num_batches
            print(f"Epoch: {epoch} Generator loss: {gen_loss_avg} Discriminator loss: {disc_loss_avg}")

            with torch.no_grad():
                fig, axes = plt.subplots(2, 5)
                data = next(iter(self.dataloader))
                batch = data[0].shape[0]
                indexes = [random.randint(0, batch) for i in range(5)]
                z = torch.rand((5, self.emb_size)).to(self.device)
                imgs = data[0][indexes]
                labels = data[1][indexes]

                imgs_fake = self.gen([z, labels.to(self.device)]).detach().cpu()
                for i in range(5):
                    axes[0][i].imshow(imgs[i].view(64, 64))
                    axes[0][i].axis('off')
                    axes[1][i].imshow(imgs_fake[i].view(64, 64))
                    axes[1][i].axis('off')
                plt.show()

    def load(self):
        pass
