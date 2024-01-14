import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint

# Define a Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )
        self.image_shape = image_shape

    def forward(self, z):
        return self.main(z).view(z.size(0), *self.image_shape)

# Define a Discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.image_shape = image_shape

    def forward(self, img):
        return self.main(img.view(img.size(0), -1))

# Define the GAN model
class GAN(pl.LightningModule):
    def __init__(self, latent_dim, image_shape):
        super(GAN, self).__init__()

        self.automatic_optimization = False

        self.latent_dim = latent_dim
        self.image_shape = image_shape

        self.generator = Generator(latent_dim, image_shape)
        self.discriminator = Discriminator(image_shape)

        self.criterion = nn.BCELoss()

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return [optimizer_G, optimizer_D], []

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)

        # Adversarial ground truth
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # Train Generator
        if batch_idx % 2 == 0:  # Example: alternate between generator and discriminator
            z = torch.randn(batch_size, self.latent_dim)
            gen_imgs = self.generator(z)
            g_loss = self.criterion(self.discriminator(gen_imgs), valid)
            self.manual_backward(g_loss)  # Manually compute gradients
            generator_optimizer = self.optimizers()[0]  # Access generator optimizer by name
            generator_optimizer.step()  # Step the generator optimizer
            self.log('loss', g_loss)
            return {"loss": g_loss}

        # Train Discriminator
        else:
            real_loss = self.criterion(self.discriminator(real_imgs.view(batch_size, -1)), valid)
            z = torch.randn(batch_size, self.latent_dim)
            fake_imgs = self.generator(z)
            fake_loss = self.criterion(self.discriminator(fake_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            discriminator_optimizer = self.optimizers()[0]  # Access generator optimizer by name
            discriminator_optimizer.step()  # Step the generator optimizer
            self.log('loss', d_loss)
            return {"loss": d_loss}

    def validation_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)

        # Adversarial ground truth
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim)
        gen_imgs = self.generator(z)

        # Calculate generator loss and discriminator loss
        g_loss = self.criterion(self.discriminator(gen_imgs), valid)
        real_loss = self.criterion(self.discriminator(real_imgs.view(batch_size, -1)), valid)
        fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # Log the losses as validation metrics
        self.log('val_loss', g_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', d_loss, on_step=False, on_epoch=True, prog_bar=True)

# Define a PyTorch Lightning DataModule
class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        datasets.MNIST(root="data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST(root="data", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(root="data", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Main function to train the GAN
def main():
    pl.seed_everything(42)
    
    # Define hyperparameters
    latent_dim = 100
    image_shape = (1, 28, 28)  # MNIST image shape
    batch_size = 64
    epochs = 100

    # Create a GAN model
    gan = GAN(latent_dim, image_shape)

    # Create a Lightning DataModule
    datamodule = SimpleDataModule(batch_size)

    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="gan-{epoch:02d}-{val_loss:.2f}",
        save_last=True,  # Save the latest checkpoint
        monitor="loss",  # Monitor validation loss
        mode="min",  # Save the checkpoint with the lowest validation loss
        save_top_k=1,  # Save only the best checkpoint
    )


    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        # gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback],  # Pass the callback here
    )


    # Start training
    trainer.fit(gan, datamodule)

if __name__ =="__main__":
    main()