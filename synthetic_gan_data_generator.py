import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import ipaddress
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# GAN-based Synthetic Network Log Generator
class SyntheticGANDataGenerator:
    def __init__(self, input_dim=100, output_dim=10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generator = Generator(input_dim, output_dim)
        self.discriminator = Discriminator(output_dim)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()

    def train(self, real_data, epochs=100, batch_size=64):
        for epoch in range(epochs):
            for i in range(0, len(real_data), batch_size):
                batch = real_data[i:i + batch_size]
                batch_size = len(batch)

                # Train Discriminator
                self.d_optimizer.zero_grad()
                label_real = torch.ones(batch_size, 1)
                label_fake = torch.zeros(batch_size, 1)
                output_real = self.discriminator(torch.FloatTensor(batch))
                d_loss_real = self.criterion(output_real, label_real)
                noise = torch.randn(batch_size, self.input_dim)
                fake_data = self.generator(noise)
                output_fake = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(output_fake, label_fake)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # Train Generator
                self.g_optimizer.zero_grad()
                output_fake = self.discriminator(fake_data)
                g_loss = self.criterion(output_fake, label_real)
                g_loss.backward()
                self.g_optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

    def generate_synthetic_logs(self, num_entries):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_entries, self.input_dim)
            synthetic_data = self.generator(noise).numpy()
        return synthetic_data

    def generate_dataset(self, num_entries=10000, anomaly_percentage=0.05, start_time=None):
        if start_time is None:
            start_time = datetime.now()

        # Generate synthetic data using GAN
        synthetic_data = self.generate_synthetic_logs(num_entries)

        # Convert synthetic data to network logs
        logs = []
        for i in range(num_entries):
            timestamp = start_time + timedelta(seconds=i)
            source_ip = str(ipaddress.IPv4Address(random.randint(1, 2**32 - 1)))
            destination_ip = str(ipaddress.IPv4Address(random.randint(1, 2**32 - 1)))
            protocol = random.choice(['TCP', 'UDP', 'HTTP', 'FTP'])
            status = 'SUCCESS' if random.random() > 0.1 else 'FAILED'
            error_code = 'NONE' if status == 'SUCCESS' else random.choice(['TIMEOUT', 'CONNECTION_REFUSED', 'HOST_UNREACHABLE'])
            bytes_sent = int(synthetic_data[i][0] * 10000)
            bytes_received = int(synthetic_data[i][1] * 10000)
            latency = int(synthetic_data[i][2] * 200)

            logs.append({
                'timestamp': timestamp,
                'source_ip': source_ip,
                'destination_ip': destination_ip,
                'protocol': protocol,
                'status': status,
                'error_code': error_code,
                'bytes_sent': bytes_sent,
                'bytes_received': bytes_received,
                'latency': latency
            })

        return pd.DataFrame(logs)

def main():
    # Example usage
    generator = SyntheticGANDataGenerator()
    real_data = np.random.rand(1000, 10)  # Example real data
    generator.train(real_data, epochs=100)
    df = generator.generate_dataset(num_entries=10000)
    print(f"Generated {len(df)} synthetic network logs")
    df.to_csv('data/synthetic_network_logs.csv', index=False)

if __name__ == "__main__":
    main() 