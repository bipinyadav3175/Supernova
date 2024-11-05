from train import train
from network import Model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    model = Model(4096 ,eps=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    train(optimizer, model, num_episodes=1, num_simulations=10, batch_size=4, path="supernova-v0", graph_freq=1)