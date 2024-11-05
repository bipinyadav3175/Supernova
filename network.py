import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Chess import Chess

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    def __init__(self, action_space=4096, eps=0.05):
        self.action_space = action_space
        self.eps_critic = eps
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=14, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(128*8*8, 512)
        self.fc2 = nn.Linear(512, action_space)

        self.fc3 = nn.Linear(128*8*8, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x, mask):
        if not isinstance(mask, (np.ndarray, torch.Tensor)):
            raise ValueError("Invalid mask type")
        # print(mask.size, self.action_space, x.size(0))
        # if (isinstance(mask, torch.Tensor) and mask.size(0) != x.size(0)) or mask.size != self.action_space:
        #     raise ValueError("Mask size is different from the logits size")
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask,dtype=torch.float).to(device)
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1) # Flatten

        logits = F.relu(self.fc1(x))
        logits = self.fc2(logits)
        
        logits = logits + (1-mask) * -1e10
        probs = F.softmax(logits, dim=1)

        value = F.relu(self.fc3(x))
        value = self.fc4(value)

        return probs, value
    
    
    @staticmethod
    def encode_board(board, player):
        piece_to_one_hot = {
            '_' :[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Pw' : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Pb' : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Rw' : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'Rb' : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'Bw' : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'Bb' : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'Nw' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'Nb' : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'Qw' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'Qb' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'Kw' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'Kb' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }       
        result = []
        for row in board:
            result.append([piece_to_one_hot[elem] for elem in row])
        player_channel = torch.ones((8, 8), dtype=torch.float).to(device) if player == 'w' else torch.zeros((8, 8), dtype=torch.float).to(device)
        player_channel = player_channel.unsqueeze(dim=2)
        result = torch.tensor(result, dtype=torch.float).to(device)
        result = torch.cat((result, player_channel), dim=2)
        return result.permute(2, 0, 1) # 14x8x8


if __name__ == "__main__":
    model = Model()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 8490145