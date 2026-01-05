
import torch.nn as nn

#def du modèle
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(),
            nn.Linear(128, 3)  # couche de sortie
        )

    def forward(self, x):
        return self.net(x)