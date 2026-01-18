
import torch.nn as nn
from rsk_neural_simulator.data.preparation_datas import MEMORY_WINDOW

#def du modèle
class SimpleNN3(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128), #entrées
            nn.Linear(128,256), 
            nn.Dropout(0.3),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 4)  # couche de sortie
        )

    def forward(self, x):
        return self.net(x)
    


#def du modèle
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 256),  #entrées
            nn.Dropout(0.3),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 4)  # couche de sortie
        )

    def forward(self, x):
        return self.net(x)


### NOUVEAU MLP QUI PREND LA MEMOIRE DES DERIVEES précédentes

base_features = 7 # orders.dx, orders.dy, orders.dtheta, derivee.x, derivee.y, derivee.theta_cos, derivee.theta_sin
input_dim = base_features + ( 4 * MEMORY_WINDOW )  # 4 dérivées (x, y, cos, sin) par instant dans la mémoire
    
#def du modèle
class SimpleNNMemory(nn.Module):
    def __init__(self, input_dimension=input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dimension, 256),  #entrées
            nn.Dropout(0.3),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 4)  # couche de sortie
        )

    def forward(self, x):
        return self.net(x)