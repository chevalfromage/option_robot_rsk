import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Définir le modèle
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Une couche linéaire qui prend 6 entrées et produit 3 sorties
        self.fc = nn.Linear(6, 3)

    def forward(self, x):
        # Passage avant : simplement appliquer la couche linéaire
        return self.fc(x)

# 2. Créer le modèle
model = SimpleNN()

# 3. Définir une fonction de perte et un optimiseur
criterion = nn.MSELoss()  # erreur quadratique moyenne
optimizer = optim.SGD(model.parameters(), lr=0.01)  # descente de gradient

# 4. Exemple de données
# 4 exemples, 6 caractéristiques chacun
inputs = torch.randn(100, 6)
print(inputs)
# 4 sorties attendues, 3 valeurs chacune
def f(x):
    # x : (N, 6)
    y = (
        5*x[:, 0]
        + 2*x[:, 1]
        + x[:, 2]
        + 4*x[:, 3]
        + x[:, 4]
        - 2*x[:, 5]
    )
    # 3 sorties identiques
    return torch.stack([y, y, y], dim=1)  # (N, 3)

targets = f(inputs)
print(targets)

# 5. Entraînement minimal : une itération

epoch=10000
for k in range(epoch):
    optimizer.zero_grad()        # réinitialiser les gradients
    outputs = model(inputs)      # calculer la sortie du modèle
    loss = criterion(outputs, targets)  # calculer la perte
    loss.backward()              # calculer les gradients
    optimizer.step()             # mettre à jour les poids
    print(f"{k}: loss : {loss.item()}")

# print(targets)

# print("Sorties du modèle :", outputs)
print("Perte :", loss.item())


inputs = torch.randn(1, 6)
print(inputs)
targets = f(inputs)
print(targets)

print(model(inputs))
