# Importation des modules

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# Definition de la transformation à appliquer sur les images d'entrée avec
# Pytorch torchvision.transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Transforme l'image en un tenseur PyTorch
    transforms.Normalize((0.1307,), (0.3081,))  # Normalise l'image avec une
    # moyenne et un ecart type pr?d?finis
])

# Chargement du dataset MNIST avec PyTorch torchvision.datasets
# Et recuperation des donnees d'entrainement

train = datasets.MNIST(root='data', train=True, download=True,
                       transform=transform)
print(len(train), 'images d''\'entrainement')
# print(len(train_dataset)) pour r?cup?rer le nombre de points (images)
# 60000 images utilis?es pour l'entrainement ici

# Dataset entrainement
test = datasets.MNIST(root='./data', train=False, download=True,
                      transform=transform)
print(len(test), 'images de test')
# 300 images utilis?es pour le test


# Definition du DataLoader avec PyTorch torch.utils.data.DataLoader
chargeur_train = DataLoader(train, batch_size=128,
                            shuffle=True) 
# Chargement de donnees pour l'entrainement

chargeur_test = DataLoader(test, batch_size=128, shuffle=False)
# Chargement de donnees pour le test


# Definition du mod?le avec PyTorch torch.nn.Module
class SNN(nn.Module):

    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        # Couche lin?aire pour passer de 784 ? 256 neurones
        self.fc2 = nn.Linear(256, 128)
        # Couche lin?aire pour passer de 256 ? 128 neurones
        self.fc3 = nn.Linear(128, 10)
        # Couche lin?aire pour passer de 128 ? 10 neurones
        # (10 etiquettes possibles)

    def forward(self, x):
        """
        Propagation avant du mod?le
        :param x: les donn?es d'entr?e
        :type x: PyTorch Tensor
        :return: valeurs de sortie
        :rtype: PyTorch Tensor
        """
        x = x.view(-1, 784)  # Aplatit l'entr?e, n?cessaire, car la premiere
        # couche est lin?aire
        x = nn.functional.relu(self.fc1(x))  # 1re Couche cach?e avec une
        # fonction d'activation ReLU
        x = nn.functional.relu(self.fc2(x))  # 2e Couche cach?e avec une
        # fonction d'activation ReLU
        x = self.fc3(x)  # Couche de sortie sans fonction d'activation, car
        # la fonction de perte CrossEntropyLos inclut Softmax
        return x


# Definition du mod?le, de la fonction de perte et de l'optimiseur
model = SNN()  # Instancie le mod?le d?fini pr?c?demment
criterion = nn.CrossEntropyLoss()  # Fonction de perte CrossEntropyLoss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                      weight_decay=1e-4)  # Optimisation, maj des
# poids ? l'aide de la fonction de descente de gradient stochastique SGD

# Entrainement du mod?le
print('Perte par epoch')
pertes_epochs = []  # initialisation de la liste des pertes dans les epochs
# qui nous servira ? faire la courbe des pertes dans les epochs

debut = time.time()  # Pour connaitre la performance en temps

for epoch in range(10):
    courbe = []
    perte_entrainement = 0.0
    for i, donnees in enumerate(chargeur_train, 0):
        entrees, etiquettes = donnees
        entrees = entrees.view(entrees.size(0), -1)  # Aplatir les images
        # D'entr?e. Dans cette partie, nous transformons les images en un
        # vecteur 1D. Cela consiste ? prendre tous les pixels de l'image
        # et ? les ranger bout ? bout dans un vecteur.

        optimizer.zero_grad()  # Reinitialise les gradients
        sorties = model(entrees)  # Propagation avant
        perte = criterion(sorties, etiquettes)  # Calcul de la perte
        perte.backward()  # Propagation arriere pour calculer les gradients
        optimizer.step()  # Mise ? jour des poids et des biais
        perte_entrainement += perte.item()
    pertes_epochs.append(perte_entrainement / len(chargeur_train))
    # Ajout de perte de l'epoch courant a la liste des pertes dans les epochs
    print(f'[Epoch {epoch + 1}] Perte: {pertes_epochs[-1]:.3f}')

# Test du mod?le
correct = 0
total = 0
model.eval()  # Desactivation du dropout et BatchNorm

with torch.no_grad():
    for donnees in chargeur_test:
        entrees, etiquettes = donnees
        entrees = entrees.view(entrees.size(0), -1)  # Aplatissement
        # des images d'entr?e
        sorties = model(entrees)  # Propagation avant
        _, predicted = torch.max(sorties.data, 1)  # Etiquette pr?dite
        total += etiquettes.size(0)  # Compte le nombre total d'images test?es
        correct += (predicted == etiquettes).sum().item()  # Compte le nombre
        # d'images correctement categorisees
print(f'Precision du r?seau sur les 10000 images de test : '
      f'{100 * correct / total:.2f} %')
fin = time.time()
temps = fin - debut
numero_epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Creation d'un graphique en utilisant Matplotlib
plt.plot(numero_epoch, pertes_epochs, )
plt.title('Graphique de la perte par Epoch')
plt.xlabel('Epoch')
plt.ylabel('Perte')
# Afficher le graphique
plt.show()
plt.savefig('graphique_perte.png')
print('Le temps total est de ' + str(int(temps)) + 's')
