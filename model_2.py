#   Importation des modules nécessaires
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.parallel import DataParallel

# Vérification de la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128

# taux d'apprentissage, qui contrôle la taille des pas effectués lors de la
# mise à jour des poids du modèle afin de minimiser la perte (loss)
learning_rate = 0.01

# itérations complètes sur l'ensemble des données d'entraînement que le modèle
# doit effectuer pendant l'entraînement
num_epochs = 10

transform = transforms.Compose([
    transforms.ToTensor(),  # transforme les données d'entrée (images) en tenseurs
    transforms.Normalize((0.5,), (0.5,))  # normalisation des données d'entrée
])

# Chargement des données d'entrainement et de test
train = datasets.MNIST(root='./data', train=True, download=True,
                       transform=transform)

train_loader = DataLoader(
    train,
    batch_size=batch_size,
    shuffle=True
)

print(len(train), 'images d''\'entrainement')

test = datasets.MNIST(root='./data', train=False, download=True,
                      transform=transform)

test_loader = DataLoader(
    test,
    batch_size=batch_size,
    shuffle=False
)

print(len(test), 'images de test')


class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=128 * 5 * 5, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        """
            Effectue une passe avant (forward) du réseau de neurones.
            Args:
                x (torch.Tensor): Le tenseur d'entrée de taille (batch_size, 3, 32, 32).

            Returns:
                torch.Tensor: Le tenseur de sortie de taille (batch_size, 10).
        """
        x = self.conv1(x)  # Convolue l'image par le premier filtre convolutionnel"""
        x = torch.relu(x)  # Applique une activation ReLU
        x = self.pool1(x)  # Effectue un sous-échantillonnage 2x2"""
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)  # Effectue un deuxième sous-échantillonnage 2x2"""
        x = x.view(-1, 128 * 5 * 5)
        """Aplatie le tenseur 4D en un tenseur 1D de 5000 éléments"""
        x = self.fc1(x)  # Fait une passe à travers une première couche
        # linéaire avec ReLU"""
        x = torch.relu(x)
        x = self.fc2(x)  # Fait une passe à travers une seconde couche linéaire
        # pour donner la sortie"""
        return x


def train(model, train_loader, optimizer, criterion, device):
    """ Entraîne le modèle sur les données d'entraînement.

    Args:
        model (nn.Module): Le modèle à entraîner.
        train_loader (DataLoader) : Le DataLoader pour les données
            d'entraînement.
        optimizer (torch.optim.Optimizer): L'optimiseur pour ajuster
            les poids du modèle.
        criterion (torch.nn.modules.loss._Loss): La fonction de perte
            pour évaluer la sortie du modèle.
        device (torch.device): Le dispositif sur lequel le modèle est entraîné.

    Returns :
        float : Le pourcentage de précision sur les données d'entraînement.
    """
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss = {loss.item()}")

    accuracy = 100 * correct / total
    return accuracy


def test(model, test_loader, device):
    """ Évalue le modèle sur les données de test
    :param model: Le modèle de réseau de neurones à impulsion
    :param test_loader: le DataLoader pour les données de test
    :param device: le dispositif sur lequel le modèle est évalué
    :return: le pourcentage de précision sur les données de test
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    # Initialisation du modèle, optimiseur et fonction de perte
    model = SNN()

    # Utilisation de DataParallel si plusieurs GPU sont disponibles
    if torch.cuda.device_count() > 1:
        print(f"Utiliser {torch.cuda.device_count()} GPU(s)")
        model = DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Entraînement du modèle
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")
        accuracy = train(model, train_loader, optimizer, criterion, device)
        print(f"Taux de réussite:{accuracy:.2f}%")

    # Évaluation du modèle sur les données de test
    test_accuracy = test(model, test_loader, device)
    print()
    print(f"Performance globale: {test_accuracy:.2f}%")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "snn.pth")


if __name__ == "__main__":
    debut = time.time()
    main()
    fin = time.time()
    temps = fin - debut
    print('Le temps total est de ' + str(int(temps)) + 's')
