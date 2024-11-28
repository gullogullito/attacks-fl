import torch
import torch.nn as nn
from flex.data import Dataset, FedDatasetConfig, FedDataDistribution
from torchvision import datasets, transforms
from flex.pool import init_server_model, FlexPool
from flex.model import FlexModel
from flex.pool import deploy_server_model
from flex.pool import FlexPool
from torchvision.models import resnet18
from torch.utils.data import TensorDataset
import copy, random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Subset

#   First we select cuda (if posible), for efficiency purposes
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on %s" % device)
torch.cuda.empty_cache()

#   Usual transformations to pytorch mnist
mnist_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
)

def get_dataset():
    #   We create our dataset ir order to federate it
    train_data = datasets.MNIST( root=".", train=True, download=True, transform=mnist_transforms)
    test_data = datasets.MNIST( root=".", train=False, download=True, transform=mnist_transforms)
    #indices = np.random.permutation(len(train_data))
    #size = len(train_data)
    #train_data = Subset(train_data, indices[:size//6])

    #   Process of federating our data
    fed_train_data = Dataset.from_torchvision_dataset(train_data)
    fed_test_data = Dataset.from_torchvision_dataset(test_data)

    #   Creating the Federated Distribution
    config = FedDatasetConfig(seed=0)
    config.replacement = False
    config.n_nodes = 2

    flex_dataset = FedDataDistribution.from_config(fed_train_data, config)

    #   Assign test data --> to server_id
    server_id = "server"
    flex_dataset[server_id] = fed_test_data

    return flex_dataset

#   Next step is to define our model, we'll use a simple convolutional NN
class SimpleNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(14*14*64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        return self.fc(x)

#   Initialization of the server model, with its hyperparameters
@init_server_model
def build_server_model():

    server_flex_model = FlexModel()

    server_flex_model["model"] = SimpleNet().to(device)
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    server_flex_model["optimizer_func"] = torch.optim.Adam
    server_flex_model["optimizer_kwargs"] = {}

    return server_flex_model

#   We create the client/server pool and print the config
@deploy_server_model
def copy_server_model_to_clients(server_flex_model: FlexModel):
    new_flex_model = FlexModel()
    new_flex_model["model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["server_model"] = copy.deepcopy(server_flex_model["model"])
    new_flex_model["criterion"] = copy.deepcopy(server_flex_model["criterion"])
    new_flex_model["optimizer_func"] = copy.deepcopy(server_flex_model["optimizer_func"])
    new_flex_model["optimizer_kwargs"] = copy.deepcopy(server_flex_model["optimizer_kwargs"])
    return new_flex_model

def getTrainedModel(client, client_data):
    model = client["model"]
    criterion = client["criterion"]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-7)
    model.train()
    model = model.to(device)

    dataloader = DataLoader(client_data, batch_size=32, shuffle=True)

    for epoch in range(20):
      for inputs, labels in dataloader:
          inputs = inputs.to(device)
          labels = labels.to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

    return model


#   For this scenario, we assume that the attacker has access to data which is similar to the
#   target model's training data. To simulate this, we'll flip the values of 15% randomly
#   selected features, to obtain a "noisy" dataset to train our shadow models.

def saltPepper(img, prob=0.5):
    noisy_img = img.clone()
    num_pixels = img.numel()
    num_noisy_pixels = int(prob * num_pixels)

    # Generar índices aleatorios y asignarles valor 0 o 1
    for _ in range(num_noisy_pixels):
        idx = random.randint(0, num_pixels - 1)
        noisy_img.view(-1)[idx] = 1.0 if random.random() < 0.5 else 0.0

    return noisy_img

def createShadowDataset(dataset):

    indices = np.random.permutation(len(dataset))
    size = len(dataset)
    dIn = [dataset[i] for i in indices[:size//2]]
    dOut = [dataset[i] for i in indices[size//2:size]]

    return dIn, dOut

def train_shadow_model( dIn, nShadowModels = 30):

    shadows = []

    for i in range(nShadowModels):

        model = SimpleNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-7)

        trainloader = torch.utils.data.DataLoader(dIn, batch_size=32, shuffle=True)


        for epoch in range(15):
            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        shadows.append(model)

        correct = 0
        total = 0

        model.eval()
        with torch.no_grad():
            for inputs, labels in trainloader:
              inputs = inputs.to(device)
              labels = labels.to(device)
              outputs = model(inputs)
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
        print(f'Accuracy of the shadow model: {100 * correct / total}%')

    return shadows


#   Custom class for the adversary dataset, we need to create a custom class because we need to store the
#   probabilities vector, the class and the IN/OUT value
class AdversaryDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.target[idx]

        return x, y

    def __iter__(self):
        return zip(self.features, self.target)

#   We need to build a dataset where:
#   1. For each sample in dIn, we compute model(x) = y
#   2. Add to the dataset the data-point [v(vector of probabilities for each class), y, IN]
#
#   Same for dOut but we change IN for OUT
def getAdversaryDataset(shadowModels, dIn, dOut, c):
    features = []
    target = []

    indices1 = [i for i, (_, label) in enumerate(dIn) if label == c]
    indices2 = [i for i, (_, label) in enumerate(dOut) if label == c]

    dataset1 = Subset(dIn, indices1)
    dataset2 = Subset(dOut, indices2)

    for model in shadowModels:
        model.eval()

        dInLoader = torch.utils.data.DataLoader(dataset1, batch_size=32, shuffle=True)
        for images, labels in dInLoader:
            images = images.to(device)
            output = model(images)
            for o in output:
                features.append(o.detach().cpu().numpy())  # Probabilities vector
                target.append(1)  # IN == 1

        dOutLoader = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=True)
        for images, labels in dOutLoader:
            images = images.to(device)
            output = model(images)
            for o in output:
                features.append(o.detach().cpu().numpy())  # Probabilities vector
                target.append(0)  # OUT == 0

    features = np.array(features)
    target = np.array(target)

    return AdversaryDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.int))

def getInput(model, train, test, c):

    indices1 = [i for i, (_, label) in enumerate(train) if label == c]
    indices2 = [i for i, (_, label) in enumerate(test) if label == c]

    dataset1 = Subset(train, indices1)
    dataset2 = Subset(test, indices2)

    trainLoader = torch.utils.data.DataLoader(dataset1, batch_size=32, shuffle=True)
    features = []
    target = []
    model.to(device)
    model.eval()

    for images, labels in trainLoader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        for o in output:
            features.append(o.detach().cpu().numpy())  # Probabilities vector
            target.append(1)  # IN == 1

    testLoader = torch.utils.data.DataLoader(dataset2, batch_size=32, shuffle=True)

    for images, labels in testLoader:
      images = images.to(device)
      labels = labels.to(device)
      output = model(images)
      for o in output:
            features.append(o.detach().cpu().numpy())  # Probabilities vector
            target.append(0)  # OUT == 0

    features = np.array(features)
    target = np.array(target)

    return AdversaryDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.int))


# Crear y entrenar el modelo adversario
class AdversaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 20)
        self.hidden2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = nn.ReLU()(x)
        x = self.hidden2(x)
        x = nn.ReLU()(x)
        x = self.output(x)
        x = nn.Sigmoid()(x)
        return x


#   We create a pool, this is not important for the attack to be done, because we assume that the adversarial
#   has information about the data and the model
pool = FlexPool.client_server_pool(fed_dataset=get_dataset(), init_func=build_server_model, server_id="server")
clients = pool.clients
servers = pool.servers
aggregators = pool.aggregators

print(
    f"Number of nodes in the pool {len(pool)}: {len(servers)} server plus {len(clients)} clients. The server is also an aggregator"
)

honest_client = pool.clients.select(1)
pool.servers.map(copy_server_model_to_clients, honest_client)

#   We create the shadow dataset, by taking the server dataset and adding noise to it
dataset = datasets.MNIST( root=".", train=True, download=True, transform=mnist_transforms)
#indices = np.random.permutation(len(dataset))
#size = len(dataset)
#dataset = Subset(dataset, indices[:size//6])

dataset_test = datasets.MNIST( root = ".", train = False, download = True, transform = mnist_transforms)
dIn, dOut = createShadowDataset(dataset)

def convert_to_tensor_dataset(data):
    inputs = torch.stack([x[0] for x in data])
    labels = torch.tensor([x[1] for x in data])
    return TensorDataset(inputs, labels)

#   We create the shadow models
shadowModels = train_shadow_model(dIn)

#   We create the adversary dataset
adversaryDataset = getAdversaryDataset(shadowModels, dIn, dOut, 1)

print("\nENTRENAMIENTO DEL ADVERSARIO:")
print("Número de shadow models: ", len(shadowModels))
print("Número de elementos en el dataset que SÍ han servido para el entrenamiendo (shadow models): ", len(dIn))
print("Número de elementos en el dataset que NO han servido para el entrenamiento (shadow models): ", len(dOut))
print("Número de elementos en el dataset del adversario de la clase positiva: ", len(adversaryDataset.target[adversaryDataset.target == 1]))
print("Número de elementos en el dataset del adversario de la clase negativa: ", len(adversaryDataset.target[adversaryDataset.target == 0]))
indices = [i for i, (_, label) in enumerate(adversaryDataset) if label == 1]
random_index = random.choice(indices)
print("Elemento aleatorio del dataset para la clase positiva: ", adversaryDataset[random_index])
indices = [i for i, (_, label) in enumerate(adversaryDataset) if label == 0]
random_index = random.choice(indices)
print("Elemento aleatorio del dataset para la clase negativa: ", adversaryDataset[random_index])

#   We create the adversary model
adversaryModel = AdversaryModel()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(adversaryModel.parameters(), lr=0.1, momentum=0.9)
dataloader = torch.utils.data.DataLoader(adversaryDataset, batch_size=32, shuffle=True)
print(len(adversaryDataset.target))

for epoch in range(20):
  correct = 0
  adversaryModel.train()
  for inputs, labels in dataloader:
      optimizer.zero_grad()
      outputs = adversaryModel(inputs)
      predicted = torch.round(outputs)
      labels = labels.unsqueeze(1).float()  # Add a dimension and convert to float
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      correct += (predicted == labels).float().sum()

  accuracy = 100 * correct / len(adversaryDataset.target)
  print(f"Accuracy en época {epoch} = {accuracy}")

correct = 0
with torch.no_grad():
    for inputs, labels in dataloader:
        labels = labels.unsqueeze(1).float()  # Add a dimension and convert to float
        outputs = adversaryModel(inputs)
        predicted = torch.round(outputs)
        correct += (predicted == labels).float().sum()

print("\nRESULTADOS del entrenamiento del modelo adversario")
accuracy = 100 * correct / len(adversaryDataset.target)
print(f'Accuracy of the adversary model on the train set: {accuracy}%')

# Evaluar el modelo adversario en el conjunto de datos original
model = honest_client.map(getTrainedModel)[0]
data = getInput(model, dataset, dataset_test, 1)
data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
correct = 0

with torch.no_grad():
    for inputs, labels in data_loader:
        labels = labels.unsqueeze(1).float()  # Add a dimension and convert to float
        outputs = adversaryModel(inputs)
        predicted = torch.round(outputs)
        correct += (predicted == labels).float().sum()

print(f'Accuracy of the adversary model on the original data set: {100 * correct / len(data.target)}%')

##TODO: Escoger mejor el clasificador adversario, ver para que devuelve en el github las labels