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

#   First we select cuda (if posible), for efficiency purposes
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on %s" % device)
torch.cuda.empty_cache()

def get_dataset():
    #   We create our dataset ir order to federate it
    train_data = datasets.CIFAR10( root=".", train=True, download=True, transform=None)
    test_data = datasets.CIFAR10( root=".", train=False, download=True, transform=None)

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

#   Usual transformations to pytorch CIFAR10
cifar_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

#   Next step is to define our model, we'll use a simple convolutional NN
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.tanh(self.conv1(x)))
        x = self.pool(self.tanh(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.tanh(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

#   Initialization of the server model, with its hyperparameters
@init_server_model
def build_server_model():

    server_flex_model = FlexModel()

    server_flex_model["model"] = CIFAR10_CNN().to(device)
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

def getTrainedModel(client, clientData):
    model = client["model"]
    criterion = client["criterion"]
    model.train()
    model = model.to(device)

#   For this scenario, we assume that the attacker has access to data which is similar to the
#   target model's training data. To simulate this, we'll flip the values of 15% randomly
#   selected features, to obtain a "noisy" dataset to train our shadow models.

def saltPepper(img, prob=0.1):
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

def train_shadow_model( dIn, nShadowModels = 1):

    shadows = []

    for i in range(nShadowModels):

        model = CIFAR10_CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-7)

        trainloader = torch.utils.data.DataLoader(dIn, batch_size=4, shuffle=True)


        for epoch in range(1):
            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        shadows.append(model)

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

#   We need to build a dataset where:
#   1. For each sample in dIn, we compute model(x) = y
#   2. Add to the dataset the data-point [v(vector of probabilities for each class), y, IN]
#
#   Same for dOut but we change IN for OUT
def getAdversaryDataset(shadowModels, dIn, dOut):
    features = []
    target = []

    for model in shadowModels:
        model.eval()

        dInLoader = torch.utils.data.DataLoader(dIn, batch_size=32, shuffle=True)
        for images, labels in dInLoader:
            images = images.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            for o, p in zip(output, predicted):
                features.append(o.detach().cpu().numpy())  # Probabilities vector
                target.append(1)  # IN == 1

        dOutLoader = torch.utils.data.DataLoader(dOut, batch_size=32, shuffle=True)
        for images, labels in dOutLoader:
            images = images.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            for o, p in zip(output, predicted):
                features.append(o.detach().cpu().numpy())  # Probabilities vector
                target.append(0)  # OUT == 0

    features = np.array(features)
    target = np.array(target)
    return AdversaryDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long))

def getInput(model, train, test):

    trainLoader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    features = []
    target = []
    model.to(device)
    model.eval()

    for images, labels in trainLoader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        for o, p in zip(output, predicted):
            features.append(o.detach().cpu().numpy())  # Probabilities vector
            target.append(1)  # IN == 1

    testLoader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

    for images, labels in testLoader:
      images = images.to(device)
      labels = labels.to(device)
      output = model(images)
      _, predicted = torch.max(output, 1)
      for o, p in zip(output, predicted):
            features.append(o.detach().cpu().numpy())  # Probabilities vector
            target.append(0)  # OUT == 0

    features = np.array(features)
    target = np.array(target)

    return AdversaryDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long))


# Crear y entrenar el modelo adversario
class AdversaryModel(nn.Module):
    def __init__(self):
        super(AdversaryModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
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

#   We create the shadow dataset, by taking the server dataset and adding noise to it
dataset = datasets.CIFAR10( root=".", train=True, download=True, transform=cifar_transforms)
dataset_test = datasets.CIFAR10( root=".", train=False, download=True, transform=cifar_transforms)
server = pool.servers.select(1)
server_dataset = server._data
dIn, dOut = createShadowDataset(dataset)

def convert_to_tensor_dataset(data):
    inputs = torch.stack([x[0] for x in data])
    labels = torch.tensor([x[1] for x in data])
    return TensorDataset(inputs, labels)

dIn = convert_to_tensor_dataset(dIn)
dOut = convert_to_tensor_dataset(dOut)

#   We create the shadow models
shadowModels = train_shadow_model(dIn)

#   We create the adversary dataset
adversaryDataset = getAdversaryDataset(shadowModels, dIn, dOut)

#   We create the adversary model
adversaryModel = AdversaryModel()

#   We train the adversary model
adversaryModel.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(adversaryModel.parameters(), lr=0.001, weight_decay=1e-7)
dataloader = torch.utils.data.DataLoader(adversaryDataset, batch_size=4, shuffle=True)

for epoch in range(1):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = adversaryModel(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

adversaryModel.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = adversaryModel(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the adversary model: {100 * correct / total}%')

# Evaluar el modelo adversario en el conjunto de datos original
data = getInput(CIFAR10_CNN().to(device), dataset, dataset_test)
data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in data_loader:
        outputs = adversaryModel(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the adversary model on the original data set: {100 * correct / total}%')

##TODO: Escoger mejor el clasificador adversario, ver para que devuelve en el github las labels