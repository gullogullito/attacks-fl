################################################################################
#   In this example we are going to focus on privacy attacks on a FL scenario.
#   Privacy attacks aim to infer sensitive information from clients, during the
#   learning process. To carry out the experiment we'll follow the next steps:
#   
#       1. Create a FL architecture, with its federated dataset, using FLEX
#       2. Create any model which will train the clients' data
#       3. Implement the attack Deep Leakage from Gradients


import torch
import torch.nn as nn
from flex.data import Dataset, FedDatasetConfig, FedDataDistribution
from torchvision import datasets, transforms
from flex.pool import init_server_model, FlexPool
from flex.model import FlexModel
from flex.pool import deploy_server_model
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from flex.pool import FlexPool
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List
import copy

#   First we select cuda (if posible), for efficiency purposes
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on %s" % device)

#   We create our dataset ir order to federate it 
train_data = datasets.MNIST( root=".", train=True, download=True, transform=None)
test_data = datasets.MNIST( root=".", train=False, download=True, transform=None)

#   Process of federating our data
fed_train_data = Dataset.from_torchvision_dataset(train_data)
fed_test_data = Dataset.from_torchvision_dataset(test_data)

#   Creating the Federated Distribution
config = FedDatasetConfig(seed=0)
config.replacement = False
config.n_nodes = 100

flex_dataset = FedDataDistribution.from_config(fed_train_data, config)

#   Assign test data --> to server_id
server_id = "server"
flex_dataset[server_id] = fed_test_data

#   Usual transformations to pytorch mnist
mnist_transforms = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)   

#   Next step is to define our model, we'll use pytorch nnModule to define a simple nn
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

#   Recovery of a client's gradient (we'll need it for the attack)
#   The client will be selected later, and we'll use its model and dataset to get the gradient
def get_gradient(client_model : FlexModel, client_data: Dataset):
    model = client_model["model"]
    criterion = client_model["criterion"]

    model.train()
    model.to(device)

    test_data = client_data.to_torchvision_dataset(transform = mnist_transforms)

    client_dataloader = DataLoader(test_data, batch_size=1, pin_memory=False)

    model.zero_grad()
    data, target = next(iter(client_dataloader))

    data, target = data.to(device), target.to(device)

    #   We want our differentiable function to be the loss
    #   and we derive it wrt the weights
    f = criterion(model(data), target)
    df = torch.autograd.grad(f, model.parameters())
    
    return data, [g.detach().clone() for g in df]


#   Creation of the pool and selection of the leaked client 
pool = FlexPool.client_server_pool(fed_dataset=flex_dataset, init_func=build_server_model)
honest_client = pool.clients.select(1)
pool.servers.map(copy_server_model_to_clients, honest_client)

#   Now that we can recover a random client's gradient its time to attack
#   Let's take a random sample and see how we can approach
data, gradients = honest_client.map(get_gradient)[0]

image = data[0].squeeze()  # .squeeze() erases the channel dimension

plt.title("Original")
plt.imshow( image, cmap="gray")
plt.show()



#   Assumption that we have the list of gradients of one client
def DLG(client_model : FlexPool, gradients):

    dummy_data = torch.randn(1, 1, 28, 28, requires_grad= True, device = device)
    dummy_label = torch.randn(1, 10, requires_grad= True, device = device)

    plt.title("Dummy Data")
    plt.imshow( dummy_data[0].detach().squeeze())
    plt.show()

    model = client_model._models[list(client_model._models.keys())[0]]
    model = model["model"]

    #   Paper uses this optimizer
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    #   We'll keep track of how the attack is running
    history = []

    #   We have everything now, lets train our dummies so that we can optimize the gradients
    def closure():
        
        optimizer.zero_grad()

        criterion = client_model._models[list(client_model._models.keys())[0]]
        criterion = criterion["criterion"]

        pred = model(dummy_data) 
        dummy_loss = criterion(pred, dummy_label)
        dummy_gradients = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        
        grad_diff = 0

        for gx, gy in zip(dummy_gradients, gradients):
            grad_diff += ((gx - gy) ** 2).sum()

        grad_diff.backward()
        
        return grad_diff

    for iters in range(300):
        optimizer.step(closure)
        if iters % 10 == 0: 
            current_loss = closure()
            print(iters, "%.4f" % current_loss.item())
        history.append(dummy_data[0].detach().squeeze()) 

    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i * 10], cmap = "gray")
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    
    plt.show()
    '''plt.title("Recovered data")
    plt.imshow(dummy_data.detach().squeeze(), cmap="gray") #    detach() is not inplace, detach_() is
    plt.show()'''

#   We reproduce the attack now
DLG(honest_client, gradients)