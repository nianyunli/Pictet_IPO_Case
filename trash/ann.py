import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 11),
            nn.ReLU(),
            nn.Linear(11, 11),
            nn.ReLU(),
            nn.Linear(11, 1),
        )
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # print(y.shape)
            # print(y.unsqueeze(1).shape)
            test_loss += loss_fn(pred, y.unsqueeze(1)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

outcome = 'high_return'
seed = 1
process = Preprocess()
process.pre_process(df = df,outcome = outcome,features = features,test_size =0.2, seed = seed)
X_train = process.X_train
y_train = process.y_train
X_test  = process.X_test
y_test  = process.y_test
X_all = process.X_all

np.array(y_train).shape
np.array(y_test).shape
train_data = TensorDataset(torch.Tensor(np.array(X_train)), torch.Tensor(np.array(y_train)))
test_data =  TensorDataset(torch.Tensor(np.array(X_test)), torch.Tensor(np.array(y_test)))

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#define dimension of the network 

input_dim = X_train.shape[1]

model = NeuralNetwork()
learning_rate = 1e-3
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")



# size = len(test_dataloader.dataset)
# num_batches = len(test_dataloader)

