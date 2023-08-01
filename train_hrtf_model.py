import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from prediction import get_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import hrir_conversion
from models import hrtfModel

dataset_x, dataset_y = get_dataset()
print(dataset_x)
for i in range(len(dataset_y)):
    dataset_y[i] = hrir_conversion.compute_magnitudte(dataset_y[i])
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device".format(device))


class HrtfDataset(Dataset):
    def __init__(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).to(torch.complex64)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


dataset_m = HrtfDataset(dataset_x, dataset_y)
total_samples = len(dataset_m)
train_size = int(0.5 * total_samples)
valid_size = total_samples - train_size
train_dataset, valid_dataset = random_split(dataset_m, [train_size, valid_size])
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

# dataloader = DataLoader(dataset_m, batch_size=4, shuffle=True)


model = hrtfModel()


# loss function
def complex_mse_loss(output, target):
    real_diff = output.real - target.real
    imag_diff = output.imag - target.imag
    return torch.mean(real_diff ** 2 + imag_diff ** 2)


# optimizer
optimizer = Adam(model.parameters(), lr=0.0001)

train_loss_list = []
val_loss_list = []


def train_model(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    train_loss_list.append(avg_loss)
    return avg_loss


def validate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    val_loss_list.append(avg_loss)
    return avg_loss


num_epochs = 10000
for epoch in range(num_epochs):
    train_loss = train_model(model, train_dataloader, optimizer, complex_mse_loss)
    valid_loss = validate_model(model, valid_dataloader, complex_mse_loss)
    print(f"Epoch {epoch + 1}/{num_epochs},training loss: {train_loss}")
    print(f"Epoch {epoch + 1}/{num_epochs}, validation loss: {valid_loss}")

fig, ax = plt.subplots()
arr = np.arange(0, num_epochs)
ax.plot(arr, train_loss_list, label='Train Loss')
ax.plot(arr, val_loss_list, label='Validation Loss')
ax2 = ax.twinx()
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax2.legend(loc='upper right')
ax.legend(loc='upper left')
plt.show()
torch.save(model.state_dict(), 'models/HRTF_model.pth')
