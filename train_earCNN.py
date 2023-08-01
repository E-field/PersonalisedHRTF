import matplotlib.pyplot as plt
from utils import read_split_data
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from my_dataset import MyDataSet
import torchvision.transforms as transforms
from models import EarCNN, init_weights
import numpy as np

root_train = 'dataset/Customized dataset/train'
root_val = 'dataset/Customized dataset/validate'
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device".format(device))


def train_ear_cnn():
    train_image_path, train_image_label, val_image_path, val_image_label = read_split_data(root_train, 0.2)

    print(len(train_image_path))
    data_transforms = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = MyDataSet(image_path=train_image_path, image_class=train_image_label, transform=data_transforms)
    print(train_dataset)
    val_dataset = MyDataSet(image_path=val_image_path, image_class=val_image_label, transform=data_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = EarCNN()
    model.apply(init_weights)
    model = model.to(device)
    epochs = 100
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    criterion_ear_cnn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Training loop
    for epoch in range(epochs):
        # Train
        print("Start training")
        print("Starting epoch {}/100".format(epoch + 1))
        model.train()
        train_loss = 0
        batch_num = 0
        train_correct = 0
        train_total = 0

        for batch, label in train_dataloader:
            batch_num += 1
            # print(f"Batch {batch_num}/{len(train_dataloader)}")
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            # print(outputs)
            loss = criterion_ear_cnn(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            # print("predicts: {}".format(predicted))
            # print("labels: {}".format(label.data))
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()

        train_loss /= len(train_dataloader)
        train_loss_list.append(train_loss)
        batch_num = 0
        train_accuracy = train_correct / train_total
        train_accuracy_list.append(train_accuracy)
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch, label in val_dataloader:
                batch_num += 1
                # print(f"Batch {batch_num}/{len(val_dataloader)}")
                batch = batch.to(device)
                label = label.to(device)
                outputs = model(batch)
                loss = criterion_ear_cnn(outputs, label)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_loss /= len(val_dataloader)
        val_loss_list.append(val_loss)
        val_accuracy = val_correct / val_total
        val_accuracy_list.append(val_accuracy)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    arr = np.arange(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(arr, train_loss_list, label='Train Loss')
    ax.plot(arr, val_loss_list, label='Validation Loss')
    ax2 = ax.twinx()
    ax2.plot(arr, train_accuracy_list, label='Train Accuracy', linestyle='--')
    ax2.plot(arr, val_accuracy_list, label='Validation Accuracy', linestyle='--')
    ax2.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax2.legend(loc='upper right')
    ax.legend(loc='upper left')
    plt.show()
    torch.save(model.state_dict(), 'models/EAR_CNN.pth')


train_ear_cnn()
