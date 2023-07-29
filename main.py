import matplotlib.pyplot as plt
from utils import read_split_data
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torchvision import transforms
from my_dataset import MyDataSet
import copy
import torchvision.models as models
from ear_CNN import Autoencoder
from PIL import Image
import torchvision.transforms as transforms

# set path and device
root = 'dataset\\Customized dataset'
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device".format(device))


def main():
    # divide dataset with rate 0.2
    train_image_path, train_image_label, val_image_path, val_image_label = read_split_data(root, 0.2)

    # transform input
    data_transforms = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # load train_dataset and val_dataset
    train_dataset = MyDataSet(image_path=train_image_path, image_class=train_image_label, transform=data_transforms)
    val_dataset = MyDataSet(image_path=val_image_path, image_class=val_image_label, transform=data_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

    # load pretrained  vgg_19 net
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
    print(model)

    # copy a new model
    modified_model = copy.deepcopy(model)

    # # remove last fc_layer
    # modified_model.classifier = nn.Sequential(*[model.classifier[i] for i in range(5)])
    #
    # # modify output layers following paper
    # modified_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # modified_model.classifier.add_module("6", nn.Linear(4096, 100))
    # # modified_model.classifier.add_module("7", nn.Softmax(dim=1))

    modified_model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=4096, out_features=100, bias=True),
    )
    print(modified_model)

    # old modification code

    # modified_model.classifier = nn.Sequential(*[model.classifier[i] for i in range(len(model.classifier) - 1)])
    # modified_model.classifier = nn.Sequential(
    #     nn.Linear(in_features=25088, out_features=4096, bias=True),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.2, inplace=False),
    #     nn.Linear(in_features=4096, out_features=4096, bias=True),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.2, inplace=False),
    #     nn.Linear(in_features=4096, out_features=100, bias=True),
    #     nn.Softmax(dim=1)
    # )

    # set cost function, optimizer, and epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modified_model.parameters(), lr=0.001)
    epochs = 50
    train_loss_list = []
    val_loss_list = []
    # start training
    modified_model = train_model(modified_model, criterion, optimizer, train_dataloader, val_dataloader,
                                 device, num_epochs=epochs)


def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, device, num_epochs=5):
    # load model to device
    model = model.to(device)
    criterion = criterion
    optimizer = optimizer
    train_dataloader = train_dataloader
    val_dataloader = val_dataloader
    train_loss_list = []
    val_loss_list = []
    batch_num = 0
    for epoch in range(num_epochs):
        print("now in epoch :{}/{}".format(epoch+1,num_epochs))
        print("=====================")
        # training
        model.train()
        train_loss = 0
        print("start training")
        for batch, label in train_dataloader:
            batch_num += 1
            print(f"Batch {batch_num}/{len(train_dataloader)}")
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            _, preds = torch.max(outputs, 1)
            print("predicts: {}".format(preds))
            print("labels: {}".format(label.data))
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_loss_list.append(train_loss)
        # validate
        model.eval()
        val_loss = 0
        batch_num = 0
        with torch.no_grad():
            for batch, label in val_dataloader:
                batch_num += 1
                print(f"Batch {batch_num}/{len(val_dataloader)}")
                batch = batch.to(device)
                label = label.to(device)
                outputs = model(batch)
                loss = criterion(outputs, label)
                val_loss += loss.item()
        val_loss /= len(val_dataloader)
        val_loss_list.append(val_loss)

    arr = np.arange(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(arr, train_loss_list, label='Train Loss')
    ax.plot(arr, val_loss_list, label='Validation Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='upper left')
    plt.show()
    torch.save(model, 'vgg_ear.pt')
    # # main training loop
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch + 1}/{num_epochs}")
    #     print('-' * 10)
    #     for phase in ['train', 'val']:
    #         if phase == 'train':
    #             dataloader = train_dataloader
    #             model.train()  # Set model to training mode
    #         else:
    #             dataloader = val_dataloader
    #             model.eval()  # Set model to evaluate mode
    #
    #         running_loss = 0.0
    #         running_corrects = 0
    #         batch_num = 0
    #
    #         # Iterate over data.
    #         for inputs, labels in dataloader:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)
    #
    #             # zero the parameter gradients
    #             optimizer.zero_grad()
    #
    #             # forward
    #             with torch.set_grad_enabled(phase == 'train'):
    #                 outputs = model(inputs)
    #                 _, preds = torch.max(outputs, 1)
    #                 print("predicts: {}".format(preds))
    #                 print("labels: {}".format(labels.data))
    #                 running_corrects += torch.sum(preds == labels.data)
    #                 loss = criterion(outputs, labels)
    #
    #                 # backward + optimize only if in training phase
    #                 if phase == 'train':
    #                     loss.backward()
    #                     optimizer.step()
    #
    #             # statistics
    #             running_loss += loss.item() * inputs.size(0)
    #             # running_corrects += torch.sum(preds == labels.data)
    #             batch_num += 1
    #             print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_num}/{len(dataloader)}")
    #
    #         epoch_loss = running_loss / len(dataloader.dataset)
    #         epoch_acc = running_corrects / len(dataloader.dataset)
    #         print(running_corrects)
    #
    #         print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    return model


main()

# model = torch.load("C:\\Users\\Gray-LAN\\Desktop\\Master\\FYP\\Project\\dataset\\sava.pt")
# model.eval()
# print(model)
# model = model.to(device)
# image_path = "Database-Master_V1-4/Database-Master_V1-4/D2/D2_Scans/D2 (R).png"  # 替换为你的图片路径
# image = Image.open(image_path).convert("RGB")
# data_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
#
# input_image = data_transforms(image).unsqueeze(0)
#
#
# # 反转换操作，以便可视化
# unnormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
#
# untransformed_image = unnormalize(input_image)
# untransformed_image = torch.squeeze(untransformed_image)
# # 将图像从 Tensor 转换回 PIL Image
# pil_image = transforms.ToPILImage()(untransformed_image)
#
# # 显示图像
# plt.imshow(pil_image)
# plt.axis('off')
# plt.show()
#
#
#
# input_image = input_image.to(device)
# with torch.no_grad():
#     output = model(input_image)
#
# print(output.shape)
#
# # 将张量转换为图像
# image = output.cpu().squeeze(0).permute(1, 2, 0).numpy()  # 将维度顺序调整为 (H, W, C) 并转换为 NumPy 数组
# image = (image * 255).astype('uint8')  # 将图像从标准化范围转换为整数值范围
# image = Image.fromarray(image)  # 创建 PIL 图像对象
#
# # 显示图像
# image.show()
