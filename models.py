import torch
import torch.nn as nn
import torch.utils.data


root_train = 'dataset/Customized dataset/train'
root_val = 'dataset/Customized dataset/validate'
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device".format(device))


# model for predict hrtf(magnitude)
class hrtfModel(nn.Module):
    def __init__(self):
        super(hrtfModel, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(100, 500)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(500, 400)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        real_part = x[:, :200]
        imag_part = x[:, 200:]
        return torch.complex(real_part, imag_part)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder
        self.dec1 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x = torch.relu(self.enc1(x))
        x = self.pool(x)
        x = torch.relu(self.enc2(x))
        x = self.pool(x)
        x = torch.relu(self.enc3(x))
        x = self.pool(x)  # the latent space representation
        # Decoder
        x = torch.relu(self.dec1(x))
        x = torch.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x


class EarCNN(nn.Module):
    def __init__(self):
        super(EarCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def main():
    return 0


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


if __name__ == "__main__":
    main()
