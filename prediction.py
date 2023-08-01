from matplotlib import pyplot as plt
from numpy.fft import fftfreq
from models import EarCNN, hrtfModel
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
import os
from pysofaconventions import SOFAFile
import numpy as np

model_cnn = EarCNN()
m_state_dict = torch.load('models/EAR_CNN.pth')
model_cnn.load_state_dict(m_state_dict)
model_cnn.eval()
data_transforms = transforms.Compose([
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
model_hrtf = hrtfModel()
h_state_dict = torch.load('models/HRTF_model.pth')
model_hrtf.eval()


def get_dataset():
    dataset_x = []
    dataset_y = []
    PATH = "dataset/cipic-hrtf-database-master/cipic-hrtf-database-master/ear_photos"
    dir_list = []
    # store file folder
    for root, dirs, files in os.walk(PATH):
        for d in dirs:
            dir_list.append(os.path.join(root, d))
    # print(dir_list)
    for i in range(len(dir_list)):
        folder_path = dir_list[i]
        folder_name = os.path.basename(folder_path)
        sofa_path = 'dataset/cipic-hrtf-database-master/cipic-hrtf-database-master/sofa_files/sofacoustics' \
                    '.org_data_database_cipic_' + folder_name + '.sofa '
        hrir01 = get_HRIR(sofa_path)[300][0]
        file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_list.append(os.path.join(root, file))
        # print(file_list)
        for i in range(len(file_list)):
            output = get_output(file_list[i])
            output = output.cpu()
            output = output.numpy()
            output = output[0]

            dataset_x.append(output)
            dataset_y.append(hrir01)
    print(dataset_x)
    return dataset_x, dataset_y


def get_output(path):
    img = Image.open(path)
    img = data_transforms(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model_cnn(img)
        print("prediction successful")
    return output


def get_output_hrtf(list):
    with torch.no_grad():
        output = model_hrtf(list)
    return output


def get_HRIR(path):
    sofa = SOFAFile(path, 'r')
    hrir = sofa.getDataIR()
    return hrir


if __name__ == "__main__":
    hrtf_magnitude = get_output_hrtf(get_output('ear_image.png'))
    print(hrtf_magnitude)
    hrtf_magnitude = np.array(hrtf_magnitude)[0]
    frequency_response = hrtf_magnitude

    hrtf = fftfreq(hrtf_magnitude.size, 1 / 44100)
    freq_p = []
    frequency_response_p = []
    for i in range(len(hrtf)):
        if hrtf[i] >= 0:
            freq_p.append(hrtf[i])
            frequency_response_p.append(frequency_response[i])
    plt.figure(figsize=(10, 5))
    plt.plot(freq_p, 20 * np.log10(frequency_response_p))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response')
    plt.show()
