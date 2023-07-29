import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile
import scipy.io
import matplotlib.pyplot as plt
import cv2
from pysofaconventions import SOFAFile
import soundfile as sf
import scipy.special

Root = 'dataset/cipic-hrtf-database-master/cipic-hrtf-database-master/standard_hrir_database'
data = scipy.io.loadmat('dataset/cipic-hrtf-database-master/cipic-hrtf-database-master/standard_hrir_database'
                        '/subject_003/hrir_final.mat')

SAMPLE_RATE = 44100


def f_transform(hrir, az, el, sample_rate):
    # 傅里叶变换
    fft_vals = fft(hrir['hrir_r'][az][el])
    frequency_response = np.abs(fft_vals)
    print("fftvals is{}".format(frequency_response))

    # 计算频率
    freqs = fftfreq(hrir['hrir_r'][az][el].size, 1 / SAMPLE_RATE)
    print(freqs)
    # 只保留正半部分
    freq_p = []

    frequency_response_p = []
    for i in range(len(freqs)):
        if freqs[i] >= 0:
            freq_p.append(freqs[i])
            frequency_response_p.append(frequency_response[i])
    print(freq_p)
    plt.figure(figsize=(10, 5))
    plt.plot(freq_p, 20 * np.log10(frequency_response_p))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response')
    plt.show()


def convert_to_wav(file_list):
    scaled_array = np.int16(file_list * (2 ** 15 - 1))
    wavfile.write('output.wav', SAMPLE_RATE, scaled_array)


def wav_to_list(file_path):
    # file_path, sample_rate = sf.read(file_path)
    #
    # normalized_data = data / np.max(np.abs(data))
    #
    # data_list = normalized_data.tolist()
    #
    # return data_list
    sample_rate, filelist = wavfile.read(file_path)
    return filelist / (2 ** 15 - 1)


def edge_detection(image_path):
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图片的宽度和高度
    height, width = gray.shape[:2]
    print(height, width)

    # 计算裁剪边界
    left = int(width / 10)
    top = int(height / 10)
    right = width - int(width / 10)
    bottom = height - int(height / 10)

    # 裁剪图片
    cropped_image = gray[top:bottom, left:right]
    cropped_image = cv2.resize(cropped_image, (64, 64))
    edges = cv2.Canny(cropped_image, 120, 155)

    # 使用 Matplotlib 显示原始图像和边缘检测后的图像
    plt.figure(figsize=(10, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge Image')

    plt.show()


# h_array = data["hrir_r"][2][3]
# print(h_array)
# convert_to_wav(h_array)
# print(wav_to_list("Database-Master_V1-4/Database-Master_V1-4/D1/D1_HRIR_WAV/44K_16bit/azi_0,0_ele_15,0.wav"))


PAHT_SOFA3 = "cipic-hrtf-database-master/cipic-hrtf-database-master/sofa_files/sofacoustics" \
             ".org_data_database_cipic_subject_003.sofa "
sofa3 = SOFAFile(PAHT_SOFA3, 'r')
print("==========================")
sofa3.printSOFAVariables()
print("==========================")
hrtf_data = sofa3.getDataIR()
print(hrtf_data)