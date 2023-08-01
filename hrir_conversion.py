import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.io import wavfile
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100


def f_transform(hrir):
    # Fourier transform
    hrtf = fft(hrir)
    frequency_response = np.abs(hrtf)

    # compute frequency
    hrtf = fftfreq(hrir.size, 1 / SAMPLE_RATE)
    # save positive half
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


def hrtf_convert(IR):
    for i in range(0, len(IR)):
        for j in range(0, len(IR[i])):
            hrir = IR[i][j]
            hrtf = fft(hrir)
            hrtf_magnitude = np.abs(hrtf)
            hrtf_no_phase = hrtf_magnitude * np.exp(1j * np.zeros_like(hrtf_magnitude))
            IR[i][j] = hrtf_no_phase
    return IR


def compute_magnitudte(hrir):
    hrtf = fft(hrir)
    hrtf_magnitude = np.abs(hrtf)
    hrtf_no_phase = hrtf_magnitude * np.exp(1j * np.zeros_like(hrtf_magnitude))
    return hrtf_no_phase


def convert_to_wav(file_list):
    scaled_array = np.int16(file_list * (2 ** 15 - 1))
    wavfile.write('output.wav', SAMPLE_RATE, scaled_array)


def wav_to_list(file_path):
    sample_rate, filelist = wavfile.read(file_path)
    return filelist / (2 ** 15 - 1)


if __name__ == "__main__":
    print("")
