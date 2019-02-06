import numpy as np
import scipy.misc
import scipy.signal
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

path_file = Path("Test_NTO_1024.bmp")
#path_file = Path("indeks.bmp")
im = Image.open(path_file).convert('L')
width, height = im.size
x = height
y = width

print("Input Matrix size = " + str(width) + "x" + str(height))
u_in = np.array(im)
print("u_in.max = " + str(u_in.max()))
u_in = u_in / u_in.max()
np.savetxt("Tablica-" + str(x) + "x" + str(y) + ".txt", u_in, fmt='%1.5f')

u_in_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
u_in_fft = np.sqrt(pow(u_in_fft.real, 2) + pow(u_in_fft.imag, 2))
u_in_fft = u_in_fft + abs(u_in_fft.min())
u_in_fft = np.rint((u_in_fft / u_in_fft.max())*255)
u_in_fft_name = "u_in_FFT_AMP_size_" + str(width) + "x" + str(height) + ".bmp"
scipy.misc.imsave(u_in_fft_name, u_in_fft)

u_fft = np.genfromtxt("1024x1024.txt", dtype=int ,delimiter='\t')
u_fft_name = "u_z_AMP_size_" + str(width) + "x" + str(height) + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)