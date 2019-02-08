import numpy as np
import scipy.misc
import scipy.signal
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

multi = 4
path_file = Path("Test_NTO_1024.bmp")
#path_file = Path("indeks.bmp")
im = Image.open(path_file).convert('L')
width, height = im.size
x = height * multi
y = width * multi

xx = x/2-width/2
yy = y/2-height/2

print("Input Matrix size = " + str(width) + "x" + str(height))
print("Output Matrix size = " + str(x) + "x" + str(y))
print("Help Start = " + str(xx) + "x" + str(yy))
print("Help End = " + str(xx+width) + "x" + str(yy+height))

u_in = np.zeros([x,y], dtype=float)
u_in[int(xx):int(xx)+width, int(yy):int(yy)+height] = im
print("u_in.max = " + str(u_in.max()))
u_in = u_in / u_in.max()
np.savetxt("Tablica-" + str(x) + "x" + str(y) + ".txt", u_in, fmt='%1.5f')

#u_in_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
#u_in_fft = np.sqrt(pow(u_in_fft.real, 2) + pow(u_in_fft.imag, 2))
#u_in_fft = u_in_fft + abs(u_in_fft.min())
#u_in_fft = np.rint((u_in_fft / u_in_fft.max())*255)
#u_in_fft_name = "u_in_FFT_AMP_size_" + str(width) + "x" + str(height) + ".bmp"
#scipy.misc.imsave(u_in_fft_name, u_in_fft)


u_name1 = "result_z_0.50000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)

u_name1 = "result_z_0.60000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)

u_name1 = "result_z_0.70000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)

u_name1 = "result_z_0.80000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)

u_name1 = "result_z_0.90000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)

u_name1 = "result_z_1.00000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)

u_name1 = "result_z_1.50000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)

u_name1 = "result_z_1.70000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)

u_name1 = "result_z_2.00000"
u_fft = np.genfromtxt(u_name1 + ".txt", dtype=int ,delimiter='\t')
u_fft_name = u_name1 + ".bmp"
scipy.misc.imsave(u_fft_name, u_fft)