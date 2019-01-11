import numpy as np
import scipy.misc
import scipy.signal
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

path_file = Path("Test_NTO_1024.bmp")
#path_file = Path("indeks.bmp")
im = Image.open(path_file).convert('L')

mx = 100
my = 100
width, height = im.size
x = height + mx
y = width + my
print("Input Matrix size = " + str(width) + "x" + str(height))

u_in = np.zeros([x,y], dtype=complex)
u_in[int(mx/2):x-int(mx/2), int(my/2):y-int(my/2)] = im
u_in = u_in / u_in.max()

h_z_kappa = np.zeros([x,y], dtype=complex)
h_z = np.zeros([x,y], dtype=complex)
H_Z = np.zeros([x,y], dtype=complex)
u_in_fft = np.zeros([x,y], dtype=complex)
u_out = np.zeros([x,y], dtype=complex)

lam = 633*(pow(10,(-9)))
k = 2*np.pi/lam
z = 1000*(pow(10,(-3)))   #odlegosc symulacji

print("Dlugosc fali = " + str(lam))
print("Wektor falowy fali = " + str(k))
print("Odleglosc propagacji = " + str(z))

sampling = 10*pow(10, (-6))
i = 0
o = 0
for i in range(x):
    for o in range(y):
        h_z_kappa[i, o] = np.exp(1j*k*(pow((i-(x/2))*sampling, 2) + pow((o-(y/2))*sampling, 2))/(2*z))
        #H_Z[i, o] = np.exp(1j*k*z) * np.exp(-1j*np.pi*lam*z*((pow(((i-(x/2))*sampling), 2) + pow(((o-(y/2))*sampling), 2))))
h_z = h_z_kappa * (np.exp(1j * k * z) / (1j * lam * z))

u_in_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
h_z = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(h_z)))

u_out = np.multiply(u_in_fft, h_z)
u_out = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_out)))

#rozwiaznie = fftshift(fft2(ifftshift()))

print("KARRAMBA WYNIK")

u_in_fft = np.sqrt(pow(u_in_fft.real, 2) + pow(u_in_fft.imag, 2))
u_in_fft = u_in_fft + abs(u_in_fft.min())
u_in_fft = np.rint((u_in_fft / u_in_fft.max())*255)
scipy.misc.imsave("u_in_fft.bmp", u_in_fft)

h_z = h_z.real
h_z = h_z + abs(h_z.min())
h_z = np.rint((h_z / h_z.max())*255)
scipy.misc.imsave("h_z.bmp", h_z)

u_out_imag = u_out

u_out = np.sqrt(pow(u_out.real, 2) + pow(u_out.imag, 2))
u_out = u_out + abs(u_out.min())
u_out = np.rint((u_out / u_out.max())*255)
u_out_name = "u_out_AMPLITUDE_size_" + str(width) + "-" + str(height) + "_dist_" + str(z) + ".bmp"
scipy.misc.imsave(u_out_name, u_out[int(mx/2):x-int(mx/2), int(my/2):y-int(my/2)])

u_out_imag = np.arctan2(u_out_imag.imag, u_out_imag.real) * 180 / np.pi
u_out_imag = u_out_imag + abs(u_out_imag.min())
u_out_imag = np.rint((u_out_imag / u_out_imag.max())*255)
u_out_name = "u_out_PHASE_size_" + str(width) + "-" + str(height) + "_dist_" + str(z) + ".bmp"
scipy.misc.imsave(u_out_name, u_out_imag[int(mx/2):x-int(mx/2), int(my/2):y-int(my/2)])

