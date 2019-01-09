import numpy as np
import scipy.misc
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

path_file = Path("Test_NTO_1024.bmp")
im = Image.open(path_file).convert('L')

u_in = np.array(im)
u_in = u_in / u_in.max()
print(u_in[0][0])

width, height = im.size
x = height
y = width
print(x,y)

h_z = np.complex
h_z = np.zeros((x,y))
lam = 633*(pow(10,(-9)))
k = 2*np.pi/lam
z = 1000*(pow(10,(-3)))   #odlegosc symulacji

print(u_in)
print(lam)
print(k)
print(z)

sampling = 10*pow(10,(-6))
i = 0
o = 0
for i in range(x):
    for o in range(y):
        h_z[i,o]= (np.exp(1j*k*z)/(1j*lam*z))*np.exp(1j*k*(pow((i-(x/2))*sampling,2) + pow((o-(y/2))*sampling,2))/(2*z))
print(h_z[0][0])
h_z_kappa = h_z / (np.exp(1j*k*z)/(1j*lam*z))
U_fft = np.array((x,y))
#u_out = np.multiply(np.fft.fft2(u_in), np.fft.fft2(h_z))
u_out = np.multiply(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in))), h_z_kappa)
#u_out = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in * h_z_kappa)))

#rozwiaznie = fftshift(fft2(ifftshift()))

u_out = np.fft.ifft2(u_out)
#u_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_fft)))

#for i in range(x):
#   for o in range(y):
#        u_out[i,o]= (u_out[i,o])*(np.exp(1j*k*z)/(1j*lam*z))*np.exp(1j*k*(pow(i*sampling,2) + pow(o*sampling,2))/(2*z))

#u_out = np.fft.ifft2(np.fft.ifftshift(U_fft))
print("KARRAMBA WYNIK")
print(u_out[0][0])

u_out = u_out.real
u_out = u_out + abs(u_out.min())
u_out = np.rint((u_out / u_out.max())*255)

h_z = h_z.real
h_z = h_z + abs(h_z.min())
h_z = np.rint((h_z / h_z.max())*255)
scipy.misc.imsave("h_z.bmp", h_z)
print(u_out[0][0])

u_out_name = "u_out_size_" + str(x) + "_dist_" + str(z) + ".bmp"
scipy.misc.imsave(u_out_name, u_out)
#plt.imshow(u_out)