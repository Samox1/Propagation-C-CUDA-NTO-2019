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
print(x,y)

u_in = np.array(im, dtype=complex)
u_in = u_in / u_in.max()
print(u_in[0][0])

h_z_kappa = np.empty([x,y], dtype=complex)
h_z = np.empty([x,y], dtype=complex)
H_Z = np.empty([x,y], dtype=complex)
u_in_fft = np.empty([x,y], dtype=complex)
u_out = np.empty([x,y], dtype=complex)

lam = 633*(pow(10,(-9)))
k = 2*np.pi/lam
z = 1000*(pow(10,(-3)))   #odlegosc symulacji

print(u_in)
print(lam)
print(k)
print(z)

sampling = 10*pow(10, (-6))
i = 0
o = 0
for i in range(x):
    for o in range(y):
        h_z_kappa[i, o] = np.exp(1j*k*(pow((i-(x/2))*sampling, 2) + pow((o-(y/2))*sampling, 2))/(2*z))
        H_Z[i, o] = np.exp(1j*k*z) * np.exp(-1j*np.pi*lam*z*((pow(((i-(x/2))*sampling), 2) + pow(((o-(y/2))*sampling), 2))))
h_z = h_z_kappa * (np.exp(1j * k * z) / (1j * lam * z))
print(h_z[0][0])

#u_out = scipy.signal.convolve2d(u_in, h_z)
#u_in_fft = np.fft.fft2(u_in)
#u_in_fft = np.fft.fftshift(np.fft.fft2(u_in))
u_in_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))


#u_out = np.multiply(u_out, np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(h_z))))
h_z = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(h_z)))
#u_out = u_out * h_z

#u_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u_out)))


#u_out = u_out * h_z
#u_out = np.fft.ifft2(u_out)

#U_fft = np.array((x,y))
#u_out = np.multiply(np.fft.fft2(u_in), np.fft.fft2(h_z))
u_out = np.multiply(u_in_fft, h_z)
u_out = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_out)))
#u_out = np.fft.ifft2(u_out)

#rozwiaznie = fftshift(fft2(ifftshift()))

#for i in range(x):
#   for o in range(y):
#        u_out[i,o]= (u_out[i,o])*(np.exp(1j*k*z)/(1j*lam*z))*np.exp(1j*k*(pow(i*sampling,2) + pow(o*sampling,2))/(2*z))

#u_out = np.fft.ifft2(np.fft.ifftshift(U_fft))
print("KARRAMBA WYNIK")
print(u_out[0][0])

u_in_fft = u_in_fft.real
u_in_fft = u_in_fft + abs(u_in_fft.min())
u_in_fft = np.rint((u_in_fft / u_in_fft.max())*255)
scipy.misc.imsave("u_in_fft.bmp", u_in_fft)

H_Z = H_Z.real
H_Z = H_Z + abs(H_Z.min())
H_Z = np.rint((H_Z / H_Z.max())*255)
scipy.misc.imsave("H_Z1.bmp", H_Z)

h_z = h_z.real
h_z = h_z + abs(h_z.min())
h_z = np.rint((h_z / h_z.max())*255)
scipy.misc.imsave("h_z.bmp", h_z)

u_out_imag = u_out.imag

u_out = u_out.real
u_out = u_out + abs(u_out.min())
u_out = np.rint((u_out / u_out.max())*255)
u_out_name = "u_out_REAL_size_" + str(x) + "_dist_" + str(z) + ".bmp"
scipy.misc.imsave(u_out_name, u_out)

#u_out_imag = u_out_imag.imag
u_out_imag = u_out_imag + abs(u_out_imag.min())
u_out_imag = np.rint((u_out_imag / u_out_imag.max())*255)
u_out_name = "u_out_IMAG_size_" + str(x) + "_dist_" + str(z) + ".bmp"
scipy.misc.imsave(u_out_name, u_out_imag)
#plt.imshow(u_out)