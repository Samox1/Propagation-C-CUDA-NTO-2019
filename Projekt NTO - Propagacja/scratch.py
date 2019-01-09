import numpy as np
import scipy.misc
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

path_file = Path("Test_NTO_1000.bmp")
im = Image.open(path_file).convert('L')

u_in = np.array(im)
u_in = u_in / u_in.max()
print(u_in[0][0])

width, height = im.size
x = height
y = width
print(x,y)

h_z = np.zeros((x,y))
lam = 633*(pow(10,(-9)))
k = 2*np.pi/lam
z = 500*(pow(10,(-3)))   #odlegosc symulacji

print(u_in)
print(lam)
print(k)
print(z)

i=0
o=0
for i in range(x):
    for o in range(y):
        h_z[i,o]= (np.exp(1j*k*z)/(1j*lam*z))*np.exp(1j*k*(pow(i,2) + pow(o,2))/(2*z))
print(h_z[0][0])

U_fft = np.array((x,y))
U_fft = scipy.multiply(np.fft.fft2(u_in), np.fft.fft2(h_z))
print(U_fft[0][0])

u_out = np.fft.ifft2((U_fft))
#u_out = np.fft.ifft2(np.fft.fftshift(U_fft))
print("KARRAMBA WYNIK")
print(u_out[0][0])

u_out = u_out.real
u_out = u_out + abs(u_out.min())
u_out = np.rint((u_out / u_out.max())*255)
print(u_out[0][0])

u_out_name = "u_out_size_" + str(x) + "_dist_" + str(z) + ".bmp"
scipy.misc.imsave(u_out_name, u_out)
#plt.imshow(u_out)