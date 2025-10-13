# %% [markdown]
# Imports

# %%
import numpy, matplotlib.pyplot as plt

# %% [markdown]
# Exercitiul 1

# %% [markdown]
# Subpunctul a, b

# %%
t = numpy.linspace(0, 0.03, int(0.03 / 0.0005))
x = numpy.cos(520 * numpy.pi * t + numpy.pi / 3)
y = numpy.cos(280 * numpy.pi * t - numpy.pi / 3)
z = numpy.cos(120 * numpy.pi * t + numpy.pi / 3)

# Semnal x(t)
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Semnal x(t)')
plt.show()
plt.savefig(fname = './ex1_semnal_x.pdf', format = 'pdf')

# Semnal y(t)
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Semnal y(t)')
plt.show()
plt.savefig(fname = './ex1_semnal_y.pdf', format = 'pdf')

# Semnal z(t)
plt.plot(t, z)
plt.xlabel('t')
plt.ylabel('z(t)')
plt.title('Semnal z(t)')
plt.show()
plt.savefig(fname = './ex1_semnal_z.pdf', format = 'pdf')


# %% [markdown]
# Subpunctul c

# %%
duration = 0.03
fs = 200 
samples = numpy.linspace(0, duration, int(duration * fs))
samples_fine = numpy.linspace(0, duration, int(duration * fs * 100))
x_samples = numpy.cos(520 * numpy.pi * samples + numpy.pi / 3)
y_samples = numpy.cos(280 * numpy.pi * samples - numpy.pi / 3)
z_samples = numpy.cos(120 * numpy.pi * samples + numpy.pi / 3)
x_samples_fine = numpy.cos(520 * numpy.pi * samples_fine + numpy.pi / 3)
y_samples_fine = numpy.cos(280 * numpy.pi * samples_fine - numpy.pi / 3)
z_samples_fine = numpy.cos(120 * numpy.pi * samples_fine + numpy.pi / 3)
# Semnal x(samples)
plt.plot(samples_fine, x_samples_fine)
plt.stem(samples, x_samples)
plt.xlabel('samples')
plt.ylabel('x(samples)')
plt.title('Semnal x(samples)')
plt.show()
plt.savefig(fname = './ex1_semnal_x_esantioane.pdf', format = 'pdf')

# Semnal y(samples)
plt.plot(samples_fine, y_samples_fine)
plt.stem(samples, y_samples)
plt.xlabel('samples')
plt.ylabel('y(samples)')
plt.title('Semnal y(samples)')
plt.show()
plt.savefig(fname = './ex1_semnal_y_esantioane.pdf', format = 'pdf')

# Semnal z(samples)
plt.plot(samples_fine, z_samples_fine)
plt.stem(samples, z_samples)
plt.xlabel('samples')
plt.ylabel('z(samples)')
plt.title('Semnal z(samples)')
plt.show()
plt.savefig(fname = './ex1_semnal_z_esantionane.pdf', format = 'pdf')

# %% [markdown]
# Exercitiul 2

# %% [markdown]
# Subpunctul a

# %%
frec_a = 400
nr_samples = 1600
t2 = numpy.linspace(0, nr_samples / frec_a, nr_samples)
x2 = numpy.sin(2 * numpy.pi * frec_a * t2)
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(t2, x2)
plt.xlabel('t2')
plt.ylabel('x2(t2)')
plt.title('Semnal x2(t2)')
plt.show()
plt.savefig(fname = './ex2_semnal_x2_1200_esantionane.pdf', format = 'pdf')

# %% [markdown]
# Subpunctul b

# %%
frec_b = 800
duration_b = 3
nr_samples_b = int(frec_b * duration_b)
t3 = numpy.linspace(0, duration_b, nr_samples_b)
x3 = numpy.sin(2 * numpy.pi * frec_b * t3)
plt.plot(t3, x3)
plt.rcParams["figure.figsize"] = (5,5)
plt.xlabel('t3')
plt.ylabel('x3(t3)')
plt.title('Semnal x3(t3)')
plt.show()
plt.savefig(fname = './ex2_semnal_x2_800frecventa.pdf', format = 'pdf')

# %% [markdown]
# Subpunctul c

# %%
frec_c = 240
duration_c = 0.03
fs_c = 7000  
t_c = numpy.linspace(0, duration_c, int(fs_c * duration_c))
saw_c = 2 * (t_c * frec_c - numpy.floor(0.5 + t_c * frec_c))  # amplitudine [-1, 1]

plt.plot(t_c, saw_c)
plt.xlabel('t_c')
plt.ylabel('sawtooth(t_c)')
plt.title('Semnal sawtooth, 240 Hz')
plt.show()
plt.savefig(fname = './ex2_semnal_sawtooth_240Hz.pdf', format = 'pdf')

# %% [markdown]
# Subpunctul d

# %%
frec_d = 300
duration_d = 0.01
fs_d = 100000
t_d = numpy.linspace(0, duration_d, int(fs_d * duration_d))
square_d = numpy.sign(numpy.sin(2 * numpy.pi * frec_d * t_d))

plt.plot(t_d, square_d)
plt.xlabel('t_d')
plt.ylabel('square(t_d)')
plt.title('Semnal square, 300 Hz')
plt.grid(True)
plt.show()
plt.savefig(fname = './ex2_semnal_square_300Hz.pdf', format = 'pdf')

# %% [markdown]
# Subpunctul e

# %%
rand_array = numpy.random.rand(128, 128)
plt.imshow(rand_array)
plt.title('Semnal 2D aleator')
plt.colorbar()
plt.show()
plt.savefig(fname = './ex2_semnal_2D_aleator.pdf', format = 'pdf')

# %% [markdown]
# Subpunctul f

# %%
semnal_2d = numpy.zeros((128, 128))
for i in range(128):
    for j in range(128):
        if (int(numpy.round(numpy.log(j + 1), decimals = 2) * 10)) % 2:
            semnal_2d[i, j] = 1
        else:
            semnal_2d[i, j] = 0

plt.imshow(semnal_2d)
plt.title('Semnal 2D propriu')
plt.colorbar()
plt.show()
plt.savefig(fname = './ex2_semnal_2D_propriu.pdf', format = 'pdf')

# %% [markdown]
# Exercitiul 3

# %% [markdown]
# Subounctul a

# %%
# fs = 1 / T => T = 1/fs
fs = 2000
T = 1 / fs
print("Timpul de esantionare este: ", T)

# %% [markdown]
# Subpunctul b

# %%
# 1 byte = 8 biti
# 2000 esantionane pe sec => 2000 * 60 * 60 esantionae pe ora = 7.2 milioane esantioane pe ora
# 7.2 milioane esantioane pe ora * 4 biti = 28.8 milioane biti pe ora
# 28.8 milioane biti pe ora / 8 = 3.6 milioane bytes
print("Dimensiunea in bytes pentru o ora de inregistrare este: ", int(2000 * 60 * 60 * 4 / 8), " bytes")


