import matplotlib.pyplot as plt
import random

config = []
with open("config.txt") as f:
    for line in f:
        config.append(float(line))

num_epochs, freq, _, N, NumberOfBases = config
num_epochs = int(num_epochs)
freq = int(freq)
N = int(N)
NumberOfBases = int(NumberOfBases)

with open("times_train.txt") as f:
    for line in f:
        type, time = line.split()
        if type == "float":
            float_time = float(time)
        elif type == "double":
            double_time = float(time)

kullbach_leibler_norm_float = []
with open("kullbach_leibler_norm_float.txt") as f:
    for line in f:
        kullbach_leibler_norm_float.append(float(line))

epochs = []
for i in range(1, num_epochs + 1):
    if i == 1 or i % freq == 0:
        epochs.append(i)

plt.title('Размер матрицы плотности N = ' + str(N) + ', число базисов - ' + str(NumberOfBases) + 
    ', время обучения float - ' + str(float_time) + ' с', fontweight = 'bold')

#plt.yscale('log')
plt.plot(epochs, kullbach_leibler_norm_float, 'g--')
plt.grid(True, linestyle='-', color='0.75')
plt.xlabel("Номер эпохи, i")
plt.ylabel(r"$ \sum_{b} \sum_{i=1}^{n-1} \rho^b_{Original}(i,i) \cdot \log \frac{\rho^b_{Original}(i,i)}{\rho_{RBM}(i,i)} $")
 
plt.show()