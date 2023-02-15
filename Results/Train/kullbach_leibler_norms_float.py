import matplotlib.pyplot as plt
import random

config = []
with open("..\\config.txt") as f:
    for line in f:
        config.append(float(line))

num_epochs, freq, _, N, NumberOfBases = config
num_epochs = int(num_epochs)
freq = int(freq)
N = int(N)
NumberOfBases = int(NumberOfBases)

colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for _ in range(NumberOfBases)]

with open("..\\times_train.txt") as f:
    for line in f:
        type, time = line.split()
        if type == "float":
            float_time = float(time)
        elif type == "double":
            double_time = float(time)

kullbach_leibler_norm_float = []
with open("..\\kullbach_leibler_norm_float.txt") as f:
    for line in f:
        kullbach_leibler_norm_float.append(float(line))

kullbach_leibler_norms_float = []
for i in range(NumberOfBases):
    with open("..\\kullbach_leibler_norm_float_" + str(i) + ".txt") as f:
        part_kullbach_leibler_norms_float = []
        for line in f:
            part_kullbach_leibler_norms_float.append(float(line))
        kullbach_leibler_norms_float.append(part_kullbach_leibler_norms_float)

epochs = []
for i in range(1, num_epochs + 1):
    if i == 1 or i % freq == 0:
        epochs.append(i)

plt.title('Размер матрицы плотности N = ' + str(N) + ', число базисов - ' + str(NumberOfBases) + 
    ', время обучения float - ' + str(float_time) + ' с', fontweight = 'bold')

plt.yscale('log')
for i in range(NumberOfBases):
    plt.plot(epochs, kullbach_leibler_norms_float[i], color = colors[i], linestyle = '-', label = 'Базис ' + str(i + 1))
plt.plot(epochs, kullbach_leibler_norm_float, 'g--', label = 'По всем базисам')

plt.grid(True, linestyle='-', color='0.75')
plt.legend()
plt.xlabel("Номер эпохи, i")
plt.ylabel(r"$ \sum_{b} \sum_{i=0}^{n-1} \rho^b_{Original}(i,i) \cdot \log \frac{\rho^b_{Original}(i,i)}{\rho_{RBM}(i,i)} $")
 
plt.show()