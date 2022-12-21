import matplotlib.pyplot as plt
import random

config = []
with open("config.txt") as f:
    for line in f:
        config.append(float(line))

num_epochs, freq, _, N, NumberOfBases, alpha = config
num_epochs = int(num_epochs)
freq = int(freq)
N = int(N)
NumberOfBases = int(NumberOfBases)

colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for _ in range(NumberOfBases)]

with open("times_train.txt") as f:
    for line in f:
        type, time = line.split()
        if type == "float":
            float_time = float(time)
        elif type == "double":
            double_time = float(time)

fidelity_float = []
for i in range(NumberOfBases):
    with open("fidelity_float_" + str(i) + ".txt") as f:
        part_fidelity_float = []
        for line in f:
            part_fidelity_float.append(float(line))
        fidelity_float.append(part_fidelity_float)

epochs = []
for i in range(1, num_epochs + 1):
    if i == 1 or i % freq == 0:
        epochs.append(i)

plt.title('Размер матрицы плотности N = ' + str(N) + ', число базисов - ' + str(NumberOfBases) + 
    r', $ p_{dep} $ - ' + str(alpha) + ', время обучения float - ' + str(float_time) + ' с', fontweight = 'bold')

for i in range(NumberOfBases):
    plt.plot(epochs, fidelity_float[i], color = colors[i], linestyle = '-', label = 'Базис ' + str(i + 1))

plt.grid(True, linestyle='-', color='0.75')
plt.legend()
plt.xlabel("Номер эпохи, i")
plt.ylabel(r"$ Tr\{\sqrt{\sqrt{\rho_{\lambda^*\mu^*}}\varrho\sqrt{\rho_{\lambda^*\mu^*}}}\}} $")
 
plt.show()