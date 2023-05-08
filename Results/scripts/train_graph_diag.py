import matplotlib.pyplot as plt
import random

config = []
with open("..\\config.txt") as f:
    for line in f:
        config.append(float(line))

num_epochs, freq, work_time, N = config
num_epochs = int(num_epochs)
freq = int(freq)

colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for _ in range(num_epochs // freq + 1)]

data_diag_f_reference = []
with open("..\\matrix_diag_float.txt") as f:
    for line in f:
        data_diag_f_reference.append(float(line))

data_diag_f = []
for i in range(1, num_epochs + 1):
    if i == 1 or i % freq == 0:
        with open("matrix_diag_float_" + str(i) + ".txt") as f:
            part_data_diag_f = []
            for line in f:
                part_data_diag_f.append(float(line))
            data_diag_f.append((part_data_diag_f, i))

plt.title("Диагональ матрицы плотности ρ при N = " + str(N) + ", число эпох - " 
    + str(num_epochs) + ", время обучения - " + str(work_time) + " сек")

plt.xlabel("Номер элемента, i")
plt.ylabel("ρ(i,i)")

plt.plot(range(N), data_diag_f_reference, label = 'Одинарная точность, эталонные значения', color = 'black')
for i in range(len(data_diag_f)):
    plt.plot(range(N), data_diag_f[i][0], label = 'Одинарная точность, epoch = ' + str(data_diag_f[i][1]), color = colors[i])

plt.legend()
plt.grid(True, linestyle='-', color='0.75')
plt.show()