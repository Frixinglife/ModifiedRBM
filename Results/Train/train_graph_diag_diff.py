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

data_diag_diff = []
for j in range(len(data_diag_f)):
	data_diag_diff_local = []
	for i in range(N):
		data_diag_diff_local.append(abs(data_diag_f_reference[i] - data_diag_f[j][0][i]))
	data_diag_diff.append((data_diag_diff_local, data_diag_f[j][1]))

plt.title("Модуль разности диагоналей эталонной матрицы плотности и матрицы плотности из RBM при N = " + str(N) 
    + ", время обучения - " + str(work_time) + " сек")
plt.xlabel("Номер элемента, i")
plt.ylabel("Модуль разности")

for i in range(len(data_diag_diff)):
    plt.plot(range(N), data_diag_diff[i][0], label = 'epoch = ' + str(data_diag_diff[i][1]), color = colors[i])

plt.legend()

plt.grid(True, linestyle='-', color='0.75')
plt.show()