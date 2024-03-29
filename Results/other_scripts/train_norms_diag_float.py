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

data_diag_norm_f = []
with open("..\\diag_norm_float.txt") as f:
    for line in f:
        data_diag_norm_f.append(float(line))

# data_diag_norm_d = []
# with open("..\\diag_norm_double.txt") as f:
#     for line in f:
#         data_diag_norm_d.append(float(line))

data_eig_norm_f = []
with open("..\\eig_norm_float.txt") as f:
    for line in f:
        data_eig_norm_f.append(float(line))

# data_eig_norm_d = []
# with open("..\\eig_norm_double.txt") as f:
#     for line in f:
#         data_eig_norm_d.append(float(line))

data_diag_norms_basis_f = []
data_eig_norms_basis_f = []
for i in range(NumberOfBases):
    with open("..\\diag_norm_float_" + str(i) + ".txt") as f:
        part_data_diag_norms_basis_f = []
        for line in f:
            part_data_diag_norms_basis_f.append(float(line))
        data_diag_norms_basis_f.append(part_data_diag_norms_basis_f)
    with open("..\\eig_norm_float_" + str(i) + ".txt") as f:
        part_data_eig_norms_basis_f = []
        for line in f:
            part_data_eig_norms_basis_f.append(float(line))
        data_eig_norms_basis_f.append(part_data_eig_norms_basis_f)

epochs = []
for i in range(1, num_epochs + 1):
    if i == 1 or i % freq == 0:
        epochs.append(i)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# labels = ["float", "double"]
labels = ["float"]
 
# fig.suptitle('Размер матрицы плотности N = ' + str(N) + ', время обучения double - ' + 
#     str(double_time) + ' c, время обучения float - ' + str(float_time) + ' с', fontweight = 'bold')

fig.suptitle('Размер матрицы плотности N = ' + str(N) + ', время обучения float - ' + str(float_time) + ' с', fontweight = 'bold')

# ax1.plot(epochs, data_diag_norm_f, 'g--', epochs, data_diag_norm_d, 'r^')
for i in range(NumberOfBases):
    ax1.plot(epochs, data_diag_norms_basis_f[i], color = colors[i], label = 'Базис ' + str(i + 1))
ax1.plot(epochs, data_diag_norm_f, 'g--', label = 'Не в базисах')
ax1.grid(True, linestyle='-', color='0.75')
ax1.set_xlabel("Номер эпохи, i")
ax1.set_ylabel(r"$|| diag(\rho_{RBM}^{i}) - diag(\rho_{Original}) || _{2}$")
ax1.legend()
ax1.set_yscale("log")

# ax2.plot(epochs, data_eig_norm_f, 'g--', epochs, data_eig_norm_d, 'r^')
for i in range(NumberOfBases):
    ax2.plot(epochs, data_eig_norms_basis_f[i], color = colors[i], label = 'Базис ' + str(i + 1))
ax2.plot(epochs, data_eig_norm_f, 'g--', label = 'Не в базисах')
ax2.grid(True, linestyle='-', color='0.75')
ax2.set_xlabel("Номер эпохи, i")
ax2.set_ylabel(r"$\max_{|\lambda|}(\rho_{RBM}^{i} - \rho_{Original})$")
ax2.legend()
ax2.set_yscale("log")
 
# fig.legend(labels=labels, loc='upper right')
 
plt.show()