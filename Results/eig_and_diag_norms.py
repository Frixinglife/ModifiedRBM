import matplotlib.pyplot as plt

config_f, config_d = [], []
with open(".\\float\\config.txt") as f:
    config_f = [line for line in f]
with open(".\\double\\config.txt") as f:
    config_d = [line for line in f]

num_epochs_f, freq_f, time_f, N_f, NumberOfBases_f, fidelity_f = config_f
num_epochs_d, freq_d, time_d, N_d, NumberOfBases_d, fidelity_d = config_d

if N_f != N_d or num_epochs_f != num_epochs_d: 
    exit()

diag_norm_d, diag_norm_f = [], []
with open(".\\double\\diag_norm.txt") as f:
    diag_norm_d = [float(line) for line in f]
with open(".\\float\\diag_norm.txt") as f:
    diag_norm_f = [float(line) for line in f]

eig_norm_d, eig_norm_f = [], []
with open(".\\double\\eig_norm.txt") as f:
    eig_norm_d = [float(line) for line in f]
with open(".\\float\\eig_norm.txt") as f:
    eig_norm_f = [float(line) for line in f]

epochs = [i for i in range(1, int(num_epochs_f) + 1) if i == 1 or i % int(freq_f) == 0]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
labels = [f'double, time = {float(time_d):.1f} s, fidelity = {float(fidelity_d):.3f}', \
          f'float, time = {float(time_f):.1f} s, fidelity = {float(fidelity_f):.3f}']

title = 'Графики зависимостей Евклидовой нормы разности диагональных элементов и \n' +\
        'максимального по модулю собственного числа разности матриц от числа эпох \n' +\
		f'для матрицы плотности размера {N_f} x {N_f} и {NumberOfBases_f} базисов'.replace('\n', '')
fig.suptitle(title, fontweight = 'bold')

ax1.grid(True, linestyle='-', color='0.75')
ax1.plot(epochs, diag_norm_d, 'r-')
ax1.plot(epochs, diag_norm_f, 'b--')
ax1.set_xlabel("Номер эпохи, i")
ax1.set_ylabel(r"$|| diag(\rho_{RBM}^{i}) - diag(\rho_{Original}) || _{2}$")

ax2.grid(True, linestyle='-', color='0.75')
ax2.plot(epochs, eig_norm_d, 'r-')
ax2.plot(epochs, eig_norm_f, 'b--')
ax2.set_xlabel("Номер эпохи, i")
ax2.set_ylabel(r"$\max_{|\lambda|}(\rho_{RBM}^{i} - \rho_{Original})$")
 
fig.legend(labels=labels, loc='upper right')
 
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()