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

eig_norm_d, eig_norm_f = [], []
with open(".\\double\\eig_norm.txt") as f:
    eig_norm_d = [float(line) for line in f]
with open(".\\float\\eig_norm.txt") as f:
    eig_norm_f = [float(line) for line in f]

epochs = [i for i in range(1, int(num_epochs_f) + 1) if i == 1 or i % int(freq_f) == 0]
title = 'График зависимости максимального по модулю собственного числа разности матриц от числа эпох \n' +\
		f'для матрицы плотности размера {N_f} x {N_f} и {NumberOfBases_f} базисов'.replace('\n', '')

plt.title(title, fontweight = 'bold')
plt.grid(True, linestyle='-', color='0.75')
plt.plot(epochs, eig_norm_d, 'r-', label = f'double, time = {float(time_d):.1f} s, fidelity = {float(fidelity_d):.3f}')
plt.plot(epochs, eig_norm_f, 'b--', label = f'float, time = {float(time_f):.1f} s, fidelity = {float(fidelity_f):.3f}')
plt.legend()
plt.xlabel("Номер эпохи, i")
plt.ylabel(r"$\max_{|\lambda|}(\rho_{RBM}^{i} - \rho_{Original})$")

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()