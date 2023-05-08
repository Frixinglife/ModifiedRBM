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

orig_ro_diag_d, orig_ro_diag_f, rbm_ro_diag_d, rbm_ro_diag_f = [], [], [], []
with open(".\\double\\orig_ro_diag.txt") as f:
    orig_ro_diag_d = [float(line) for line in f]
with open(".\\float\\orig_ro_diag.txt") as f:
    orig_ro_diag_f = [float(line) for line in f]
with open(".\\double\\rbm_ro_diag.txt") as f:
    rbm_ro_diag_d = [float(line) for line in f]
with open(".\\float\\rbm_ro_diag.txt") as f:
    rbm_ro_diag_f = [float(line) for line in f]

N = int(N_d)
diff_d = [abs(orig_ro_diag_d[i] - rbm_ro_diag_d[i]) for i in range(N)]
diff_f = [abs(orig_ro_diag_f[i] - rbm_ro_diag_f[i]) for i in range(N)]

title = 'График зависимости модуля разности диагональных элементов после обучения\n' +\
    	f'для матриц плотности размера {N_f} x {N_f} и {NumberOfBases_f} базисов'.replace('\n', '')

plt.title(title, fontweight = 'bold')
plt.grid(True, linestyle='-', color='0.75')
plt.plot(range(N), diff_d, 'r-', label = f'double, time = {float(time_d):.1f} s, fidelity = {float(fidelity_d):.3f}')
plt.plot(range(N), diff_f, 'b--', label = f'float, time = {float(time_f):.1f} s, fidelity = {float(fidelity_f):.3f}')
plt.xticks(range(N))
plt.legend()
plt.xlabel("Номер элемента, i")
plt.ylabel(r"$|\rho_{Original}(i,i) - \rho_{RBM}(i,i)|$")

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()