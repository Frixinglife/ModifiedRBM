import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 18})

data_diag_d = []
with open("matrix_diag_double.txt") as f:
	for line in f:
		data_diag_d.append(float(line))

data_diag_f = []
with open("matrix_diag_float.txt") as f:
	for line in f:
		data_diag_f.append(float(line))

N = len(data_diag_d)

data_diag_diff = []
for i in range(N):
	data_diag_diff.append(abs(data_diag_d[i] - data_diag_f[i]))
         
plt.title("Модуль разности диагоналей матриц плотности ρ для double и float при N = " + str(N))
plt.xlabel("Номер элемента, i")
plt.ylabel("Модуль разности")

plt.plot(range(N), data_diag_diff, color = 'r')

plt.grid(True, linestyle='-', color='0.75')

# plt.savefig('graph_diag_diff_' + str(len(data_diag_d))+ '.png')
plt.show()