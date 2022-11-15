import matplotlib.pyplot as plt

data_diag_d = []
with open("matrix_diag_double.txt") as f:
	for line in f:
		data_diag_d.append(float(line))

data_diag_f = []
with open("matrix_diag_float.txt") as f:
	for line in f:
		data_diag_f.append(float(line))
         
plt.title("Диагональ матрицы плотности ρ при N = " + str(len(data_diag_d)))
plt.xlabel("Номер элемента, i")
plt.ylabel("ρ(i,i)")

ax = plt.gca()

plt.plot(range(len(data_diag_d)), data_diag_d, label = 'Двойная точность', color = 'g')
plt.plot(range(len(data_diag_f)), data_diag_f, label = 'Одинарная точность', color = 'b')

plt.legend()
plt.grid(True, linestyle='-', color='0.75')

# plt.savefig('graph_diag_' + str(len(data_diag_d))+ '.png')
plt.show()