import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('new_train.csv', delimiter=',')

pos = train[:, -1] > 0
train_pos = train[pos, :]
train_neg = train[~pos, :]

plt.scatter(train_neg[:, 0], train_neg[:, 1],marker='o', color='', edgecolors='r')
plt.scatter(train_pos[:, 0], train_pos[:, 1],marker='o', color='', edgecolors='g')
# plt.xticks(())
# plt.yticks(())
plt.show()


# '+': 466;   '-': 7819
# splite '-': 488*5+489*11
# 16 pairs training set

# s_num = [488] * 5 + [489] * 11
# s_index1 = [0, 488, 976, 1464, 1952, 2440, 2929, 3418, 3907, 4396, 4885, 5374, 5863, 6352, 6841, 7330]
# s_index2 = [488, 976, 1464, 1952, 2440, 2929, 3418, 3907, 4396, 4885, 5374, 5863, 6352, 6841, 7330, 7819]

# for i in range(16):
#     temp = train_neg[s_index1[i]:s_index2[i], :]
#     data = np.vstack((train_pos, temp))
#     name = 'data_after_pre/new_pre' + str(i) + '.csv'
#     np.savetxt(name, data, delimiter = ',')