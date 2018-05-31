import numpy as np
from sklearn.svm import SVC

data_file = 'data_after_pre/new_pre' + str(1) + '.csv'
train = np.loadtxt(data_file, delimiter=',')
pre = np.loadtxt('new_pre.csv', delimiter=',')
x_train = train[:, 0:2]
y_train = train[:, -1]
x_pre = pre[0:235, 0:2]
y_pre = pre[0:235, -1]
# print(x_train.shape, y_train.shape)
clf = SVC()
clf.fit(x_train, y_train)
print(clf.score(x_pre, y_pre))