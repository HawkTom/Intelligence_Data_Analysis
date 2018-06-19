import numpy as np
from svm_cvxopt import SVM, gaussian_kernel
from auc import roc


train = np.loadtxt('new_train.csv', delimiter=',')
x_train = train[:, 0:-1]
y_train = train[:, -1]

C = 2**6.33
sigma = 2**1.67
clf = SVM(kernel=gaussian_kernel, C=C, sigma=sigma)
clf.fit(x_train, y_train)

test = np.loadtxt('new_pre.csv', delimiter=',')
x_test = test[:, 0:-1]
y_test = np.ravel(test[:, -1])

proj = clf.project(x_test)
print(proj)
r = roc(y_test, proj)
r.plot()
print(r.auc)
# auc = 0.5175989682383516
y_pre = np.sign(proj)
actual = y_test > 0
pred = y_pre > 0
TP = sum(actual & pred)
FN = sum(actual) - TP
TN = sum(~(actual | pred))
FP = sum(pred) - TP
# accuracy
acc = (TP + TN) / (TP + FN + FP + TN)
# recall
recall = 1.0 * TP / (TP + FN)
# precision
precision = 1.0 * TP / (TP + FP )
# F1- score
F1_score = (2.0 * recall * precision) / (recall + precision)
record = [TP, FN, TN, FP, recall, precision, F1_score, acc]
print(record)
# [0, 117, 1954, 1, 0.0, 0.0, nan, 0.943050193050193]