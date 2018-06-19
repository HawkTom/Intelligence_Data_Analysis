import numpy as np
from sklearn.svm import SVC
from sklearn import metrics

data_file = 'new_train' + '.csv'
train = np.loadtxt(data_file, delimiter=',')
pre = np.loadtxt('new_pre.csv', delimiter=',')
x_train = train[:, 0:2]
y_train = train[:, -1]
x_test  = pre[:, 0:2]
y_test = pre[:, -1]
# print(x_train.shape, y_train.shape)
clf = SVC(kernel="rbf", class_weight={1:20},C = 2**6)
clf.fit(x_train, y_train)
y_pre= clf.predict(x_test)

test_auc = metrics.roc_auc_score(y_test, y_pre)
print(test_auc)

# F1: 2*TP / (samples + TP - TN)
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
