import numpy as np
import shelve
from auc import roc

test = np.loadtxt('new_pre.csv', delimiter=',')

x_test = test[:, 0:-1]
y_test = np.ravel(test[:, -1])

y_pre_vote = np.zeros((16, len(y_test)))

models = shelve.open('classifier/model')

for i in range(16):
    clf_name = 'clf' + str(i)
    clf = models[clf_name]
    y_pre_vote[i, :] = clf.project(x_test)

np.savetxt('y_pre_vote.csv', y_pre_vote, delimiter=',')
proj = np.sum(y_pre_vote, axis=0)
print(proj)
r = roc(y_test, proj)
r.plot()
print(r.auc)
# auc: 0.7687586071217752
y_pre = np.sign(np.sum(y_pre_vote, axis=0))

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
# [84, 33, 1388, 567, 0.717948717948718, 0.12903225806451613, 0.21875, 0.7104247104247104]