import numpy as np
from svm_cvxopt import SVM, gaussian_kernel

train = np.loadtxt('data_after_pre/new_pre0.csv', delimiter=',')
x = train[:, 0:-1]
y = train[:, -1]
pos = y > 0
train_pos = train[pos]
train_neg = train[~pos]

# use 10-fold cross-validation
# splite train data into 10 subsamples

pos_split_set = np.array_split(train_pos, 10, axis=0)
neg_split_set = np.array_split(train_neg, 10, axis=0)

train_split = []
for i in range(10):
    temp = np.vstack((pos_split_set[i], neg_split_set[i]))
    train_split.append(temp)

# print(train_split)
train_split = np.array(train_split)

est_C_s = np.zeros((11, 10))

C = 2**6
sigma = 2**1.66
F1 = np.zeros((1, 10))
for i in range(10):
    test_i = train_split[i]
    if i is 0:
        train_i = np.concatenate((train_split[i+1:]))
    elif i is 9:
        train_i = np.concatenate((train_split[:i]))
    else:
        tmp1 = np.concatenate((train_split[:i]))
        tmp2 = np.concatenate((train_split[i+1:]))
        train_i = np.vstack((tmp1, tmp2))
    x_train_i = train_i[:, 0:-1]
    y_train_i = np.ravel(train_i[:, -1])
    x_test_i = test_i[:, 0:-1]
    y_test_i = np.ravel(test_i[:, -1])

    clf = SVM(kernel=gaussian_kernel, C=C, sigma=sigma)
    clf.fit(x_train_i, y_train_i)
    y_pred_i = clf.predict(x_test_i)
    # F1: 2*TP / (samples + TP - TN)
    actual = y_test_i > 0
    pred = y_pred_i > 0
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
    F1_score1 = 2 * TP / (TP + FN + FP + TN + TP - TN)
    # print(TP, FN, FP, TN, x_test_i.shape[0], F1)
    F1[0, i] = F1_score
    record = [TP, FN, TN, FP, recall, precision, F1_score, F1_score1, acc]
    print(record)
    # np.save('clf0', clf)