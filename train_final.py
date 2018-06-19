import numpy as np
from svm_cvxopt import SVM, gaussian_kernel
import shelve

para = np.loadtxt('para_star.csv', delimiter=',')
model = shelve.open('classifier/model')
for i in range(16):
    file_name = 'data_after_pre/new_pre' + str(i) + '.csv'
    train = np.loadtxt(file_name, delimiter=',')
    x_train = train[:, 0:-1]
    y_train = train[:, -1]

    C = 2**para[0, i]
    sigma = 2**para[1, i]  

    clf = SVM(kernel=gaussian_kernel, C=C, sigma=sigma)
    clf.fit(x_train, y_train)

    save_name = 'clf' + str(i)
    model[save_name] = clf


