import numpy as np
from numpy import linalg
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

def gaussian_kernel(x, y, sigma):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

class SVM(object):

    def __init__(self, kernel, C=None, sigma=0.1):
        # select kernel type
        self.kernel = kernel
        self.sigma = sigma
        # determine the coeffients of slack verables  
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        # X: n_samples x n_dimension
        # y: n_samples (vector)
        n_samples, _ = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j], self.sigma)
        # P_ij =  y_i*y_j*K(x_i, x_j)
        P = matrix(np.outer(y,y) * K)
        # q = [-1, ..., -1]
        q = matrix(np.ones(n_samples) * -1)
        # A = [y_1, y_2, ..., y_n]
        A = matrix(y, (1,n_samples))
        # b = 0
        b = matrix(0.0)

        if self.C is None:
            # G = -I
            G = matrix(np.diag(np.ones(n_samples) * -1))
            # h = [0, ..., 0]
            h = matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            # G = [-I; I]
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            # h = [0,...,0 ,C, ..., C];
            h = matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        # np.ravel() change matrix into one vector
        a = np.ravel(solution['x'])

        # model with support vector
        sv = a > 1e-5 
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # calculate b
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

    def project(self, X):
        y_project = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv, self.sigma)
            y_project[i] = s + self.b
        return y_project


    def predict(self, X):
        y_predict = np.sign(self.project(X))
        return y_predict

def result_plot(x, y, svm):
    place = np.argwhere(y == 1)
    label1 = x[place[:, 0]]
    place = np.argwhere(y == -1)
    label2 = x[place[:, 0]]
    plt.plot(label1[:, 0], label1[:, 1], '.', color='orange')
    plt.plot(label2[:, 0], label2[:, 1], '.', color='blue')

    x1 = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
    x2 = np.linspace(min(x[:, 1]), max(x[:, 1]), 100)
    X, Y = np.meshgrid(x1, x2)
    X_grid = np.vstack((np.ravel(X), np.ravel(Y))).transpose()
    vals = svm.predict(X_grid)
    vals = vals.reshape((100, 100))
    plt.contour(X, Y, vals)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def main():
    # read data from the data file
    with open("svm_data2.txt", 'r') as f:
        datas = f.read()
        datas = datas.split('\n')
        x, y = [], []
        for sample in datas:
            s = sample.split('\t')
            x.append([float(s[0]), float(s[1])])
            y.append(float(s[2]))
    x = np.array(x)
    y = np.array(y)
    y.shape = (y.shape[0], 1)
    place = np.argwhere(y == 0)
    y[place[:, 0]] = -1
    y = np.ravel(y)
    clf = SVM(kernel=gaussian_kernel, C=0.5)
    clf.fit(x, y)
    result_plot(x, y, clf)


if __name__ == '__main__':
    main()
