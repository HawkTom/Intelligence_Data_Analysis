import numpy as np
import matplotlib.pyplot as plt

class roc(object):

    def __init__(self, y_test, y_proj):
        n = len(y_test)
        y_test = y_test[y_proj.argsort()]

        y_roc = np.ones_like(y_test)
        
        self.roc = np.zeros((2, n+3))

        for i in range(n+1):
            if i is not 0:
                y_roc[i-1] = -1
            actual = y_test > 0
            pred = y_roc > 0
            TP = sum(actual & pred)
            FN = sum(actual) - TP
            TN = sum(~(actual | pred))
            FP = sum(pred) - TP
            FPR = 1.0 * FP / (TN + FP)
            TPR = 1.0 * TP / (FN + TP)
            self.roc[0, i] = FPR
            self.roc[1, i] = TPR
        

        self.auc = 0
        for i in range(n+1):
            self.auc += (self.roc[0, i] - self.roc[0, i+1]) * (self.roc[1, i] + self.roc[1, i+1])

        self.auc /= 2.0
    
    def plot(self):
        self.roc[0, -1] = 1
        self.roc[1, -1] = 0
        plt.plot(self.roc[0, 0:-1], self.roc[1, 0:-1], 'b' , alpha=0.3)
        plt.plot(self.roc[0, 0:-1], self.roc[1, 0:-1], '.', color = 'g' )
        plt.fill(self.roc[0, :], self.roc[1, :], 'b' , alpha=0.3)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC')
        plt.show()