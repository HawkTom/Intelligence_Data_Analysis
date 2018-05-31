import numpy as np
# from sklearn.decomposition import PCA

# read data from the orignal data file
x_train = []
y_train = []
with open('dataset/train.data', 'r') as f:
    for line in f.readlines():
        data = line.split(' ')
        x_train.append(list(map(float,data[0:-1])))
        y_train.append(int(data[-1][0:-1]))

x_train = np.array(x_train)
y_train = np.array(y_train, dtype=int)[:,np.newaxis]


x_pre = []
y_pre = []
with open('dataset/test.data', 'r') as f:
    for line in f.readlines():
        data = line.split(' ')
        x_pre.append(list(map(float,data[0:-1])))
        y_pre.append(int(data[-1][0:-1]))

x_pre = np.array(x_pre)
y_pre = np.array(y_pre, dtype=int)[:,np.newaxis]


# calculate the mean of each features
x_mean = np.mean(x_train, axis=0)
x_train_new = x_train - x_mean
x_pre_new = x_pre - x_mean

# calculate the covariance of features
x_var = np.cov(x_train, rowvar=0)

# calculate the eigen value and eigen vector of covariance matrix
x_eig_val, x_eig_vec = np.linalg.eig(x_var)

# sort the eigen value 
eig_index = np.argsort(x_eig_val)

# select the 2 eigen vector correspond to max 2 eigen value
eig_select_index = eig_index[-1: -4: -1]
eigvec_select = x_eig_vec[:, eig_select_index]

# reduce dimension by the eigen vector for train data
low_dim_train_data = np.dot(x_train_new, eigvec_select)
low_dim_pre_data = np.dot(x_pre_new, eigvec_select)

# write data to a new file after pre-processing
train = np.hstack((low_dim_train_data, y_train))
pre = np.hstack((low_dim_pre_data, y_pre))
np.savetxt('new.csv', train, delimiter = ',')
np.savetxt('new_pre.csv', pre, delimiter = ',')




