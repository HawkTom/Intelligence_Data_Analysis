import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

data_file = 'data_after_pre/new_pre' + str(0) + '.csv'
train = np.loadtxt(data_file, delimiter=',')
x_train = train[:, 0:3]
y_train = train[:, -1]
y_train[y_train == -1] = 0

model = Sequential()
model.add(Dense(1, input_dim=3, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=100, epochs=200, verbose=2, validation_split=0.8)

# plot curve of loss and val_los
fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower left')
#
plt.show()