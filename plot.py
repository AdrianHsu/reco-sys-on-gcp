import os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn')
f = open("loss.txt", 'r')

cnt = 0
train_loss = []
test_loss = []
for line in f:
    line = line.split(",")

    train_loss.append(float(line[0]))
    test_loss.append(float(line[1]))

train_loss = train_loss[::-1]
test_loss = test_loss[::-1]

plt.title("Min Square Error (MSE Loss) for MovieLens 100k")
plt.plot(train_loss, label="training set")
plt.plot(test_loss, label="testing set")

plt.legend()
plt.xlabel("Epoch #")
plt.ylabel("MSE Loss")
plt.show()