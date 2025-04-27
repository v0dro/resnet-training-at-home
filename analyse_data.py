import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.out")
test = pd.read_csv("test.out")
num_epochs = 50

print(f"Total training time: {train['iter_time'].sum() / 1e3 / 60 / 60} hours")
print(f"Avg. test accuracy: {test['accuracy'].mean() * 100:.2f}%")

average_accuracy = list()
for epoch in range(num_epochs):
    average_accuracy.append(train[train['epoch'] == epoch]['accuracy'].mean() * 100)

plt.plot(list(range(num_epochs)),  average_accuracy, label='Train Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy of ResNet101 on CIFAR-10")
plt.grid(which="both")
plt.legend()
plt.savefig("train_accuracy.png")
plt.show()