import pandas as pd
import matplotlib.pyplot as plt

file_path = "C:/Users/User/Desktop/Python/deep_learning/generative_point_net/src/pointnet_plus2/output/log.csv"
df = pd.read_csv(file_path)

epochs = df.iloc[:, 0]
train_loss = df.iloc[:, 1]
train_acc = df.iloc[:, 2]
test_loss = df.iloc[:, 3]
test_acc = df.iloc[:, 4]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Train Loss", marker="o")
plt.plot(epochs, test_loss, label="Test Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
plt.plot(epochs, test_acc, label="Test Accuracy", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()
plt.grid()
plt.show()
