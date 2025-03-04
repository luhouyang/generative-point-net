import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# file_path = r"C:\Users\User\Desktop\Python\deep_learning\generative_point_net\src\pointnet\output\classification\no_normal\modelnet10\log.csv"
# file_path = r"C:\Users\User\Desktop\Python\deep_learning\generative_point_net\src\pointnet\output\classification\no_normal\modelnet40\log.csv"
# file_path = r"C:\Users\User\Desktop\Python\deep_learning\generative_point_net\src\pointnet\output\classification\normal\modelnet10\log.csv"
# file_path = r"C:\Users\User\Desktop\Python\deep_learning\generative_point_net\src\pointnet\output\classification\normal\modelnet40\log.csv"

# file_path =r"C:\Users\User\Desktop\Python\deep_learning\generative_point_net\src\pointnet\output\part_seg\no_normal\log.csv"
file_path = r"C:\Users\User\Desktop\Python\deep_learning\generative_point_net\src\pointnet\output\part_seg\normal\log.csv"

df = pd.read_csv(file_path)

epochs = df.iloc[:, 0]
train_loss = df.iloc[:, 1]
train_acc = df.iloc[:, 2]
test_loss = df.iloc[:, 3]
test_acc = df.iloc[:, 4]

mn_test_loss = np.min(test_loss)
mx_test_acc = np.max(test_acc)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Train Loss", marker="o")
plt.plot(epochs, test_loss, label="Test Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Training and Test Loss | min: {mn_test_loss:.4f}")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
plt.plot(epochs, test_acc, label="Test Accuracy", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(f"Training and Test Accuracy | max: {mx_test_acc:.4f}")
plt.legend()
plt.grid()
plt.show()
