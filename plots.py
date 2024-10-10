import numpy as np
import matplotlib.pyplot as plt
from model import *


train_height = train_df["Hight"].values
train_shoe_size = train_df["Shoe size"].values

# Extract height and shoe size from test data
test_height = test_df["Hight"].values
test_shoe_size = test_df["Shoe size"].values


# Generate predictions for both training and test data
with torch.no_grad():
    train_predictions = model(height_weight_feature).numpy().flatten()
    test_predictions = model(test_height_weight_feature).numpy().flatten()

# Sort the training data by height for a smoother regression line plot
sorted_indices = np.argsort(train_height)
train_height_sorted = train_height[sorted_indices]
train_predictions_sorted = train_predictions[sorted_indices]


# Sort the test data by height for a smoother regression line plot
sorted_indices_test = np.argsort(test_height)
test_height_sorted = test_height[sorted_indices_test]
test_predictions_sorted = test_predictions[sorted_indices_test]


# Plot training data and regression line
plt.figure(figsize=(10, 6))  # Increase the figure size for better visibility
plt.scatter(train_height, train_shoe_size, label="Training Data", color="blue")
plt.plot(train_height_sorted, train_predictions_sorted, color="red", label="Training Regression Line")

# Plot test data and regression line
plt.scatter(test_height, test_shoe_size, label="Test Data", color="green")
plt.plot(test_height_sorted, test_predictions_sorted, color="orange", label="Test Regression Line")

plt.grid(True, linestyle = "--", alpha = 0.3)

plt.xlabel("Height")
plt.ylabel("Shoe Size")
plt.title("Training and Test Data with Regression Lines")
plt.legend()
plt.show()