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

# Plot training data and regression line
plt.figure(figsize=(10, 6))  # Increase the figure size for better visibility
plt.scatter(train_height, train_shoe_size, label="Training Data", color="blue", alpha=0.7)  # Adjust alpha for transparency
plt.plot(train_height, train_predictions, color="red", label="Training Regression Line", linewidth=2)  # Increase linewidth

# Plot test data and regression line
plt.scatter(test_height, test_shoe_size, label="Test Data", color="green", alpha=0.7)
plt.plot(test_height, test_predictions, color="orange", label="Test Regression Line", linewidth=2)

plt.xlabel("Height (cm)", fontsize=14)  # Increase font size for labels
plt.ylabel("Shoe Size (EU)", fontsize=14)
plt.title("Regression of Shoe Size on Height", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)  # Add a grid for better readability
plt.tight_layout()  # Adjust layout for better spacing
plt.show()