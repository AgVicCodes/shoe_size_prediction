import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# from google.colab import drive
# drive.mount('/content/drive')

# df = pd.read_csv("/content/drive/MyDrive/Colab/csv/height.csv")

df = pd.read_csv("height.csv")

df.head()

df.shape

# Method 1

test_df = df.sample(10, random_state = 23)
train_df = df.drop(test_df.index)

test_df.head()

train_df.head()

# Method 2

train_df, test_df = train_test_split(df, test_size = 0.1, random_state = 42)

# print("Test DataFrame:")
# test_df.head()

# print("\nTraining DataFrame:")
# train_df.head()

"""
    - Create a sequential layer
    - Convert series to tensor inputs
    - Create a model (ignore activation function for now)
"""

height_weight_feature = torch.tensor((train_df[["Hight", "Weight"]].values).astype("float32"))
target = torch.tensor((train_df["Shoe size"].values).astype("float32"))
# height_input = height_input.type(torch.float32)
# weight_input = torch.tensor(train_df["Weight"].values)

linear_layer = nn.Linear(in_features = 2, out_features = 1)

output = linear_layer(height_weight_feature)

# model = nn.Sequential(
#     nn.Linear(in_features = 2, out_features = 10),
#     nn.Linear(in_features = 10, out_features = 5),
#     nn.Linear(in_features = 5, out_features = 1)
# )


model = nn.Sequential(
	nn.Linear(2, 4),
	nn.LeakyReLU(),
	nn.Linear(4, 8),
	nn.LeakyReLU(),
	nn.Linear(8, 16),
	nn.LeakyReLU(),
	nn.Linear(16, 8),
	nn.LeakyReLU(),
	nn.Linear(8, 2),
	nn.LeakyReLU(),
	nn.Linear(2, 1)
)



# model = nn.Sequential(
#     nn.Linear(2, 4),
#     nn.Linear(4, 8),
#     nn.Linear(8, 16),
#     nn.Linear(16, 8),
#     nn.Linear(8, 2),
#     nn.Linear(2, 1)
#     # nn.Sigmoid()
# )

pre_activation_output = model(height_weight_feature)

# sigmoid = nn.Sigmoid()

# model = nn.Sequential(
#     nn.Linear(2, 10),
#     nn.Linear(10, 5),
#     nn.Linear(5, 1)
#     # nn.Sigmoid()
# )

output = model(height_weight_feature)

# - **Loss Function and Optimizer:** You haven't defined a loss function (e.g., MSE for regression) or an optimizer (e.g., SGD, Adam) to train the model.
# - **Training Loop:**  You've created the model, but you haven't implemented the training loop that iterates through the data, computes the loss, and updates the model's parameters.
# - **Evaluation:**  You haven't included a way to evaluate the performance of your model on the test set (e.g., calculating the mean squared error).
# - **Data Loading (DataLoader):** While you converted your data to tensors, consider using a DataLoader from PyTorch to make the training process more efficient (batching data and shuffling).
# - **Activation Function:** You've created the model but haven't added an activation function within it. This should be added between the linear layers to introduce non-linearity, which is essential for learning complex patterns. A common activation function for regression tasks is ReLU (nn.ReLU).

loss = nn.MSELoss()

# prompt: generate a code to implement the training loop

optimizer = optim.Adam(model.parameters(), lr = 0.01)

epochs = 500

for epoch in range(epochs):
	optimizer.zero_grad()  # Reset gradients

	# Forward pass
	predictions = model(height_weight_feature)
	loss_value = loss(predictions, target)

	# Backward pass and optimization
	loss_value.backward()
	optimizer.step()

	# if (epoch + 1) % 10 == 0:
	#   print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_value.item():.4f}")

# prompt: Generate a code that can perform model evaluations on test set

# Prepare the test data
test_height_weight_feature = torch.tensor((test_df[["Hight", "Weight"]].values).astype("float32"))
test_target = torch.tensor((test_df["Shoe size"].values).astype("float32"))

# Set the model to evaluation mode
model.eval()

# Make predictions on the test set
with torch.no_grad():  # Disable gradient calculations during evaluation
	test_predictions = model(test_height_weight_feature)

# Calculate the loss on the test set
test_loss = loss(test_predictions, test_target)
print(f"Test Loss: {test_loss.item():.4f}")

# You can also calculate other evaluation metrics like R-squared or Mean Absolute Error (MAE)
# Here's an example of calculating MAE
mae = nn.L1Loss()(test_predictions, test_target)
print(f"Mean Absolute Error (MAE): {mae.item():.4f}")

# prompt: Generate a code that will perform Data Loading

# Assuming you have your training data in 'train_height_weight_feature' and 'train_target' tensors
train_dataset = TensorDataset(height_weight_feature, target)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# You can create a DataLoader for your test data similarly
test_dataset = TensorDataset(test_height_weight_feature, test_target)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Now you can iterate through your training data in batches during the training loop:
# for batch_features, batch_targets in train_loader:
#     # Your training code here...

# prompt: Generate a code that will add Activation Functions to the linear layer

# model = nn.Sequential(
#     nn.Linear(2, 10),
#     nn.ReLU(),  # Add ReLU activation after the first linear layer
#     nn.Linear(10, 5),
#     nn.ReLU(),
#     nn.Linear(5, 1)
#     # nn.Sigmoid()
# )

# prompt: Can I now make predictions

# Sample input data for prediction (replace with your actual data)
new_data = torch.tensor([[177.8, 68.6]], dtype = torch.float32)

# Set the model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
	predictions = model(new_data)

# print("Predictions:", predictions.item())

# Loss funtion might be what's missing

# Create a model
# Choose loss function
# Create dataset
# Define optimiser
# Run training loop

# Mean Square Error Loss nn.MSELoss()

# # Creating dataset and dataloader
# dataset = TensorDataset(height_weight_feature, target)
# dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

# # Creating model
# model = nn.Sequential(
#     nn.Linear(2, 4),
#     nn.Linear(4, 1)
# )

# # Loss optimizer
# criterion = nn.MSELoss()
# optimiser = optim.SGD(model.parameters(), lr = 0.001)

# for epoch in range(4):
#     for data in dataloader:
#         optimiser.zero_grad()

#         feature, target = data

#         pred = model(feature)

#         loss = criterion(pred, target)

#         loss.backward()

#         optimiser.step()

# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OggnrA8XDBW6ycgnODDiqms6vCuhm-8r
"""

