import numpy as np


# --------------------------------------------------
# 1. Dataset
# --------------------------------------------------

customer_characteristics = np.array([
    [22, 35, 42, 28],      # Age
    [1, 6, 8, 2],          # Visits
    [200, 700, 1000, 300]  # Average spend
], dtype=float)

purchase_success = np.array([0, 1, 1, 0], dtype=float)


# --------------------------------------------------
# 2. Normalize the input features
# --------------------------------------------------

min_values = np.min(customer_characteristics, axis=1, keepdims=True)
max_values = np.max(customer_characteristics, axis=1, keepdims=True)

normalized_customer_characteristics = (
    (customer_characteristics - min_values)
    / (max_values - min_values)
)


# --------------------------------------------------
# 3. Initialize the neuron
# --------------------------------------------------

weight = np.array([3.0, 3.0, 3.0])
bias = 2.0
learning_rate = 0.01
epochs = 1000


# --------------------------------------------------
# 4. Training
# --------------------------------------------------

for epoch in range(epochs):

    # Forward pass:
    # weighted inputs + bias
    z_raw = weight @ normalized_customer_characteristics + bias

    # ReLU activation
    prediction = np.maximum(0, z_raw)

    # Mean Squared Error
    loss = np.mean((prediction - purchase_success) ** 2)

    # Gradient of MSE
    gradient = (
        2 * (prediction - purchase_success)
        / len(purchase_success)
    )

    # Gradient through ReLU
    gradient = gradient * (z_raw > 0)

    # Gradient of weights and bias
    gradient_weight = (
        gradient @ normalized_customer_characteristics.T
    )
    gradient_bias = np.sum(gradient)

    # Gradient descent update
    weight = weight - learning_rate * gradient_weight
    bias = bias - learning_rate * gradient_bias


# --------------------------------------------------
# 5. Final results
# --------------------------------------------------

final_raw_output = (
    weight @ normalized_customer_characteristics + bias
)

final_prediction = np.maximum(0, final_raw_output)

# Convert continuous predictions into 0/1 decisions.
# This threshold is used only for interpretation,
# not during training.
final_classification = (final_prediction >= 0.5).astype(int)


print("Neural Network From Scratch")
print("----------------------------")
print("Final Weight:", weight)
print("Final Bias:", bias)
print("Final Loss:", loss)
print("Final Raw Output:", final_raw_output)
print("Final Prediction:", final_prediction)
print("Final Classification:", final_classification)
print("Target:", purchase_success)
