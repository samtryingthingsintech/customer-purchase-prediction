# Neural Network From Scratch

A tiny neural network built from scratch using **Python and NumPy**, without PyTorch, TensorFlow, or any machine-learning framework.

The purpose of this project is educational: to understand what is happening inside a neural network rather than treating a neural-network library as a black box.

## Problem

We use three customer characteristics:

- Age
- Number of visits
- Average spend

The model tries to learn whether a customer successfully purchases a product.

### Input

A `3 × 4` matrix:

- 3 features
- 4 customers

### Target

A `1 × 4` array:

```text
[0, 1, 1, 0]
```

Each value represents the purchase outcome for one customer.

## Model

This project uses a **single neuron**.

The neuron calculates:

```text
z = weights × inputs + bias
```

Then applies ReLU:

```text
ReLU(z) = max(0, z)
```

The model is trained using Mean Squared Error:

```text
loss = mean((prediction - target)²)
```

The weights and bias are updated using gradient descent:

```text
new parameter = old parameter - learning rate × gradient
```

## What this project demonstrates

1. Representing data as NumPy arrays
2. Feature normalization
3. Matrix multiplication
4. A neuron
5. Weights and bias
6. ReLU activation
7. Mean Squared Error
8. Gradients
9. Backpropagation for a simple network
10. Gradient descent
11. Training over multiple iterations
12. Converting continuous model output into a final 0/1 interpretation

## Why normalization?

The original features have very different scales:

- Age: around 20–40
- Visits: around 1–8
- Average spend: around 200–1000

If these values are used directly, the largest-scale feature can dominate the calculation.

We therefore use min-max normalization:

```text
x_normalized = (x - x_min) / (x_max - x_min)
```

This puts every feature between 0 and 1 for this dataset.

## Training flow

```text
Customer data
      ↓
Normalization
      ↓
Weighted sum + bias
      ↓
ReLU
      ↓
Prediction
      ↓
MSE Loss
      ↓
Gradient
      ↓
Gradient descent
      ↓
Updated weights and bias
      ↺
```

## Example result

With the current dataset and initialization, the model converges to approximately:

```text
Final Weight:
[0.5434, 0.6569, 0.5851]

Final Bias:
-0.5279

Final Loss:
0.0455
```

The continuous predictions are approximately:

```text
[0.0000, 0.6603, 1.2576, 0.0000]
```

Using `0.5` only as a final interpretation threshold gives:

```text
[0, 1, 1, 0]
```

which matches the training targets for this tiny dataset.

**Important:** this is a toy educational example with only four customers. It is not a useful production prediction model.

## Project structure

```text
neural-network-from-scratch/
├── neural_network.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Run locally

Install the dependency:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python neural_network.py
```

## What I learned

This project helped me move from understanding neural networks conceptually to implementing the basic mechanics myself:

- how inputs become predictions
- what weights and bias actually do
- why loss is needed
- how gradients tell us how parameters should change
- why the learning rate matters
- how matrix dimensions fit together
- how backpropagation connects the error back to the weights



