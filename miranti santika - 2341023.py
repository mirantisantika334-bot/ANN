import math
import random

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Turunan sigmoid
def dsigmoid(y):
    return y * (1 - y)

# Data XOR
data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# Inisialisasi bobot dan bias
w_input_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
w_hidden_output = [random.uniform(-1, 1) for _ in range(2)]
bias_hidden = [random.uniform(-1, 1) for _ in range(2)]
bias_output = random.uniform(-1, 1)

learning_rate = 0.5
epoch = 5000

# Training ANN
for _ in range(epoch):
    for x, target in data:

        # ---- FEEDFORWARD ----
        hidden = []
        for i in range(2):
            net = x[0] * w_input_hidden[0][i] + x[1] * w_input_hidden[1][i] + bias_hidden[i]
            hidden.append(sigmoid(net))

        net_output = hidden[0] * w_hidden_output[0] + hidden[1] * w_hidden_output[1] + bias_output
        output = sigmoid(net_output)

        # ---- BACKPROPAGATION ERROR ----
        error = target - output
        delta_output = error * dsigmoid(output)

        delta_hidden = []
        for i in range(2):
            delta_hidden.append(delta_output * w_hidden_output[i] * dsigmoid(hidden[i]))

        # ---- UPDATE BOBOT ----
        for i in range(2):
            w_hidden_output[i] += learning_rate * delta_output * hidden[i]
        bias_output += learning_rate * delta_output

        for i in range(2):
            for j in range(2):
                w_input_hidden[j][i] += learning_rate * delta_hidden[i] * x[j]
            bias_hidden[i] += learning_rate * delta_hidden[i]

# Testing hasil
print("Hasil Testing XOR:")
for x, target in data:
    hidden = []
    for i in range(2):
        net = x[0] * w_input_hidden[0][i] + x[1] * w_input_hidden[1][i] + bias_hidden[i]
        hidden.append(sigmoid(net))

    net_output = hidden[0] * w_hidden_output[0] + hidden[1] * w_hidden_output[1] + bias_output
    output = sigmoid(net_output)

    print(f"Input {x} -> Output: {round(output, 3)}")
