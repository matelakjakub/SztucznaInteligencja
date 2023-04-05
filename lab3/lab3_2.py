import numpy as np

#zadanie 1
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[0,0], [0,1], [1,0], [1,1]])
d = np.array([0,0,0,1])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

print("zadanie 2.4.1:",perceptron.W)

#zadanie 2
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[0], [1]])
d = np.array([1,0])

perceptron = Perceptron(input_size=1)
perceptron.fit(X, d)

print("zadanie 2.4.2:", perceptron.W)

#zadanie 3
class Perceptron:
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = X[i]
                y = self.predict(x)
                e = d[i] - y
                x = np.insert(x, 0, 1)
                self.W = self.W + self.lr * e * x

X = np.array([[1,0], [1,1], [0,1], [0,0]])
d = np.array([1,0,0,0])

perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

print("zadanie 2.4.3:",perceptron.W)

#zadanie 5
print("zadanie 2.4.5:")
# Funkcja aktywacji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Wejścia sieci
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Oczekiwane wyniki
targets = np.array([[0], [1], [1], [0]])

# Wagi warstwy ukrytej
hidden_weights = np.random.uniform(size=(2, 2))

# Wagi warstwy wyjściowej
output_weights = np.random.uniform(size=(2, 1))

# Współczynnik uczenia
learning_rate = 0.1

# Liczba epok
epochs = 10000

# Trenowanie sieci
for i in range(epochs):
    # Propagacja w przód
    hidden_layer_activation = sigmoid(np.dot(inputs, hidden_weights))
    output_layer_activation = sigmoid(np.dot(hidden_layer_activation, output_weights))

    # Obliczanie błędów
    output_layer_error = targets - output_layer_activation
    hidden_layer_error = np.dot(output_layer_error, output_weights.T) * hidden_layer_activation * (
                1 - hidden_layer_activation)

    # Propagacja wsteczna
    output_weights += learning_rate * np.dot(hidden_layer_activation.T, output_layer_error)
    hidden_weights += learning_rate * np.dot(inputs.T, hidden_layer_error)

# Testowanie sieci
print("Dane:  Wynik:")
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for i in range(len(test_inputs)):
    hidden_layer_activation = sigmoid(np.dot(test_inputs[i], hidden_weights))
    output_layer_activation = np.round(sigmoid(np.dot(hidden_layer_activation, output_weights)),0)
    inter = int(output_layer_activation)
    print(test_inputs[i],"  ",inter)