class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss):
        self.loss = loss

    def predict(self, samples):
        predictions = []

        for sample in samples:
            output = sample
            for layer in self.layers:
                output = layer.forward_propagation(output)
            predictions.append(output)

        return predictions

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for epoch in range(1, epochs + 1):
            err = 0
            for i in range(samples):
                # Forward propagation:
                output = x_train[i]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Compute the total loss:
                err += self.loss(y_train[i], output)

                # Backward propagation:
                error = self.loss(y_train[i], output, prime=True)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # Evaluate the average error on all samples:
            err /= samples
            print(f"Epoch {epoch:4d} of {epochs:<4d} \t Error = {err:.6f}")
