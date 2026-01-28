import numpy as np
from PIL import Image
import os

# ========== STEP 1: LOAD THE DATA ==========

def load_comnist_data(data_path):
    """Load images and labels from the CoMNIST dataset"""
    images = []
    labels = []
    
    for class_folder in sorted(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_path):
            for img_file in os.listdir(class_path):
                img = Image.open(os.path.join(class_path, img_file)).convert('L')
                img_array = np.array(img).astype(np.float32) / 255.0
                images.append(img_array.flatten())  # Flatten 28x28 to 784
                labels.append(int(class_folder))
    
    return np.array(images), np.array(labels)


# ========== STEP 2: ACTIVATION FUNCTIONS ==========
# These add non-linearity so the network can learn complex patterns

def relu(x):
    """ReLU: If positive keep it, if negative make it zero"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative for backpropagation"""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax: Converts numbers to probabilities that sum to 1"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ========== STEP 3: HELPER FUNCTIONS ==========

def one_hot_encode(labels, num_classes):
    """Convert labels to one-hot vectors
    Example: label 2 with 5 classes becomes [0, 0, 1, 0, 0]
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def initialize_weights(input_size, output_size):
    """Initialize weights randomly (small values)"""
    return np.random.randn(input_size, output_size) * 0.01

def initialize_bias(output_size):
    """Initialize biases to zero"""
    return np.zeros((1, output_size))


# ========== STEP 4: THE NEURAL NETWORK CLASS ==========

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=33):
        """
        Create a simple 3-layer network:
        - Input layer: 784 neurons (28x28 pixels)
        - Hidden layer: 128 neurons
        - Output layer: 33 neurons (one per Cyrillic letter)
        """
        # Initialize weights and biases for each layer
        self.W1 = initialize_weights(input_size, hidden_size)
        self.b1 = initialize_bias(hidden_size)
        
        self.W2 = initialize_weights(hidden_size, output_size)
        self.b2 = initialize_bias(output_size)
        
        # Store intermediate values for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def forward(self, X):
        """
        Forward pass: push data through the network
        
        Think of it like this:
        1. Input image (784 numbers) → multiply by weights → add bias → ReLU
        2. Hidden values (128 numbers) → multiply by weights → add bias → Softmax
        3. Output probabilities (33 numbers, one per letter)
        """
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1  # Linear combination
        self.a1 = relu(self.z1)                  # Activation
        
        # Second layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear combination
        self.a2 = softmax(self.z2)                     # Get probabilities
        
        return self.a2
    
    def backward(self, X, y, learning_rate=0.01):
        """
        Backward pass: adjust weights based on errors
        
        This is where the learning happens:
        1. Calculate how wrong we were (error)
        2. Figure out how much each weight contributed to the error
        3. Adjust weights in the opposite direction of the error
        """
        m = X.shape[0]  # Number of examples
        
        # Calculate error at output layer
        dz2 = self.a2 - y  # Difference between prediction and truth
        
        # Calculate gradients for second layer
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Calculate error at hidden layer
        dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        
        # Calculate gradients for first layer
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights and biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def compute_loss(self, predictions, y):
        """
        Calculate cross-entropy loss
        This measures how wrong our predictions are
        """
        m = y.shape[0]
        # Add small epsilon to avoid log(0)
        log_likelihood = -np.log(predictions[range(m), np.argmax(y, axis=1)] + 1e-8)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def predict(self, X):
        """Make predictions on new data"""
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)


# ========== STEP 5: TRAINING FUNCTION ==========

def train_network(X_train, y_train, X_test, y_test, epochs=20, learning_rate=0.1, batch_size=32):
    """
    Train the network using mini-batch gradient descent
    
    Instead of using all data at once, we:
    1. Split data into small batches
    2. Train on each batch
    3. Repeat for multiple epochs
    """
    num_classes = y_train.shape[1]
    input_size = X_train.shape[1]
    
    # Create the network
    nn = NeuralNetwork(input_size=input_size, hidden_size=128, output_size=num_classes)
    
    num_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Train on mini-batches
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            predictions = nn.forward(X_batch)
            
            # Backward pass and update weights
            nn.backward(X_batch, y_batch, learning_rate)
        
        # Calculate metrics every epoch
        train_predictions = nn.forward(X_train)
        train_loss = nn.compute_loss(train_predictions, y_train)
        train_acc = nn.accuracy(X_train, y_train)
        test_acc = nn.accuracy(X_test, y_test)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - "
              f"Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    return nn


def split_train_test(X, y, test_ratio=0.2):
    """Split data into training and test sets"""
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Calculate split point
    split_idx = int(len(X) * (1 - test_ratio))
    
    X_train = X_shuffled[:split_idx]
    y_train = y_shuffled[:split_idx]
    X_test = X_shuffled[split_idx:]
    y_test = y_shuffled[split_idx:]
    
    return X_train, y_train, X_test, y_test



# ========== STEP 6: MAIN EXECUTION ==========


def main():
    print("Loading data...")
    
    # Load data
    X_all, y_all_raw, label_map = load_comnist_data('Cyrillic/')
    
    # Split into train and test
    X_train, y_train_raw, X_test, y_test_raw = split_train_test(X_all, y_all_raw, test_ratio=0.2)
    
    # Flatten images for the neural network (28x28 -> 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Determine number of classes
    num_classes = len(label_map)
    print(f"Found {num_classes} classes: {list(label_map.keys())}")
    
    # Convert labels to one-hot encoding
    y_train = one_hot_encode(y_train_raw, num_classes)
    y_test = one_hot_encode(y_test_raw, num_classes)
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Train the network
    print("\nTraining network...")
    model = train_network(X_train, y_train, X_test, y_test, 
                         epochs=20, learning_rate=0.1, batch_size=32)
    
    # Create reverse mapping for display
    reverse_map = {v: k for k, v in label_map.items()}
    
    # Test on a few examples
    print("\nTesting predictions...")
    test_predictions = model.predict(X_test[:10])
    true_labels = np.argmax(y_test[:10], axis=1)
    
    print("Predicted letters:", [reverse_map[p] for p in test_predictions])
    print("True letters:", [reverse_map[t] for t in true_labels])

if __name__ == "__main__":
    main()