import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

def load_comnist_data(data_path, target_size=(28, 28)):
    """Load images and create numeric labels from Cyrillic folder names"""
    images = []
    labels = []
    
    # Get all class folders and sort them to ensure consistent label assignment
    class_folders = sorted([f for f in os.listdir(data_path) 
                           if os.path.isdir(os.path.join(data_path, f))])
    
    # Create a mapping from Cyrillic letter to numeric label
    label_map = {folder: idx for idx, folder in enumerate(class_folders)}
    
    print(f"Found {len(label_map)} classes:")
    print(label_map)
    
    for class_folder in class_folders:
        class_path = os.path.join(data_path, class_folder)
        label = label_map[class_folder]  # Get numeric label
        
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Only image files
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                
                # Resize to target size
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                img_array = np.array(img) / 255.0  # Normalize to 0-1
                images.append(img_array)
                labels.append(label)
    
    return np.array(images), np.array(labels), label_map


# Build the network
def create_cyrillic_model(num_classes):
    model = keras.Sequential([
        # First layer: looks at raw pixels
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Second layer: looks at features
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # Final layers: makes decisions
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# Prepare data for TensorFlow
X_train, y_train, label_map = load_comnist_data('Cyrillic/')

print(f"\nLoaded {len(X_train)} images")
print(f"Image shape after loading: {X_train[0].shape}")

# Get actual number of classes
num_classes = len(label_map)
print(f"Number of classes: {num_classes}")

# Add channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
print(f"Shape after reshape: {X_train.shape}")

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
print(f"Labels shape: {y_train.shape}")

# Create and compile the model
model = create_cyrillic_model(num_classes=num_classes)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Show model summary
model.summary()

# Train the model
print("\nTraining on CPU...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# Save the model
model.save('cyrillic_model.keras')
print("\nModel saved as 'cyrillic_model.keras'")

# Test predictions on validation data
# Get some test data (last 100 samples)
test_samples = X_train[-100:]
test_labels = y_train[-100:]

predictions = model.predict(test_samples)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Create reverse mapping for display
reverse_map = {v: k for k, v in label_map.items()}

print("\nSample predictions (first 10):")
for i in range(10):
    print(f"Predicted: {reverse_map[predicted_classes[i]]} | True: {reverse_map[true_classes[i]]} | Match: {predicted_classes[i] == true_classes[i]}")

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes)
print(f"\nTest accuracy on last 100 samples: {accuracy:.2%}")