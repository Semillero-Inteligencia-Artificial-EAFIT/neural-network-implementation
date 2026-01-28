import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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


# Define the network structure
class CyrillicNet(nn.Module):
    def __init__(self, num_classes):
        super(CyrillicNet, self).__init__()
        
        # First layer: looks at the raw pixels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second layer: looks at features from first layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Final layers: makes the decision
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x


# Prepare data for PyTorch
X_train, y_train, label_map = load_comnist_data('Cyrillic/')  # Fixed: unpack 3 values

print(f"\nLoaded {len(X_train)} images")
print(f"Image shape: {X_train[0].shape}")

# Get actual number of classes
num_classes = len(label_map)
print(f"Number of classes: {num_classes}")

# Split into train and validation
split_idx = int(len(X_train) * 0.8)
X_val = X_train[split_idx:]
y_val = y_train[split_idx:]
X_train = X_train[:split_idx]
y_train = y_train[:split_idx]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val).unsqueeze(1)
y_val = torch.LongTensor(y_val)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create and set up the model
model = CyrillicNet(num_classes=num_classes)  # Use actual number of classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nModel architecture:")
print(model)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for images, labels in train_loader:
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass: network makes predictions
        outputs = model(images)
        
        # Calculate how wrong the predictions are
        loss = criterion(outputs, labels)
        
        # Backward pass: adjust the network
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_predictions = torch.argmax(val_outputs, dim=1)
        val_accuracy = (val_predictions == y_val).float().mean()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), "cyrillicnet.pth")
print("\nModel saved as 'cyrillicnet.pth'")

# Create reverse mapping for displaying predictions
reverse_map = {v: k for k, v in label_map.items()}

# Test on validation set
model.eval()
with torch.no_grad():
    test_outputs = model(X_val[:10])
    predictions = torch.argmax(test_outputs, dim=1)
    
print("\nSample predictions (first 10 validation samples):")
for i in range(10):
    pred_label = predictions[i].item()
    true_label = y_val[i].item()
    print(f"Predicted: {reverse_map[pred_label]} | True: {reverse_map[true_label]} | "
          f"Match: {pred_label == true_label}")

# Overall validation accuracy
with torch.no_grad():
    all_outputs = model(X_val)
    all_predictions = torch.argmax(all_outputs, dim=1)
    accuracy = (all_predictions == y_val).float().mean()
    
print(f"\nFinal validation accuracy: {accuracy:.2%}")