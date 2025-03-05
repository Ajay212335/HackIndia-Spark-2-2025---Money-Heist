import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import create_model
import os

# Define dataset path
DATA_DIR = os.path.join(os.getcwd(), "data")  # Ensure the correct dataset folder
print("Checking dataset path:", DATA_DIR)

# Check if dataset directory exists
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory '{DATA_DIR}' not found. Please check the path.")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define model parameters
NUM_CLASSES = 5  # Ensure this matches the number of classes in your dataset
EPOCHS = 10
LEARNING_RATE = 0.001

# Create model
model = create_model(NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "rural_problem_model.pth")
print("Model saved successfully!")
