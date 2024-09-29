import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from phe import paillier  # Paillier library
import logging

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784  # 28x28 images
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 64
learning_rate = 0.001
max_encrypted_weights = 10  # Set a constant limit on the number of weights to encrypt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.encrypted_weights = None  # Placeholder for encrypted weights
    
    def simple_linear(self, x):
        """Simple linear transformation as an activation function."""
        return x + 1  # Simple transformation to avoid complex operations

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten the image
        x = self.fc1(x)
        x = self.simple_linear(x)  # Apply simple linear transformation
        x = self.fc2(x)
        return x
    
    def encrypt_fc1(self, public_key, max_weights_to_encrypt=max_encrypted_weights):
        """Encrypt a limited number of weights of the first fully connected layer."""
        original_weights = self.fc1.weight.data.cpu().numpy()
        encrypted_weights = []
        
        num_encrypted = 0
        total_weights = original_weights.size
        
        logging.info(f"Starting encryption of weights. Total weights: {total_weights}, Max to encrypt: {max_weights_to_encrypt}")
        
        for row in original_weights:
            encrypted_row = []
            for w in row:
                if num_encrypted < max_weights_to_encrypt:
                    encrypted_row.append(public_key.encrypt(float(w)))
                    num_encrypted += 1
                else:
                    # If we reached the limit, append the original weight (unencrypted)
                    encrypted_row.append(w)
            encrypted_weights.append(encrypted_row)
            
            # Log progress every 100 weights
            if num_encrypted % 100 == 0:
                logging.info(f"Encrypted {num_encrypted}/{max_weights_to_encrypt} weights.")
            
            # Stop encrypting if the limit is reached
            if num_encrypted >= max_weights_to_encrypt:
                break
        
        self.encrypted_weights = encrypted_weights
        logging.info(f"Encryption completed. Total encrypted weights: {num_encrypted}")

    def forward_encrypted(self, x, public_key):
        """Forward pass with encrypted weights."""
        x = x.view(-1, input_size)  # Flatten the image

        # Perform computations with encrypted weights in the first layer
        encrypted_output = []
        for row in self.encrypted_weights:
            encrypted_row_output = [public_key.encrypt(0)] * x.size(0)
            for i, encrypted_weight in enumerate(row):
                for batch_idx in range(x.size(0)):
                    if isinstance(encrypted_weight, paillier.EncryptedNumber):
                        # Ciphertext * Plaintext
                        encrypted_row_output[batch_idx] += encrypted_weight * float(x[batch_idx, i])
                    else:
                        # Plaintext * Plaintext
                        encrypted_row_output[batch_idx] += encrypted_weight * float(x[batch_idx, i])
            encrypted_output.append(encrypted_row_output)

        # Apply the simple linear transformation (x + 1) to the encrypted data
        transformed_output = [
            [val + public_key.encrypt(1) for val in row]  # Apply linear transformation (x + 1)
            for row in encrypted_output
        ]

        # Flatten transformed_output for passing to the next layer
        flat_transformed_output = [val for row in transformed_output for val in row]

        # Perform the forward pass through the second layer using encrypted outputs
        num_samples = len(flat_transformed_output) // hidden_size
        num_classes = self.fc2.out_features
        encrypted_final_output = [public_key.encrypt(0)] * num_samples * num_classes

        # Perform the second layer linear combination
        fc2_weights = self.fc2.weight.data.cpu().numpy()
        fc2_bias = self.fc2.bias.data.cpu().numpy()

        for i in range(num_classes):
            for j in range(num_samples):
                encrypted_sum = public_key.encrypt(0)
                for k in range(hidden_size):
                    if isinstance(flat_transformed_output[j * hidden_size + k], paillier.EncryptedNumber):
                        # Ciphertext * Plaintext
                        encrypted_sum += flat_transformed_output[j * hidden_size + k] * fc2_weights[i, k]
                    else:
                        # Plaintext * Plaintext
                        encrypted_sum += flat_transformed_output[j * hidden_size + k] * fc2_weights[i, k]

                # Add bias term
                encrypted_sum += public_key.encrypt(fc2_bias[i])
                encrypted_final_output[j * num_classes + i] = encrypted_sum

        return encrypted_final_output




# Initialize model, loss function, and optimizer
model = MLP(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Test the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    logging.info(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

# Paillier encryption setup
public_key, private_key = paillier.generate_paillier_keypair()

# Encrypt the weights of the first layer after training, respecting the encryption limit
model.encrypt_fc1(public_key)

# Now use a sample from the MNIST test dataset for inference with encrypted weights
test_iter = iter(test_loader)
sample_images, sample_labels = next(test_iter)
sample_images = sample_images.to(device)

# Perform forward pass with encrypted weights using the sample from the test dataset
encrypted_output = model.forward_encrypted(sample_images, public_key)

plain_output = model.forward(sample_images)
logging.info("Forward pass with encrypted weights using an MNIST test sample completed.")

# decrypted_output = private_key.decrypt(encrypted_output)

logging.info(f"Decrypted output after inference: {encrypted_output}")
logging.info(f"plain output after inference: {plain_output}")