import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import tenseal as ts
import numpy as np


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # The step function: outputs 1 for x >= 0, and 0 for x < 0
        output = torch.where(input >= 0, torch.tensor(1.0), torch.tensor(-1.0))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # The gradient of the step function is undefined, but we return 0
        grad_input = grad_output.clone()
        return grad_input

# Example of how to use the custom StepFunction
class StepActivation(torch.nn.Module):
    def __init__(self):
        super(StepActivation, self).__init__()

    def forward(self, input):
        return StepFunction.apply(input)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(128, 128)
        # self.fc4 = nn.Linear(128, 128)
        # self.fc5 = nn.Linear(256, 64)
        # self.fc6 = nn.Linear(256,512)
        self.fc7 = nn.Linear(1024, 10)
        self.step_activation = StepActivation()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        x = self.fc7(x)
        # print(x.shape)
        return x

# Training and validation setup
def train_and_validate(model, train_loader, test_loader, optimizer, criterion, epochs=2):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%')

    # Save the model parameters
    return model

# Test the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Load MNIST data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=1000, shuffle=False
)

# Initialize the model, loss function, and optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# Train the model and print validation accuracy
model = train_and_validate(model, train_loader, test_loader, optimizer, criterion)

# Test the trained model with the MNIST test dataset and print test accuracy
test_model(model, test_loader)

# Encrypt the model parameters with a smaller global scale
context = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = pow(2, 40)
context.generate_galois_keys()

weights = []
biases = []

for name, param in model.named_parameters():
    param_np = param.detach().numpy()
    if len(param_np.shape) == 2:  # Weight matrix
        if(name == 'fc3.weight'):
            plain1 = ts.plain_tensor(param_np.T)
            encrypted = ts.ckks_tensor(context, plain1)
            weights.append(encrypted)
            # print(encrypted.shape)
        else:
            weights.append(param_np)
    else:  # Bias vector
        biases.append(param_np)

# Send encrypted biases and plaintext weights to server
# In a real scenario, these would be serialized and sent over a network


print("Client Done. Moving to Server ...")

#______________SERVER____________________

def decrypt(enc):
    return enc.decrypt().tolist()

class EncryptedMLP:
    def __init__(self, weights, encrypted_biases, context):
        self.fc1_weight = weights[0]
        self.fc1_bias = encrypted_biases[0]  # CKKSVector
        self.fc2_weight = weights[1]
        self.fc2_bias = encrypted_biases[1]  # CKKSVector
        self.fc3_weight = weights[2]
        self.fc3_bias = encrypted_biases[2]  # CKKSVector
        self.context = context  # Save context to create CKKS vectors
        self.step_activation = StepActivation()

    def relu_approx(self, x):
        # Quadratic approximation of ReLU: (x^2 + x) / 2
        x_squared = x * x  # Square the CKKSVector
        x_sum = x + x_squared  # Add the original x to x^2
        return x_sum * 0.5  # Divide by 2 (manual scaling)

    def sigmoid_approx(self,x):
        return x.polyval([0.5, 0.197, 0, -0.004])

    def forward(self, x):
        # First layer: x @ W1 + b1
        x = np.dot(x, self.fc1_weight.T)
        x = self.fc1_bias + x  
        # x = self.relu_approx(x)  
        x = F.relu(torch.tensor(x)).tolist()

        # Second layer: x @ W2 + b2
        x = np.dot(x, self.fc2_weight.T)
        x = self.fc2_bias + x
        # x = self.relu_approx(x)
        # x = F.hardtanh(torch.tensor(x)).tolist()
        x = self.step_activation(x)
        
        x = ts.ckks_tensor(context,x)
        x = x.mm(self.fc3_weight) 
        # print(x.shape)
        # print(self.fc3_weight.decrypt().tolist())
        # print(type(x))
        return x

# Load MNIST test data
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=1, shuffle=True  # Load one image at a time
)

# Assuming 'weights' and 'encrypted_biases' are received from the client
server_model = EncryptedMLP(weights, biases, context)

for i in range(100):
    # Fetch one image and its label
    images, labels = next(iter(test_loader))
    test_image = images.view(1, -1).numpy()  # Flatten the image to match input size

    # Perform inference on the encrypted model
    result = server_model.forward(test_image)
    result_decrypted = result.decrypt().tolist() # Decrypt the result
    # print(result_decrypted.tolist())
    # print(result)
    # Get the predicted label
    predicted_label = np.argmax(result_decrypted)
    print(f'Predicted Label: {predicted_label}, True Label: {labels.item()}')