import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import nn, optim
import matplotlib.pyplot as plt

class SimpleNN(nn.Module):
    model_file = "MNIST_trained_model.pth"
    num_epochs = 5
    _device = None

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        return self.linear2(x)

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    @staticmethod
    def get_device():
        if SimpleNN._device is None:
            SimpleNN._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return SimpleNN._device

    @staticmethod
    def initialize_data_loaders():
        transform = SimpleNN.get_transform()
        mnist_dataset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
        train_size = int(0.8 * len(mnist_dataset))
        val_size = len(mnist_dataset) - train_size
        train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        return train_loader, val_loader

    @staticmethod
    def get_trained_model():
        try:
            model = torch.load(SimpleNN.model_file)
            print("model loaded")
            return model
        except FileNotFoundError:
            print("model not found. Training...")
            model = SimpleNN().full_train()
            torch.save(model, SimpleNN.model_file)
            return model

    def full_train(self):
        device = SimpleNN.get_device()
        train_loader, val_loader = SimpleNN.initialize_data_loaders()

        model = self.to(device)
        opt = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Lists to store the training and validation statistics
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(SimpleNN.num_epochs):
            model.train()
            for x, labels in train_loader:
                x, labels = x.to(device), labels.to(device)
                opt.zero_grad()
                y = model.forward(x)
                loss = criterion(y, labels)
                loss.backward()
                opt.step()

            # Validation loop
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            model.train()

            # Calculate and print training and validation statistics
            avg_train_loss = loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total

            print(f'Epoch {epoch + 1}/{SimpleNN.num_epochs}, Avg. Train Loss: {avg_train_loss:.4f}, Avg. Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

            # Append values for plotting
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

        # SimpleNN.plot_training(SimpleNN.num_epochs, train_losses, val_losses, val_accuracies)
        return model

    @staticmethod
    def plot_training(num_epochs, train_losses, val_losses, val_accuracies):
        # Plotting
        plt.figure(figsize=(10, 5))

        # Plot Training and Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    model = SimpleNN.get_trained_model()
    model.eval()
    mnist_testing_dataset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=SimpleNN.get_transform())
    test_loader = DataLoader(mnist_testing_dataset, batch_size=64, shuffle=True)
    with torch.no_grad():
        x, label = next(iter(test_loader))
        x = x.to(SimpleNN.get_device())
        y = model.forward(x)

        # Plot the first 10 images along with their predicted labels
        plt.figure(figsize=(15, 6))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            input_image = x[i].cpu().numpy().squeeze()
            prediction = torch.argmax(y[i]).item()
            plt.imshow(input_image, cmap='gray')
            plt.title(f'Predicted Label: {prediction}, {label[i]}')
            plt.axis('off')
        
        plt.show()
