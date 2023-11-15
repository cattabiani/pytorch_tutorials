import torch
from torch import nn, optim
import matplotlib.pyplot as plt

class KModel(nn.Module):
    def __init__(self, degree=3):
        super(KModel, self).__init__()

        # Using ParameterList to create a list of learnable parameters
        self.coeff = nn.ParameterList([nn.Parameter(torch.randn(())) for _ in range(degree + 1)])

    def forward(self, x):
        # Create powers of x up to the specified degree
        powers = torch.stack([x**i for i in range(len(self.coeff))], dim=1)

        # Stack the coefficients along a new dimension
        coefficients_stacked = torch.stack(tuple(self.coeff), dim=0)

        # Multiply element-wise to obtain the final polynomial
        ans = torch.sum(coefficients_stacked * powers, dim=1)

        return ans

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 1000

# Instantiate the model
m = KModel(degree=3).to(device)
opt = optim.SGD(m.parameters(), lr=0.01)

# Generate input data
x = torch.linspace(-2, 2, 1000).to(device)

target = torch.sin(x)

losses = []

for i in range(num_epochs):
    # Forward pass
    y = m(x)

    loss = nn.MSELoss()(y, target)

    loss.backward()

    # Clip gradients to prevent numerical instability
    nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

    opt.step()
    opt.zero_grad()

    losses.append(loss.item())

    if i % 100 == 0:
        print(f"Epoch {i}/{num_epochs}, Loss: {loss.item()}")

# Plot the results
plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label='Model Output')
plt.plot(x.detach().cpu().numpy(), target.detach().cpu().numpy(), label='Target (sin)')

# Plot the loss
plt.figure()
plt.plot(range(len(losses)), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
