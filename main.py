import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from model import VariationalAutoEncoder
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 784
h_dim = 400
latent_dim = 200

dataset = torchvision.datasets.MNIST(root="dataset/", train = True, transform = transforms.ToTensor(), download=True)
dataset_test = torchvision.datasets.MNIST(root="dataset/", train=False, transform = transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle= True)
test_loader = DataLoader(dataset=dataset_test, batch_size=32, shuffle=True)
model = VariationalAutoEncoder(input_dim, h_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)



def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(input_dim, optimizer, device):
    # loss_fn = nn.BCELoss(reduction="sum")
    for epoch in range(50):
        print(epoch)
        for i, (x,_) in enumerate(train_loader):
            x = x.to(device).view(x.shape[0], input_dim)
            x_reconstructed, mu, sigma = model(x)

            optimizer.zero_grad()

            # reconstruction_loss = loss_fn(x_reconstructed, x)
            loss = loss_function(x,x_reconstructed, mu, sigma)
            # kl_div = torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            #
            # loss = reconstruction_loss + kl_div
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'model_weights.pth')


train(input_dim, optimizer, device)

model.load_state_dict(torch.load('model_weights.pth',weights_only=True))
model.eval()

px = 1/plt.rcParams['figure.dpi']
figure = plt.figure(figsize=(6,6))

for i in range(0,5):
    test_img, test_label = next(iter(test_loader))

    predict_img = torch.flatten(test_img[0].squeeze()).to(device)
    x, mu, sigma = model(predict_img)
    x = x.detach().cpu()
    x = torch.unflatten(x, 0, (28,28))

    figure.add_subplot(6, 2, 1+i*2)
    plt.title(str(test_label[0].item()) + " (actual)")
    plt.imshow(test_img[0].squeeze(), cmap="gray")
    plt.axis("off")

    figure.add_subplot(6, 2, 2+i*2)
    plt.title(str(test_label[0].item()) + " (prediction)")
    plt.imshow(x, cmap="gray")
    plt.axis("off")
    if i==4:
        print(mu.size())

# z = torch.tensor([2,1], dtype=torch.float).to(device)
# x_decoded = model.decode(z)
# digit = x_decoded.detach().cpu().reshape(28,28)
# figure.add_subplot(6, 2, 11)
# plt.imshow(digit,cmap='gray')
plt.show()

#
# num = 11
# predict = torch.flatten(dataset[num][0]).to(device)
#
# # print(predict)
# mu,sigma = model.encode(predict)
# epsilon = torch.randn_like(sigma)
# z = mu + sigma * epsilon
# # z_sample = torch.tensor([mu,sigma],dtype=torch.float).to(device)
# # x = model.decode(z)
# x = model(predict)
# x = x.detach().cpu()
# x = torch.unflatten(x,0,(28,28))
#
#
# figure = plt.figure(figsize=(5,5))
# img,label = dataset[num]
# figure.add_subplot(5, 2, 1)
# plt.imshow(img.squeeze(), cmap= "gray")
#
# figure.add_subplot(5, 2, 2)
# plt.imshow(x, cmap= "gray")
#
# plt.show()
