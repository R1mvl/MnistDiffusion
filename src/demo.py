import torch
import torch.nn as nn
from unet import ContextUnet, DDPM
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 10
n_feat = 128
n_T = 400

# Load the model
model = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
model.load_state_dict(torch.load('./results/model_0.pth', map_location=device))
model.eval()

median = 0.2

# Generate a sample
with torch.no_grad():
    n_sample = 1
    input_num = input("Enter a number: ")
    n = int(input_num)
    digits = []
    while n > 0:
        digits.append(n % 10)
        n = n // 10
    digits.reverse()

    # create a matrix for concatenating the generated images in a row
    output_num = np.zeros((1, 28, 28 * len(digits)))

    for i in range(len(digits)):
        print("Generating the {}th digit".format(i + 1))
        x_gen, _ = model.sample_single_label(n_sample, (1, 28, 28), device, digits[i], guide_w=2)
        x_gen = x_gen[0, 0, :, :].cpu().numpy()
        # make the image binary
        # get the median value of the image
        x_gen[x_gen > median] = 1
        output_num[0, :, i * 28:(i + 1) * 28] = x_gen

# Visualize the generated samples
fig = plt.figure(figsize=(10, 10))

plt.imshow(output_num[0, :, :], cmap='gray')

plt.show()