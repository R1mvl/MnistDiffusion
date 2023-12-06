import torch
import torch.nn as nn
from unet import ContextUnet, DDPM
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# change port streamlit to 1224
import os
os.environ['STREAMLIT_SERVER_PORT'] = '1224'

st.title("Number Generator")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 10
n_feat = 128
n_T = 400

# Load the model
model = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
model.load_state_dict(torch.load('./results/model_39.pth', map_location=device))
model.eval()

median = 0.2

# Generate a sample
with torch.no_grad():
    n_sample = 1

    st.subheader("Input a number")
    input_num = st.text_input("Input a number")

    if st.button("Generate"):

        if input_num == "":
            input_num = 0

        digits = []
        for i in range(len(input_num)):
            digits.append(int(input_num[i]))

        # create a matrix for concatenating the generated images in a row
        output_num = np.zeros((1, 28, 28 * len(digits)))

        textP = st.text("Generating the {}th digit".format(1))
        barP = st.progress(0)
        for i in range(len(digits)):
            
            # update the progress bar
            barP.progress((i + 1) / len(digits))
            textP.text("Generating the {}th digit".format(i + 1))

            print("Generating the {}th digit".format(i + 1))
            x_gen, _ = model.sample_single_label(n_sample, (1, 28, 28), device, digits[i], guide_w=2)
            x_gen = x_gen[0, 0, :, :].cpu().numpy()
            # invert all pixels
            x_gen = 1 - x_gen
            output_num[0, :, i * 28:(i + 1) * 28] = x_gen
        
        textP.text("Generating the result")
        barP.empty()

        st.subheader("Generated number")

        plt.imshow(output_num[0, :, :], cmap='gray')
        plt.axis('off')
        plt.savefig('./results/generated.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        st.image('./results/generated.png', caption=f'Generated number : {input_num}', use_column_width=True)
        

        textP.empty()

