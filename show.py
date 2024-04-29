import numpy as np
import scipy.io as sio
import spectral
from utils import applyPCA,padWithZeros
from net import vit, sslsm
import torch
import matplotlib.pyplot as plt
from ColorFunctions import *

patch_size = 17
pca_components = 30
num_class = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_patches = (patch_size-8)**2

v = vit.ViT(
    image_size = 9,
    patch_size = 3,
    num_classes = num_class,
    dim = patch_size ** 2,
    depth = 2,
    heads = 8,
    mlp_dim = 2048,
    bands=pca_components,  # 波段数
)

net = sslsm.SSLSM(
    encoder = v,
    num_classes=num_class,
    masking_ratio = 0.00001,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 8,       # anywhere from 1 to 8
    bands=pca_components,  # 波段数
    dim=patch_size ** 2,
).to(device)
net.eval()
net_params = torch.load("./log/net_ip_fine.pkl")
net.load_state_dict(net_params)  # 加载模型可学习参数

# load the original image
X = sio.loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('./data/Indian_pines_gt.mat')['indian_pines_gt']
# X = sio.loadmat('./data/paviaU.mat')['paviaU']
# y = sio.loadmat('./data/paviaU_gt.mat')['Data_gt']
# X = sio.loadmat('./data/zy_yc_0609_data_samples.mat')['zy_yc_0906_data']
# y = sio.loadmat('./data/zy_yc_0609_data_samples.mat')['zy_yc_0906_samples']
height = y.shape[0]
width = y.shape[1]
# predict_image = spectral.imshow(classes = y.astype(int),figsize =(8,8))
X = applyPCA(X, numComponents= pca_components)
X = padWithZeros(X, patch_size//2)

# 逐像素预测类别
outputs = np.zeros((height,width))
for i in range(height):
    for j in range(width):
        if int(y[i,j]) == 0:
            continue
        else :
            image_patch = X[i:i+patch_size, j:j+patch_size, :]
            image_patch = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2])
            X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2)).to(device)
            # X_test_image = X_test_image.squeeze(dim=0)
            _,prediction = net(X_test_image)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction+1
    if i % 20 == 0:
        print('... ... row ', i, ' handling ... ...')
img = DrawResult(np.reshape(outputs,-1),2)
plt.savefig("results/ip/ip_res" + ".png", dpi=1000, bbox_inches='tight')  # 保存图像
# plt.imsave(r'res_our.png',img)

# predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(8,8))
# # spectral.save_rgb('rgb.jpg', outputs)
# plt.pause(10000)