from train import train
from utils import applyPCA,createImageCubes
import scipy.io as sio
import torch
import numpy as np

#定义超参数
class_num = 16  # 地物类别
EPOCH = 100 #迭代次数
ratio_self = 0.5 #自训练测试样本比例
ratio_fine = 0.95 #微调测试样本比例
patch_size = 17   # 每个像素周围提取 patch 的尺寸
N_RUNS = 5 #跑多次取平均精度
pca_components = 30 # 使用 PCA 降维，得到主成分的数量
batch_size = 128  #批处理量
mask_ratio = 0.5 #掩膜比例
encoder_depth = 8 #编码器深度
decoder_depth = 2 #解码器深度

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#路径信息
#--数据集路径
X = sio.loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('./data/Indian_pines_gt.mat')['indian_pines_gt']

# X = sio.loadmat('./data/paviaU.mat')['paviaU']
# y = sio.loadmat('./data/paviaU_gt.mat')['Data_gt']

# X = sio.loadmat('./data/zy_yc_0609_data_samples.mat')['zy_yc_0906_data']
# y = sio.loadmat('./data/zy_yc_0609_data_samples.mat')['zy_yc_0906_samples']

train_path = r'./log/net_ip_train.pkl'  #自训练模型保存路径
fine_path= r'./log/net_ip_fine.pkl'  #微调模型保存路径
# accuracy_path = r'./results/ip/ratio_fine/' #精度保存路径
accuracy_path = r'./results/indian/' #精度保存路径

print('Hyperspectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

# ratio_fine_total = [0.95,0.94,0.93,0.92,0.91,0.90,0.89,0.88]
# model_depth = [(2,2),(2,3),(2,4),(4,2),(4,3),(4,4),(6,2),(6,3),(6,4),(8,2),(8,3),(8,4)]
# model_depth = [(6,2),(6,3),(6,4),(8,2),(8,3),(8,4)]
#进行不同参数的训练
# for encoder_depth, decoder_depth in model_depth:
# for patch_size in range(3,19,2):
# for ratio_self in range(5, 100, 5):
# for ratio_fine in range(1, 11, 1):
# for ratio_fine in ratio_fine_total:
# for ratio_self in range(5, 100, 5):
print('当前微调大小为：',(ratio_fine))
# #对数据进行预处理
X, Y = createImageCubes(X_pca, y, windowSize=patch_size)

#训练测试阶段
accuracy = train(X,Y,ratio_self,ratio_fine,class_num,N_RUNS,
      patch_size,pca_components,batch_size,device,EPOCH,train_path,fine_path,mask_ratio,encoder_depth,decoder_depth)
print(accuracy)
with open(accuracy_path + str('acc') + '.csv', 'a', encoding='utf-8') as f:
      f.write(accuracy)
      #将精度写入文件
      # with open(accuracy_path + str(encoder_depth)+'-'+str(decoder_depth) + '.csv', 'a', encoding='utf-8') as f:
      # with open(accuracy_path + str(ratio_fine) + '.csv', 'a', encoding='utf-8') as f:
      # #     f.write(accuracy)
      # # with open(accuracy_path + str(ratio_self/100) + '.csv', 'a', encoding='utf-8') as f:
      # # with open(accuracy_path + str(1-ratio_self/100) + '.csv', 'a', encoding='utf-8') as f:
      #       f.write(accuracy)