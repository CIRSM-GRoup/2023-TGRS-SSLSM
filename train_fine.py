import torch
from utils import splitTrainTestSet,TrainDS,TestDS
import os
import torch.nn as nn
import numpy as np
import time
from net import vit, sslsm
def train_fine(X,Y, test_ratio,class_num,run,
               patch_size,pca_components,batch_size,device,EPOCH,train_path,fine_path,encoder_depth,decoder_depth):
    # 定义模型
    v = vit.ViT(
        image_size=9,
        patch_size=3,
        num_classes=class_num,
        dim=patch_size ** 2,
        depth=encoder_depth,
        heads=8,
        mlp_dim=2048,
        bands=pca_components,  # 波段数
    )
    model = sslsm.SSLSM(
        encoder=v,
        num_classes=class_num,
        masking_ratio=0.00001,  # the paper recommended 75% masked patches
        decoder_dim=512,  # paper showed good results with just 512
        decoder_depth=decoder_depth,  # anywhere from 1 to 8
        bands=pca_components,  # 波段数
        dim=patch_size ** 2,
    ).to(device)
    #数据预处理
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, Y, test_ratio)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components)
    Xtest  = Xtest.reshape(-1, patch_size, patch_size, pca_components)
    Xtrain = Xtrain.transpose(0, 3, 1, 2)
    Xtest  = Xtest.transpose(0, 3, 1, 2)

    # 创建 trainloader 和 testloader
    trainset = TrainDS(Xtrain,ytrain)
    testset  = TestDS(Xtest,ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=batch_size, shuffle=False, num_workers=0)

    #加载模型参数
    net_params = torch.load(train_path)
    model.load_state_dict(net_params,False)  # 加载模型可学习参数

    #定义优化函数
    opt = torch.optim.Adam(model.parameters(), lr = 3e-4)
    criterion = nn.CrossEntropyLoss()

    #初始化时间
    time1 = time.time()
    #迭代训练
    model.train()
    for epoch in range(EPOCH):
        correct = 0.0
        total = 0.0
        total_loss = 0
        for _, (data, label) in enumerate(train_loader):
            data = data.to(device)
            data = data.squeeze()
            label = label.to(device)
            _,outs = model(data)
            loss = criterion(outs, label)

            # outputs = linear(outputs)
            # _, predicted = torch.max(outputs.data, 1)  # 预测结果
            # correct += ((predicted == label).squeeze().sum()).item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            # mae.update_moving_average() # update moving average of teacher encoder and teacher centers
            # total_loss += loss.item()
            # _, predicted = torch.max(outs.data, 1)  # 预测结果
            # total += label.size(0)  # 统计每批有多少数据
            # correct += ((predicted == label).squeeze().sum()).item()
        # print('[Epoch: %d]   [loss avg: %.6f]   [acc: %.2f]' % (epoch + 1, total_loss / (len(train_loader)), correct / total))
    time2 = time.time()
    fine_time = (time2 - time1)

    # 模型测试
    count = 0
    model.eval()
    #初始化混淆矩阵
    conf_mat = np.zeros([class_num, class_num])
    for inputs, label in test_loader:
        inputs = inputs.to(device)
        inputs = inputs.squeeze()
        _,outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test =  outputs
            count = 1
        else:
            y_pred_test = np.concatenate( (y_pred_test, outputs) )
        #存储混淆矩阵
        for i in range(len(label)):
            cate_i = label[i]
            pre_i = outputs[i]
            conf_mat[cate_i, pre_i] += 1.0
    #保存模型
    cnn_save_path = os.path.join(fine_path)
    torch.save(model.state_dict(), cnn_save_path)
    print("patch_size为{}时的第{}次运行微调阶段完成".format(patch_size, run))

    return conf_mat,fine_time



