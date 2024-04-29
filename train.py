import torch
from utils import splitTrainTestSet,TrainDS,TestDS,kappa
import os
from train_fine import train_fine
import numpy as np
import time

def train(X,Y, test_ratio,ratio_fine,class_num,N_RUNS,
          patch_size,pca_components,batch_size,device,EPOCH,path,fine_path,mask_ratio,encoder_depth,decoder_depth):
    #定义初始存储
    text = ''
    Kappa = []
    CA = np.zeros([class_num, N_RUNS])
    OA = []
    AA = []
    total_time = []
    from net import vit, sslsm
    for run in range(N_RUNS):

        #定义模型
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
            masking_ratio=mask_ratio,  # the paper recommended 75% masked patches
            decoder_dim=512,  # paper showed good results with just 512
            decoder_depth=decoder_depth,  # anywhere from 1 to 8
            bands = pca_components, #波段数
            dim=patch_size ** 2,
        ).to(device)
        #划分数据集
        Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, Y, test_ratio)
        #数据预处理
        Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components)
        Xtest  = Xtest.reshape(-1, patch_size, patch_size, pca_components)
        Xtrain = Xtrain.transpose(0, 3, 1, 2)
        Xtest  = Xtest.transpose(0, 3, 1, 2)

        # 创建 trainloader 和 testloader
        trainset = TrainDS(Xtrain,ytrain)
        # testset  = TestDS(Xtest,ytest)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        # test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=batch_size, shuffle=False, num_workers=0)

        #定义优化函数
        opt = torch.optim.Adam(model.parameters(), lr = 3e-4)

        #初始化损失和时间
        epoch_loss = 0
        time1 = time.time()
        #进行迭代训练
        for epoch in range(EPOCH*2):
            for _,(data, label) in enumerate(train_loader):
                data = data.to(device)
                # label = label.to(device)
                loss,_ = model(data)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # mae.update_moving_average() # update moving average of teacher encoder and teacher centers
                epoch_loss += loss / len(train_loader)
                # print(epoch_loss)
        #计算训练时间
        time2 = time.time()
        train_time = (time2 - time1)
        #保存训练模型
        cnn_save_path = os.path.join(path)
        torch.save(model.state_dict(), cnn_save_path)

        print("patch_size为{}时的第{}次运行自训练阶段完成".format(patch_size, run))

        # 微调阶段
        conf_mat,fine_time = train_fine( X, Y, ratio_fine, class_num,run,
                                        patch_size,pca_components, batch_size, device,
                                        EPOCH, path, fine_path,encoder_depth,decoder_depth)

        #计算微调测试的精度
        m = 0
        for i in range(class_num):
            ca = conf_mat[i, i] / np.sum(conf_mat[i, :])  #计算ca
            # print("class:{:<2},Recall: {:.2%}".format(
            #     classes_name[i], conf_mat[i, i] / np.sum(conf_mat[i, :])))
            text += ("class:{:<2},{:.4}\n".format(i, ca)) #将每一次的ca加入到text中
            m += ca #将多类的ca进行相加
            CA[i,run] = ca   #统计多次训练下的ca
        #计算Kappa
        K = kappa(conf_mat)
        Kappa.append(K)
        #计算OA
        oa = np.trace(conf_mat) / np.sum(conf_mat)
        OA.append(oa)
        #计算AA
        text += ('AA,{:.4}\n kappa,{:.4}\n OA,{:.4} ,\n'.format(
            m / class_num, K, oa))
        AA.append(m / class_num)
        #计算训练时间
        text += ('train_time,{:.4}\n'.format(train_time+fine_time))
        total_time.append(train_time+fine_time)
        # print('AA,{:.4}\n kappa,{:.4}\n OA,{:.4} ,\n'.format( m / class_num, K, oa))
    #统计多次训练的平均精度
    average_ca = np.sum(CA, axis=1)/N_RUNS
    for i in range(class_num):
        text += ("total_ca:{:<2},{:.4}\n".format(i, average_ca[i]))
    average_kappa = np.sum(Kappa)/N_RUNS
    average_oa = np.sum(OA)/N_RUNS
    average_aa = np.sum(AA) / N_RUNS
    average_time = np.sum(total_time) / N_RUNS
    text += ('average_aa,{:.4}\n average_kappa,{:.4}\n average_oa,{:.4} ,\n average_time,{:.4} ,\n'
        .format(average_aa, average_kappa, average_oa, average_time))

    return text

