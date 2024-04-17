import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from Model import AlexNet  # 导入AlexNet 模型


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    data_transform = {
        'Images': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # 设置图像数据集路径
    image_path = r"D:\Download\qq\UCMerced_LandUse\Images"

    # 创建完整的数据集
    full_dataset = datasets.ImageFolder(root=image_path,
                                         transform=data_transform['Images'])

    # 获取数据集的长度和分类标签信息
    dataset_size = len(full_dataset)
    flower_list = full_dataset.class_to_idx

    # 划分训练集和测试集的索引
    indices = list(range(dataset_size))
    split_size = dataset_size // 5  # 每折的样本数量
    np.random.shuffle(indices)  # 随机打乱索引顺序

    # 循环进行四折交叉验证
    for fold in range(5):
        # 划分训练集和测试集的索引
        test_indices = indices[fold * split_size:(fold + 1) * split_size]
        train_indices = indices[:fold * split_size] + indices[(fold + 1) * split_size:]

        # 创建训练数据集和测试数据集
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        # 创建数据加载器
        batch_size = 32
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=nw)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

        # 初始化 AlexNet 模型
        net = AlexNet(num_classes=21, init_weights=True)
        net.to(device)

        # 定义损失函数和优化器
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.0002)

        epochs = 10
        save_path = f'./AlexNet_fold{fold + 1}.pth'
        #保存训练过程中得到的最佳模型权重的文件路径，模型训练完成后会将最佳模型的参数保存到这个路径下的文件中，以便后续的模型测试或使用。
        best_loss = float('inf')  # 初始化最佳损失为正无穷大
        train_steps = len(train_loader)

        print(f"Starting fold {fold + 1}...")

        for epoch in range(epochs):
            # 训练模型
            net.train()
            running_loss = 0.0
            for step, data in enumerate(train_loader, start=0):
                images, labels = data
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # 统计损失
                running_loss += loss.item()

            # 计算平均训练损失
            train_loss = running_loss / train_steps
            print('[Epoch {}] Fold {} Train Loss: {:.4f}'.format(epoch + 1, fold + 1, train_loss))

            # 保存最佳模型
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(net.state_dict(), save_path)

        print(f'Finished Training Fold {fold + 1}\n')


if __name__ == '__main__':
    main()
