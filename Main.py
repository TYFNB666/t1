import math
import sys

import hiddenlayer as h
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from resnet3d import ResNet3D, BasicBlock
from Res3D import resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet200,BasicBlock,Bottleneck


from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
#from ResNet50 import ResNet, ResidualBlock
# Import the CreateNiiDataset class or paste its code here
from Dataset3D import CreateNiiDataset, RandomRotation3D
from utils import read_split_data, plot_data_loader_image
import SimpleITK as sitk

class Resize3D:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = sitk.GetImageFromArray(img)
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()

        new_size = self.size
        new_spacing = [original_spacing[i] * (original_size[i] / new_size[i]) for i in range(len(original_size))]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        img_resampled = resampler.Execute(img)
        img_array = sitk.GetArrayFromImage(img_resampled)
        return img_array

class ToTensor3D:
    def __call__(self, img):
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.from_numpy(img)
        return img_tensor.float()

class Normalize3D:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img_normalized = transforms.functional.normalize(img, self.mean, self.std)
        return img_normalized


# def resnet50_3d(pretrained=True):
#     model = ResNet(Bottleneck, [3, 4, 6, 3])
#
#     if pretrained:
#         model.load_state_dict(torch.load('./resnet50-19c8e357.pth'))
#
#     # Modify the first convolution layer to accept 3D input
#     model.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=1, bias=False)
#
#     # Modify the final fully connected layer to match the number of classes
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Linear(num_ftrs, 1024),
#         nn.Linear(1024, 2),
#     )
#
#     # Modify the average pooling layer to work with 3D input
#     model.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
#
#     return model
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda

    num_epochs = 50  # 50轮
    learning_rate = 0.001  # 学习率

    root = "/kaggle/input/res-3d-t1/3D_Data_T1_8"
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    data_transform = {
        "train": transforms.Compose([
            RandomRotation3D(15),
            Resize3D([112, 112, 112]),
            ToTensor3D(),
            Normalize3D((0.5,), (0.5,)),
        ]),
        "val": transforms.Compose([
            Resize3D([112, 112, 112]),
            ToTensor3D(),
            Normalize3D((0.5,), (0.5,)),
        ])
    }

    train_data_set = CreateNiiDataset(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    Val_data_set = CreateNiiDataset(images_path=val_images_path,
                             images_class=val_images_label,

                             transform=data_transform["val"])

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(Val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             collate_fn=Val_data_set.collate_fn)

    # Resnet-50 3-4-6-3 总计(3+4+6+3)*3=48 个conv层 加上开头的两个Conv 一共50层
    # model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=2).to(device)
    #model = resnet50_3d().to(device)

    #model = ResNet3D(BasicBlock, [3, 4, 6, 3], num_classes=2).to(device)

    # # 加载预训练权重文件
    # checkpoint = torch.load('r3d50_K_200ep.pth')
    #
    # # 从预训练权重文件中仅提取模型权重
    # pretrained_weights = checkpoint['state_dict']
    #
    # # 加载提取的模型权重
    # model.load_state_dict(pretrained_weights)
    model = resnet50(
        sample_input_D=112,
        sample_input_H=112,
        sample_input_W=112,
        num_seg_classes=2,
        shortcut_type='B',
        no_cuda=False
    ).to(device)
    #model.load_state_dict(torch.load('./resnet_50.pth'))        #用预先训练好的权重进行训练
    # model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
    #  #卷积层
    # num_ftrs = model.fc.in_features     #获取最后一个fc的特征数，用于定义新的fc
    # model.fc = nn.Sequential(
    #     nn.Linear(num_ftrs, 1024),
    #     nn.Linear(1024, 2),
    # )
    # model.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
  # 修改avgpool层以适应较小的输入尺寸
    # model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    print(model)

    # vis_graph = h.build_graph(model, torch.zeros([1, 1, 224, 224]))  # 获取绘制图像的对象
    # vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
    # vis_graph.save("./demo1.png")  # 保存图像的路径
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #adam优化器，用于更新参数

    # state_dict = torch.load('resnet50-19c8e357.pth')
    # model.load_state_dict(state_dict)

    # 更新学习率
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # 训练数据集
    total_step = len(train_loader)
    val_num = len(Val_data_set)
    curr_lr = learning_rate
    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)     #展现进度条
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()        #更新权重
            torch.cuda.empty_cache()  # 添加这一行以清理 GPU 缓存
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     num_epochs,
                                                                     loss)  #显示信息
        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        label_all = []
        # F1_score
        output_all = []
        # AUC
        predict_scores = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # loss = loss_function(outputs, test_labels)

                # Predict score for AUC
                pred = outputs.cpu().numpy()
                output_all.extend(np.argmax(pred, axis=1))
                label_all.extend(val_labels)

                # F1 Score
                predict_scores.extend(outputs[:, 1].cpu().numpy())

                predict_y = torch.max(outputs, dim=1)[1]
                acc += float(torch.eq(predict_y, val_labels.to(device)).sum().item())

        val_accurate = acc / val_num
        print('[epoch %d]  val_accuracy: %.3f' %
              (epoch + 1, val_accurate))

        print("F1-Score:{:.4f}".format(f1_score(label_all, output_all)))
        print("AUC:{:.4f}".format(roc_auc_score(label_all, predict_scores)))


        if val_accurate > best_acc:
            best_acc = val_accurate

        # 延迟学习率
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
    print('Best accuracy of the model on the validation images: ' + str(best_acc))
    # S将模型保存
    # torch.save(model.state_dict(), 'resnet50_T2_50_2FC.ckpt')


if __name__ == '__main__':
    main()

# import math
# import sys
#
# import hiddenlayer as h
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from tqdm import tqdm
# from PIL import Image
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import f1_score
# from ResNet50 import ResNet, ResidualBlock
# from my_dataset import MyDataSet
# from utils import read_split_data, plot_data_loader_image
# from scipy.ndimage import zoom
#
# class Resize3D:
#     def __init__(self, output_size):
#         assert isinstance(output_size, tuple)
#         self.output_size = output_size
#
#     def __call__(self, volume):
#         resized_volume = []
#         original_dtype = volume.dtype
#         for img_np in volume:
#             img = Image.fromarray(img_np)
#             img_resized = img.resize(self.output_size)
#             resized_volume.append(np.array(img_resized, dtype=original_dtype))
#         return np.stack(resized_volume, axis=0)
#
#
# def collate_fn_3d(batch):
#     images, labels = zip(*batch)
#     max_image_count = max([image.shape[2] for image in images])
#
#     padded_images = []
#     for image in images:
#         pad_count = max_image_count - image.shape[2]
#         padded_image = torch.nn.functional.pad(image, (0, pad_count))
#         padded_images.append(padded_image)
#
#     images_stacked = torch.stack(padded_images)
#     labels_stacked = torch.tensor(labels, dtype=torch.long)
#
#     return images_stacked, labels_stacked
#
#
# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda
#
#     num_epochs = 50  # 50轮
#     learning_rate = 0.01  # 学习率0.01
#
#     root = "Data_T2"
#     train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
#     data_transform = {
#         "train": transforms.Compose([
#             transforms.Lambda(lambda volume: Resize3D((224, 224, volume.shape[0]))(volume)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#         ]),
#         "val": transforms.Compose([
#             transforms.Lambda(lambda volume: Resize3D((224, 224, volume.shape[0]))(volume)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#         ])
#     }
#     #预处理
#     train_data_set = MyDataSet(images_path=train_images_path,
#                                images_class=train_images_label,
#                                transform=data_transform["train"])
#     Val_data_set = MyDataSet(images_path=val_images_path,
#                              images_class=val_images_label,
#
#                              transform=data_transform["val"])
#
#     batch_size = 48
#
#     train_loader = torch.utils.data.DataLoader(train_data_set,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                collate_fn=collate_fn_3d)
#     val_loader = torch.utils.data.DataLoader(Val_data_set,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              collate_fn=collate_fn_3d)
#
#     # Resnet-50 3-4-6-3 总计(3+4+6+3)*3=48 个conv层 加上开头的两个Conv 一共50层
#     # model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=2).to(device)
#     model = models.resnet50()
#     model.load_state_dict(torch.load('./resnet50-19c8e357.pth'))        #用预先训练好的权重进行训练
#     model.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(1, 1, 1), bias=False) #卷积层
#     num_ftrs = model.fc.in_features     #获取最后一个fc的特征数，用于定义新的fc
#     model.fc = nn.Sequential(
#         nn.Linear(num_ftrs, 1024),
#         nn.Linear(1024, 2),
#     )
#     # model.fc = nn.Linear(num_ftrs, 2)
#     model.to(device)
#     print(model)
#
#     # vis_graph = h.build_graph(model, torch.zeros([1, 1, 224, 224]))  # 获取绘制图像的对象
#     # vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
#     # vis_graph.save("./demo1.png")  # 保存图像的路径
#     # 损失函数
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #adam优化器，用于更新参数
#
#     # state_dict = torch.load('resnet50-19c8e357.pth')
#     # model.load_state_dict(state_dict)
#
#     # 更新学习率
#     def update_lr(optimizer, lr):
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#
#     # 训练数据集
#     total_step = len(train_loader)
#     val_num = len(Val_data_set)
#     curr_lr = learning_rate
#     loss_function = nn.CrossEntropyLoss()
#     best_acc = 0.0
#     for epoch in range(num_epochs):
#         # train
#         model.train()
#         running_loss = 0.0
#         train_bar = tqdm(train_loader, file=sys.stdout)     #展现进度条
#         for step, data in enumerate(train_bar):
#             images, labels = data
#             optimizer.zero_grad()
#             logits = model(images.to(device))
#             loss = loss_function(logits, labels.to(device))
#             loss.backward()
#             optimizer.step()        #更新权重
#
#             # print statistics
#             running_loss += loss.item()
#
#             train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
#                                                                      num_epochs,
#                                                                      loss)  #显示信息
#         # validate
#         model.eval()
#         acc = 0.0  # accumulate accurate number / epoch
#         label_all = []
#         # F1_score
#         output_all = []
#         # AUC
#         predict_scores = []
#         with torch.no_grad():
#             val_bar = tqdm(val_loader, file=sys.stdout)
#             for val_data in val_bar:
#                 val_images, val_labels = val_data
#                 outputs = model(val_images.to(device))
#                 # loss = loss_function(outputs, test_labels)
#
#                 # Predict score for AUC
#                 pred = outputs.cpu().numpy()
#                 output_all.extend(np.argmax(pred, axis=1))
#                 label_all.extend(val_labels)
#
#                 # F1 Score
#                 predict_scores.extend(outputs[:, 1].cpu().numpy())
#
#                 predict_y = torch.max(outputs, dim=1)[1]
#                 acc += float(torch.eq(predict_y, val_labels.to(device)).sum().item())
#
#         val_accurate = acc / val_num
#         print('[epoch %d]  val_accuracy: %.3f' %
#               (epoch + 1, val_accurate))
#
#         print("F1-Score:{:.4f}".format(f1_score(label_all, output_all)))
#         print("AUC:{:.4f}".format(roc_auc_score(label_all, predict_scores)))
#
#
#         if val_accurate > best_acc:
#             best_acc = val_accurate
#
#         # 延迟学习率
#         if (epoch + 1) % 20 == 0:
#             curr_lr /= 3
#             update_lr(optimizer, curr_lr)
#     print('Best accuracy of the model on the validation images: ' + str(best_acc))
#     # S将模型保存
#     # torch.save(model.state_dict(), 'resnet50_T2_50_2FC.ckpt')
#
#
# if __name__ == '__main__':
#     main()
