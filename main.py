import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # 迭代数据.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 梯度置零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播 + 优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        scheduler.step()

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']

        print(f'Epoch {epoch}/{num_epochs - 1} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model
def main():
    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'C:\\Users\\62616\\Desktop\\图片'  # 替换为您的数据集路径
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(class_names)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 使用预训练的模型
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # 这里每个输出类别的数量是2，因为我们是在区分星球大战和星际迷航
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 观察所有参数都被优化
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 每7个epochs衰减LR通过设置gamma=0.1
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型的函数


    # 训练模型
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)

    # 可以保存模型
    torch.save(model.state_dict(), 'starwars_vs_startrek.pth')

if __name__ == '__main__':
    main()