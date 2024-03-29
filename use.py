import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.models import ResNet18_Weights

# 定义模型结构，确保与训练时一致
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features

# 假设我们的模型是针对2个类别训练的
model.fc = nn.Linear(num_ftrs, 2)

# 加载之前保存的模型权重
model.load_state_dict(torch.load('starwars_vs_startrek.pth'))
model.eval()  # 设置为评估模式

# 图像预处理转换
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    # 打开图像文件
    image = Image.open(image_path).convert('RGB')
    # 应用预处理转换
    input_tensor = data_transforms(image).unsqueeze(0)  # 创建batch维度

    # 如果有GPU，将数据移到GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model.to(device)

    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    # 返回类别名称
    class_names = ['star_trek', 'star_wars']  # 替换为你的类别名称
    return class_names[preds[0]]

# 使用模型进行预测
image_path = '11.jpeg'  # 替换成你要预测的图片路径
predicted_class = predict_image(image_path)
print(f'Predicted class for the input image: {predicted_class}')