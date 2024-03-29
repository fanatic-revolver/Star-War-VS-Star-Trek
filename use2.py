# 其他之前定义的导入和函数保持不变...
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont,ImageTk
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


def show_image_with_label(image_path, label):
    # Open the image
    image = Image.open(image_path)
    # 创建一个可以在上面绘图的空白图层
    draw = ImageDraw.Draw(image)

    # 设置字体和大小
    font = ImageFont.truetype('arial.ttf', 20)

    # 添加标签文本
    draw.text((10, 10), label, font=font, fill=(255, 255, 255))

    # 显示图像
    image.show()


# 创建GUI窗口来选择图片并显示预测结果
class ImageClassifierApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # 创建一个按钮，用于打开图像文件
        self.btn_open = tk.Button(window, text="Open Image", command=self.open_image)
        self.btn_open.pack()

        # 创建一个标签，用于显示图像
        self.lbl_image = tk.Label(window)
        self.lbl_image.pack()

        # 创建一个标签，用于显示预测结果
        self.lbl_result = tk.Label(window, text="")
        self.lbl_result.pack()

        self.window.mainloop()

    def open_image(self):
        # 使用文件对话框选择图像文件
        file_path = filedialog.askopenfilename()
        if file_path:
            # 显示选择的图像
            self.display_image(file_path)

            # 进行预测并显示结果
            predicted_class = predict_image(file_path)
            self.lbl_result.config(text="Predicted Class: " + predicted_class)

    def display_image(self, image_path):
        # 打开图像并将其转换为Tkinter可以显示的格式
        image = Image.open(image_path)
        image = ImageTk.PhotoImage(image)

        # 如果标签中已有图像，则先删除
        if hasattr(self.lbl_image, 'image'):
            self.lbl_image.config(image='')
            self.lbl_image.image = None

        # 将图像配置到标签中并显示
        self.lbl_image.image = image
        self.lbl_image.config(image=image)


# 创建窗口和应用
root = tk.Tk()
app = ImageClassifierApp(root, "Image Classifier")