# 星际迷航与星球大战图片区分
这个项目训练了一个模型，用以区分星球大战和星际迷航的图片，这只是个初学者的简单尝试
## 依赖安装
### 用anaconda
```bash
conda env create -f environment.yml
```
或者
### 用pip安装
```python
pip install -r requirements.txt
```
## 使用方法
运行main.py可以训练模型
运行use时，该方法需要手动修改图片路径，运行后会弹出图片以及预测标签
```
image_path = '11.jpeg'  # 替换成你要预测的图片路径
predicted_class = predict_image(image_path)
show_image_with_label(image_path, predicted_class)
```

运行use2时，可通过GUI图形界面加载图片进行预测，需要注意的是有时图片太大会导致标签被挤到看不到的地方


## 模型基于ResNet18架构，并在特定数据集上进行了训练。


