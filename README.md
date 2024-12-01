# 水泥裂缝检测系统

基于 SAM(Segment Anything Model)的水泥裂缝检测系统。

## 安装步骤

1. 运行 install.bat 安装所需依赖
2. 首次运行时会自动下载 SAM 模型（约 2.4GB）
3. 将需要检测的图片放在程序同目录下

## 使用方法

1. 运行 crack_detection.py
2. 选择要处理的图片
3. 等待处理完成
4. 结果将保存在 results 目录下

## 注意事项

- 建议使用 GPU 运行以获得更好的性能
- 支持的图片格式：jpg、jpeg、png、bmp
- 首次运行需要下载模型文件，请确保网络通畅
