import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch
import requests
from tqdm import tqdm

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


class CrackDetector:
    def __init__(self, model_type="vit_h"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")

        # 模型检查点路径
        self.checkpoint = "sam_vit_h_4b8939.pth"
        self.model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        # 如果模型文件不存在，自动下载
        if not os.path.exists(self.checkpoint):
            self._download_model()

        # 加载SAM模型
        print("正在加载SAM模型...")
        sam = sam_model_registry[model_type](checkpoint=self.checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print("模型加载完成！")

    def _download_model(self):
        """下载SAM模型权重"""
        print(f"正在下载SAM模型权重文件...")
        response = requests.get(self.model_url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(self.checkpoint, "wb") as file, tqdm(
            desc=self.checkpoint,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)

    def detect_cracks(self, image_path, use_box=True):
        """
        检测图像中的裂缝
        Args:
            image_path: 图像路径
            use_box: 是否使用框选模式
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("无法读取图像文件")

        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 设置图像
        self.predictor.set_image(image_rgb)

        if use_box:
            # 创建窗口和trackbar
            window_name = "框选裂缝区域"
            cv2.namedWindow(window_name)
            roi = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(window_name)

            # 获取框选区域的坐标
            x, y, w, h = roi
            input_box = np.array([x, y, x + w, y + h])

            # 使用框选区域进行预测
            masks, scores, _ = self.predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=True)
        else:
            # 使用多点提示模式
            h, w = image.shape[:2]
            num_points = 5
            input_points = []
            for i in range(num_points):
                x = w // (num_points + 1) * (i + 1)
                y = h // 2
                input_points.append([x, y])

            input_points = np.array(input_points)
            input_labels = np.ones(len(input_points))

            masks, scores, _ = self.predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)

        # 选择最佳掩码
        best_mask_idx = scores.argmax()
        best_mask = masks[best_mask_idx]

        # 应用形态学操作改善结果
        kernel = np.ones((3, 3), np.uint8)
        best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # 在原图上标记裂缝
        result = image.copy()
        result[best_mask == 1] = [0, 255, 0]  # 用绿色标记裂缝

        # 添加半透明效果
        overlay = image.copy()
        overlay[best_mask == 1] = [0, 255, 0]
        result = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

        return result, best_mask

    def train_on_image(self, image_path):
        """
        交互式训练：让用户框选裂缝区域
        Args:
            image_path: 图像路径
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("无法读取图像文件")

        # 存储框选区域
        boxes = []
        labels = []

        window_name = "框选裂缝区域 (左键:开始框选 Enter:完成 Esc:取消 r:重置)"
        cv2.namedWindow(window_name)
        display_image = image.copy()

        print("\n=== 交互式框选说明 ===")
        print("- 用鼠标框选裂缝区域")
        print("- 可以框选多个区域")
        print("- 按r键重置所有框选")
        print("- 按Enter键完成框选")
        print("- 按Esc键取消操作")

        while True:
            cv2.imshow(window_name, display_image)

            # 获取用户框选的区域
            roi = cv2.selectROI(window_name, display_image, fromCenter=False, showCrosshair=True)
            if roi[2] > 0 and roi[3] > 0:  # 如果框选了有效区域
                x, y, w, h = roi
                boxes.append([x, y, x + w, y + h])
                labels.append(1)  # 1表示裂缝区域

                # 在图像上画框
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 询问是否继续框选
                print("\n是否继续框选？")
                print("- 按Enter继续框选")
                print("- 按Esc完成框选")
                print("- 按r重置所有框选")

                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # Esc键
                    break
                elif key == ord("r"):  # r键重置
                    display_image = image.copy()
                    boxes = []
                    labels = []
                    continue

        cv2.destroyWindow(window_name)

        if not boxes:
            print("未框选任何区域！")
            return None, None

        # 转换为numpy数组
        input_boxes = np.array(boxes)

        # 设置图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        # 使用框选区域进行预测
        masks = []
        scores = []

        for box in input_boxes:
            mask, score, _ = self.predictor.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=True)
            masks.append(mask[score.argmax()])
            scores.append(score.max())

        # 合并所有掩码
        final_mask = np.zeros_like(masks[0])
        for mask in masks:
            final_mask = final_mask | mask

        return [final_mask], np.array([1.0])  # 返回合并后的掩码和置信度


def process_image(detector, image_path):
    """处理单张图片并显示结果"""
    try:
        # 让用户选择模式
        print("\n请选择模式：")
        print("1. 框选模式（手动框选裂缝区域）")
        print("2. 自动模式（使用多点提示）")
        print("3. 训练模式（交互式标注裂缝）")

        while True:
            try:
                mode = input("请输入模式编号 (1/2/3): ")
                if mode in ["1", "2", "3"]:
                    break
                print("无效的选择，请重新输入！")
            except ValueError:
                print("请输入有效的数字！")

        if mode == "3":
            # 训练模式
            masks, scores = detector.train_on_image(image_path)
            if masks is None:
                return False

            # 选择最佳掩码
            best_mask_idx = scores.argmax()
            best_mask = masks[best_mask_idx]

            # 读取原始图像
            image = cv2.imread(image_path)

            # 在原图上标记裂缝
            result = image.copy()
            result[best_mask == 1] = [0, 255, 0]

            # 添加半透明效果
            overlay = image.copy()
            overlay[best_mask == 1] = [0, 255, 0]
            result = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        else:
            # 原有的框选或自动模式
            use_box = mode == "1"
            result, best_mask = detector.detect_cracks(image_path, use_box=use_box)

        # 显示结果
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(best_mask, cmap="gray")
        plt.title("裂缝掩码")
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("检测结果")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # 保存结果
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mode_str = {"1": "box", "2": "auto", "3": "train"}[mode]

        cv2.imwrite(f"{output_dir}/{base_name}_{mode_str}_result.png", result)
        cv2.imwrite(f"{output_dir}/{base_name}_{mode_str}_mask.png", best_mask * 255)
        print(f"结果已保存到 {output_dir} 目录")

        return True
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return False


def main():
    detector = CrackDetector()
    picpath = "./pics"

    if not os.path.exists(picpath):
        os.makedirs(picpath)
        print(f"已创建图片目录: {picpath}")
        print("请将要处理的图片放入该目录")
        return

    # 支持多种图像格式
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [f for f in os.listdir(picpath) if any(f.lower().endswith(fmt) for fmt in supported_formats)]

    if not image_files:
        print(f"在 {picpath} 目录下没有找到支持的图像文件！")
        print(f"支持的格式: {', '.join(supported_formats)}")
        return

    while True:
        print("\n发现以下图像文件：")
        for i, f in enumerate(image_files):
            print(f"{i+1}. {f}")
        print("0. 退出程序")

        try:
            choice = input("\n请选择要处理的图像编号（0-{}）: ".format(len(image_files)))
            if choice == "0":
                break

            idx = int(choice) - 1
            if 0 <= idx < len(image_files):
                image_path = os.path.join(picpath, image_files[idx])
                print(f"\n正在处理图像: {image_files[idx]}")
                process_image(detector, image_path)
            else:
                print("无效的选择！")
        except ValueError:
            print("请输入有效的数字！")
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()
