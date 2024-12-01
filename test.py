from crack_detection import CrackDetector
import os
import sys


def test_detector():
    # 测试图片目录
    test_dir = "test_images"

    # 确保测试图片目录存在
    if not os.path.exists(test_dir):
        print(f"请创建 {test_dir} 目录并放入测试图片")
        return

    # 创建检测器实例
    detector = CrackDetector()

    # 获取所有测试图片
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    test_images = [f for f in os.listdir(test_dir) if any(f.lower().endswith(fmt) for fmt in supported_formats)]

    if not test_images:
        print(f"在 {test_dir} 目录中没有找到支持的图片文件")
        return

    # 处理每张测试图片
    for image_file in test_images:
        image_path = os.path.join(test_dir, image_file)
        print(f"\n处理图片: {image_file}")

        try:
            result_image, mask = detector.detect_cracks(image_path)
            print(f"成功处理图片: {image_file}")
        except Exception as e:
            print(f"处理图片 {image_file} 时出错: {str(e)}")


if __name__ == "__main__":
    test_detector()
