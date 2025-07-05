import os
import time
import requests
from tqdm import tqdm
from typing import List, Dict
import statistics


class ClassificationClient:
    def __init__(self, api_url: str = "http://localhost:8000/api/v1/predict"):
        self.api_url = api_url
        self.candidate_labels = ["corpse", "animal", "person", "vehicle", "plant"]

    def send_request(self, image_path: str, bbox: Dict = None) -> Dict:
        """发送单个预测请求并返回耗时和结果"""
        start_time = time.perf_counter()

        try:
            payload = {
                "image_path": image_path,
                "candidate_labels": self.candidate_labels,
                "priority": 1
            }

            # 添加bbox如果存在
            if bbox:
                payload["bbox"] = bbox

            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()

            elapsed = time.perf_counter() - start_time
            return {
                "success": True,
                "time": elapsed,
                "result": response.json(),
                "image": os.path.basename(image_path)
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return {
                "success": False,
                "time": elapsed,
                "error": str(e),
                "image": os.path.basename(image_path)
            }

    def benchmark(self, image_dir: str, max_images: int = None) -> Dict:
        """批量测试并返回统计信息"""
        image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if max_images:
            image_files = image_files[:max_images]

        results = []
        print(f"开始测试 {len(image_files)} 张图片...")

        for img_file in tqdm(image_files, desc="Processing images"):
            result = self.send_request(img_file)
            results.append(result)

            if result["success"]:
                # 格式化预测结果输出
                predictions = result["result"].get("results", [])
                pred_str = ", ".join(
                    [f"{p['label']}:{p['score']:.4f}" for p in predictions]
                )
                tqdm.write(
                    f"{os.path.basename(img_file)}: "
                    f"{result['time']:.3f}s -> "
                    f"[{pred_str}]"
                )
            else:
                tqdm.write(
                    f"{os.path.basename(img_file)}: "
                    f"失败 ({result['error']})"
                )

        # 计算统计信息
        success_times = [r["time"] for r in results if r["success"]]
        failed_count = len([r for r in results if not r["success"]])

        stats = {
            "total_images": len(image_files),
            "success_count": len(success_times),
            "failed_count": failed_count,
            "avg_time": statistics.mean(success_times) if success_times else 0,
            "min_time": min(success_times) if success_times else 0,
            "max_time": max(success_times) if success_times else 0,
            "std_dev": statistics.stdev(success_times) if len(success_times) > 1 else 0
        }

        return {
            "details": results,
            "statistics": stats
        }


if __name__ == "__main__":
    # 使用示例
    client = ClassificationClient()

    # 测试单个图像（取消注释即可使用）
    # test_image = r"C:\Users\lanqi\Desktop\classFod\ClassDataset\corpse\example.jpg"
    # if os.path.exists(test_image):
    #     print("\n测试单个图像:")
    #     single_result = client.send_request(test_image)
    #     print(f"耗时: {single_result['time']:.3f}秒")
    #     print(f"结果: {single_result.get('result', {}).get('results', 'N/A')}")
    # else:
    #     print(f"测试图像不存在: {test_image}")

    # 批量测试
    image_directory = r"C:\Users\lanqi\Desktop\classFod\ClassDataset\corpse"
    if os.path.isdir(image_directory):
        print("\n开始批量测试...")
        benchmark_results = client.benchmark(image_directory, max_images=50)

        print("\n测试结果统计:")
        stats = benchmark_results["statistics"]
        print(f"总测试图像: {stats['total_images']}")
        print(f"成功次数: {stats['success_count']}")
        print(f"失败次数: {stats['failed_count']}")
        print(f"平均耗时: {stats['avg_time']:.3f}秒")
        print(f"最短耗时: {stats['min_time']:.3f}秒")
        print(f"最长耗时: {stats['max_time']:.3f}秒")
        print(f"标准差: {stats['std_dev']:.3f}秒")
    else:
        print(f"图像目录不存在: {image_directory}")