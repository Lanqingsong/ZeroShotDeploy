import os
import json
import logging
from PIL import Image
from redis_manager import RedisManager
from config import Config
from model_utils import ClipClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import redis
import time

class AIWorker:
    def __init__(self):
        self.redis = RedisManager(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB
        )
        self.classifier = ClipClassifier()
        self.queue_name = Config.REDIS_QUEUE

    def process_image_task(self, task_data: dict) -> dict:
        """处理图像路径任务"""
        try:
            # 验证图像路径
            if not os.path.exists(task_data["image_path"]):
                raise FileNotFoundError(f"Image not found at {task_data['image_path']}")

            # 读取图像
            image = Image.open(task_data["image_path"]).convert("RGB")

            # 处理bbox
            if task_data.get("bbox"):
                bbox = task_data["bbox"]
                image = image.crop((
                    bbox["x1"], bbox["y1"],
                    bbox["x2"], bbox["y2"]
                ))

            # 执行预测
            results = self.classifier.predict(
                image=image,
                candidate_labels=task_data["candidate_labels"],
                top_k=Config.TOP_K_RESULTS
            )

            return {
                "task_id": task_data["task_id"],
                "status": "completed",
                "predictions": results,
                "image_path": task_data["image_path"]
            }
        except Exception as e:
            logger.error(f"Task failed: {str(e)}")
            return {
                "task_id": task_data.get("task_id", "unknown"),
                "status": "failed",
                "error": str(e),
                "image_path": task_data.get("image_path", "")
            }

    def run(self):
        """主工作循环"""
        logger.info(f"Worker started. Listening on queue: {self.queue_name}*")
        while True:
            try:
                # 从所有优先级队列获取任务
                for priority in [0, 1, 2]:
                    queue_name = f"{self.queue_name}:{priority}"
                    task_json = self.redis.get_connection().blpop(queue_name, timeout=1)

                    if task_json:
                        _, task_json = task_json
                        task_data = json.loads(task_json)
                        logger.info(f"Processing task {task_data['task_id']} from {queue_name}")

                        # 处理任务
                        result = self.process_image_task(task_data)

                        # 发布结果
                        self.redis.publish_result(
                            task_id=task_data["task_id"],
                            result=result
                        )
                        break

            except redis.exceptions.ConnectionError:
                logger.error("Redis connection error. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logger.exception(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    worker = AIWorker()
    worker.run()