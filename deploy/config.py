import os
import torch
from typing import Optional


class Settings:
    # Redis配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")

    # 队列配置（修复REDIS_QUEUE缺失问题）
    REDIS_QUEUE: str = "clip_classification_queue"  # 添加队列名称
    RESULT_EXPIRE: int = 3600  # 结果过期时间(秒)

    # 模型配置
    MODEL_PATH: str = r".\clip_model"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 任务配置
    TASK_TIMEOUT: int = 30  # 任务超时时间(秒)
    TOP_K_RESULTS: int = 4  # 返回前N个结果

    # 图像处理
    MAX_IMAGE_SIZE: int = 2048  # 图像最大尺寸(像素)

    # 性能配置
    REDIS_POOL_SIZE: int = 20  # Redis连接池大小
    WORKER_PREFETCH: int = 10  # Worker预取任务数

    # 处理器配置
    USE_FAST_PROCESSOR: bool = True  # 解决CLIP处理器警告

    @classmethod
    def validate(cls):
        """配置验证"""
        assert cls.REDIS_PORT > 0, "REDIS_PORT must be positive"
        assert os.path.exists(cls.MODEL_PATH), f"Model path {cls.MODEL_PATH} not exists"


# 实例化配置并验证
Config = Settings()
Config.validate()