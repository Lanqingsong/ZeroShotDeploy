from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from redis_manager import RedisManager
from config import Config
from pydantic import BaseModel
import uuid
from io import BytesIO
from PIL import Image
from typing import Optional, List  # 添加这行导入
app = FastAPI()
redis_mgr = RedisManager(
    host=Config.REDIS_HOST,
    port=Config.REDIS_PORT,
    db=Config.REDIS_DB
)


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class PredictionRequest(BaseModel):
    candidate_labels: list
    bbox: Optional[BBox] = None
    # priority: Optional[int] = 0


class ImageRequest(BaseModel):
    image_path: str
    candidate_labels: List[str]
    bbox: Optional[dict] = None
    priority: Optional[int] = 0

import os

@app.post("/api/v1/predict")
async def predict(request: ImageRequest):
    """接收图像路径的预测接口"""
    try:
        # 验证图像路径
        if not os.path.exists(request.image_path):
            raise HTTPException(400, f"Image not found at {request.image_path}")

        # 创建任务数据
        task_data = {
            "task_id": str(uuid.uuid4()),
            "image_path": request.image_path,
            "candidate_labels": request.candidate_labels,
            "bbox": request.bbox,
            "priority": request.priority
        }

        # 根据优先级选择队列
        queue_name = f"{Config.REDIS_QUEUE}:{min(request.priority, 2)}"
        redis_mgr.push_request(queue_name, task_data)

        # 等待结果
        result = redis_mgr.get_result(task_data["task_id"], timeout=Config.TASK_TIMEOUT)
        if not result:
            raise HTTPException(408, "Processing timeout")

        if result["status"] != "completed":
            raise HTTPException(500, result.get("error", "Processing failed"))

        return JSONResponse({
            "task_id": task_data["task_id"],
            "image_path": request.image_path,
            "results": result["predictions"]
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Internal server error: {str(e)}")


@app.get("/api/v1/task/{task_id}")
async def get_task_status(task_id: str):
    """任务状态查询接口"""
    status = redis_mgr.get_task_status(task_id)
    return {"task_id": task_id, "status": status}


@app.get("/system/health")
async def health_check():
    """健康检查端点"""
    try:
        metrics = redis_mgr.get_queue_metrics(Config.TASK_QUEUE_NAME)
        return {
            "status": "healthy",
            "redis_queue_length": metrics["length"]
        }
    except Exception as e:
        raise HTTPException(500, f"Service unavailable: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    from contextlib import asynccontextmanager


    # 增强生命周期管理
    @asynccontextmanager
    async def app_lifespan(app: FastAPI):
        """应用生命周期管理"""
        # 启动时初始化
        print("🟢 应用初始化中...")
        print(f"• Redis连接: {Config.REDIS_HOST}:{Config.REDIS_PORT} (DB:{Config.REDIS_DB})")
        print(f"• 模型路径: {Config.MODEL_PATH}")
        print(f"• 任务队列: {Config.REDIS_QUEUE}* (0-2优先级)")  # 这里使用 REDIS_QUEUE
        print(f"• 超时设置: {Config.TASK_TIMEOUT}s")

        # 测试Redis连接
        try:
            if not redis_mgr.get_connection().ping():
                raise RuntimeError("Redis连接测试失败")
            print("🟢 Redis连接成功")
        except Exception as e:
            print(f"🔴 Redis连接异常: {str(e)}")
            sys.exit(1)

        yield  # 应用运行期

        # 关闭时清理
        print("\n🔴 应用关闭中...")
        redis_mgr.get_connection().close()


    # 信号处理器
    def handle_shutdown(signum, frame):
        print(f"\n🛑 接收到终止信号 ({signal.Signals(signum).name})")
        sys.exit(0)


    # 注册信号
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # 创建带生命周期的app实例
    app_with_lifespan = FastAPI(
        title="CLIP Image Classification Service",
        description="基于Redis任务队列的分布式图像分类服务",
        version="1.0.0",
        lifespan=app_lifespan
    )

    # 注册现有路由
    app_with_lifespan.include_router(app.router)

    # 配置UVicorn
    server_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1 if Config.DEVICE == "cuda" else 4,
        "log_level": "info",
        "timeout_keep_alive": Config.TASK_TIMEOUT + 10,
    }

    # 启动横幅
    print("\n" + "=" * 50)
    print(f"🚀 CLIP分类服务 v1.0")
    print(f"📡 监听: {server_config['host']}:{server_config['port']}")
    print(f"🖥️ Workers: {server_config['workers']} | Device: {Config.DEVICE.upper()}")
    print(f"📚 文档: http://{server_config['host']}:{server_config['port']}/docs")
    print("=" * 50 + "\n")

    # 启动服务
    uvicorn.run(app_with_lifespan, **server_config)