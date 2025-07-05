import redis
import json
import uuid
from typing import Optional, Dict, Any, Union
from functools import wraps
import logging

class RedisManager:
    """
    Redis任务队列高级管理器
    功能：
    - 请求/响应式队列管理
    - 自动任务ID生成
    - 结果过期处理
    - 连接池管理
    """
    def __init__(self, host='127.0.0.1', port=6379, db=0, password=None, max_connections=10):
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=False  # 保持二进制数据兼容性
        )
        self.logger = logging.getLogger(__name__)

    def get_connection(self):
        """获取Redis连接（建议使用with语句）"""
        return redis.Redis(connection_pool=self.pool)

    def _handle_connection(func):
        """连接管理装饰器"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with self.get_connection() as conn:
                try:
                    return func(self, conn, *args, **kwargs)
                except redis.RedisError as e:
                    self.logger.error(f"Redis operation failed: {str(e)}")
                    raise
        return wrapper

    @_handle_connection
    def push_request(self, conn, queue_name: str, data: Union[Dict, str],
                   expire: int = 3600) -> str:
        """
        提交任务到处理队列
        :param queue_name: 队列名称
        :param data: 任务数据（自动添加task_id）
        :param expire: 结果过期时间（秒）
        :return: 任务ID
        """
        task_id = f"task_{uuid.uuid4().hex}"
        if isinstance(data, dict):
            data['task_id'] = task_id
            payload = json.dumps(data)
        else:
            payload = data

        # 使用事务保证原子性
        pipe = conn.pipeline()
        pipe.rpush(queue_name, payload)
        pipe.set(f"task_status:{task_id}", "queued", ex=expire)
        pipe.execute()

        return task_id

    @_handle_connection
    def get_result(self, conn, task_id: str,
                 timeout: int = 30) -> Optional[Dict]:
        """
        获取任务结果（阻塞式）
        :param task_id: 任务ID
        :param timeout: 最长等待时间（秒）
        :return: 结果字典或None
        """
        result = conn.blpop(f"result:{task_id}", timeout=timeout)
        if result:
            try:
                return json.loads(result[1])
            except json.JSONDecodeError:
                return {"status": "error", "message": "Invalid result format"}
        return None

    @_handle_connection
    def publish_result(self, conn, task_id: str,
                     result: Dict, expire: int = 3600):
        """
        发布处理结果
        :param task_id: 任务ID
        :param result: 结果字典
        :param expire: 结果过期时间（秒）
        """
        pipe = conn.pipeline()
        pipe.lpush(f"result:{task_id}", json.dumps(result))
        pipe.set(f"task_status:{task_id}", "completed", ex=expire)
        pipe.execute()

    @_handle_connection
    def get_task_status(self, conn, task_id: str) -> str:
        """获取任务状态（queued/processing/completed/failed）"""
        return conn.get(f"task_status:{task_id}") or "unknown"

    @_handle_connection
    def get_queue_metrics(self, conn, queue_name: str) -> Dict:
        """获取队列监控指标"""
        return {
            "length": conn.llen(queue_name),
            "memory_usage": conn.memory_usage(queue_name),
            "first_item": conn.lindex(queue_name, 0)
        }