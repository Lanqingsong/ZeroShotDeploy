# ParallelCLIP - 工业级零样本图像分类系统

## 🚀 项目概述

**ParallelCLIP** 是基于OpenAI CLIP模型构建的高性能图像分类系统，支持：
- **零样本分类**：无需训练直接识别新类别
- **定制化微调**：适配特定业务场景
- **分布式部署**：高并发生产级推理方案




# 第一章：CLIP零样本图像分类技术

## 1.1 CLIP模型概述

### 模型背景
CLIP（Contrastive Language-Image Pretraining）是OpenAI提出的多模态预训练模型，通过400M图像-文本对训练获得跨模态理解能力。其核心创新在于：

- **双流架构**：并行处理视觉和文本输入
- **对比学习**：拉近匹配图文对的嵌入距离
- **零样本迁移**：无需微调即可适应新任务

### 技术特性
| 特性                | 说明                          |
|---------------------|-------------------------------|
| 视觉 backbone       | ViT-B/32 或 ResNet50          |
| 文本编码器          | 63M参数 Transformer           |
| 最大输入分辨率      | 224x224（ViT版本）            |
| 典型推理延迟        | 50-200ms（取决于硬件）        |

## 1.2 零样本分类原理

### 工作流程
1. **文本编码**  
   将候选标签转换为提示文本（如`"a photo of a {label}"`）
2. **图像编码**  
   提取图像视觉特征向量
3. **相似度计算**  
   在共享嵌入空间计算图文相似度
4. **结果排序**  
   按相似度得分输出分类结果

## 1.3 应用优势

### 业务价值
- **即时可用**：无需收集标注数据即可部署
- **动态扩展**：通过修改文本提示增加新类别
- **多语言支持**：原生支持跨语言分类

### 典型场景
- 社交媒体内容审核
- 电商产品自动归类
- 医学影像初步筛查
- 智能相册管理

### 代码运行

```bash
python CLIP_Classifier.py
```

# 第二章：CLIP微调

## 2.1 数据标注
   第一步 使用labelme进行矩形框标注，得到json文件
   第二步 使用提供的createDataset.py自动进行图像区域截取和类别管理
    ```bash
    python createDataset.py
    ```
    需要根据实际情况修改json_root_dir，脚本自动将数据整理到ClassDataset目录下
## 2.2 启动微调程序
    ```bash
    python finutine_zeroshot.py
    ```
   根据实际情况修改finutine_zeroshot.py中 Config中的内容
## 2.3 测试微调结果

 修改createDataset.py中的配置，将模型改为微调后的模型
    ```python
     model_path = "./fine_tuned_clip"
    ```

    ```bash
    python createDataset.py
    ```


# 第三章：高并发GPU部署
## 3.1 目录结构
   ```bash
      ClipClassifier/
        ├── deploy/                  # 部署相关代码
        │   ├── __init__.py          # Python包初始化文件
        │   ├── config.py            # 配置文件（含路径白名单/超时设置等）
        │   ├── httpApi.py           # FastAPI主服务入口
        │   ├── model_utils.py       # 模型加载与预测工具
        │   ├── worker.py            # Redis任务处理器
        │   └── redis_manager.py     # Redis连接池管理
        ├── models/                  # 模型存储目录
        │   └── fine_tuned_clip/     # 微调后的CLIP模型
        │       ├── config.json
        │       ├── pytorch_model.bin
        │       └── preprocessor_config.json
        ├── tests/                   # 测试套件
        │   ├── unit/                # 单元测试
        │   └── integration/         # 集成测试
        ├── docs/                    # 文档
        │   └── API_REFERENCE.md     # API接口文档
        └── requirements.txt         # 依赖清单
   ```
    
## 3.2 注意修改
   1. 本项目只做简单的示例，使用的本地的图像路径，实际生产过程应该使用共享的存储服务器上
   2. 注意配置文件尤其是模型的地址 
   3. 需要按照redis，并且注意相关配置


## 3.3 启动推理服务端程序
    ```python
        # 启动API服务
        nohup python httpApi.py --port 8000 --workers 4 &
        
        # 启动Worker,可启动多个，
        python worker.py --gpu 0 --batch_size 16
    '''


## 参考链接
https://github.com/openai/CLIP