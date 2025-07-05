import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import numpy as np


# 1. 配置参数
class Config:
    # 模型设置
    model_name = "openai/clip-vit-base-patch32"
    local_model_path = "./clip_model"  # 本地模型路径
    fine_tuned_path = "./fine_tuned_clip"  # 微调后模型保存路径

    # 数据集设置
    dataset_path = "./ClassDataset"  # 你的分类数据集路径
    test_image_path = "./000000001296_bbox0.jpg"  # 测试图像路径

    # 训练参数
    batch_size = 16
    learning_rate = 1e-5
    num_epochs = 5
    warmup_steps = 100
    logging_steps = 50

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 图像预处理
    image_size = 224  # CLIP的标准输入尺寸


# 2. 自定义数据集类
class CustomCLIPDataset(Dataset):
    def __init__(self, dataset_path, processor):
        self.dataset_path = dataset_path
        self.processor = processor
        self.classes = sorted(os.listdir(dataset_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.dataset_path, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    samples.append((img_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        class_name = self.classes[label]

        # 使用类别名作为文本输入
        return {
            "image": image,
            "label": label,
            "text": class_name,
            "class_name": class_name
        }


# 3. 自定义数据整理函数
def collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch])

    # 使用CLIP处理器处理图像和文本
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
        return_attention_mask=True
    )

    return {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }


# 4. 加载模型和处理器
def load_model_and_processor():
    # 检查本地是否有模型
    if os.path.exists(Config.local_model_path):
        print("从本地加载模型...")
        model = CLIPModel.from_pretrained(Config.local_model_path)
        processor = CLIPProcessor.from_pretrained(Config.local_model_path)
    else:
        print("下载模型并保存到本地...")
        model = CLIPModel.from_pretrained(Config.model_name)
        processor = CLIPProcessor.from_pretrained(Config.model_name)
        model.save_pretrained(Config.local_model_path)
        processor.save_pretrained(Config.local_model_path)

    return model, processor


# 5. 微调模型
def fine_tune_clip(model, processor, train_loader):
    model.to(Config.device)
    model.train()

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # 损失函数 - 对比损失
    criterion = nn.CrossEntropyLoss()

    print("\n开始微调...")
    for epoch in range(Config.num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # 移到设备
            input_ids = batch["input_ids"].to(Config.device)
            pixel_values = batch["pixel_values"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_loss=True
            )

            # 计算损失
            loss = outputs.loss
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

            # 更新进度条
            progress_bar.set_postfix({"loss": loss.item()})

            # 日志记录
            if batch_idx % Config.logging_steps == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} 完成 | 平均损失: {avg_loss:.4f}")

    return model


# 6. 预测函数（保留零样本能力）
def zero_shot_predict(model, processor, image_path, candidate_labels):
    """
    既能识别训练过的类别，也能识别新类别的预测函数

    参数:
        model: 微调后的CLIP模型
        processor: CLIP处理器
        image_path: 要预测的图像路径
        candidate_labels: 候选标签列表（可以包含训练集类别和新类别）
    """
    image = Image.open(image_path).convert("RGB")

    # 预处理
    inputs = processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(Config.device)

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 计算概率
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # 整理结果
    results = [{"label": label, "score": float(prob)}
               for label, prob in zip(candidate_labels, probs)]
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


# 7. 主函数
def main():
    # 加载模型和处理器
    model, processor = load_model_and_processor()

    # 准备数据集
    print("\n准备数据集...")
    train_dataset = CustomCLIPDataset(Config.dataset_path, processor)

    # 修改collate_fn以接收processor参数
    def wrapped_collate_fn(batch):
        return collate_fn(batch, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=wrapped_collate_fn
    )

    print(f"数据集统计:")
    print(f"- 总类别数: {len(train_dataset.classes)}")
    print(f"- 总样本数: {len(train_dataset)}")
    print(f"- 类别列表: {train_dataset.classes}")

    # 微调模型
    model = fine_tune_clip(model, processor, train_loader)

    # 保存微调后的模型
    model.save_pretrained(Config.fine_tuned_path)
    processor.save_pretrained(Config.fine_tuned_path)
    print(f"\n微调后的模型已保存到 {Config.fine_tuned_path}")

    # 测试预测
    test_labels = train_dataset.classes + ["unknown_object", "vehicle", "building"]  # 包含训练类别和新类别
    predictions = zero_shot_predict(model, processor, Config.test_image_path, test_labels)

    print("\n测试预测结果:")
    for pred in predictions[:5]:  # 显示前5个结果
        print(f"{pred['label']}: {pred['score']:.4f}")


if __name__ == "__main__":
    main()