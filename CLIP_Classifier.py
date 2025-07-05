from transformers import pipeline
from PIL import Image
import os

# 指定本地模型路径
model_path = "./clip_model"

# 检查模型是否存在，不存在则下载
if not os.path.exists(model_path):
    print("本地模型不存在，正在下载...")
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.save_pretrained(model_path)
    processor.save_pretrained(model_path)

# 从本地加载模型创建管道
classifier = pipeline(
    "zero-shot-image-classification",
    model=model_path,
    device="cuda"  # 如果有GPU可以使用 device="cuda"
)

# 使用本地图片
image_path = "29.png"  # 替换为你的图片路径
try:
    image = Image.open(image_path)

    # 定义候选标签
    candidate_labels = ["car", "truck", "motorcycle",
                        "bicycle", "bus", "cat", "dog",
                        "branches","leaf","golf",
                        "grass","person","animal",
                        "metal","bottle","other",
                        "airplane","tools","corpse",
                        "plastics","tyre","Iron","belt",
                        "rubber","paper","ice","stone",
                        "glass","ball","key","fissure",
                        "lamplight","mouse","insect","bird","phone"
                        ]

    # 进行分类预测
    results = classifier(image, candidate_labels=candidate_labels)

    # 打印结果
    print(f"\n图片 '{image_path}' 的分类结果:")
    for result in results:
        print(f"{result['label']}: {result['score']:.4f}")

except FileNotFoundError:
    print(f"错误: 图片文件 '{image_path}' 未找到")
except Exception as e:
    print(f"发生错误: {str(e)}")