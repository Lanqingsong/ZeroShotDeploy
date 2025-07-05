from transformers import CLIPModel, CLIPProcessor
import torch
from config import Config


class ClipClassifier:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(Config.MODEL_PATH).to(Config.DEVICE)
        self.processor = CLIPProcessor.from_pretrained(Config.MODEL_PATH)
        self.model.eval()

    def predict(self, image, candidate_labels, top_k=4):
        """返回top_k个类别及其概率"""
        inputs = self.processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(Config.DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        results = sorted(zip(candidate_labels, probs),
                         key=lambda x: x[1], reverse=True)[:top_k]

        return [{"label": label, "score": float(score)} for label, score in results]