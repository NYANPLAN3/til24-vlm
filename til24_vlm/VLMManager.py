"""VLM Manager."""

from typing import List

import torch
import torchvision
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_PATH = "./models/google_owlv2-large-patch14-ensemble"


class VLMManager:
    """VLM Manager."""

    def __init__(self):
        """Init."""
        self.processor: Owlv2Processor = Owlv2Processor.from_pretrained(MODEL_PATH)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_PATH)
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def identify(self, image: bytes, caption: str) -> List[int]:
        """Identify."""
        # TODO: The eval code wastefully repeats images for each eval call, within a batch, might be able to cache image?
        raw = torch.frombuffer(image, dtype=torch.uint8)
        im = torchvision.io.decode_image(raw)
        _, ih, iw = im.shape

        inp = self.processor(text=[caption], images=im, return_tensors="pt")
        inp.to(self.device)
        out = self.model(**inp)

        result = self.processor.post_process_object_detection(
            outputs=out, target_sizes=[(ih, iw)], threshold=0.01
        )[0]
        boxes, scores = result["boxes"], result["scores"]
        if len(boxes) == 0:
            return 0, 0, 0, 0
        box = sorted(zip(boxes, scores), key=lambda x: x[1], reverse=True)[0][0]

        x1, y1, x2, y2 = box.tolist()
        l, t, w, h = x1, y1, x2 - x1, y2 - y1
        return l, t, w, h
