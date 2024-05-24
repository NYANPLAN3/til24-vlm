"""VLM Manager."""

import io
from typing import List

import open_clip
import torch
from open_clip import pretrained
from PIL import Image
from ultralytics import YOLO

YOLO_PATH = "./models/yolov9c-til24ufo.pt"
CLIP_PATH = (
    "./models/Interpause_ViT-H-14-quickgelu-dfn5b-til24id/open_clip_pytorch_model.bin"
)

YOLO_OPTS = dict(
    conf=0.05,
    iou=0.2,
    imgsz=1536,
    half=True,
    device="cuda",
    verbose=False,
    save_dir=None,
)


class VLMManager:
    """VLM Manager."""

    def __init__(self):
        """Init."""
        self.yolo = YOLO(YOLO_PATH, task="detect").cuda()
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            "ViT-H-14-quickgelu",
            pretrained=CLIP_PATH,
            # pretrained="dfn5b",
            device="cuda",
            precision="fp16",
            image_resize_mode="longest",
            image_interpolation="bicubic",
        )
        self.model.cuda().eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")

    @torch.inference_mode()
    @torch.autocast("cuda")
    def identify(self, image: bytes, caption: str) -> List[int]:
        """Identify."""
        # TODO: The eval code wastefully repeats images for each eval call, within a batch, might be able to cache image?
        file = io.BytesIO(image)
        im = Image.open(file)

        # Get bboxes using YOLO.
        results = self.yolo.predict(im, **YOLO_OPTS)
        bboxes = results[0].boxes.xyxy.tolist()
        crops = [im.crop((l, t, r, b)) for l, t, r, b in bboxes]
        crop_tens = []
        for crop in crops:
            crop_tens.append(self.preprocess(crop))

        # NOTE: We purposefully return invalid input if not found; That way, the eval system leaks how many failed altogether.
        if len(crop_tens) == 0:
            return None

        crop_batch = torch.stack(crop_tens).to("cuda")
        caption_tens = self.tokenizer(caption).to("cuda")

        crop_embs = self.model.encode_image(crop_batch)
        caption_emb = self.model.encode_text(caption_tens)

        crop_embs /= crop_embs.norm(dim=-1, keepdim=True)
        caption_emb /= caption_emb.norm(dim=-1, keepdim=True)

        crop_probs = (100.0 * caption_emb @ crop_embs.T).softmax(dim=-1)
        idx = crop_probs.argmax().item()

        x1, y1, x2, y2 = bboxes[idx]
        l, t, w, h = x1, y1, x2 - x1, y2 - y1
        return l, t, w, h
