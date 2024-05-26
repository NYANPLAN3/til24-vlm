"""VLM Manager."""

import io
from functools import partial
from typing import List

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import xxhash
from PIL import Image
from ultralytics import YOLO

YOLO_PATH = "./models/yolov9c-til24ufo.pt"
CLIP_PATH = "./models/v1_e28_fp16.bin"
MODEL_ARCH = "ViT-H-14-quickgelu"

YOLO_OPTS = dict(
    conf=0.05,
    iou=0.2,
    imgsz=1536,
    half=True,
    device="cuda",
    verbose=False,
    save_dir=None,
    max_det=16,
)


class VLMManager:
    """VLM Manager."""

    def __init__(self):
        """Init."""
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            MODEL_ARCH,
            pretrained=CLIP_PATH,
            # pretrained="dfn5b",
            # jit=True, # NOTE: eval has 1000 samples, JiT isn't worth
            device="cuda",
            precision="fp16",
            image_resize_mode="longest",
            image_interpolation="bicubic",
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_ARCH)
        self.model.cuda().eval()
        yolo = YOLO(YOLO_PATH, task="detect")
        self.det = partial(yolo.predict, **YOLO_OPTS)
        self.hasher = xxhash.xxh64_hexdigest
        self._cache = dict()

    def _calc_im(self, im: Image.Image):
        # Get bboxes using YOLO.
        results = self.det(im)
        bboxes = results[0].boxes.xyxy.tolist()
        tens = []
        for l, t, r, b in bboxes:
            if r - l < 3 or b - t < 3:
                continue
            crop = im.crop((l, t, r, b))
            tens.append(self.preprocess(crop))

        # NOTE: We purposefully return invalid input if not found; That way, the eval system leaks how many failed altogether.
        if len(tens) == 0:
            return None

        # Normalize & cache crop embeddings.
        bat = torch.stack(tens).to("cuda")
        out = F.normalize(self.model.encode_image(bat))
        embs: np.ndarray = out.numpy(force=True)
        return bboxes, embs.T

    def _calc_txt(self, caption):
        tens = self.tokenizer(caption).to("cuda")
        out = F.normalize(self.model.encode_text(tens))
        emb: np.ndarray = out.numpy(force=True)
        return emb

    @torch.inference_mode()
    @torch.autocast("cuda")
    def identify(self, image: bytes, caption: str) -> List[int]:
        """Identify."""
        imhash = self.hasher(image)
        if imhash in self._cache:
            bboxes, crop_embs = self._cache[imhash]
        else:
            file = io.BytesIO(image)
            im = Image.open(file)
            self._cache[imhash] = bboxes, crop_embs = self._calc_im(im)

        caption_emb = self._calc_txt(caption)
        crop_probs = caption_emb @ crop_embs
        idx = crop_probs.argmax().item()

        x1, y1, x2, y2 = bboxes[idx]
        l, t, w, h = x1, y1, x2 - x1, y2 - y1
        return l, t, w, h
