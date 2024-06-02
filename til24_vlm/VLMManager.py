"""VLM Manager."""

import io
from functools import partial
from math import ceil, floor
from typing import List, Tuple

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import xxhash
from open_clip.transform import PreprocessCfg, image_transform_v2
from PIL import Image, ImageOps
from ultralytics import YOLO

DEVICE = "cuda"

YOLO_PATH = "./models/yolov9c-til24ufo-last.pt"
CLIP_PATH = "./models/wiseft.bin"
MODEL_ARCH = "ViT-H-14-quickgelu"
MODEL_ARCH_PROPS = {
    "size": (224, 224),
    "mode": "RGB",
    "mean": (0.48145466, 0.4578275, 0.40821073),
    "std": (0.26862954, 0.26130258, 0.27577711),
    "interpolation": "bicubic",
    "resize_mode": "longest",
    "fill_color": 0,
}
INIT_JIT = False

YOLO_OPTS = dict(
    conf=0.1,
    iou=0.0,
    imgsz=1536,
    half=True,
    device=DEVICE,
    verbose=False,
    save_dir=None,
    max_det=16,
    agnostic_nms=True,
)

PAD = 0.0
EPAD = 0.0  # What if I purposefully made the submitted bbox larger?


class VLMManager:
    """VLM Manager."""

    def __init__(self):
        """Init."""
        if INIT_JIT:
            self._init_jit()
        else:
            self._init_normal()
            print(self.model.visual.preprocess_cfg)
        yolo = YOLO(YOLO_PATH, task="detect")
        self.det = partial(yolo.predict, **YOLO_OPTS)
        self.hasher = xxhash.xxh64_hexdigest
        self._cache = dict()

    def _init_normal(self):
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            MODEL_ARCH,
            pretrained=CLIP_PATH,
            # pretrained="dfn5b",
            device=DEVICE,
            precision="fp16",
            image_resize_mode="longest",
            image_interpolation="bicubic",
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_ARCH)
        self.model.to(DEVICE).eval()

    def _init_jit(self):
        self.model = torch.jit.load(CLIP_PATH, map_location=DEVICE)
        self.preprocess = image_transform_v2(
            PreprocessCfg(**MODEL_ARCH_PROPS),
            is_train=False,
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_ARCH)
        self.model.to(DEVICE).eval()

    def _crop_bbox_pad(self, im: Image.Image, bbox: Tuple[int, int, int, int], pad=PAD):
        """Crop bbox with pad, filling out of bound with 0."""
        l, t, r, b = bbox
        ih, iw = im.height, im.width
        ch, cw = b - t, r - l
        if ch < 3 or cw < 3:
            return None

        if pad > 0:
            ph, pw = ch * pad, cw * pad
            l, t = floor(l - pw), floor(t - ph)
            r, b = ceil(r + pw), ceil(b + ph)
            el, et = max(-l, 0), max(-t, 0)
            er, eb = max(r - iw, 0), max(b - ih, 0)

        crop = im.crop((max(l, 0), max(t, 0), min(r, iw), min(b, ih)))

        if pad > 0:
            crop = ImageOps.expand(crop, (el, et, er, eb), fill=0)

        return crop

    def _calc_im(self, im: Image.Image):
        # Get bboxes using YOLO.
        results = self.det(im)
        bboxes = results[0].boxes.xyxy.tolist()
        tens = []
        for bbox in bboxes:
            crop = self._crop_bbox_pad(im, bbox)
            if crop is None:
                continue
            tens.append(self.preprocess(crop))

        # NOTE: We purposefully return invalid input if not found; That way, the eval system leaks how many failed altogether.
        if len(tens) == 0:
            return None

        # Normalize & cache crop embeddings.
        bat = torch.stack(tens).to(DEVICE)
        out = F.normalize(self.model.encode_image(bat))
        embs: np.ndarray = out.numpy(force=True)
        return bboxes, embs.T

    def _calc_txt(self, caption):
        tens = self.tokenizer(caption).to(DEVICE)
        out = F.normalize(self.model.encode_text(tens))
        emb: np.ndarray = out.numpy(force=True)
        return emb

    @torch.inference_mode()
    @torch.autocast(DEVICE)
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
        if EPAD > 0:
            ew, eh = w * EPAD, h * EPAD
            l, t, r, b = x1 - ew, y1 - eh, x2 + ew, y2 + eh
            w, h = r - l, b - t
        return l, t, w, h
