from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

MODEL_ID = "google/paligemma2-3b-pt-224"
ADAPTER_DIR = Path(__file__).resolve().parent.parent.parent / "paligemma2-lora-out"
MODEL_LOADED = False
processor = None
model = None

YOLO_MODEL = None
HOG_DETECTOR = None
DETECTOR_BACKEND = "yolo"

YOLO_CONF = 0.4
MIN_BOX_W = 18
MIN_BOX_H = 24
MIN_BOX_AREA_FRAC = 0.003
MAX_BOX_AREA_FRAC = 0.8
MAX_BOX_WH_RATIO = 4.5
NMS_IOU_THRESHOLD = 0.45
MAX_PEOPLE = 8

DETECTION_PROMPT = "detect swimming ; drowning"

LOC_LABEL_PATTERN = re.compile(
    r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([a-zA-Z_]+)"
)

CLASSIFICATION_PROMPT_BASE = (
    "You are a pool-safety vision assistant analyzing ONE tracked person crop from a surveillance video. "
    "Classify this person's state using behavior cues only for this person: "
    "head-above-water control, horizontal propulsion, coordinated strokes, "
    "vertical struggling/treading with little progress, repeated submersion, frantic ineffective arm motion, "
    "or limp/unresponsive floating. "
    "Avoid using labels from other people in the frame. "
    "When uncertain or evidence is weak/occluded, return unknown."
)

CLASSIFICATION_PROMPT = (
    "You are a pool-safety vision assistant analyzing ONE tracked person crop. "
    "Classify this person's state using behavior cues: head-above-water control, "
    "horizontal propulsion, coordinated strokes, vertical struggling/treading, "
    "repeated submersion, frantic ineffective arm motion, or limp/unresponsive floating. "
    "Output format (required): label=<drowning|swimming|unknown>."
)


def _parse_roi(roi_text: str, width: int, height: int) -> tuple[int, int, int, int]:
    """Parse ROI "x1,y1,x2,y2" or return full frame."""
    if not roi_text or not roi_text.strip():
        return (0, 0, width, height)
    vals = [int(v.strip()) for v in roi_text.split(",")]
    if len(vals) != 4:
        return (0, 0, width, height)
    x1, y1, x2, y2 = vals
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return (x1, y1, x2, y2)


def _in_roi(point: tuple[float, float], roi: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = roi
    return x1 <= x < x2 and y1 <= y < y2


def parse_detections(text: str, width: int, height: int) -> list[dict]:
    detections = []
    for match in LOC_LABEL_PATTERN.finditer(text):
        y1n, x1n, y2n, x2n, label = match.groups()
        y1 = int(int(y1n) / 1024.0 * height)
        x1 = int(int(x1n) / 1024.0 * width)
        y2 = int(int(y2n) / 1024.0 * height)
        x2 = int(int(x2n) / 1024.0 * width)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 > x1 and y2 > y1:
            detections.append(
                {"label": label.lower(), "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )
    return detections


def _infer_raw(image: Image.Image, prompt: str) -> str:
    """Run PaliGemma inference and return the raw decoded text."""
    global MODEL_LOADED, processor, model
    if not MODEL_LOADED:
        return ""
    if image.mode != "RGB":
        image = image.convert("RGB")
    device = next(
        (p.device for p in model.parameters() if p.device.type != "meta"),
        _get_compute_device(),
    )
    inputs = processor(text=f"<image> {prompt}", images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    continuation = generated[:, input_len:]
    decoded = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
    if decoded:
        return decoded
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _build_track_prompt(in_water_now: bool) -> str:
    """Track-pipeline style: add in_water/on_deck context to improve drowning classification."""
    location_text = "in_water" if in_water_now else "on_deck"
    return (
        f"{CLASSIFICATION_PROMPT_BASE}\n"
        f"Context: tracked_person_location={location_text}.\n"
        "Decision rules: drowning = distress or loss of control in water; "
        "swimming = controlled purposeful movement; "
        "unknown = insufficient/ambiguous evidence or not in water.\n"
        "Output format (required): label=<drowning|swimming|unknown>.\n"
        "Do not repeat instructions or explain."
    )


def label_to_p_distress(label: str) -> float:
    label = label.strip().lower()
    if label == "drowning":
        return 0.95
    elif label == "swimming":
        return 0.05
    else:
        return 0.5


def _empty_result(error: str | None = None) -> dict:
    payload = {
        "detections": [],
        "threat_detected": False,
        "threat_count": 0,
    }
    if error:
        payload["error"] = error
    return payload


def _get_compute_device() -> torch.device:
    """Single device to avoid meta-device lazy-load issues with device_map='auto'."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _initialize_model(skip_adapter: bool = False) -> None:
    global MODEL_LOADED, processor, model

    try:
        device = _get_compute_device()
        print(f"Loading PaliGemma 2 from HuggingFace ({MODEL_ID})...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=False,
        )
        model = model.to(device)

        try:
            from . import config as _cfg
            override = getattr(_cfg, "ADAPTER_DIR_OVERRIDE", "") or ""
        except Exception:
            override = ""
        use_dir = Path(override).resolve() if override and str(override).strip() else ADAPTER_DIR
        adapter_config = use_dir / "adapter_config.json"
        if not skip_adapter and adapter_config.exists():
            try:
                print(f"Loading LoRA adapter from {ADAPTER_DIR}...")
                model = PeftModel.from_pretrained(model, str(use_dir))
                print("Fine-tuned adapter loaded")
            except Exception as adapter_err:
                print(f"Adapter load failed (vision/arch mismatch): {adapter_err}")
                print("Continuing with base model only (zero-shot).")
        else:
            print(f"No adapter found at {use_dir} - running zero-shot")

        model.eval()
        MODEL_LOADED = True
        print(f"PaliGemma 2 ready on {device}.")
    except Exception as e:
        MODEL_LOADED = False
        processor = None
        model = None
        print("ERROR: Could not load PaliGemma 2")
        print("Check: huggingface-cli login and license accepted")
        print(f"Detail: {e}")


def _initialize_detector(backend: str = "yolo") -> None:
    global YOLO_MODEL, HOG_DETECTOR, DETECTOR_BACKEND
    DETECTOR_BACKEND = backend

    if backend == "yolo":
        try:
            from ultralytics import YOLO
            YOLO_MODEL = YOLO("yolov8n.pt")
            print("YOLOv8 person detector loaded")
        except Exception as e:
            print(f"YOLO load failed: {e}, falling back to HOG")
            DETECTOR_BACKEND = "hog"
            _initialize_hog()
    else:
        _initialize_hog()


def _initialize_hog() -> None:
    global HOG_DETECTOR
    HOG_DETECTOR = cv2.HOGDescriptor()
    HOG_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    print("HOG person detector loaded")


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)


def nms(boxes: list, scores: list, iou_threshold: float = 0.45) -> list:
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        cur = idxs.pop(0)
        keep.append(cur)
        idxs = [i for i in idxs if iou(boxes[cur], boxes[i]) < iou_threshold]
    return keep


def is_plausible_person_box(
    box: tuple,
    frame_w: int,
    frame_h: int,
    min_w: int = MIN_BOX_W,
    min_h: int = MIN_BOX_H,
    min_area_frac: float = MIN_BOX_AREA_FRAC,
    max_area_frac: float = MAX_BOX_AREA_FRAC,
    max_wh_ratio: float = MAX_BOX_WH_RATIO,
) -> bool:
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    if w < min_w or h < min_h:
        return False
    ratio = max(w / float(h), h / float(w))
    if ratio > max_wh_ratio:
        return False
    frame_area = max(1, frame_w * frame_h)
    area_frac = (w * h) / float(frame_area)
    if area_frac < min_area_frac or area_frac > max_area_frac:
        return False
    return True


def detect_people_yolo(frame: np.ndarray, conf_thresh: float = YOLO_CONF) -> tuple[list, list]:
    global YOLO_MODEL
    if YOLO_MODEL is None:
        return [], []

    pred = YOLO_MODEL.predict(
        source=frame, conf=conf_thresh, classes=[0], verbose=False, imgsz=640
    )
    if not pred or pred[0].boxes is None:
        return [], []

    boxes_xyxy = pred[0].boxes.xyxy.detach().cpu().numpy()
    scores = pred[0].boxes.conf.detach().cpu().numpy()
    out_boxes, out_scores = [], []

    for (bx1, by1, bx2, by2), score in zip(boxes_xyxy, scores):
        box = (int(bx1), int(by1), int(bx2), int(by2))
        if box[2] > box[0] and box[3] > box[1]:
            out_boxes.append(box)
            out_scores.append(float(score))

    return out_boxes, out_scores


def detect_people_hog(frame: np.ndarray, conf_thresh: float = 0.0) -> tuple[list, list]:
    global HOG_DETECTOR
    if HOG_DETECTOR is None:
        _initialize_hog()

    boxes, weights = HOG_DETECTOR.detectMultiScale(
        frame, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    out_boxes, out_scores = [], []

    for (px, py, pw, ph), score in zip(boxes, weights):
        score = float(score)
        if score < conf_thresh:
            continue
        box = (int(px), int(py), int(px + pw), int(py + ph))
        if box[2] > box[0] and box[3] > box[1]:
            out_boxes.append(box)
            out_scores.append(score)

    return out_boxes, out_scores


def classify_from_text(text: str) -> str:
    lower = text.lower().strip()

    label_match = re.search(r"\blabel\s*[:=]\s*(drowning|swimming|unknown)\b", lower)
    if label_match:
        return label_match.group(1)

    token_match = re.match(r"^\W*(drowning|swimming|unknown)\W*$", lower)
    if token_match:
        return token_match.group(1)

    words = re.findall(r"[a-z]+", lower)
    if words:
        first = words[0]
        if first in {"drowning", "swimming", "unknown"}:
            return first

    has_drown = bool(re.search(r"\bdrowning\b", lower))
    has_swim = bool(re.search(r"\bswimming\b", lower))
    if has_drown and not has_swim:
        return "drowning"
    if has_swim and not has_drown:
        return "swimming"
    return "unknown"


def _classify_single(crop_image: Image.Image, prompt: str) -> str:
    """Internal: classify one crop. Used by classify_crop and as batch fallback."""
    global MODEL_LOADED, processor, model
    if not MODEL_LOADED:
        return "unknown"
    try:
        if crop_image.mode != "RGB":
            crop_image = crop_image.convert("RGB")
        inputs = processor(
            text=f"<image> {prompt}",
            images=crop_image,
            return_tensors="pt",
        )
        device = next(
            (p.device for p in model.parameters() if p.device.type != "meta"),
            _get_compute_device(),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        continuation = generated[:, input_len:]
        decoded = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
        return classify_from_text(decoded) if decoded else classify_from_text(
            processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        )
    except Exception as e:
        print(f"classify error: {e}")
        return "unknown"


def classify_crop(crop_image: Image.Image, prompt: str = CLASSIFICATION_PROMPT) -> str:
    """Classify a single person crop."""
    return _classify_single(crop_image, prompt)


def classify_crops_batch(
    crop_images: list[Image.Image],
    prompt: str | list[str] = CLASSIFICATION_PROMPT,
) -> list[str]:
    """Classify multiple person crops. prompt: single str or list of str (one per crop)."""
    global MODEL_LOADED, processor, model
    if not MODEL_LOADED:
        return ["unknown"] * len(crop_images)
    crops = [c.convert("RGB") if c.mode != "RGB" else c for c in crop_images]
    if not crops:
        return []

    prompts = [prompt] * len(crops) if isinstance(prompt, str) else prompt
    if len(prompts) != len(crops):
        prompts = [prompts[0] if prompts else CLASSIFICATION_PROMPT] * len(crops)

    if len(crops) == 1:
        return [_classify_single(crops[0], prompts[0])]

    try:
        inputs = processor(
            text=[f"<image> {p}" for p in prompts],
            images=crops,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        device = next(
            (p.device for p in model.parameters() if p.device.type != "meta"),
            _get_compute_device(),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)

        labels: list[str] = []
        for i in range(generated.shape[0]):
            continuation = generated[i : i + 1, input_len:]
            decoded = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
            labels.append(classify_from_text(decoded) if decoded else "unknown")
        return labels
    except Exception as e:
        print(f"classify_crops_batch error ({len(crops)} crops): {e}, falling back to sequential")
        return [_classify_single(c, p) for c, p in zip(crops, prompts)]


def analyze_frame(image: Image.Image) -> dict:
    try:
        if not MODEL_LOADED:
            return _empty_result(error="Model not loaded")

        if image.mode != "RGB":
            image = image.convert("RGB")

        frame_w, frame_h = image.size

        decoded_text = _infer_raw(image, DETECTION_PROMPT)
        parsed = parse_detections(decoded_text, frame_w, frame_h)

        if not parsed:
            return _empty_result()

        if len(parsed) > MAX_PEOPLE:
            parsed = parsed[:MAX_PEOPLE]

        detections: list[dict] = []
        for d in parsed:
            label = d["label"]
            is_threat = label == "drowning"
            detections.append({
                "label": label,
                "bbox": [d["x1"], d["y1"], d["x2"], d["y2"]],
                "is_threat": is_threat,
                "p_distress": label_to_p_distress(label),
                "p_unresponsive": 0.0,
            })

        threat_count = sum(1 for d in detections if d["is_threat"])
        return {
            "detections": detections,
            "threat_detected": threat_count > 0,
            "threat_count": threat_count,
        }
    except Exception as e:
        print(f"analyze_frame error: {e}")
        return _empty_result(error=str(e))


_initialize_model(skip_adapter=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test image")
    parser.add_argument("--no-adapter", action="store_true", help="Skip adapter loading")
    args = parser.parse_args()

    if args.no_adapter:
        print("Reloading model without adapter...")
        _initialize_model(skip_adapter=True)

    img = Image.open(Path(args.test))
    print(f"Image size: {img.size}")
    print(f"Running PaliGemma detection with prompt: {DETECTION_PROMPT}")

    result = analyze_frame(img)

    print(json.dumps(result, indent=2))
    print(f"Threat detected: {result['threat_detected']}")
    print(f"Detection count: {len(result['detections'])}")
