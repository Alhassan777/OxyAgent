#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


LOC_LABEL_PATTERN = re.compile(
    r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([a-zA-Z_]+)"
)


def parse_roi(roi_text: str, width: int, height: int):
    if not roi_text:
        return (0, 0, width, height)
    parts = [int(p.strip()) for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return (x1, y1, x2, y2)


def iou(a, b):
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


def nms(boxes, scores, iou_threshold=0.45):
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        current = idxs.pop(0)
        keep.append(current)
        idxs = [i for i in idxs if iou(boxes[current], boxes[i]) < iou_threshold]
    return keep


def parse_detections(text: str, width: int, height: int):
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


def classify_crop_label(text: str):
    lower = text.lower().strip()
    label_match = re.search(r"\blabel\s*[:=]\s*(drowning|swimming|unknown)\b", lower)
    if label_match:
        label = label_match.group(1)
        return label if label in {"drowning", "swimming"} else None
    token_match = re.match(r"^\W*(drowning|swimming|unknown)\W*$", lower)
    if token_match:
        label = token_match.group(1)
        return label if label in {"drowning", "swimming"} else None

    words = re.findall(r"[a-z]+", lower)
    if words:
        first = words[0]
        if first in {"drowning", "swimming"}:
            return first
        if first == "unknown":
            return None

    has_drowning = bool(re.search(r"\bdrowning\b", lower))
    has_swimming = bool(re.search(r"\bswimming\b", lower))
    if has_drowning and not has_swimming:
        return "drowning"
    if has_swimming and not has_drowning:
        return "swimming"
    return None


def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def average_motion(centers):
    if len(centers) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i - 1][0]
        dy = centers[i][1] - centers[i - 1][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total / (len(centers) - 1)


def infer_label_on_image(model, processor, image, prompt, device, max_new_tokens):
    inputs = processor(text=f"<image> {prompt}", images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    continuation = generated[:, input_len:]
    decoded = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
    if decoded:
        return decoded
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def detect_people_hog(frame, roi, hog, person_conf):
    x1, y1, x2, y2 = roi
    roi_frame = frame[y1:y2, x1:x2]
    if roi_frame.size == 0:
        return [], []
    boxes, weights = hog.detectMultiScale(
        roi_frame, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    out_boxes = []
    out_scores = []
    for (px, py, pw, ph), score in zip(boxes, weights):
        conf = float(score)
        if conf < person_conf:
            continue
        gx1 = x1 + int(px)
        gy1 = y1 + int(py)
        gx2 = gx1 + int(pw)
        gy2 = gy1 + int(ph)
        if gx2 > gx1 and gy2 > gy1:
            out_boxes.append((gx1, gy1, gx2, gy2))
            out_scores.append(conf)
    return out_boxes, out_scores


def update_tracks(
    tracks,
    detections,
    frame_idx,
    next_track_id,
    iou_threshold,
    track_max_age,
    motion_window,
):
    detection_boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in detections]
    track_ids = list(tracks.keys())
    matches = []
    used_tracks = set()
    used_detections = set()

    pairs = []
    for track_id in track_ids:
        for detection_index, detection_box in enumerate(detection_boxes):
            score = iou(tracks[track_id]["bbox"], detection_box)
            if score >= iou_threshold:
                pairs.append((score, track_id, detection_index))
    pairs.sort(reverse=True)
    for _, track_id, detection_index in pairs:
        if track_id in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_id)
        used_detections.add(detection_index)
        matches.append((track_id, detection_index))

    for track_id, detection_index in matches:
        det = detections[detection_index]
        box = (det["x1"], det["y1"], det["x2"], det["y2"])
        track = tracks[track_id]
        track["bbox"] = box
        track["last_frame"] = frame_idx
        label = det["label"]
        track["label_counts"][label] = track["label_counts"].get(label, 0) + 1
        if label == track["last_label"]:
            track["label_run"] += 1
        else:
            track["last_label"] = label
            track["label_run"] = 1
        centers = track["centers"]
        centers.append(get_center(box))
        if len(centers) > motion_window:
            del centers[0]

    for detection_index, det in enumerate(detections):
        if detection_index in used_detections:
            continue
        box = (det["x1"], det["y1"], det["x2"], det["y2"])
        track_id = next_track_id
        next_track_id += 1
        tracks[track_id] = {
            "id": track_id,
            "bbox": box,
            "last_frame": frame_idx,
            "label_counts": {det["label"]: 1},
            "last_label": det["label"],
            "label_run": 1,
            "centers": [get_center(box)],
        }

    stale_ids = [
        tid
        for tid, track in tracks.items()
        if (frame_idx - track["last_frame"]) > track_max_age
    ]
    for tid in stale_ids:
        del tracks[tid]

    return next_track_id


def stable_track_detections(tracks, persist_frames, drowning_min_motion_px):
    stable = []
    for track in tracks.values():
        counts = track["label_counts"]
        if not counts:
            continue
        majority_label = max(counts, key=counts.get)
        if track["last_label"] != majority_label:
            continue
        if track["label_run"] < persist_frames:
            continue
        motion = average_motion(track["centers"])
        if majority_label == "drowning" and motion < drowning_min_motion_px:
            continue
        x1, y1, x2, y2 = track["bbox"]
        stable.append(
            {
                "track_id": track["id"],
                "label": majority_label,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "motion_px": round(motion, 3),
                "votes": dict(counts),
            }
        )
    return stable


def draw_detections(frame, detections):
    for det in detections:
        label = det["label"]
        color = (0, 255, 255) if label == "swimming" else (0, 0, 255)
        cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, 2)
        track_text = f"#{det.get('track_id', '-')}"
        text = f"{track_text} {label}"
        cv2.putText(
            frame,
            text,
            (det["x1"], max(18, det["y1"] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Run PaliGemma2 + LoRA on video.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--base_model", default="google/paligemma2-3b-pt-224")
    parser.add_argument(
        "--adapter_path", default="/home/hackathon/paligemma2-lora-out-v2"
    )
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument(
        "--prompt",
        default=(
            "You are a pool-safety vision analyst. Analyze ONE PERSON crop from a pool scene and classify only this person. "
            "Use posture and behavior cues: head-above-water control, coordinated horizontal propulsion, vertical struggling, "
            "repeated submersion, frantic arm motion without forward progress, or limp/unresponsive floating. "
            "If evidence is weak, occluded, or the person is not clearly in water, choose unknown. "
            "Avoid guessing from background context or other people. "
            "Output format (required): label=<drowning|swimming|unknown>. "
            "Do not add explanation."
        ),
    )
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--max_seconds", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--output_video", default="video_output_annotated.mp4")
    parser.add_argument("--output_json", default="video_output_detections.json")

    parser.add_argument(
        "--single_stage",
        action="store_true",
        help="Disable person-proposal stage and run Gemma on full frame only.",
    )
    parser.add_argument(
        "--roi",
        default="",
        help="Region of interest x1,y1,x2,y2 in pixels. Default: full frame.",
    )
    parser.add_argument("--person_conf", type=float, default=0.0)
    parser.add_argument("--max_people", type=int, default=8)
    parser.add_argument("--track_iou", type=float, default=0.3)
    parser.add_argument("--track_max_age", type=int, default=20)
    parser.add_argument("--persist_frames", type=int, default=1)
    parser.add_argument("--motion_window", type=int, default=8)
    parser.add_argument(
        "--drowning_min_motion_px",
        type=float,
        default=0.0,
        help="Drowning labels require at least this average track motion.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading base model: {args.base_model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.base_model)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=dtype
    )
    if args.no_adapter:
        print("Running with base model only", flush=True)
        model = base_model.to(device).eval()
    else:
        print(f"Loading LoRA adapter: {args.adapter_path}", flush=True)
        model = PeftModel.from_pretrained(base_model, args.adapter_path).to(device).eval()
    print(f"Inference prompt: {args.prompt}", flush=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(args.max_seconds * fps) if args.max_seconds > 0 else None
    roi = parse_roi(args.roi, width, height)

    writer = cv2.VideoWriter(
        args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    tracks = {}
    next_track_id = 1
    last_stable_detections = []
    results = []
    frame_idx = 0

    print(
        f"Running inference on {video_path.name} | frames={total_frames} | stride={args.frame_stride} | roi={roi}",
        flush=True,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        run_this_frame = (frame_idx % max(1, args.frame_stride)) == 0
        raw_prediction_texts = []
        raw_detections = []

        if run_this_frame:
            if args.single_stage:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded = infer_label_on_image(
                    model,
                    processor,
                    Image.fromarray(rgb),
                    args.prompt,
                    device,
                    args.max_new_tokens,
                )
                parsed = parse_detections(decoded, width, height)
                x1, y1, x2, y2 = roi
                parsed = [
                    d
                    for d in parsed
                    if x1 <= ((d["x1"] + d["x2"]) // 2) < x2
                    and y1 <= ((d["y1"] + d["y2"]) // 2) < y2
                ]
                raw_detections = parsed
                raw_prediction_texts = [decoded]
            else:
                person_boxes, person_scores = detect_people_hog(
                    frame, roi, hog, args.person_conf
                )
                if person_boxes:
                    keep = nms(person_boxes, person_scores, iou_threshold=0.45)
                    person_boxes = [person_boxes[i] for i in keep[: args.max_people]]
                for box in person_boxes:
                    x1, y1, x2, y2 = box
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    decoded = infer_label_on_image(
                        model,
                        processor,
                        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                        args.prompt,
                        device,
                        args.max_new_tokens,
                    )
                    label = classify_crop_label(decoded)
                    raw_prediction_texts.append(decoded)
                    if not label:
                        continue
                    raw_detections.append(
                        {"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    )

                if not raw_detections:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    decoded = infer_label_on_image(
                        model,
                        processor,
                        Image.fromarray(rgb),
                        args.prompt,
                        device,
                        args.max_new_tokens,
                    )
                    parsed = parse_detections(decoded, width, height)
                    x1, y1, x2, y2 = roi
                    parsed = [
                        d
                        for d in parsed
                        if x1 <= ((d["x1"] + d["x2"]) // 2) < x2
                        and y1 <= ((d["y1"] + d["y2"]) // 2) < y2
                    ]
                    raw_prediction_texts.append(decoded)
                    raw_detections = parsed

            next_track_id = update_tracks(
                tracks,
                raw_detections,
                frame_idx,
                next_track_id,
                args.track_iou,
                args.track_max_age,
                args.motion_window,
            )
            last_stable_detections = stable_track_detections(
                tracks, args.persist_frames, args.drowning_min_motion_px
            )
        else:
            stale_ids = [
                tid
                for tid, track in tracks.items()
                if (frame_idx - track["last_frame"]) > args.track_max_age
            ]
            for tid in stale_ids:
                del tracks[tid]
            last_stable_detections = stable_track_detections(
                tracks, args.persist_frames, args.drowning_min_motion_px
            )

        annotated = frame.copy()
        rx1, ry1, rx2, ry2 = roi
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 120, 0), 2)
        annotated = draw_detections(annotated, last_stable_detections)
        writer.write(annotated)

        results.append(
            {
                "frame_idx": frame_idx,
                "ran_inference": run_this_frame,
                "raw_prediction_texts": raw_prediction_texts,
                "raw_detections": raw_detections,
                "stable_detections": last_stable_detections,
            }
        )

        if frame_idx % 50 == 0:
            counts = Counter(d["label"] for d in last_stable_detections)
            print(
                f"Processed frame {frame_idx}/{max(0, total_frames - 1)} | stable={dict(counts)}",
                flush=True,
            )
        frame_idx += 1

    cap.release()
    writer.release()
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(
        f"Done.\nAnnotated video: {args.output_video}\nDetections JSON: {args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
