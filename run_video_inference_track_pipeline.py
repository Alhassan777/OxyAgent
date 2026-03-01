#!/usr/bin/env python3
import argparse
import json
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


def parse_detections(text, width, height):
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
        cur = idxs.pop(0)
        keep.append(cur)
        idxs = [i for i in idxs if iou(boxes[cur], boxes[i]) < iou_threshold]
    return keep


def parse_roi(roi_text, width, height):
    if not roi_text:
        return (0, 0, width, height)
    vals = [int(v.strip()) for v in roi_text.split(",")]
    if len(vals) != 4:
        raise ValueError("ROI format: x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return (x1, y1, x2, y2)


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def in_roi(point, roi):
    x, y = point
    x1, y1, x2, y2 = roi
    return x1 <= x < x2 and y1 <= y < y2


def update_tracks(tracks, detections, frame_idx, next_track_id, iou_thresh, max_age):
    det_boxes = [d["bbox"] for d in detections]
    pairs = []
    for track_id, track in tracks.items():
        for di, box in enumerate(det_boxes):
            ov = iou(track["bbox"], box)
            if ov >= iou_thresh:
                pairs.append((ov, track_id, di))
    pairs.sort(reverse=True)

    used_t, used_d = set(), set()
    for _, track_id, det_idx in pairs:
        if track_id in used_t or det_idx in used_d:
            continue
        used_t.add(track_id)
        used_d.add(det_idx)
        det = detections[det_idx]
        track = tracks[track_id]
        track["bbox"] = det["bbox"]
        track["last_frame"] = frame_idx

        cls = det["cls"]
        track["class_votes"][cls] = track["class_votes"].get(cls, 0) + 1
        if cls == track["last_cls"]:
            track["class_run"] += 1
        else:
            track["last_cls"] = cls
            track["class_run"] = 1

        cur_in_water = det["in_water"]
        if (not track["in_water"]) and cur_in_water:
            track["jump_events"] += 1
        track["in_water"] = cur_in_water

    for di, det in enumerate(detections):
        if di in used_d:
            continue
        tracks[next_track_id] = {
            "id": next_track_id,
            "bbox": det["bbox"],
            "last_frame": frame_idx,
            "class_votes": {det["cls"]: 1},
            "last_cls": det["cls"],
            "class_run": 1,
            "in_water": det["in_water"],
            "jump_events": 0,
        }
        next_track_id += 1

    stale = [tid for tid, t in tracks.items() if frame_idx - t["last_frame"] > max_age]
    for tid in stale:
        del tracks[tid]
    return next_track_id


def track_role(track, water_roi):
    cx, cy = center(track["bbox"])
    in_water_now = in_roi((cx, cy), water_roi)
    if in_water_now:
        return "in_water"
    return "deck"


def build_display_tracks(tracks, min_persist):
    out = []
    for t in tracks.values():
        if not t["class_votes"]:
            continue
        majority = max(t["class_votes"], key=t["class_votes"].get)
        if min_persist > 1 and t["class_run"] < min_persist:
            continue
        x1, y1, x2, y2 = t["bbox"]
        out.append(
            {
                "track_id": t["id"],
                "label": majority,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "role": t.get("role", "deck"),
                "jump_events": t.get("jump_events", 0),
                "votes": dict(t["class_votes"]),
            }
        )
    return out


def draw_tracks(frame, tracks):
    for t in tracks:
        color = (0, 255, 255) if t["label"] == "swimming" else (0, 0, 255)
        cv2.rectangle(frame, (t["x1"], t["y1"]), (t["x2"], t["y2"]), color, 2)
        txt = f"#{t['track_id']} {t['label']} {t['role']}"
        cv2.putText(
            frame,
            txt,
            (t["x1"], max(18, t["y1"] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def infer_text(model, processor, image, prompt, device, max_tokens):
    inputs = processor(text=f"<image> {prompt}", images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    continuation = generated[:, input_len:]
    decoded = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
    if decoded:
        return decoded
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--base_model", default="google/paligemma2-3b-pt-224")
    parser.add_argument("--adapter_path", default="/home/hackathon/paligemma2-lora-out-v2")
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument(
        "--prompt",
        default="detect swimming ; drowning",
    )
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--max_seconds", type=float, default=30.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_video", default="video_output_track_pipeline.mp4")
    parser.add_argument("--output_json", default="video_output_track_pipeline.json")

    parser.add_argument("--det_nms_iou", type=float, default=0.45)
    parser.add_argument("--max_people", type=int, default=8)
    parser.add_argument("--track_iou", type=float, default=0.3)
    parser.add_argument("--track_max_age", type=int, default=20)
    parser.add_argument("--persist_frames", type=int, default=1)
    parser.add_argument("--roi", default="")
    parser.add_argument("--water_roi", default="")
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
        model = base_model.to(device).eval()
    else:
        print(f"Loading LoRA adapter: {args.adapter_path}", flush=True)
        model = PeftModel.from_pretrained(base_model, args.adapter_path).to(device).eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(args.max_seconds * fps) if args.max_seconds > 0 else None
    roi = parse_roi(args.roi, width, height)
    water_roi = parse_roi(args.water_roi, width, height) if args.water_roi else roi

    writer = cv2.VideoWriter(
        args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tracks = {}
    next_track_id = 1
    display = []
    results = []
    frame_idx = 0

    print(
        f"Running track-pipeline on {video_path.name} | frames={total_frames} | stride={args.frame_stride}",
        flush=True,
    )
    print(f"Track prompt: {args.prompt}", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        raw_detections = []
        raw_texts = []
        run_this_frame = (frame_idx % max(1, args.frame_stride)) == 0

        if run_this_frame:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded_text = infer_text(
                model,
                processor,
                Image.fromarray(rgb),
                args.prompt,
                device,
                args.max_new_tokens,
            )
            raw_texts.append(decoded_text)

            parsed = parse_detections(decoded_text, width, height)
            rx1, ry1, rx2, ry2 = roi
            for d in parsed:
                cx = (d["x1"] + d["x2"]) / 2.0
                cy = (d["y1"] + d["y2"]) / 2.0
                if not (rx1 <= cx < rx2 and ry1 <= cy < ry2):
                    continue
                in_water_now = in_roi((cx, cy), water_roi)
                raw_detections.append(
                    {
                        "bbox": (d["x1"], d["y1"], d["x2"], d["y2"]),
                        "cls": d["label"],
                        "in_water": in_water_now,
                    }
                )

            if len(raw_detections) > args.max_people:
                raw_detections = raw_detections[: args.max_people]

            next_track_id = update_tracks(
                tracks,
                raw_detections,
                frame_idx,
                next_track_id,
                args.track_iou,
                args.track_max_age,
            )
            for t in tracks.values():
                t["role"] = track_role(t, water_roi)
            display = build_display_tracks(tracks, args.persist_frames)
        else:
            stale = [
                tid
                for tid, t in tracks.items()
                if frame_idx - t["last_frame"] > args.track_max_age
            ]
            for tid in stale:
                del tracks[tid]
            for t in tracks.values():
                t["role"] = track_role(t, water_roi)
            display = build_display_tracks(tracks, args.persist_frames)

        annotated = frame.copy()
        rx1, ry1, rx2, ry2 = roi
        wx1, wy1, wx2, wy2 = water_roi
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 120, 0), 2)
        cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 255, 0), 2)
        draw_tracks(annotated, display)
        writer.write(annotated)

        results.append(
            {
                "frame_idx": frame_idx,
                "ran_inference": run_this_frame,
                "raw_detections": [
                    {"bbox": d["bbox"], "cls": d["cls"], "in_water": d["in_water"]}
                    for d in raw_detections
                ],
                "raw_prediction_texts": raw_texts,
                "track_detections": display,
            }
        )

        if frame_idx % 50 == 0:
            counts = Counter(d["label"] for d in display)
            print(f"Processed {frame_idx}/{max(0, total_frames - 1)} tracks={dict(counts)}", flush=True)
        frame_idx += 1

    cap.release()
    writer.release()
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Done.\nAnnotated video: {args.output_video}\nDetections JSON: {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
