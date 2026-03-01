"""Main runtime orchestrator for local real-time drowning detection."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

try:
    from . import config
    from .capture import CameraCapture
    from .display import build_display_frame, draw_minimap, draw_overlay
    from .orchestrator import Orchestrator
    from .outputs import OutputLogger
except ImportError:
    import config  # type: ignore
    from capture import CameraCapture  # type: ignore
    from display import build_display_frame, draw_minimap, draw_overlay  # type: ignore
    from orchestrator import Orchestrator  # type: ignore
    from outputs import OutputLogger  # type: ignore

# Ensure local agent_workflow directory can import detect.py directly.
sys.path.insert(0, os.path.dirname(__file__))

MODEL_LOADED = False
analyze_frame = None


def _load_detect():
    """Load detect module (and thus model). Call after config overrides."""
    global MODEL_LOADED, analyze_frame
    try:
        from detect import MODEL_LOADED as loaded, analyze_frame as af  # type: ignore
        MODEL_LOADED = loaded
        analyze_frame = af
        if MODEL_LOADED:
            print("Real inference active")
        else:
            print("WARNING: Model failed to load - mock mode")
    except ImportError as e:
        print(f"FATAL: detect.py import failed: {e}")
        MODEL_LOADED = False
        analyze_frame = lambda image: {"detections": [], "threat_detected": False, "threat_count": 0}


def _capture_worker(capture: CameraCapture, shared: dict, stop_event: threading.Event) -> None:
    last_frame_id = -1
    while not stop_event.is_set():
        frame, frame_id = capture.get_frame_with_id()
        if frame is None:
            time.sleep(0.005)
            continue
        with shared["lock"]:
            shared["frame"] = frame
            shared["frame_id"] = frame_id
            if frame_id != last_frame_id:
                shared["frame_count"] += 1
                last_frame_id = frame_id


def _inference_worker(
    orchestrator: Orchestrator,
    camera: CameraCapture,
    shared: dict,
    stop_event: threading.Event,
    run_log_path: Path,
    output_logger: OutputLogger | None,
) -> None:
    last_inferred_id = -1
    last_logged_frame = -1

    while not stop_event.is_set():
        current_frame, current_id = camera.get_frame_with_id()

        if current_frame is None:
            time.sleep(0.01)
            continue

        if current_id - last_inferred_id < max(1, int(config.INFERENCE_EVERY)):
            time.sleep(0.005)
            continue

        last_inferred_id = current_id
        frame = current_frame
        frame_count = current_id

        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_shape = (frame.shape[0], frame.shape[1])

        result = orchestrator.run_inference(pil_image, frame_count, frame_shape)

        detections = result["detections"]
        actions = result["actions"]
        dispatch_plan = result["dispatch_plan"]
        swimmers = result["swimmer_positions"]
        victim = result["victim_pos"]
        tracks = result["tracks"]

        p_distress = actions.get("p_distress", 0.0)

        with shared["lock"]:
            shared["detections"] = detections
            shared["p_distress"] = p_distress
            shared["tracks"] = tracks
            shared["last_inferred_frame_id"] = current_id
            shared["agent_actions"] = actions
            shared["agent_state"] = str(actions.get("state", "MONITOR"))
            shared["dispatch_plan"] = dispatch_plan
            shared["swimmer_positions"] = swimmers
            shared["victim_pos"] = victim

        if output_logger:
            ems_payload = actions.get("ems_payload")
            if ems_payload:
                output_logger.log_ambulance(ems_payload)
            state = actions.get("state", "MONITOR")
            last_saved = shared.get("last_path_save_frame", -999)
            save_path = (
                ems_payload is not None
                or (state in ("DISPATCH", "ESCALATE") and dispatch_plan and victim
                   and (frame_count - last_saved) >= 60)
            )
            if save_path and dispatch_plan and victim:
                lg = dispatch_plan.get("lifeguard")
                lg_pos = config.LIFEGUARD_A if lg == "A" else config.LIFEGUARD_B
                output_logger.save_path_image(
                    lg_pos, victim, dispatch_plan.get("jump_point", (0, 0)),
                    swimmers, dispatch_plan,
                )
                with shared["lock"]:
                    shared["last_path_save_frame"] = frame_count

        if frame_count > 0 and frame_count % 100 == 0 and frame_count != last_logged_frame:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "frame": frame_count,
                "state": actions.get("state", "MONITOR"),
                "p_distress": p_distress,
                "eta": None if dispatch_plan is None else float(dispatch_plan.get("eta_seconds", 0.0)),
                "dispatch_plan": dispatch_plan,
                "track_count": len(orchestrator.tracker.tracks),
            }
            with run_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            last_logged_frame = frame_count


def _display_worker(
    orchestrator: Orchestrator,
    shared: dict,
    stop_event: threading.Event,
    headless: bool,
    max_seconds: float,
    output_logger: OutputLogger | None,
) -> None:
    start = time.time()
    target_dt = 1.0 / 15.0

    while not stop_event.is_set():
        loop_start = time.time()

        with shared["lock"]:
            frame = None if shared["frame"] is None else shared["frame"].copy()
            detections = dict(shared.get("detections", {}))
            actions = dict(shared.get("agent_actions", {}))
            dispatch_plan = None if shared.get("dispatch_plan") is None else dict(shared["dispatch_plan"])
            swimmers = list(shared.get("swimmer_positions", []))
            victim = shared.get("victim_pos")
            agent_state = shared.get("agent_state", "MONITOR")
            frame_id = int(shared.get("frame_id", 0))
            last_inferred_frame_id = int(shared.get("last_inferred_frame_id", -1))

        if frame is None:
            time.sleep(0.01)
            continue

        state = actions.get("state", "MONITOR")
        explanation = actions.get("explanation", "")
        detections_stale = frame_id - last_inferred_frame_id > config.STALE_DETECTION_FRAMES
        overlay = draw_overlay(
            frame, detections, state, dispatch_plan, explanation, detections_stale=detections_stale
        )
        minimap = draw_minimap(
            swimmers,
            {"A": config.LIFEGUARD_A, "B": config.LIFEGUARD_B},
            victim,
            None if dispatch_plan is None else dispatch_plan.get("jump_point"),
            dispatch_plan,
            agent_state=agent_state,
        )
        combined = build_display_frame(overlay, minimap)

        if output_logger:
            output_logger.write_video_frame(combined)

        if not headless:
            cv2.imshow("Lifeguard Agent", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                stop_event.set()
            elif key in (ord("a"), ord("A")):
                orchestrator.lifeguard_acknowledged()
            elif key in (ord("r"), ord("R")):
                orchestrator.reset()

        if max_seconds > 0 and (time.time() - start) >= max_seconds:
            stop_event.set()

        elapsed = time.time() - loop_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=os.environ.get("DROWNING_VIDEO_SOURCE", config.SOURCE),
        help="Video source: 0 for webcam, or path to video file. Overridable via DROWNING_VIDEO_SOURCE env.",
    )
    parser.add_argument("--pool_w", type=float, default=config.POOL_W)
    parser.add_argument("--pool_l", type=float, default=config.POOL_L)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Exit with error if PaliGemma fails to load; otherwise run in mock mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Save outputs: ambulance_calls.log, path_images/, detections_*.mp4 video.",
    )
    parser.add_argument(
        "--water-roi",
        default="",
        help="Pool region x1,y1,x2,y2 (pixels). Enables track-pipeline detection. e.g. 0,200,1280,720",
    )
    parser.add_argument(
        "--adapter",
        default="",
        help="Override LoRA adapter path (e.g. /home/hackathon/paligemma2-lora-out-v2). Must set before model load.",
    )
    args = parser.parse_args()

    if not args.headless and not os.environ.get("DISPLAY"):
        print("No DISPLAY detected; switching to headless mode.")
        args.headless = True

    source = int(args.source) if isinstance(args.source, str) and args.source.isdigit() else args.source
    if isinstance(source, str):
        source = str(Path(source))
    # Replace placeholder with actual video path
    if source == "/path/to/video.mp4" or (isinstance(source, str) and "path/to" in source):
        source = str(Path(__file__).resolve().parent.parent.parent / "Video Playback (1).mp4")
    # Existence check for file sources; fallback to webcam
    if isinstance(source, str) and not Path(source).exists():
        print(f"WARNING: Video file not found: {source}. Falling back to webcam (0).")
        source = 0

    # Update config at runtime using CLI overrides. Must set before detect import loads model.
    config.POOL_W = args.pool_w
    config.POOL_L = args.pool_l
    if getattr(args, "water_roi", None) is not None:
        config.WATER_ROI = args.water_roi or ""
    if getattr(args, "adapter", None):
        config.ADAPTER_DIR_OVERRIDE = args.adapter

    _load_detect()

    model_mode = "paligemma2-live" if MODEL_LOADED else "fallback-empty"
    if args.require_model and not MODEL_LOADED:
        print("ERROR: --require-model set but PaliGemma failed to load. Exiting.")
        sys.exit(1)
    print(
        f"Starting Lifeguard Agent | source={source} | pool=({config.POOL_W}m x {config.POOL_L}m) | model={model_mode}"
    )

    ems_callback = getattr(config, "EMS_CALLBACK", "log")
    capture = CameraCapture(source)
    orchestrator = Orchestrator(analyze_frame_fn=analyze_frame, ems_callback=ems_callback)
    stop_event = threading.Event()

    _workflow_dir = Path(__file__).resolve().parent
    output_logger = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_logger = OutputLogger(output_dir)
        fps = capture.video_fps if hasattr(capture, "video_fps") else 15
        output_logger.start_video_writer(fps=fps, frame_w=config.FRAME_W, frame_h=config.FRAME_H)
        print(f"Outputs will be saved to: {output_dir}")

    shared = {
        "frame": None,
        "frame_id": 0,
        "last_inferred_frame_id": -1,
        "last_path_save_frame": -999,
        "detections": {},
        "agent_actions": {},
        "dispatch_plan": None,
        "p_distress": 0.0,
        "swimmer_positions": [],
        "victim_pos": None,
        "frame_count": 0,
        "tracks": [],
        "lock": threading.Lock(),
    }

    run_log_path = _workflow_dir / "run_log.jsonl"

    threads = [
        threading.Thread(target=_capture_worker, args=(capture, shared, stop_event), daemon=True),
        threading.Thread(
            target=_inference_worker,
            args=(orchestrator, capture, shared, stop_event, run_log_path, output_logger),
            daemon=True,
        ),
        threading.Thread(
            target=_display_worker,
            args=(
                orchestrator, shared, stop_event, bool(args.headless), float(args.max_seconds),
                output_logger,
            ),
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Shutting down...")
        stop_event.set()
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        capture.stop()
        cv2.destroyAllWindows()
        if output_logger:
            output_logger.stop_video_writer()
            summary = output_logger.get_output_summary()
            print("Output summary:")
            for k, v in summary.items():
                print(f"  {k}: {v}")

        with shared["lock"]:
            final_state = shared.get("agent_actions", {}).get("state", "MONITOR")
            total_frames = int(shared.get("frame_count", 0))
            final_dispatch = shared.get("dispatch_plan")
            track_count = len(shared.get("tracks", []))

        print("Final state summary:")
        print(f"  frames={total_frames}")
        print(f"  state={final_state}")
        print(f"  active_tracks={track_count}")
        print(f"  dispatch_plan={final_dispatch}")


if __name__ == "__main__":
    main()
