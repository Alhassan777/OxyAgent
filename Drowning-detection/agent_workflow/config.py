"""Configuration values for local drowning detection edge app."""

from pathlib import Path

# Pool dimensions (meters)
POOL_W = 25.0
POOL_L = 50.0

# Speeds
DECK_SPEED = 3.0   # m/s
SWIM_SPEED = 1.0   # m/s

# Path agent: crowd density and swimming speed variation
PATH_CROWD_SIGMA = 1.5         # Gaussian sigma for per-swimmer crowd penalty
PATH_CROWD_AMPLITUDE = 2.0    # Gaussian amplitude for crowd penalty
PATH_DENSITY_SPEED_K = 0.5    # Speed reduction factor: eff_speed = SWIM_SPEED / (1 + k * density)
PATH_N_BOUNDARY_POINTS = 80   # Points sampled on pool perimeter
SWIM_SPEED_SLOW_ZONE = 0.6    # Optional: reduced speed in crowded zones (unused if density-based)

# EMS: pluggable callback ("log" | "null" | custom name)
EMS_CALLBACK = "log"          # "log" = write to alerts.log, "null" = no-op

# Lifeguard positions in pool coords (meters from top-left)
LIFEGUARD_A = (2.0, 25.0)   # left side midpoint
LIFEGUARD_B = (23.0, 25.0)  # right side midpoint

# Agent thresholds
ALERT_THRESHOLD = 0.6      # p_distress to trigger ALERT
DISPATCH_THRESHOLD = 0.75  # p_distress to trigger DISPATCH
ESCALATE_THRESHOLD = 0.9   # p_distress to trigger ESCALATE
UNRESPONSIVE_SECONDS = 5   # seconds above threshold to escalate
TEMPORAL_WINDOW = 10       # frames to smooth p_distress over

# Tracking parameters
TRACK_IOU_THRESHOLD = 0.3  # IoU threshold for track association
TRACK_MAX_AGE = 20         # frames before track is deleted
TRACK_MIN_PERSIST = 1      # min consecutive frames to consider stable

# Person detection parameters
DETECTOR_BACKEND = "yolo"  # "yolo" or "hog"
YOLO_CONF = 0.4            # YOLO confidence threshold
MAX_PEOPLE = 8             # max people to track

# Track-pipeline style: water ROI and contextual prompt (from run_video_inference_track_pipeline)
# WATER_ROI: "x1,y1,x2,y2" in pixels, pool region. Empty = full frame. People in water_roi get in_water context.
WATER_ROI = ""             # e.g. "0,200,1280,720" for lower half = pool

# ADAPTER_DIR override: set to use different LoRA (e.g. paligemma2-lora-out-v2 on VM). Empty = use default.
ADAPTER_DIR_OVERRIDE = ""  # e.g. "/home/hackathon/paligemma2-lora-out-v2"

# Display
FRAME_W = 1280
FRAME_H = 720
MINIMAP_W = 400
MINIMAP_H = 300
INFERENCE_EVERY = 5        # run PaliGemma every N frames
STALE_DETECTION_FRAMES = 15  # show stale indicator when display frame - inferred frame > this

# Source: 0 = webcam, or path to video file
SOURCE = str(Path(__file__).resolve().parent.parent.parent / "Video Playback (1).mp4")
