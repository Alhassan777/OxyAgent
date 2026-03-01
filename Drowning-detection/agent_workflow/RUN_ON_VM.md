# Running on Hackathon VM

## SSH Connection

```bash
ssh hackathon@34.58.67.151
# Password: 0f48c998
```

## Setup (first time on VM)

```bash
cd ~/InstaLily-Hackathon/Drowning-detection/agent_workflow  # or your project path
python -m pip install -r requirements.txt   # if exists
# Or: pip install opencv-python numpy pillow torch transformers peft ultralytics
```

## Run with Outputs (Track-Pipeline Detection)

Uses the same detection as `run_video_inference_track_pipeline.py`: water_roi + in_water context + Decision rules for better drowning classification.

```bash
# With pool region (x1,y1,x2,y2 pixels) and v2 adapter (same as track pipeline):
python main.py --source "/home/hackathon/Video Playback (1).mp4" --output-dir ./outputs \
  --water-roi "0,200,1280,720" --adapter /home/hackathon/paligemma2-lora-out-v2 \
  --headless --max-seconds 60
```

**Outputs created:**
- `outputs/ambulance_calls.log` - log when ambulance is called (severe/unresponsive)
- `outputs/path_images/` - images of shortest path: stickman (lifeguard), victim, deck + swim segments
- `outputs/detections_*.mp4` - annotated video with detections and status labels

## Run from Video File

```bash
python main.py --source /path/to/video.mp4 --output-dir ./outputs
```

## Run Headless (no display, e.g. on server)

```bash
python main.py --source 0 --headless --output-dir ./outputs --max-seconds 60
```

## Keys (when not headless)

- `Q` - Quit
- `A` - Acknowledge / resolve
- `R` - Reset agent
