 #!/usr/bin/env python3
import time
import logging
import statistics
from collections import defaultdict, deque

import cv2
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# ---------------- CONFIG ----------------
VIDEO_PATH   = r"video_used_for_processing_the_vehicle_count.mp4"      # <-- put your video path here
MODEL_PATH   = "yolov8s.pt"                 # YOLOv8 small
OUTPUT_CSV   = "vehicle_counts.csv"
OUTPUT_VIDEO = "demo.mp4"

# Accuracy controls
CONF_THRES     = 0.35         # detection confidence threshold
IOU_THRES      = 0.5          # NMS IoU threshold
MIN_BOX_AREA   = 1600         # ignore tiny boxes (w*h)
FRAME_STRIDE   = 1            # 1 = process every frame (best accuracy)
MIN_HITS       = 3            # track must appear in >= MIN_HITS processed frames before eligible to count
LANE_VOTE_WIN  = 5            # rolling window of lane votes
LANE_VOTE_MIN  = 3            # need at least this many agreeing votes to fix lane

# COCO classes to keep: car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [2, 3, 5, 7]

# ---------------- QUIET YOLO LOGS ----------------
LOGGER.setLevel(logging.ERROR)

# ---------------- INIT ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define 4 vertical lanes as equal-width bands (edit if needed)
lane_bounds = [0,
               int(width * 0.25),
               int(width * 0.50),
               int(width * 0.75),
               width]  # 5 numbers -> 4 segments
num_lanes   = 4

# Video writer (write every processed frame to keep timing natural)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps / max(1, FRAME_STRIDE), (width, height))

# Detector + tracker
model = YOLO(MODEL_PATH)

# State
counts             = {i: 0 for i in range(1, num_lanes + 1)}    # per-lane counts
counted_ids        = set()                                      # IDs already counted (only once total)
track_hits         = defaultdict(int)                           # frames seen for each id
lane_votes         = defaultdict(lambda: deque(maxlen=LANE_VOTE_WIN))  # last K lane votes
fixed_lane_for_id  = {}                                         # id -> fixed lane once stable
rows_csv           = []                                         # CSV rows

def which_lane(cx: int) -> int | None:
    for i in range(num_lanes):
        if lane_bounds[i] <= cx < lane_bounds[i + 1]:
            return i + 1
    return None

print(f"✅ Processing {VIDEO_PATH} with BoT-SORT (accuracy mode)...")

start = time.time()
frame_idx = -1

# Use Ultralytics streaming tracker to preserve IDs
results_gen = model.track(
    source=VIDEO_PATH,
    stream=True,
    tracker="botsort.yaml",
    persist=True,
    conf=CONF_THRES,
    iou=IOU_THRES,
    classes=VEHICLE_CLASSES,
    agnostic_nms=False,
    verbose=False
)

for result in results_gen:
    frame_idx += 1
    if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE != 0):
        # still need to write the frame for smooth output timing
        frame_bgr = result.orig_img.copy()
        out.write(frame_bgr)
        continue

    frame_bgr = result.orig_img.copy()

    if result.boxes is None or result.boxes.id is None:
        # draw lane lines and counts only
        for i in range(1, num_lanes):
            x = lane_bounds[i]
            cv2.line(frame_bgr, (x, 0), (x, height), (0, 0, 255), 2)
        for lane_id, c in counts.items():
            x_left = lane_bounds[lane_id - 1]
            cv2.putText(frame_bgr, f"Lane {lane_id}: {c}", (x_left + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out.write(frame_bgr)
        continue

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    ids        = result.boxes.id.int().cpu().numpy()
    clses      = result.boxes.cls.int().cpu().numpy()
    confs      = result.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), tid, cls, conf in zip(boxes_xyxy, ids, clses, confs):
        # filter by class and confidence (Ultralytics also filters, this is a second guard)
        if int(cls) not in VEHICLE_CLASSES or float(conf) < CONF_THRES:
            continue

        w = max(0, int(x2 - x1))
        h = max(0, int(y2 - y1))
        if w * h < MIN_BOX_AREA:
            continue

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        lane = which_lane(cx)
        if lane is None:
            continue

        # track maturity
        track_hits[tid] += 1
        lane_votes[tid].append(lane)

        # fix the lane for this id once we have a strong consensus
        if tid not in fixed_lane_for_id:
            votes = list(lane_votes[tid])
            if len(votes) >= LANE_VOTE_MIN:
                try:
                    mode_lane = statistics.mode(votes)
                except statistics.StatisticsError:
                    # no unique mode yet
                    mode_lane = None
                if mode_lane is not None and votes.count(mode_lane) >= LANE_VOTE_MIN:
                    fixed_lane_for_id[tid] = mode_lane

        # counting rule: count once per unique ID when
        #  - track is mature (MIN_HITS),
        #  - lane is fixed,
        #  - and we haven't counted this ID before
        if (tid not in counted_ids
            and track_hits[tid] >= MIN_HITS
            and tid in fixed_lane_for_id):
            lane_final = fixed_lane_for_id[tid]
            counts[lane_final] += 1
            counted_ids.add(tid)

            # timestamp based on frame index
            ts_seconds = int(round(frame_idx / max(1.0, fps)))
            rows_csv.append({
                "Vehicle_ID": int(tid),
                "Lane": int(lane_final),
                "Frame": int(frame_idx),
                "Timestamp": f"{ts_seconds//3600:02d}:{(ts_seconds%3600)//60:02d}:{ts_seconds%60:02d}"
            })

        # draw box and id
        lane_disp = fixed_lane_for_id.get(tid, lane)
        cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"ID {int(tid)} L{lane_disp}",
                    (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # draw lane lines and counts
    for i in range(1, num_lanes):
        x = lane_bounds[i]
        cv2.line(frame_bgr, (x, 0), (x, height), (0, 0, 255), 2)
    for lane_id, c in counts.items():
        x_left = lane_bounds[lane_id - 1]
        cv2.putText(frame_bgr, f"Lane {lane_id}: {c}", (x_left + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(frame_bgr)

# finalize
cap.release()
out.release()

# Save CSV
pd.DataFrame(rows_csv).to_csv(OUTPUT_CSV, index=False)

elapsed = time.time() - start
proc_fps = max(0.0001, (frame_idx + 1) / elapsed)

print(f"✅ vehicle_counts.csv written with {len(counted_ids)} unique vehicles")
print(f"✅ {OUTPUT_VIDEO} saved")
print(f"✅ Final lane counts: {counts}")
print(f"✅ Average processing speed: {proc_fps:.1f} FPS")
