from ultralytics import YOLO

# Loads your trained checkpoint. best.pt bundles the model weights AND the class
# names it was trained on (0=ball, 1=goalkeeper, 2=player, 3=referee, 4=other,
# per configs/train.yaml's expected_classes) — that mapping travels with the file,
# you don't set it separately.
model = YOLO("models/best.pt")

# YouTube's bot detection blocks pytubefix's streaming-URL approach on many videos,
# so download locally first (yt-dlp, not pytubefix) and point at the file instead:
#   yt-dlp -f "best[ext=mp4]" -o "data/samples/highlight.mp4" "https://youtu.be/EufrVkKKCwE"
source = "data/samples/vidssave.com Chelsea 2-0 Luton _ HIGHLIGHTS _ Carabao Cup 2026_27 1080P.mp4"

results = model(
    source,
    stream=True,       # see explanation below — required for video
    save=True,          # writes an annotated output video to disk
    conf=0.25,           # drop detections below 25% confidence
    iou=0.45,             # NMS overlap threshold (only matters for non-end2end heads;
                           # harmless to pass either way for YOLO26)
    imgsz=640,             # match training resolution — mismatched imgsz shifts accuracy
    device="mps",            # your Mac's GPU
    project="runs/detect",
    name="youtube_test",      # output lands in runs/detect/youtube_test/
)

# results is a *generator* — nothing has actually run yet. Iterating it is what
# pulls each frame through the model, one at a time.
for i, r in enumerate(results):
    if i % 30 == 0:  # ~once a second at 30fps, so this doesn't spam your terminal
        n_ball = sum(1 for b in r.boxes if int(b.cls[0]) == 0)
        print(f"frame {i}: {len(r.boxes)} detections, {n_ball} ball")
