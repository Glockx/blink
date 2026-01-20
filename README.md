## Live Webcam

```bash
uv run main.py
```

## Process all frames in video sample:

```bash
uv run main.py --mode video --video "C:/Users/nijat/Pictures/Camera Roll/WIN_20260120_15_35_45_Pro.mp4" --output "output_blink_detected.mp4"
```

## Process every other frame (2x faster):

```bash
uv run main.py --mode video --video "C:/Users/nijat/Pictures/Camera Roll/WIN_20260120_15_35_45_Pro.mp4" --output "output_blink_detected.mp4" --skip 2
```

## Process every 5th frame (5x faster):

```bash
uv run main.py --mode video --video "C:/Users/nijat/Pictures/Camera Roll/WIN_20260120_15_35_45_Pro.mp4" --output "output_blink_detected.mp4" --skip 4
```
