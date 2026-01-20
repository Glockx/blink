# Blink Detection (Only for a fun project)

## Available Options

### Mode Selection

- `--mode`: Choose between `camera` (live webcam) or `video` (file processing)
  - Default: `camera`

### Video Mode Options

- `--video`: Path to input video file (required for video mode)
- `--output`: Path to save annotated output video (optional)
- `--skip`: Process every n-th frame for faster processing
  - `0` = process all frames (default)
  - `1` = every other frame (2x faster)
  - `2` = every 3rd frame (3x faster)
  - etc.

### Camera Mode Options

- `--camera`: Camera device ID
  - Default: `0` (first camera)

### Detection Parameters

- `--threshold`: EAR (Eye Aspect Ratio) threshold for blink detection
  - Default: `0.21`
  - Lower values = more sensitive (detects smaller eye closures)
  - Higher values = less sensitive (requires more pronounced blinks)
- `--consecutive`: Number of consecutive frames below threshold to count as a blink
  - Default: `2`
  - Higher values = reduces false positives

### Advanced Options

- `--model`: Path to custom `face_landmarker.task` model file
  - If not provided, downloads automatically to `~/.mediapipe/models/`
- `--stop-on-blink`: Stop processing immediately when a blink is detected
  - Useful for testing or quick validation

## Usage Examples

### Basic Examples

#### Live Webcam (Default)

```bash
uv run main.py
```

#### Live Webcam with Stop-on-Blink

```bash
uv run main.py --stop-on-blink
```

#### Process Video File

```bash
uv run main.py --mode video --video "input.mp4"
```

#### Process Video and Save Output

```bash
uv run main.py --mode video --video "input.mp4" --output "output.mp4"
```

### Performance Optimization

#### Process Every Other Frame (2x Faster)

```bash
uv run main.py --mode video --video "input.mp4" --output "output.mp4" --skip 1
```

#### Process Every 3rd Frame (3x Faster)

```bash
uv run main.py --mode video --video "input.mp4" --output "output.mp4" --skip 2
```

#### Process Every 5th Frame (5x Faster)

```bash
uv run main.py --mode video --video "input.mp4" --output "output.mp4" --skip 4
```

### Custom Detection Parameters

#### More Sensitive Detection

```bash
uv run main.py --threshold 0.25
```

#### Less Sensitive Detection

```bash
uv run main.py --threshold 0.18
```

#### Reduce False Positives

```bash
uv run main.py --consecutive 3
```

### Advanced Combinations

#### Fast Processing with Stop-on-Blink

```bash
uv run main.py --mode video --video "input.mp4" --skip 2 --stop-on-blink
```

#### Custom Threshold + Save Output

```bash
uv run main.py --mode video --video "input.mp4" --output "output.mp4" --threshold 0.25 --consecutive 3
```

#### Use Specific Camera

```bash
uv run main.py --mode camera --camera 1
```

#### Custom Model Path

```bash
uv run main.py --model "path/to/face_landmarker.task"
```

#### All Options Combined (Video Mode)

```bash
uv run main.py --mode video --video "input.mp4" --output "output.mp4" --skip 2 --threshold 0.22 --consecutive 3 --stop-on-blink
```

#### All Options Combined (Camera Mode)

```bash
uv run main.py --mode camera --camera 0 --threshold 0.22 --consecutive 3 --stop-on-blink
```

## Keyboard Controls (During Execution)

- **`q`**: Quit the program
- **`r`**: Reset blink counter (camera mode only)

## Output Information

The program displays real-time information on the video/camera feed:

- **EAR**: Current Eye Aspect Ratio
- **Blinks**: Total blink count
- **Left EAR**: Left eye aspect ratio
- **Right EAR**: Right eye aspect ratio
- **BLINK!**: Indicator when a blink is detected
- **Frame Progress**: Current frame / total frames (video mode only)
