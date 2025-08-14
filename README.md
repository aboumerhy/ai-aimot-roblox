
# Player Detector Overlay

A from-scratch screen-based detector and transparent overlay that can learn to localize a specific kind of target
(e.g., a character silhouette) on your screen **without hooking or injecting** into any client. It uses:

- **Your own dataset** of positive (target) vs. negative (background) crops.
- A **small Keras CNN** trained as a binary classifier.
- A **sliding-window scanner** to run the CNN across the screen.
- A **Tkinter transparent overlay** to visualize detections.
- (Optional) **mouse clicking** on detected targets (disabled by default).

> ⚠️ **Use responsibly.** Automating or augmenting interactions with software may violate terms of service.
> Enable clicking only in contexts where you have permission (e.g., your own software, testing tools, or accessibility use).
> By default, clicking is **OFF**.

---

## Quickstart

1. **Install**
```bash
pip install -r requirements.txt
```

2. **Collect data**
- Gather screenshots from your scene.
- Crop examples of targets into `data/train/player/`.
- Crop background/no-target patches into `data/train/background/`.
- Target crop size used in this repo is **64x128** (you can change it, but keep code consistent).

3. **Train**
```bash
python src/train_cnn.py --epochs 10 --img_w 64 --img_h 128 --out models/player_detector.h5
```

4. **Run overlay (scan + visualize)**
```bash
python src/overlay.py --model models/player_detector.h5 --stride 32 --threshold 0.9
```
Add `--click` to enable clicking on the highest-confidence detection (disabled by default).

---

## Repo Layout

```
player-detector-overlay/
├─ src/
│  ├─ overlay.py                # Screen capture + sliding window + overlay + (optional) clicks
│  ├─ train_cnn.py              # Train a simple CNN classifier on your crops
│  ├─ utils/
│  │  ├─ sliding_window.py      # Sliding window + pyramid utilities
│  └─ dataset_tools/
│     ├─ screenshot_capture.py  # Quick tool to capture full-screen PNGs for dataset creation
├─ data/
│  └─ train/
│     ├─ player/                # put positive crops here
│     └─ background/            # put negative/background crops here
├─ models/                       # will be created by scripts
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## Notes

- The model is intentionally tiny for fast scanning. For better accuracy:
  - Add **data augmentation**.
  - Increase model capacity.
  - Use a **region proposal** pre-filter (e.g., motion or color thresholding) to reduce windows.
- On HiDPI or ultra-wide displays, consider scanning only a **region of interest** or lowering resolution.

