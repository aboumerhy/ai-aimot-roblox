
import argparse
import time
import tkinter as tk
import numpy as np
import pyautogui
import mss
import cv2
import tensorflow as tf
from utils.sliding_window import sliding_window

# Disable fail-safe to allow programmatic mouse move (use with caution)
pyautogui.FAILSAFE = True  # keep True so you can slam mouse to corner to stop

def load_model(path):
    print(f"Loading model from {path} ...")
    return tf.keras.models.load_model(path)

def capture_screen(sct, width, height):
    monitor = {"top": 0, "left": 0, "width": width, "height": height}
    img = np.array(sct.grab(monitor))
    # mss returns BGRA
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def preprocess_patch(patch, img_w, img_h):
    patch = cv2.resize(patch, (img_w, img_h), interpolation=cv2.INTER_AREA)
    patch = patch.astype(np.float32) / 255.0
    return np.expand_dims(patch, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to .h5 model')
    parser.add_argument('--img_w', type=int, default=64)
    parser.add_argument('--img_h', type=int, default=128)
    parser.add_argument('--stride', type=int, default=32, help='sliding window stride in pixels')
    parser.add_argument('--threshold', type=float, default=0.9, help='confidence threshold')
    parser.add_argument('--click', action='store_true', help='enable clicking the best detection')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('left','top','width','height'), help='limit scanning region')
    parser.add_argument('--max_windows', type=int, default=4000, help='cap windows per frame to limit CPU')
    args = parser.parse_args()

    model = load_model(args.model)

    # Tk overlay
    root = tk.Tk()
    root.title("Detection Overlay")
    root.attributes("-transparentcolor", "black")
    root.attributes("-topmost", True)
    root.overrideredirect(True)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg="black", highlightthickness=0)
    canvas.pack()

    sct = mss.mss()

    # ROI setup
    if args.roi:
        left, top, roi_w, roi_h = args.roi
    else:
        left, top, roi_w, roi_h = 0, 0, screen_w, screen_h

    window_size = (args.img_w, args.img_h)
    last_fps_time = time.time()
    frames = 0

    best_box = None
    best_score = 0.0

    def loop():
        nonlocal frames, last_fps_time, best_box, best_score

        frame_bgr = capture_screen(sct, screen_w, screen_h)
        roi = frame_bgr[top:top+roi_h, left:left+roi_w]

        canvas.delete("all")
        best_box, best_score = None, 0.0

        # Sliding window over ROI
        count = 0
        for (x, y, patch) in sliding_window(roi, stepSize=args.stride, windowSize=(window_size[0], window_size[1])):
            # safety cap
            count += 1
            if count > args.max_windows:
                break

            inp = preprocess_patch(patch, args.img_w, args.img_h)
            conf = float(model.predict(inp, verbose=0)[0][0])

            if conf >= args.threshold and conf > best_score:
                best_score = conf
                # Map back to full-screen coords
                x1 = left + x
                y1 = top + y
                x2 = x1 + window_size[0]
                y2 = y1 + window_size[1]
                best_box = (x1, y1, x2, y2)

        # Draw best detection (or all, adjust as preferred)
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3)
            canvas.create_text(x1+4, y1-10, anchor='w', text=f"{best_score:.2f}", fill="red")

            if args.click:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                try:
                    pyautogui.moveTo(cx, cy, duration=0.0)
                    pyautogui.click()
                except pyautogui.FailSafeException:
                    print("PyAutoGUI fail-safe triggered; move mouse to corner to abort.")

        # FPS meter
        frames += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frames / (now - last_fps_time)
            frames = 0
            last_fps_time = now
            canvas.create_text(20, 20, text=f"FPS: {fps:.1f}", anchor='w', fill="white")

        # Schedule next frame
        root.after(1, loop)

    loop()
    root.mainloop()

if __name__ == "__main__":
    main()
