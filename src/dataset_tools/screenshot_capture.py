
import os
import time
import argparse
import mss
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/raw_screens', help='folder to save PNGs')
    parser.add_argument('--interval', type=float, default=1.0, help='seconds between shots')
    parser.add_argument('--prefix', default='shot')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    sct = mss.mss()
    monitor = sct.monitors[0]

    i = 0
    try:
        while True:
            img = sct.grab(monitor)
            im = Image.frombytes("RGB", img.size, img.rgb)
            path = os.path.join(args.out, f"{args.prefix}_{i:05d}.png")
            im.save(path)
            print("Saved", path)
            i += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    main()
