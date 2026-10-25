
import cv2
import numpy as np
import requests
import time
from pathlib import Path

LEFT_IP = "10.87.132.203"  # Update if changed
RIGHT_IP = "10.87.132.205"

OUTPUT_DIR = Path("outputs/stereo_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def capture(ip):
    url = f"http://{ip}/capture"

    start = time.perf_counter()
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    image = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise RuntimeError(f"Failed to decode image from {ip}")

    elapsed = time.perf_counter() - start
    return image, elapsed


def main():
    for i in range(10):
        try:
            left, left_time = capture(LEFT_IP)
            right, right_time = capture(RIGHT_IP)

            if left.shape != right.shape:
                raise RuntimeError("Camera resolutions differ")

            cv2.imwrite(str(OUTPUT_DIR / f"left_{i:02d}.jpg"), left)
            cv2.imwrite(str(OUTPUT_DIR / f"right_{i:02d}.jpg"), right)

            print(
                f"Pair {i + 1}: "
                f"Left={left_time:.2f}s, "
                f"Right={right_time:.2f}s"
            )

            time.sleep(1)

        except Exception as e:
            print(f"Capture failed: {e}")

    print("Test complete.")


if __name__ == "__main__":
    main()
