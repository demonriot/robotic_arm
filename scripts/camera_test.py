
import cv2
import time

# Replace with your ESP32-CAM's current IP address
CAMERA_URL = "http://10.87.132.203/stream"

def main():
    print(f"Connecting to {CAMERA_URL}")

    cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("ERROR: Could not connect to ESP32-CAM")
        return

    print("Camera connected!")

    frames = 0
    fps = 0.0
    start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Failed to receive frame")
                break

            frames += 1
            elapsed = time.perf_counter() - start

            if elapsed >= 2.0:
                fps = frames / elapsed
                frames = 0
                start = time.perf_counter()
                print(f"FPS: {fps:.2f}")

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.imshow("ESP32-CAM | GC2145", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                filename = f"capture_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera disconnected.")

if __name__ == "__main__":
    main()
