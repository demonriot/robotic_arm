
import cv2
import time
import threading
from pathlib import Path

LEFT_URL = "http://192.168.0.206/stream"
RIGHT_URL = "http://192.168.0.205/stream"


class CameraReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {url}")

        self.frame = None
        self.timestamp = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(
            target=self._read_loop, daemon=True
        )
        self.thread.start()

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            with self.lock:
                self.frame = frame
                self.timestamp = time.monotonic()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None, None
            return self.frame.copy(), self.timestamp

    def stop(self):
        self.running = False
        self.cap.release()
        self.thread.join(timeout=2)


def main():
    left = CameraReader(LEFT_URL)
    right = CameraReader(RIGHT_URL)

    output_dir = Path("outputs/stereo_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            frame_l, time_l = left.get_frame()
            frame_r, time_r = right.get_frame()

            if frame_l is None or frame_r is None:
                time.sleep(0.05)
                continue

            frame_r = cv2.resize(
                frame_r,
                (frame_l.shape[1], frame_l.shape[0])
            )

            # Difference between frame reception timestamps.
            # This is NOT hardware exposure synchronization.
            delta_ms = abs(time_l - time_r) * 1000

            combined = cv2.hconcat([frame_l, frame_r])

            cv2.putText(
                combined,
                f"Reception delta: {delta_ms:.0f} ms",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            cv2.imshow("Stereo ESP32-CAM", combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                cv2.imwrite(
                    str(output_dir / "left.jpg"), frame_l
                )
                cv2.imwrite(
                    str(output_dir / "right.jpg"), frame_r
                )
                print("Stereo pair saved!")

            time.sleep(0.02)

    finally:
        left.stop()
        right.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
