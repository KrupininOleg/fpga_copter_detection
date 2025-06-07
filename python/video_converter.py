from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def frame_serialize(frame: npt.NDArray[np.uint8]) -> bytes:
    buffer = b""
    # buffer += np.uint16(frame.shape[:2]).tobytes()
    # buffer += np.uint8(frame.shape[2]).tobytes()
    buffer += frame.tobytes()
    return buffer


def video_to_bin_frames(src: Path, dst: Path, frames: slice) -> None:
    cap = cv2.VideoCapture(src)
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num < frames.start:
            continue
        if frame_num == frames.stop:
            break

        frame_buf = frame_serialize(frame[:, :, ::-1])
        with open(dst, "ab") as f:
            print(frame_num, frame.shape)
            f.write(frame_buf)

    cap.release()


def bin_frames_to_video(src: Path) -> None:
    frames = []
    with open(src, "rb") as f:
        while True:
            # w, h = map(int, np.frombuffer(f.read(2 * 2), dtype=np.uint16))
            # ch = int(np.frombuffer(f.read(1 * 1), dtype=np.uint8)[0])
            w, h, ch = (480, 848, 3)
            size = w * h * ch
            values = np.frombuffer(f.read(size * 1), dtype=np.uint8)
            print(values[:100])
            frame = values.reshape(w, h, ch)
            # print(frame[:h:10, 0])
            frames.append(frame)
            # break
            plt.imshow(frame)
            plt.show()


if __name__ == "__main__":
    data_path = Path(r"C:\Users\Public\OwnPrograms\stereo_vision\fpga\fpga_copter_detection\sim\data")
    src_path = data_path / "running.mp4"
    dst_path = data_path / "running_0_2.bin"
    # dst_path.unlink(missing_ok=True)
    # video_to_bin_frames(src_path, dst_path, slice(1, 4))
    bin_frames_to_video(dst_path)
