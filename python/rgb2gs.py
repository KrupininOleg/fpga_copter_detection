import numpy as np
from pathlib import Path


def rgb2gs(r: int, g: int, b: int) -> int:
    return ((r + b) // 2 + g) // 2


if __name__ == "__main__":
    data_path = Path(r"C:\Users\Public\OwnPrograms\stereo_vision\fpga\fpga_copter_detection\sim\data")
    src_path = data_path / "running_0_2.bin"

    n_points = 300
    with open(src_path, "rb") as f:
        for i in range(n_points):
            rgb = np.frombuffer(f.read(3), dtype=np.uint8).astype(np.int32)
            gs = rgb2gs(*rgb)
            print(f"{(i + 1) * 24}: {rgb.tolist()} -> {gs}")
