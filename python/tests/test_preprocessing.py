import numpy as np
import numpy.typing as npt
import numpy.testing as nptest
import math


def frame_rgb2gs(frame_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    frame_rgb = frame_rgb.astype(np.uint32)
    frame_gs = ((frame_rgb[:, 0] + frame_rgb[:, 2]) // 2 + frame_rgb[:, 1]) // 2
    return frame_gs.astype(np.uint8)


def hex_to_uint8(hex_values: bytes) -> npt.NDArray[np.uint8]:
    return np.asarray(
        [int(hex_values[i:i + 2], 16) for i in range(0, len(hex_values), 2)],
        dtype=np.uint8
    )


def test_frame_rgb2gs() -> None:
    src_path = r"C:/Users/Public/OwnPrograms/stereo_vision/fpga/fpga_copter_detection/sim/data/running_0_2.bin"
    eval_path = r"C:/Users/Public/OwnPrograms/stereo_vision/fpga/fpga_copter_detection/sim/data/results/gs.bin"

    frame_shape = (480, 848, 3)
    with open(src_path, "rb") as f:
        frame_bin = f.read(math.prod(frame_shape))
    frame_values = np.frombuffer(frame_bin, dtype=np.uint8)
    frame_rgb = frame_values.reshape((-1, frame_shape[2]))
    frame_gs_exp = frame_rgb2gs(frame_rgb)

    with open(eval_path, "rb") as f:
        frame_bin = f.read(math.prod(frame_shape))
    frame_gs_eval = hex_to_uint8(frame_bin)
    nptest.assert_equal(frame_gs_exp[:len(frame_gs_eval)], frame_gs_eval)

