"""Exercise Chm02 solver ordinals with the devices visible to the CUDA driver."""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def visible_devices():
    if os.name == "nt":
        driver = ctypes.WinDLL("nvcuda.dll")
    else:
        driver = ctypes.CDLL("libcuda.so.1")
    # CUDA_ERROR_NO_DEVICE is a valid result on GPU-less build hosts.
    status = driver.cuInit(0)
    if status == 100:
        return []
    if status:
        raise RuntimeError(f"cuInit failed with CUDA status {status}")
    count = ctypes.c_int()
    status = driver.cuDeviceGetCount(ctypes.byref(count))
    if status:
        raise RuntimeError(f"cuDeviceGetCount failed with CUDA status {status}")
    devices = []
    for ordinal in range(count.value):
        handle = ctypes.c_int()
        status = driver.cuDeviceGet(ctypes.byref(handle), ordinal)
        if status:
            raise RuntimeError(f"cuDeviceGet({ordinal}) failed with CUDA status {status}")
        devices.append(handle.value)
    return devices


def run(exe, source, output, ordinals, expected_devices=(), file_io=False, expected_status=0):
    output.mkdir()
    keys = output / source.name
    shutil.copyfile(source, keys)
    args = [exe, keys, output, "Chm02", "Mulshrolate3RX", "And", "2",
            "--CuConcurrency=2", "--DisableCsvOutputFile",
            "--Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847"]
    if ordinals is not None:
        args.append(f"--CuDevices={ordinals}")
    if not file_io:
        args.append("--NoFileIo")
    proc = subprocess.run(args, capture_output=True, timeout=45,
                          env={**os.environ, "PH_DEBUG_CUDA_CHM02": "1"})
    if os.name != "nt":
        expected_status &= 0xff
    if proc.returncode != expected_status or (
        expected_status == 0 and b"PH_CHM02_CUDA_VERIFY_OK" not in proc.stderr
    ):
        raise AssertionError(
            f"devices={ordinals}, file_io={file_io}, exit={proc.returncode:#x}\n"
            + proc.stderr.decode(errors="replace")
        )
    for index, device in enumerate(expected_devices):
        marker = f"[Chm02] Solver {index} CUDA device {device}".encode()
        if marker not in proc.stderr.splitlines():
            raise AssertionError(f"Missing device assignment {marker!r}: {proc.stderr!r}")
    if file_io:
        table, = output.rglob("*.pht1")
        if table.stat().st_size == 0:
            raise AssertionError(f"Empty saved table: {table}")


def main():
    exe, keys = map(Path, sys.argv[1:])
    devices = visible_devices()
    if not devices:
        print("No visible CUDA devices")
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(exe, keys, root / "default", None, [devices[0]] * 2)
        run(exe, keys, root / "short", "0", [devices[0]] * 2)
        # PH_E_INVALID_CU_DEVICES on an unavailable ordinal.
        run(exe, keys, root / "invalid", str(len(devices)), expected_status=0xe00403a4)
        if len(devices) >= 2:
            run(exe, keys, root / "forward", "0,1", devices[:2])
            run(exe, keys, root / "reverse", "1,0", [devices[1], devices[0]], file_io=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
