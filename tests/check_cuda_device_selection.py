"""Exercise Chm02 solver ordinals with the devices visible to the CUDA driver."""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def device_count():
    if os.name == "nt":
        driver = ctypes.WinDLL("nvcuda.dll")
    else:
        driver = ctypes.CDLL("libcuda.so.1")
    # CUDA_ERROR_NO_DEVICE is a valid result on GPU-less build hosts.
    status = driver.cuInit(0)
    if status == 100:
        return 0
    if status:
        raise RuntimeError(f"cuInit failed with CUDA status {status}")
    count = ctypes.c_int()
    status = driver.cuDeviceGetCount(ctypes.byref(count))
    if status:
        raise RuntimeError(f"cuDeviceGetCount failed with CUDA status {status}")
    return count.value


def run(exe, source, output, devices, file_io=False, expected_status=0):
    output.mkdir()
    keys = output / source.name
    shutil.copyfile(source, keys)
    args = [exe, keys, output, "Chm02", "Mulshrolate3RX", "And", "2",
            "--CuConcurrency=2", "--DisableCsvOutputFile",
            "--Seeds=0xF0192B55,0xD9C83970,0x0C1E0D10,0xD11A5847"]
    if devices is not None:
        args.append(f"--CuDevices={devices}")
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
            f"devices={devices}, file_io={file_io}, exit={proc.returncode:#x}\n"
            + proc.stderr.decode(errors="replace")
        )
    if file_io:
        table, = output.rglob("*.pht1")
        if table.stat().st_size == 0:
            raise AssertionError(f"Empty saved table: {table}")


def main():
    exe, keys = map(Path, sys.argv[1:])
    count = device_count()
    if not count:
        print("No visible CUDA devices")
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(exe, keys, root / "default", None)
        run(exe, keys, root / "short", "0")
        # PH_E_INVALID_CU_DEVICES on an unavailable ordinal.
        run(exe, keys, root / "invalid", str(count), expected_status=0xe00403a4)
        if count >= 2:
            run(exe, keys, root / "forward", "0,1")
            run(exe, keys, root / "reverse", "1,0", file_io=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
