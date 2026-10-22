#!/usr/bin/env bash

set -u

section() {
    printf '\n===== %s =====\n' "$1"
}

show_version() {
    label="$1"
    shift
    if command -v "$1" >/dev/null 2>&1; then
        section "$label"
        "$@" 2>&1 || true
    else
        section "$label"
        printf '%s: not found\n' "$1"
    fi
}

section "Timestamp"
date -u '+%Y-%m-%dT%H:%M:%SZ'

section "Operating system"
if [ -r /etc/os-release ]; then
    cat /etc/os-release
else
    printf '/etc/os-release: not found\n'
fi
uname -a

section "CPU and memory"
if command -v lscpu >/dev/null 2>&1; then
    lscpu 2>&1 || true
else
    printf 'lscpu: not found\n'
fi
if command -v free >/dev/null 2>&1; then
    free -h 2>&1 || true
else
    printf 'free: not found\n'
fi

show_version "Git" git --version

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    PYTHON_BIN=
fi

if [ -n "$PYTHON_BIN" ]; then
    section "Python"
    "$PYTHON_BIN" --version 2>&1 || true
    "$PYTHON_BIN" -m pip --version 2>&1 || true
else
    section "Python"
    printf 'python3/python: not found\n'
fi

show_version "hipcc" hipcc --version
show_version "hipconfig" hipconfig --full
show_version "rocminfo" rocminfo
show_version "rocm-smi" rocm-smi
show_version "clinfo" clinfo

if [ -n "$PYTHON_BIN" ]; then
    section "Python packages and devices"
    "$PYTHON_BIN" - <<'PY' 2>&1 || true
try:
    import torch

    print(f"torch: {torch.__version__}")
    print(f"torch.version.hip: {getattr(torch.version, 'hip', None)}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(f"device[{index}]: {torch.cuda.get_device_name(index)}")
            print(f"device[{index}] properties: {properties}")
except Exception as error:
    print(f"torch check failed: {type(error).__name__}: {error}")

try:
    import triton

    print(f"triton: {getattr(triton, '__version__', 'unknown')}")
except Exception as error:
    print(f"triton check failed: {type(error).__name__}: {error}")
PY
fi

section "Relevant environment variables"
for name in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES \
    ROCM_PATH HIP_PATH DEEPINFER_ACCELERATOR TILELANG_DEFAULT_TARGET; do
    value="$(printenv "$name" 2>/dev/null || true)"
    if [ -n "$value" ]; then
        printf '%s=%s\n' "$name" "$value"
    else
        printf '%s=<unset>\n' "$name"
    fi
done

section "Snapshot complete"
