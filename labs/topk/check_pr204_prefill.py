#!/usr/bin/env python3
"""Check PR #204 BF16 indexer-prefill TopK contract on a ROCm device."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROFILES = {
    "smoke": {"rows": 8, "width": 4096, "topk": 512},
    "prefill": {"rows": 64, "width": 65536, "topk": 512},
}


def make_end(rows: int, width: int, topk: int) -> torch.Tensor:
    base = [0, 1, max(1, topk // 3), topk, min(width, topk + 1), width]
    values = [base[index % len(base)] for index in range(rows)]
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def check_contract(
    x: torch.Tensor,
    end: torch.Tensor,
    offset: torch.Tensor | None,
    output: torch.Tensor,
    values: torch.Tensor | None,
    topk: int,
) -> None:
    rows, width = x.shape
    assert values is None, "return_value=False must return None"
    assert output.shape == (rows, topk)
    assert output.dtype == torch.int32

    valid_len = end.long().clamp(min=0, max=width)
    valid_count = torch.minimum(valid_len, torch.full_like(valid_len, topk))
    positions = torch.arange(topk, device=x.device)
    valid_mask = positions[None, :] < valid_count[:, None]
    fill_mask = ~valid_mask

    assert torch.all(output[fill_mask] == -1), "invalid output positions must use the fill value"

    selected = output.long()
    if offset is not None:
        selected = selected - offset.long()[:, None]

    in_range = (selected >= 0) & (selected < valid_len[:, None])
    assert torch.all(in_range | ~valid_mask), "selected index is outside the valid row"

    safe = selected.masked_fill(~valid_mask, 0)
    sorted_selected = safe.masked_fill(~valid_mask, width).sort(dim=1).values
    duplicate = sorted_selected[:, 1:] == sorted_selected[:, :-1]
    duplicate &= valid_mask[:, 1:]
    assert not torch.any(duplicate), "selected indices contain duplicates"

    ordered = output[:, 1:] >= output[:, :-1]
    assert torch.all(ordered | ~valid_mask[:, 1:]), "sorted_index=True outputs must be ascending"

    picked = x.gather(1, safe).float()
    selected_min = picked.masked_fill(~valid_mask, float("inf")).amin(dim=1)

    remainder = x.float().clone()
    remainder.scatter_(1, safe, float("-inf"))
    wide_positions = torch.arange(width, device=x.device)
    remainder.masked_fill_(wide_positions[None, :] >= valid_len[:, None], float("-inf"))
    unselected_max = remainder.amax(dim=1)
    assert torch.all(selected_min >= unselected_max), "top-k threshold condition failed"


def invoke(
    deep_select: object,
    x: torch.Tensor,
    end: torch.Tensor,
    offset: torch.Tensor | None,
    output: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    return deep_select.topk(  # type: ignore[attr-defined]
        x,
        topk,
        sorted=False,
        end=end,
        indices_type=torch.int32,
        sorted_index=True,
        output_idx=output,
        output_idx_offset=offset,
        idx_oob_fill_value=-1,
        return_value=False,
    )


def run_case(deep_select: object, rows: int, width: int, topk: int, with_offset: bool) -> None:
    torch.manual_seed(0)
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    end = make_end(rows, width, topk)
    offset = torch.arange(1000, 1000 + rows, dtype=torch.int32, device="cuda") if with_offset else None
    output = torch.empty((rows, topk), dtype=torch.int32, device="cuda")

    input_alignment, output_alignment = deep_select.get_stride_requirement()  # type: ignore[attr-defined]
    assert x.stride(0) * x.element_size() % input_alignment == 0, "input row stride is misaligned"
    assert output.stride(0) * output.element_size() % output_alignment == 0, "output row stride is misaligned"

    values, returned = invoke(deep_select, x, end, offset, output, topk)
    assert returned.data_ptr() == output.data_ptr(), "output_idx must be reused"
    check_contract(x, end, offset, returned, values, topk)
    torch.cuda.synchronize()
    print(f"PASS correctness rows={rows} width={width} topk={topk} offset={with_offset}")


def run_graph_case(deep_select: object, rows: int, width: int, topk: int) -> None:
    torch.manual_seed(1)
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    end = make_end(rows, width, topk)
    offset = torch.arange(2000, 2000 + rows, dtype=torch.int32, device="cuda")
    output = torch.empty((rows, topk), dtype=torch.int32, device="cuda")

    for _ in range(5):
        invoke(deep_select, x, end, offset, output, topk)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(deep_select, x, end, offset, output, topk)

    output.fill_(-12345)
    graph.replay()
    torch.cuda.synchronize()
    check_contract(x, end, offset, output, None, topk)
    print("PASS graph capture and first replay")

    end.copy_(make_end(rows, width, topk).flip(0))
    output.fill_(-12345)
    graph.replay()
    torch.cuda.synchronize()
    check_contract(x, end, offset, output, None, topk)
    print("PASS graph replay after end update")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-path", required=True, help="Path to DeepInfer/vendor/rocm")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    parser.add_argument("--skip-graph", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vendor_path = Path(args.vendor_path).resolve()
    if not vendor_path.is_dir():
        raise SystemExit(f"vendor path does not exist: {vendor_path}")

    sys.path.insert(0, str(vendor_path))
    import deep_select  # noqa: PLC0415

    module_path = Path(deep_select.__file__).resolve()
    if vendor_path not in module_path.parents:
        raise SystemExit(f"deep_select was not imported from the requested vendor path: {module_path}")

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is false")

    profile = PROFILES[args.profile]
    print(f"deep_select={module_path}")
    print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")

    with torch.inference_mode():
        run_case(deep_select, **profile, with_offset=True)
        run_case(deep_select, **profile, with_offset=False)
        if not args.skip_graph:
            run_graph_case(deep_select, **profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

