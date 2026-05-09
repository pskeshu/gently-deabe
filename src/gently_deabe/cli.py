"""CLI entry points: `gently-deabe` (one-shot inference) and
`gently-deabe-server` (long-running FastAPI service)."""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import tifffile

from .convert import convert
from .inference import apply, normalize
from .registry import ModelRegistry


def _add_inference_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", required=True,
                   help="path to .hdf5, dir of .hdf5, or .pt state_dict")
    p.add_argument("-i", "--input", required=True, type=pathlib.Path)
    p.add_argument("-o", "--output", required=True, type=pathlib.Path)
    p.add_argument("-B", "--block-shape", default="32,128,128",
                   help="Z,Y,X tile size (default 32,128,128)")
    p.add_argument("-b", "--bpp", type=int, choices=[8, 16, 32], default=16)
    p.add_argument("--scale-value", type=float, default=2000.0)
    p.add_argument("--p-min", type=float, default=2.0)
    p.add_argument("--p-max", type=float, default=99.9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--single-channel", action="store_true",
                   help="write (Z,Y,X) prediction only; default writes (Z,2,Y,X) [raw, pred]")


def _parse_tuple(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _resolve_model(path: pathlib.Path):
    """Resolve a model arg (file/dir/.pt) to a loaded RCAN."""
    if path.suffix == ".pt":
        from .model import RCAN
        import torch
        m = RCAN()
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        m.load_state_dict(state)
        return m

    if path.is_dir():
        candidates = list(path.glob("*.hdf5"))
        if not candidates:
            raise FileNotFoundError(f"no .hdf5 in {path}")

        def _val_loss(p: pathlib.Path) -> float:
            try:
                return float(p.stem.split("_")[-1])
            except (ValueError, IndexError):
                return float("inf")
        path = min(candidates, key=_val_loss)
        print(f"selected best checkpoint: {path.name}")

    model, _ = convert(path)
    return model


def _cmd_apply(args: argparse.Namespace) -> int:
    import torch
    block_shape = _parse_tuple(args.block_shape)

    print(f"loading model from {args.model}")
    model = _resolve_model(pathlib.Path(args.model))
    model.to(args.device).eval()
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"loading input: {args.input}")
    raw = tifffile.imread(str(args.input))
    if raw.ndim != 3:
        raise SystemExit(f"expected 3D input, got shape {raw.shape}")
    raw_norm = normalize(raw, args.p_min, args.p_max)

    print(f"applying (block={block_shape})")
    t0 = time.time()
    pred = apply(model, raw_norm, block_shape, device=args.device, verbose=True)
    print(f"inference: {time.time() - t0:.1f}s")
    np.clip(pred, 0, None, out=pred)

    if args.single_channel:
        result = pred
    else:
        result = np.stack([raw_norm, pred], axis=1)

    if args.bpp == 8:
        result = np.clip(255 * result, 0, 255).astype(np.uint8)
    elif args.bpp == 16:
        result = np.clip(args.scale_value * result, 0, 65535).astype(np.uint16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(args.output), result, imagej=not args.single_channel)
    print(f"wrote {args.output}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    import torch
    model, state = convert(args.hdf5)
    torch.save(state, str(args.out))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"saved {args.out} ({n_params:,} params)")
    return 0


def main() -> int:
    """Entry point for the `gently-deabe` console script."""
    p = argparse.ArgumentParser(
        prog="gently-deabe",
        description="DeAbe / 3D-RCAN PyTorch inference and weight conversion utility.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_apply = sub.add_parser("apply", help="run single-step inference on one TIFF")
    _add_inference_args(p_apply)
    p_apply.set_defaults(func=_cmd_apply)

    p_conv = sub.add_parser("convert", help="convert Keras .hdf5 to PyTorch state_dict")
    p_conv.add_argument("hdf5", type=pathlib.Path)
    p_conv.add_argument("out", type=pathlib.Path)
    p_conv.set_defaults(func=_cmd_convert)

    args = p.parse_args()
    return args.func(args)


def server_main() -> int:
    """Entry point for the `gently-deabe-server` console script."""
    import uvicorn
    from .server import create_app

    p = argparse.ArgumentParser(
        prog="gently-deabe-server",
        description="Run the gently-deabe FastAPI inference service.",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8091)
    p.add_argument("--registry", required=True, type=pathlib.Path,
                   help="YAML file mapping logical model names to .hdf5 paths")
    p.add_argument("--device", default="cuda")
    p.add_argument("--preload", action="append", default=[],
                   help="model name to warm-load at startup (repeatable)")
    args = p.parse_args()

    app = create_app(args.registry, device=args.device)

    # Synchronously preload the requested models so the first user request is fast.
    if args.preload:
        registry: ModelRegistry = app.state.registry
        for name in args.preload:
            print(f"preloading {name}...")
            t0 = time.time()
            registry.get(name)
            print(f"  loaded in {time.time() - t0:.1f}s")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
