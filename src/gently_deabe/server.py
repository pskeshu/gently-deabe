"""FastAPI service for on-demand DeAbe inference.

Designed for `gently` (the diSPIM annotator) to call: run as a long-lived
process with one or more models loaded into GPU memory; gently POSTs an
input volume path and gets back a denoised volume path.

Endpoints
---------
GET  /healthz                 — liveness check
GET  /models                  — list registered + currently-loaded models
POST /models/{name}/load      — preload a model (warms the cache)
POST /models/{name}/evict     — drop a model to free GPU memory
POST /infer                   — single-step inference (one model)
POST /pipeline                — multi-step pipeline (DeAbe -> Decon [-> Expan])

Inference is serialised by an asyncio lock — only one volume on the GPU at
a time. For multi-volume jobs, queue them client-side and submit serially.
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import time
from typing import Optional

import numpy as np
import tifffile
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import apply, normalize
from .pipeline import Pipeline, PipelineStep, to_uint16
from .registry import ModelRegistry

log = logging.getLogger("gently_deabe.server")


def create_app(registry_path: pathlib.Path | str, device: str = "cuda") -> FastAPI:
    app = FastAPI(
        title="gently-deabe",
        description="On-demand 3D-RCAN deep-learning inference for diSPIM volumes.",
        version="0.1.0",
    )
    app.state.registry = ModelRegistry.from_yaml(registry_path, device=device)
    app.state.gpu_lock = asyncio.Lock()
    app.state.device = device

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; pass device='cpu' to run on CPU")
        torch.backends.cudnn.benchmark = True

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    @app.get("/models")
    async def list_models():
        registry: ModelRegistry = app.state.registry
        return {
            "registered": [
                {
                    "name": e.name,
                    "kind": e.kind,
                    "description": e.description,
                    "path": str(e.path),
                }
                for e in (registry.get_entry(n) for n in registry.names())
            ],
            "loaded": registry.loaded(),
        }

    @app.post("/models/{name}/load")
    async def load_model(name: str):
        async with app.state.gpu_lock:
            t0 = time.time()
            try:
                _ = app.state.registry.get(name)
            except KeyError:
                raise HTTPException(404, f"unknown model {name!r}")
            return {"loaded": name, "elapsed_seconds": round(time.time() - t0, 2)}

    @app.post("/models/{name}/evict")
    async def evict_model(name: str):
        evicted = app.state.registry.evict(name)
        return {"evicted": evicted}

    @app.post("/infer", response_model=InferResponse)
    async def infer(req: InferRequest):
        return await _do_infer(app, req)

    @app.post("/pipeline", response_model=InferResponse)
    async def pipeline(req: PipelineRequest):
        return await _do_pipeline(app, req)

    return app


# ---- request / response models ---------------------------------------------

class InferRequest(BaseModel):
    input_path: str = Field(..., description="Path (server-local or UNC) to the input TIFF")
    output_path: str = Field(..., description="Where to write the resulting TIFF")
    model: str = Field(..., description="Registered model name to apply")
    block_shape: tuple[int, int, int] = (32, 128, 128)
    overlap_shape: Optional[tuple[int, int, int]] = None
    p_min: float = 2.0
    p_max: float = 99.9
    scale_value: float = Field(2000.0, description="Multiplier on [0,1] prediction before uint16 clipping")
    bit_depth: int = Field(16, description="8, 16, or 32")
    interp_ratio: Optional[tuple[float, float, float]] = Field(
        None,
        description="Optional zoom factor applied before inference (e.g. (6.154, 1, 1) for Z-isotropic)",
    )


class StepSpec(BaseModel):
    model: str
    block_shape: tuple[int, int, int] = (32, 128, 128)
    overlap_shape: Optional[tuple[int, int, int]] = None
    interp_ratio: Optional[tuple[float, float, float]] = None


class PipelineRequest(BaseModel):
    input_path: str
    output_path: str
    steps: list[StepSpec]
    p_min: float = 2.0
    p_max: float = 99.9
    scale_value: float = 2000.0
    bit_depth: int = 16


class InferResponse(BaseModel):
    output_path: str
    output_shape: list[int]
    elapsed_seconds: float
    model_load_seconds: float = 0.0


# ---- internals --------------------------------------------------------------

def _read_tif(path: pathlib.Path) -> np.ndarray:
    if not path.exists():
        raise HTTPException(404, f"input not found: {path}")
    arr = tifffile.imread(str(path))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise HTTPException(400, f"expected 3D input, got shape {arr.shape}")
    return arr.astype(np.float32)


def _quantize(pred: np.ndarray, scale: float, bit_depth: int) -> np.ndarray:
    if bit_depth == 8:
        return np.clip(scale * pred, 0, 255).astype(np.uint8)
    if bit_depth == 16:
        return np.clip(scale * pred, 0, 65535).astype(np.uint16)
    if bit_depth == 32:
        return (scale * pred).astype(np.float32)
    raise HTTPException(400, f"bit_depth must be 8, 16, or 32; got {bit_depth}")


async def _do_infer(app: FastAPI, req: InferRequest) -> "InferResponse":
    in_path = pathlib.Path(req.input_path)
    out_path = pathlib.Path(req.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _read_tif(in_path)

    async with app.state.gpu_lock:
        t0 = time.time()
        try:
            model = app.state.registry.get(req.model)
        except KeyError:
            raise HTTPException(404, f"unknown model {req.model!r}")
        load_seconds = time.time() - t0

        from scipy.ndimage import zoom
        x = normalize(raw, req.p_min, req.p_max)
        if req.interp_ratio is not None:
            x = zoom(x, req.interp_ratio, order=1)
            x = normalize(x, req.p_min, req.p_max)

        t1 = time.time()
        pred = apply(model, x, req.block_shape, req.overlap_shape, device=app.state.device)
        np.clip(pred, 0, None, out=pred)
        infer_seconds = time.time() - t1

    out = _quantize(pred, req.scale_value, req.bit_depth)
    tifffile.imwrite(str(out_path), out, imagej=False)

    return InferResponse(
        output_path=str(out_path),
        output_shape=list(out.shape),
        elapsed_seconds=round(infer_seconds, 2),
        model_load_seconds=round(load_seconds, 2),
    )


async def _do_pipeline(app: FastAPI, req: PipelineRequest) -> "InferResponse":
    in_path = pathlib.Path(req.input_path)
    out_path = pathlib.Path(req.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = _read_tif(in_path)

    async with app.state.gpu_lock:
        load_seconds = 0.0
        steps: list[PipelineStep] = []
        for s in req.steps:
            t0 = time.time()
            try:
                _ = app.state.registry.get(s.model)
            except KeyError:
                raise HTTPException(404, f"unknown model {s.model!r}")
            load_seconds += time.time() - t0
            steps.append(
                PipelineStep(
                    name=s.model,
                    model=s.model,
                    block_shape=tuple(s.block_shape),
                    overlap_shape=tuple(s.overlap_shape) if s.overlap_shape else None,
                    interp_ratio=tuple(s.interp_ratio) if s.interp_ratio else None,
                )
            )

        pipeline = Pipeline(app.state.registry, device=app.state.device)
        t1 = time.time()
        pred = pipeline.run(raw, steps)
        infer_seconds = time.time() - t1

    out = _quantize(pred, req.scale_value, req.bit_depth)
    tifffile.imwrite(str(out_path), out, imagej=False)

    return InferResponse(
        output_path=str(out_path),
        output_shape=list(out.shape),
        elapsed_seconds=round(infer_seconds, 2),
        model_load_seconds=round(load_seconds, 2),
    )
