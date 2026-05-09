# gently-deabe

PyTorch port of the deep-learning aberration / deconvolution pipeline from
**Guo et al., *Nat. Commun.* 16:313 (2025)** — *"Deep learning-based aberration
compensation improves contrast and resolution in fluorescence microscopy."*

The original [3D-RCAN](https://github.com/AiviaCommunity/3D-RCAN) and
[DeAbePlus](https://github.com/eguomin/DeAbePlus) repos are TensorFlow 1.15 +
CUDA 10.0 / cuDNN 7.6.5. On Ampere-class or newer GPUs (RTX 3090, A5000,
A6000, H100) the cold-start cost on the first volume is several minutes of
PTX-JIT compilation. This package reimplements the model in PyTorch 2 +
CUDA 12 + cuDNN 9, loads the existing `.hdf5` checkpoints unmodified, and
produces output that matches the reference to **0.9947 correlation
(0.9963 on bright/foreground voxels)** with a **~10× steady-state speedup**
and a **~340× cold-start speedup** on RTX A5000.

## Install

```bash
# In a fresh Python 3.10+ env
pip install -e .

# CUDA-enabled PyTorch wheel must match your driver:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Quick start

### 1. Set up a model registry

Copy the example config and fill in your weight paths:

```bash
cp configs/highNA_diSPIM_Nuclear.example.yaml configs/local.yaml
# edit configs/local.yaml so each `path:` points at your weights
```

`configs/local.yaml` is in `.gitignore` — your local paths stay local.

### 2. One-shot inference

```bash
gently-deabe apply \
  -m /path/to/RCAN_Model_SPIMB \
  -i input.tif \
  -o output.tif \
  -B 32,128,128 -b 16
```

`-m` accepts a directory of `weights_<epoch>_<loss>.hdf5` files (best is
auto-picked) or a single `.hdf5`/`.pt` file.

### 3. Long-running HTTP service (the gently use case)

```bash
gently-deabe-server \
  --registry configs/local.yaml \
  --port 8091 \
  --preload step1_spimb --preload step2_decon
```

Then from a client:

```python
import requests

resp = requests.post("http://127.0.0.1:8091/pipeline", json={
    "input_path":  "/path/to/input.tif",
    "output_path": "/path/to/output_dl.tif",
    "steps": [
        {"model": "step1_spimb", "block_shape": [32, 128, 128]},
        {"model": "step2_decon", "block_shape": [128, 128, 128],
         "interp_ratio": [6.154, 1.0, 1.0]},
    ],
    "scale_value": 2000.0,
}).json()

print(resp)
# {'output_path': '...', 'output_shape': [314, 275, 425],
#  'elapsed_seconds': 60.4, 'model_load_seconds': 0.0}
```

### 4. Library use

```python
from gently_deabe import ModelRegistry, Pipeline, PipelineStep
import tifffile

reg = ModelRegistry.from_yaml("configs/local.yaml", device="cuda")
pipeline = Pipeline(reg)

raw = tifffile.imread("input.tif")
pred = pipeline.run(raw, [
    PipelineStep(name="DeAbe", model="step1_spimb", block_shape=(32, 128, 128)),
    PipelineStep(name="Decon", model="step2_decon", block_shape=(128, 128, 128),
                 interp_ratio=(6.154, 1.0, 1.0)),
])
```

## API reference

| HTTP endpoint | Purpose |
|---|---|
| `GET  /healthz` | Liveness + GPU info |
| `GET  /models` | List registered + currently-loaded models |
| `POST /models/{name}/load` | Warm-load a model (returns load time) |
| `POST /models/{name}/evict` | Drop a loaded model from VRAM |
| `POST /infer` | Single-step inference with one model |
| `POST /pipeline` | Multi-step pipeline (DeAbe → Decon → Expan) |

Inference is serialised by an asyncio lock — only one volume is on the GPU
at a time. For multi-volume jobs, queue them client-side and submit
sequentially. There is no built-in async job queue.

## Scope

This package replicates **only the deep-learning portion** of the upstream
pipeline. Cropping, background subtraction, and 3D registration (typically
done with the Janelia MIPAV plugin) and any post-DL registration remain in
their existing tooling. Pass an already-cropped, optionally already-registered
volume as input.

## License

MIT.
