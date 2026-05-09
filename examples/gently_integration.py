"""Sketch of how `gently-annotator` calls the gently-deabe HTTP service to
denoise a volume on demand.

Drop something like this into `gently/annotator/routes/...` to expose a
"denoise this timepoint" action in the annotator UI.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import httpx


class DeabeClient:
    """Thin wrapper over the gently-deabe FastAPI service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8091", timeout_s: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def healthz(self) -> dict:
        return self._client.get("/healthz").raise_for_status().json()

    def loaded_models(self) -> list[str]:
        return self._client.get("/models").raise_for_status().json()["loaded"]

    def warm(self, name: str) -> None:
        self._client.post(f"/models/{name}/load").raise_for_status()

    def denoise(
        self,
        input_path: pathlib.Path,
        output_path: pathlib.Path,
        view: str = "spimb",
        full_pipeline: bool = False,
    ) -> dict:
        """Denoise a single volume.

        Parameters
        ----------
        input_path : path to a 3D TIFF (single view, bg-subtracted, optionally Z-isotropic)
        output_path: path to write the denoised TIFF
        view       : "spima" or "spimb" — determines which Step1 model is used
        full_pipeline : if True, runs DeAbe -> Decon (with Z-interp). If False, DeAbe only.
        """
        steps = [{"model": f"step1_{view}", "block_shape": [32, 128, 128]}]
        if full_pipeline:
            steps.append({
                "model": "step2_decon",
                "block_shape": [128, 128, 128],
                "interp_ratio": [6.154, 1.0, 1.0],
            })

        payload = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "steps": steps,
            "scale_value": 2000.0,
            "bit_depth": 16,
        }
        return self._client.post("/pipeline", json=payload).raise_for_status().json()


# --- example wiring inside a gently route ---------------------------------------

def example_gently_route(volume_path: pathlib.Path, deabe_cache_dir: pathlib.Path) -> pathlib.Path:
    """Return the denoised version of `volume_path`, computing it on demand.

    The denoised output is cached under `deabe_cache_dir/<volume_basename>` so
    repeated requests are free.
    """
    out = deabe_cache_dir / f"{volume_path.stem}_deabe.tif"
    if out.exists():
        return out

    client = DeabeClient()
    client.denoise(volume_path, out, view="spimb", full_pipeline=False)
    return out


if __name__ == "__main__":
    # Quick check that the service is up.
    c = DeabeClient()
    print(c.healthz())
    print("loaded:", c.loaded_models())
