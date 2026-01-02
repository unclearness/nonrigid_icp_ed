Nonrigid ICP with Embedded Deformation
=====================================

Python reference implementation of non‑rigid ICP using an embedded deformation graph. It provides small, reproducible demos for mesh/point‑cloud alignment:

- **Watertight demo** (`demo_watertight.py`): deform a sphere mesh to the spot/cow mesh.
- **Face demo** (`demo_face.py`): align a frontal face mesh to a head mesh using sparse landmarks.

Project layout
--------------
- `nonrigid_icp_ed/`: core library (graph construction, losses, registration loop, I/O helpers).
- `config/`: YAML configs for the demos.
- `data/`: demo assets (meshes and landmark JSON).
- `output/`: results written by the demos (ignored by git).
- `demo_watertight.py`, `demo_face.py`: runnable examples.

Requirements
------------
- Python 3.12+
- PyTorch 2.9.0 (CPU works; CUDA 12.6 wheels are configured in `pyproject.toml`)
- Open3D, FAISS (CPU), NumPy, OmegaConf, Loguru

Setup
-----
The project uses `uv` for dependency management (see `pyproject.toml`).

```bash
# Create the environment and install dependencies
uv sync

# Optionally, activate the environment for manual runs
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
```

If you prefer `pip`, install with `pip install -e .`, but ensure a matching PyTorch wheel for your platform/GPU.

Running the demos
-----------------
All paths are relative to the repo root. Outputs are written under `output/`.

### Watertight mesh-to-mesh
```bash
uv run python demo_watertight.py
```
Produces:
- `output/watertight/warped_sphere_mesh.obj`: deformed source mesh.
- `output/watertight/graph_*.(obj|ply)`: deformation graph at start/end.
- `output/watertight/optimization_histories/`: per-iteration checkpoints (if enabled).

### Face alignment with landmarks
```bash
uv run python demo_face.py
```
Produces:
- `output/face/aligned_mediapipe_face.obj`: initial Umeyama-aligned source.
- `output/face/warped_mediapipe_face.obj`: final warped mesh.
- `output/face/reconstructed_warped_mediapipe_face.obj`: reconstruction from saved histories.
- `output/face/graph_*.(obj|ply)` and `output/face/optimization_histories/` as above.

Configuration
-------------
Each demo loads a YAML config (`config/demo_watertight.yaml`, `config/demo_face.yaml`) into `NonrigidIcpEdConfig`. Key fields:
- `num_iterations`: outer ICP iterations.
- `graph_conf`: graph degree, node assignment weights, and per-point node count.
- `correspondence_conf.k`: nearest neighbors for correspondence search.
- `minimization_conf`: inner optimizer settings (learning rate, ARAP/Chamfer/landmark/regularization weights, truncation threshold, etc.).
- `write_history_dir` / `keep_history_on_memory`: control saving optimization histories.

Feel free to copy a config, tweak weights or iteration counts, and point `write_history_dir` to a custom run directory.

Notes
-----
- GPU is optional; the demos auto-select CUDA if available (`torch.cuda.is_available()`).
- Random seeds are set in the demos for repeatability; adjust `node_num` or sampling strategy inside the scripts to change graph density.
- Input meshes for the demos live in `data/`; replace them to experiment with other shapes while reusing the configs.
