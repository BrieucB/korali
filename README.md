# Korali — gpu-batch-eval fork

This is a modified fork of [Korali](https://github.com/cselab/korali) used by the
`Hierarchical_UQ_compression_dev` inference pipeline.

**Repository**: `BrieucB/korali`
**Branch**: `gpu-batch-eval`

The fork adds a batched-evaluation path so the full TMCMC particle population is
forwarded to the PyTorch surrogate in a single GPU tensor call instead of N serial
calls. Korali itself remains a CPU/MPI library; the GPU acceleration lives in the
Python surrogate layer. See
[`LOCAL_GPU_BATCHING_NOTES.md`](LOCAL_GPU_BATCHING_NOTES.md) for the exact modified
source files.

---

## Installing on a standalone Linux workstation (o369-style)

These instructions assume:

- a standalone workstation with no job scheduler,
- system Python ≥ 3.8,
- system OpenMPI ≥ 4.0,
- no root privileges required,
- GPU optional (the build does not need CUDA; GPU is used at runtime via PyTorch).

### 1. Clone this repo

```bash
# Recommended layout — keep korali/ next to the project repo
cd /path/to/workspace/UQ_DPD
git clone -b gpu-batch-eval git@github.com:BrieucB/korali.git korali
```

If you already have the clone, make sure you are on the right branch:

```bash
cd /path/to/workspace/UQ_DPD/korali
git checkout gpu-batch-eval
git pull
```

### 2. Create a Python virtual environment

Korali's Python bindings are tested against Python 3.8–3.11. On o369 the system
Python is 3.8.

```bash
python3 -m venv /path/to/workspace/myenv
source /path/to/workspace/myenv/bin/activate

pip install --upgrade pip setuptools wheel
```

> If you already have a working venv (e.g. `myenv` on o369), activate it and skip
> this step.

### 3. Install Python dependencies

```bash
# Core scientific stack
pip install numpy scipy matplotlib pyyaml pydantic h5py

# Build tools needed by Korali
pip install meson ninja pybind11

# mpi4py — build against the system OpenMPI so it matches mpirun
MPICC=mpicc pip install --no-binary=mpi4py mpi4py

# PyTorch — for the surrogate forward pass
# CPU-only (simpler, always works):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 12.x), if the machine has a compatible GPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import mpi4py; print('mpi4py OK:', mpi4py.__file__)"
python -c "import torch; print('torch OK:', torch.__version__)"
```

### 4. Install the HUQ project in editable mode

```bash
pip install -e /path/to/workspace/UQ_DPD/Hierarchical_UQ_compression_dev --no-deps
```

### 5. Install build dependencies for Korali (GSL + Eigen)

Korali needs GSL ≥ 2.5 and Eigen ≥ 3.3.

**Check if they are already available:**

```bash
pkg-config --modversion gsl       # should print e.g. 2.7
pkg-config --modversion eigen3    # should print e.g. 3.4.0
```

On o369 both are installed system-wide and `pkg-config` finds them automatically
— skip the manual install steps below if `pkg-config` succeeds.

**Manual install (no root, fallback):**

```bash
mkdir -p $HOME/software && cd $HOME/software

# GSL
wget https://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz
tar -xzf gsl-latest.tar.gz && cd gsl-*/
./configure --prefix=$HOME/software/gsl
make -j$(nproc) && make install
cd ..

# Eigen (header-only)
git clone https://gitlab.com/libeigen/eigen.git
cd eigen && cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/software/eigen
cmake --build build --target install -j$(nproc)
cd ..

# Export so Meson/CMake can find them
export PKG_CONFIG_PATH=$HOME/software/gsl/lib/pkgconfig:${PKG_CONFIG_PATH:-}
export CMAKE_PREFIX_PATH=$HOME/software/eigen:$HOME/software/gsl:${CMAKE_PREFIX_PATH:-}
export LD_LIBRARY_PATH=$HOME/software/gsl/lib:${LD_LIBRARY_PATH:-}
```

### 6. Build and install Korali

```bash
source /path/to/workspace/myenv/bin/activate
cd /path/to/workspace/UQ_DPD/korali

# Choose an install prefix (does not need root)
export KORALI_PREFIX=$HOME/Programs/korali/.local

meson setup build \
  --wipe \
  --buildtype=release \
  --prefix="$KORALI_PREFIX" \
  -Dmpi=true \
  -Dmpi4py=true \
  -Dopenmp=false

meson install -C build
```

> `--wipe` is safe to use on an existing `build/` directory; it discards cached
> configuration and starts fresh. Omit it on subsequent rebuilds if you only changed
> Python files.

### 7. Set KORALI_PYTHONPATH

You have two options:

**Option A — use the installed site-packages (recommended, required for `libkorali`):**

```bash
export KORALI_PYTHONPATH=$(find "$KORALI_PREFIX" -type d -path '*/site-packages' | head -1)
# On o369: /temp/brieuc/Programs/korali/.local/lib/python3.8/site-packages
```

This is what the local workflow scripts default to. The compiled `libkorali.so`
extension lives here — without it `import korali` will raise `ModuleNotFoundError`.

**Option B — use the source tree (Python-only changes, no rebuild needed):**

```bash
export KORALI_PYTHONPATH=/path/to/workspace/UQ_DPD/korali/python
```

⚠ **This only works if `libkorali.so` is present in `korali/python/korali/`.**
After a fresh clone or a rebuild without install, the `.so` is not there.
Option B is useful only if you symlink the compiled extension into the source tree,
or if you have already run `meson install` and want to test Python-layer edits in-place.

### 8. Verify the installation

```bash
source /path/to/workspace/myenv/bin/activate
export KORALI_PYTHONPATH=/path/to/workspace/UQ_DPD/korali/python
export LD_LIBRARY_PATH="$KORALI_PREFIX/lib64:$KORALI_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -c "
import sys
sys.path.insert(0, '$KORALI_PYTHONPATH')
import korali
import mpi4py
print('korali  :', korali.__file__)
print('mpi4py  :', mpi4py.__file__)
print('OK')
"
```

Expected output:

```
korali  : /path/to/workspace/UQ_DPD/korali/python/korali/__init__.py
mpi4py  : /path/to/workspace/myenv/lib/python3.8/site-packages/mpi4py/__init__.py
OK
```

Also test MPI:

```bash
mpirun -np 2 python -c "
import sys; sys.path.insert(0, '$KORALI_PYTHONPATH')
from mpi4py import MPI
import korali
print(f'rank {MPI.COMM_WORLD.Get_rank()} — korali {korali.__file__}')
"
```

---

## Exact o369 state (reference)

| Item | Value |
|------|-------|
| Python | 3.8.1 (`/usr/bin/python3`) |
| venv | `/temp/brieuc/workspace/myenv/` |
| OpenMPI | 4.0.2 (`/usr/bin/mpirun`) |
| Korali source | `/temp/brieuc/workspace/UQ_DPD/korali` — branch `gpu-batch-eval` |
| Korali install prefix | `/temp/brieuc/Programs/korali/.local/` |
| KORALI_PYTHONPATH | `/temp/brieuc/workspace/UQ_DPD/korali/python` |
| Build flags | `mpi=true`, `mpi4py=true`, `openmp=false` |

---

## Rebuilding after a source change

If you pull new commits or modify the C++ source, rebuild with:

```bash
source /path/to/workspace/myenv/bin/activate
cd /path/to/workspace/UQ_DPD/korali
meson install -C build   # incremental, only recompiles changed files
```

If the `meson.build` or `meson_options.txt` changed, re-run `meson setup` first:

```bash
meson setup build --wipe --buildtype=release --prefix="$KORALI_PREFIX" \
  -Dmpi=true -Dmpi4py=true -Dopenmp=false
meson install -C build
```

---

## Troubleshooting

**`import korali` raises `ImportError`**

`KORALI_PYTHONPATH` is not in `sys.path`. The local workflow scripts set this
automatically via the `--korali-pythonpath` argument. If running manually:

```bash
export PYTHONPATH=/path/to/workspace/UQ_DPD/korali/python:$PYTHONPATH
```

**`mpirun` launches but `mpi4py` crashes on init**

`mpi4py` was built against a different MPI version than `mpirun`. Rebuild:

```bash
MPICC=mpicc pip install --force-reinstall --no-binary=mpi4py mpi4py
```

**`meson setup` cannot find `gsl` or `eigen3`**

Export `PKG_CONFIG_PATH` and `CMAKE_PREFIX_PATH` as shown in step 5, then rerun
`meson setup --wipe`.

**`meson setup` cannot find `pybind11`**

```bash
pip install pybind11
# Then tell Meson where to find it:
export CMAKE_PREFIX_PATH=$(python -c "import pybind11; print(pybind11.get_cmake_dir())"):${CMAKE_PREFIX_PATH:-}
```

---

## Related documentation

- [`LOCAL_GPU_BATCHING_NOTES.md`](LOCAL_GPU_BATCHING_NOTES.md) — which Korali files
  the `gpu-batch-eval` patch modifies
- [`../Hierarchical_UQ_compression_dev/docs/HPC_GPU_BATCHED_REDUCED_INDENTATION.md`](../Hierarchical_UQ_compression_dev/docs/HPC_GPU_BATCHED_REDUCED_INDENTATION.md)
  — equivalent guide for HPC/SLURM clusters (conda, module system)
- [`../Hierarchical_UQ_compression_dev/inference/submit/local/`](../Hierarchical_UQ_compression_dev/inference/submit/local/)
  — local (no-SLURM) workflow scripts that use this Korali installation
