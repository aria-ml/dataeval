# Container Reference

DataEval publishes container images to Harbor at
`harbor.jatic.net/aria/dataeval`. The image is a base execution environment: it
contains a Python interpreter, DataEval, and DataEval's runtime dependencies,
with no application layered on top.

The image declares no entrypoint or command. Provide the command when running
the container:

```bash
docker run --rm harbor.jatic.net/aria/dataeval:cpu \
  python -c "import dataeval; print(dataeval.__version__)"
```

To run a containerized workflow instead of building on a base image, use
[DataEval-Flow](https://github.com/aria-ml/dataeval-flow) at
`harbor.jatic.net/aria/dataeval-flow`.

## Variants

Images are published in three variants based on PyTorch and ONNX Runtime
support:

```{list-table}
:widths: 15 25 60
:header-rows: 1

* - Variant
  - Contents
  - Use when
* - `cpu`
  - PyTorch CPU build, `onnxruntime`
  - No GPU required. Smallest image size.
* - `cu126`
  - PyTorch CUDA 12.6 build, `onnxruntime-gpu`
  - Host has a CUDA 12.6 capable driver.
* - `cu130`
  - PyTorch CUDA 13.0 build, `onnxruntime-gpu`
  - Host has a CUDA 13.0 capable driver.
```

All variants use Ubuntu. CUDA variants install the CUDA runtime and cuDNN via
Python wheels, so no CUDA toolkit installation is needed in the container.
Running CUDA variants requires a compatible host NVIDIA driver and GPU access
flags such as `--gpus all`.

## Tags

```{list-table}
:widths: 35 65
:header-rows: 1

* - Tag
  - Points at
* - `1.2.0-cpu`
  - Exact release. Immutable.
* - `1.2-cpu`
  - Latest patch release on the 1.2 release line.
* - `cpu`
  - Latest stable release across all versions.
* - `edge-cpu`
  - Latest build from the `main` branch. Unstable.
```

Replace `cpu` with `cu126` or `cu130` for CUDA variants. Release candidates use
exact version tags (e.g. `1.2.0-rc1-cpu`) and do not update floating tags.

Images are signed with [cosign](https://docs.sigstore.dev/cosign/overview/) and
include a CycloneDX SBOM attestation. The public key is in the repository at
`docker/cosign.pub` (the same key used by DataEval-Flow).

To verify the image and its attestation:

```bash
curl -sSfLO \
  https://gitlab.jatic.net/jatic/aria/dataeval/-/raw/main/docker/cosign.pub

cosign verify --key cosign.pub harbor.jatic.net/aria/dataeval:1.2.0-cpu
cosign verify-attestation --key cosign.pub --type cyclonedx \
  harbor.jatic.net/aria/dataeval:1.2.0-cpu
```

To view the SBOM contents:

```bash
cosign download attestation --predicate-type https://cyclonedx.org/bom \
  harbor.jatic.net/aria/dataeval:1.2.0-cpu \
  | jq -r '.payload' | base64 -d | jq '.predicate'
```

## Configuration

DataEval is configured through its Python API rather than through environment
variables, configuration files, or CLI flags.

- **Environment variables:** None required or read. The image only sets `PATH`
  to prioritize `/app/.venv/bin`.
- **Secrets:** None used. If user code requires secrets, mount them as files.
- **Input precedence and parameter dependencies:** Not applicable.
- **Usage output:** No `--help` flag is provided because the image contains no
  CLI application.
- **Health checks:** Not applicable. The image is not a long-running service.

## Volume Mounts

The image includes standard directory paths (IR-2.5). DataEval does not default
to these locations; specify paths in your code. Unmounted directories contain a
`.not_mounted` marker file.

```{list-table}
:widths: 30 70
:header-rows: 1

* - Path
  - Purpose
* - `/input/data`
  - Input datasets. Mount read-only when possible.
* - `/input/models`
  - Input model files (such as ONNX models).
* - `/output/data`
  - Exported or modified datasets.
* - `/output/models`
  - Exported models.
* - `/output/results`
  - Metric results, reports, and generated output.
```

All mount directories are owned by the non-root `dataeval` user.

```bash
docker run --rm \
  -v "$(pwd)/data:/input/data:ro" \
  -v "$(pwd)/results:/output/results" \
  harbor.jatic.net/aria/dataeval:cpu \
  python /input/data/analyze.py
```

## Input Formats

DataEval operates on MAITE `AnnotatedDataset` objects provided by user code.
Supported file formats depend on your dataset loader. Refer to
[What data does each tool need?](../getting-started/input-requirements.md) for
in-memory data structure requirements.

## Running as Non-Root User

The container runs as user `dataeval` (uid 1001). Files written to bind-mounted
host directories are owned by uid 1001. If this differs from your host user,
either pre-create output directories with appropriate permissions or pass
`--user "$(id -u):$(id -g)"`.

## Recommended Minimum Specifications

Resource requirements depend primarily on dataset size:

```{list-table}
:widths: 20 40 40
:header-rows: 1

* - Resource
  - Minimum
  - Notes
* - CPU
  - 2 cores
  - Additional cores benefit parallel metric algorithms and CPU embedding
    extraction.
* - Memory
  - 8 GB
  - DataEval keeps embeddings and statistics in memory. Scale memory with
    dataset size.
* - Storage
  - 10 GB free space
  - CPU image size is ~2.5 GB. CUDA variants are larger.
* - GPU
  - None required
  - Optional. Accelerates embedding extraction. Use a `cu126` or `cu130` variant
    with `--gpus all`.
```

## Architectures

Published images support `linux/amd64` only. The DataEval Python library can be
installed via `pip` on other platforms supported by its dependencies.

## Internet Access

The image requires no runtime internet access. DataEval makes no outbound
network calls. Network access is only required if user scripts download
external models or data. Pulling the image requires access to
`harbor.jatic.net`.
