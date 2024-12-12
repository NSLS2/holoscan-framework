
The `Containerfile` for ubuntu image with `nvcc` compiler and `MPI` capabilities:
```
FROM docker.io/nvidia/cuda:12.6.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y curl vim libopenmpi-dev

RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash
```

To build an image:
```
docker build -t my-image .
```

After putting `pixi` files and possibly `ptycho` code into a directory called `pytcho_dir`

```
podman run --rm --net host -it -v ./ptycho_dir:/ptycho_dir --device nvidia.com/gpu=all my-image
```

Create the conda environment with `pixi` via `pixi shell` command. If running `MPI` complains about root access, export the following commands on the terminal:

```
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
```

*** A side note about `Cython` compilation, if it is required to compile `.so` file again, one can do it via `cythonize -a -i ptycho_cython.pyx` command.
