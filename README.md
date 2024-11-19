# holoscan-framework
In order to run the `eiger_connect_sample.py`, we run the holoscan application inside an ubuntu container. However, since basic tools like `vim`, `curl`, `pixi` does not exit in the pure ubuntu container, we create a wrapper-like container with a `Dockerfile` defined as below
```
FROM docker.io/nvidia/cuda:12.6.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y curl vim
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash
```
We build a container named `hxn-ptycho-holoscan`:
 ```
 docker build . -t hxn-ptycho-holoscan --network host
 ```

To run turn the Docker container into a podman container, we run the following command:
```
podman pull docker-daemon:hxn-ptycho-holoscan:latest
```
You can now verify that podman sees the correct image via:
```
podman image ls
```
The output should look like this:
```
REPOSITORY                               TAG                      IMAGE ID      CREATED        SIZE
docker.io/library/hxn-ptycho-holoscan    latest                   9777387459f9  22 hours ago   411 MB

```

After successfully building the container, we run it via 
```
podman run --rm --net host -it -v ./eiger_dir:/eiger_dir -w /eiger_dir --device nvidia.com/gpu=all hxn-ptycho-holoscan
```

Since it is easier to manage virtual environment, packaging and version control via `pixi`, we use the following `pixi.toml` to generate a virtual conda environment inside a directory named `eiger_dir` we mounted while starting the container

```
[project]
channels = ["conda-forge"]
description = "Add a short description here"
name = "eiger_test"
platforms = ["linux-64"]
version = "0.1.0"

[tasks]

[system-requirements]
libc = { family = "glibc", version = "2.35" }

[dependencies]
python = ">=3.11,<3.12"
#cupy = ">=13.3.0,<14"
numpy = ">=1.20,<1.27"
ipython = ">=8.28.0,<9"
cuda-cudart = ">=12.6.77,<13"
libcublas = ">=12.6.3.3,<13"
libcufft = ">=11.3.0.4,<12"
libcurand = ">=10.3.7.77,<11"
libcusparse-dev = ">=12.5.4.2,<13"
libcusolver = ">=11.7.1.2,<12"
pyzmq = ">=26.0"
scipy = ">=1.14"
gcc = ">=14.0"

[pypi-dependencies]
holoscan = ">=2.5.0,<2.6.0"
dectris-compression = ">=0.3"
```
The `pixi shell` command will start executing the `pixi.toml` file to create the virtual environment and create a file named `pixi.lock` which shows the information about installed packages. If one needs to remove the existing environment to start a fresh one, it is possible to do it via the `pixi clean` command.

To run the holoscan example,  `python3 eiger_connect_sample.py` and to allow gpu execution pass `--cuda` argument.

Note to the holoscan developers: the current holoscan package installs `cupy` version `12.2.0` which does not have the `resample` functionality. Therefore, this script works fine on CPU but not on GPU at the moment.
