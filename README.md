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
The output should look something like this:
```
REPOSITORY                               TAG                      IMAGE ID      CREATED        SIZE
docker.io/library/hxn-ptycho-holoscan    latest                   9777387459f9  22 hours ago   411 MB

```

After successfully building the container, we run it via 
```
podman run --rm --net host -it -v ./eiger_dir:/eiger_dir -v ./eiger_simulation/test_data:/test_data -w /eiger_dir --device nvidia.com/gpu=all hxn-ptycho-holoscan
```

Since it is easier to manage virtual environment, packaging and version control via `pixi`, we use the following `pixi.toml` to generate a virtual conda environment inside a directory named `eiger_dir` we mounted while starting the container

The `pixi shell` command will start executing the `pixi.toml` file to create the virtual environment and create a file named `pixi.lock` which shows the information about installed packages. If one needs to remove the existing environment to start a fresh one, it is possible to do it via the `pixi clean` command.

To run the holoscan example,  `python3 eiger_connect_sample.py`.
Additional parameters can be passed to the holoscan script to streamline testing and deployment in different environments. For instance, to run the holoscan pipeline with a simulated stream (see below), one needs to change the default settings for eiger ip address and port. In addition, depending on the Simplon API version, the zmq messages can be encoded differently (json vs cbor) and message format can be passed as well:
```
python3 eiger_connect_sample.py --eiger_ip 0.0.0.0 --eiger_port 5555 -m cbor
```

# Simulating data stream using test data from HXN
To test/develop the holoscan pipeline, we can run a simulated data stream.
Test ptychography scan data recoreded by Eiger should be placed to `/eiger_simulation/test_data/`. Currently, the test files include `scan_257331_raw.h5` and `scan_257331.h5`.

To emulate the Eiger data stream, a simulated SimplonAPI 1.8 is used.

To launch the simulated API go to the `./eiger_simulation` folder and build the container:

```
cd ./eiger_simulation
docker build . -t eiger_sim:test --network
```

The API uses ports 8000 and 5555 for the simulated detector control and data stream, respectively.
To see the API output run the container interactively:

```
docker run -it -p 8000:8000 -p 5555:5555 eiger_sim:test
```
Otherwise, run it in the detached mode:

```
docker run -d -p 8000:8000 -p 5555:5555 eiger_sim:test
```

After launching the container (if container is running interactively, open a separate terminal), find the container ID with `docker ps` command. The output should look like this:
```
CONTAINER ID   IMAGE            COMMAND                  CREATED              STATUS              PORTS                                                                                  NAMES
d270120da233   eiger_sim:test   "/bin/sh -c 'uvicorn…"   About a minute ago   Up About a minute   0.0.0.0:5555->5555/tcp, :::5555->5555/tcp, 0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   peaceful_meitner
```
Connect to the container:
```
docker exec -it d270120da233 sh
```
To trigger the detector use the following command:
```
python trigger_detector.py -n 10
```
parameter `-n` controls how many images will be transmitted by the API. Once executed, you will see the frame sending status in the API window (if it is open in the interactive mode). The holoscan application window will show frame receiving status.



