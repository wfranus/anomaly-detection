# C3D Feature extraction
Steps to install C3D on Ubuntu and use pre-trained models to extract video features.

## Installation
>Tested on: Ubuntu 18.04, GeForce 930M card, nvidia-driver 418.43, CUDA Toolkit 9.2, **C3D v1.0**

Detailed installation steps (also for MacOS) can be found
[here](https://github.com/facebook/C3D/blob/master/C3D-v1.0/docs/installation.md).

#### Prerequisites
* Install CUDA (detailed steps [here](https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/)):
    * [install nvidia drivers](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)
        * after reboot, verify installation with: `nvidia-smi`
    * install depencencies: `sudo apt-get install freeglut3 freeglut3-dev libxi-dev libxmu-dev`
    * download [CUDA Toolkit 9.2 **runfile**](https://developer.nvidia.com/cuda-92-download-archive)
        * select Linux > x84_64 > Ubuntu > 17.10 > runfile(local)
        * download both files: Base Installer and Patch 1
    * run the runfile `sudo sh cuda_9.2.88_396.26_linux.run`
        * **answer 'no' when asked for installing nvidia driver!!!** and 'yes' for other questions
    * install cuBLAS patch `sudo sh cuda_9.2.148.1_linux.run`
    * (optional) verify your installation by performing steps 6 and 7 from [here](https://www.pugetsystems.com/labs/hpc/How-to-install-CUDA-9-2-on-Ubuntu-18-04-1184/)
* Install ATLAS: `sudo apt-get install libatlas-base-dev`

* Install Additional dependencies:
`sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev`

* Install Google logging library
```bash
curl -OL https://github.com/google/glog/archive/v0.4.0.zip
unzip v0.4.0.zip && cd glog-0.4.0
./autogen.sh && ./configure && make && sudo make install  # use make -j2 for parallel
cd .. && rm -f GLOG_ZIP && rm -r glog-0.4.0
```

* Install Protobuf compiler: `sudo apt install protobuf-compiler`.

If C3D compilation fails because of protoc, try installing protoc like this:
```bash
PROTOC_ZIP=protoc-3.3.0-linux-x86_64.zip
curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
rm -f $PROTOC_ZIP

```
Instruction for MacOS [here](http://google.github.io/proto-lens/installing-protoc.html)

#### Install C3D
Download C3D code from github: https://github.com/facebook/C3D
```bash
cd C3D-v1.0/
cp Makefile.config.example Makefile.config
```
Adjust Makefile.config:
```bash
# comment compute_20 in line 10
CUDA_ARCH :=    -gencode arch=compute_30,code=sm_30 \
                -gencode arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                #-gencode arch=compute_20,code=sm_20 \
                #-gencode arch=compute_20,code=sm_21 \
                #-gencode=arch=compute_50,code=compute_50
                
# uncomment line 18
OPENCV_VERSION := 3

# in line 50 add path to hdf5.h, in my case it was:
# /usr/include/hdf5/serial/
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/

# in line 51 add path to libhdf5.so, in my case it was:
# /usr/lib/x86_64-linux-gnu/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
```
Compile C3D
```bash
make all -j 4  # specify number of parallel threads for compilation
make test
make runtest
```
If some targets fail to compile (some examples and tests), try to exclude (or delete) related .cpp files from
compilation process.
Eg. `mv examples/mnist/convert_mnist_data.cpp examples/mnist/convert_mnist_data.cpp_turnoff`

## Extract C3D features
Follow [this](https://docs.google.com/document/d/1-QqZ3JHd76JfimY4QKqOojcEaf5g3JS0lNh-FHTxLag/edit) doc with instructions.

> check http://vra.github.io/2016/03/03/c3d-use/