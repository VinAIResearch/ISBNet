# ISBNet Setup Using Docker

Follow the steps below to set up ISBNet with Docker, including pulling the necessary Docker image, cloning the repository, and installing required dependencies.

## 1. Pull the Docker Image

First, pull the Docker image `rayyoh/oneformer3d`:

```bash

docker pull rayyoh/oneformer3d
```

## 2. Clone ISBNet Repository

```bash
git clone https://github.com/VinAIResearch/ISBNet/tree/master
```

## 3. Run Docker Container

```bash
docker run --gpus all -it --shm-size=8g --cpus=4 -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 -v C:\Users\Perception-Team\ISBNet:/workspace -p 6008:6008 --name isnbet check
```

## 4. Install Python Requirements Inside the container:

```bash
pip3 install -r requirements.txt
```
## 5. Build Segmentator Again Inside Docker

```bash
git clone https://github.com/Karbo123/segmentator.git
cd segmentator/csrc
mkdir build && cd build
cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch; print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'`
make && make install
 ```

## 6. Install libsparsehash-dev

```bash
apt-get install libsparsehash-dev
```
## 7. Setup PointNet2

```bash

cd isbnet/pointnet2
python3 setup.py bdist_wheel
cd ./dist
pip3 install pointnet2-0.0.0-cp310-cp310-linux_x86_64.whl
```
## 8. Final Build

```bash
cd ../../..
python3 setup.py build_ext develop
```

## Todo
### Explanation:
- The **Alternative Method** section provides the user an option to directly download the Dockerfile and skip the manual image pull process. 
- You can link to the Dockerfile in your GitHub repository [link](https://github.com/Smith-S-S/ISBNet/blob/master/Docker%20Setup/Dockerfile) and indicate to the user that they can skip the manual `docker pull` command if they choose this method.




