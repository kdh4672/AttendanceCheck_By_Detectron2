# AttendanceCheck_By_Detectron2

test result video : https://www.youtube.com/watch?v=GfHGqiB_RqE

You should first install Detectron2 tool : https://github.com/facebookresearch/detectron2

Second, Download my test.py , test_videos, model_weight at releases

Finally just implement test.py


==My development environment==

sys.platform: linux
Python: 3.8.2 (default, Mar 26 2020, 15:53:00) [GCC 7.3.0]
CUDA available: True
CUDA_HOME: /usr/local/cuda-10.1
NVCC: Cuda compilation tools, release 10.1, V10.1.105
GPU 0: GeForce GTX 1660
GCC: gcc (Ubuntu 7.5.0-3ubuntu118.04) 7.5.0
PyTorch: 1.4.0

==Dependecy==
(Name/Version/BuildChannel)

_libgcc_mutex/0.1/main  
absl-py                   0.9.0                    pypi_0    pypi
attrs                     19.3.0                     py_0    conda-forge
backcall                  0.1.0                      py_0    conda-forge
blas                      1.0                         mkl  
bleach                    3.1.4              pyh9f0ad1d_0    conda-forge
bzip2                     1.0.8                h516909a_2    conda-forge
ca-certificates           2020.1.1                      0    anaconda
cachetools                4.0.0                    pypi_0    pypi
cairo                     1.16.0            hcf35c78_1003    conda-forge
certifi                   2020.4.5.1               py38_0    anaconda
chardet                   3.0.4                    pypi_0    pypi
cloudpickle               1.3.0                    pypi_0    pypi
cudatoolkit               10.1.243             h6bb024c_0  
cycler                    0.10.0                   pypi_0    pypi
cython                    0.29.16                  pypi_0    pypi
dbus                      1.13.6               he372182_0    conda-forge
decorator                 4.4.2                      py_0    conda-forge
defusedxml                0.6.0                      py_0    conda-forge
detectron2                0.1.1                     dev_0    <develop>
entrypoints               0.3             py38h32f6830_1001    conda-forge
expat                     2.2.9                he1b5a44_2    conda-forge
ffmpeg                    4.1.3                h167e202_0    conda-forge
fontconfig                2.13.1            h86ecdb6_1001    conda-forge
freetype                  2.9.1                h8a8886c_1  
future                    0.18.2                   pypi_0    pypi
fvcore                    0.1.dev200407            pypi_0    pypi
gettext                   0.19.8.1          hc5be6a0_1002    conda-forge
giflib                    5.2.1                h516909a_2    conda-forge
glib                      2.58.3          py38h73cb85d_1003    conda-forge
gmp                       6.2.0                he1b5a44_2    conda-forge
gnutls                    3.6.5             hd3a4fd2_1002    conda-forge
google-auth               1.13.1                   pypi_0    pypi
google-auth-oauthlib      0.4.1                    pypi_0    pypi
graphite2                 1.3.13            he1b5a44_1001    conda-forge
grpcio                    1.28.1                   pypi_0    pypi
gst-plugins-base          1.14.5               h0935bb2_2    conda-forge
gstreamer                 1.14.5               h36ae1b5_2    conda-forge
harfbuzz                  2.4.0                h9f30f68_3    conda-forge
hdf5                      1.10.5          nompi_h3c11f04_1104    conda-forge
icu                       64.2                 he1b5a44_1    conda-forge
idna                      2.9                      pypi_0    pypi
importlib-metadata        1.6.0            py38h32f6830_0    conda-forge
importlib_metadata        1.6.0                         0    conda-forge
intel-openmp              2020.0                      166  
ipykernel                 5.2.0            py38h23f93f0_1    conda-forge
ipython                   7.13.0           py38h32f6830_2    conda-forge
ipython_genutils          0.2.0                      py_1    conda-forge
jasper                    1.900.1           h07fcdf6_1006    conda-forge
jedi                      0.16.0           py38h32f6830_1    conda-forge
jinja2                    2.11.1                     py_0    conda-forge
jpeg                      9c                h14c3975_1001    conda-forge
jsonschema                3.2.0            py38h32f6830_1    conda-forge
jupyter_client            6.1.2                      py_0    conda-forge
jupyter_core              4.6.3            py38h32f6830_1    conda-forge
kiwisolver                1.2.0                    pypi_0    pypi
lame                      3.100             h14c3975_1001    conda-forge
ld_impl_linux-64          2.33.1               h53a641e_7  
libblas                   3.8.0                    15_mkl    conda-forge
libcblas                  3.8.0                    15_mkl    conda-forge
libclang                  9.0.1           default_hde54327_0    conda-forge
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libiconv                  1.15              h516909a_1006    conda-forge
liblapack                 3.8.0                    15_mkl    conda-forge
liblapacke                3.8.0                    15_mkl    conda-forge
libllvm9                  9.0.1                hc9558a2_0    conda-forge
libopencv                 4.2.0                    py38_3    conda-forge
libpng                    1.6.37               hbc83047_0  
libsodium                 1.0.17               h516909a_0    conda-forge
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.1.0                h2733197_0  
libuuid                   2.32.1            h14c3975_1000    conda-forge
libwebp                   1.0.2                h56121f0_5    conda-forge
libxcb                    1.13              h14c3975_1002    conda-forge
libxkbcommon              0.10.0               he1b5a44_0    conda-forge
libxml2                   2.9.10               hee79883_0    conda-forge
markdown                  3.2.1                    pypi_0    pypi
markupsafe                1.1.1            py38h1e0a361_1    conda-forge
matplotlib                3.2.1                    pypi_0    pypi
mistune                   0.8.4           py38h1e0a361_1001    conda-forge
mkl                       2020.0                      166  
mkl-service               2.3.0            py38he904b0f_0  
mkl_fft                   1.0.15           py38ha843d7b_0  
mkl_random                1.1.0            py38h962f231_0  
nbconvert                 5.6.1            py38h32f6830_1    conda-forge
nbformat                  5.0.4                      py_0    conda-forge
ncurses                   6.2                  he6710b0_0  
nettle                    3.4.1             h1bed415_1002    conda-forge
ninja                     1.9.0            py38hfd86e86_0  
notebook                  6.0.3                    py38_0    conda-forge
nspr                      4.25                 he1b5a44_0    conda-forge
nss                       3.47                 he751ad9_0    conda-forge
numpy                     1.18.1           py38h4f9e942_0  
numpy-base                1.18.1           py38hde5b4d6_1  
oauthlib                  3.1.0                    pypi_0    pypi
olefile                   0.46                       py_0  
opencv                    4.2.0                    py38_3    conda-forge
openh264                  1.8.0             hdbcaa40_1000    conda-forge
openssl                   1.1.1g               h7b6447c_0    anaconda
pandoc                    2.9.2                         0    conda-forge
pandocfilters             1.4.2                      py_1    conda-forge
parso                     0.6.2                      py_0    conda-forge
pcre                      8.44                 he1b5a44_0    conda-forge
pexpect                   4.8.0            py38h32f6830_1    conda-forge
pickleshare               0.7.5           py38h32f6830_1001    conda-forge
pillow                    7.0.0            py38hb39fc2d_0  
pip                       20.0.2                   py38_1  
pixman                    0.38.0            h516909a_1003    conda-forge
portalocker               1.6.0                    pypi_0    pypi
prometheus_client         0.7.1                      py_0    conda-forge
prompt-toolkit            3.0.5                      py_0    conda-forge
protobuf                  3.11.3                   pypi_0    pypi
pthread-stubs             0.4               h14c3975_1001    conda-forge
ptyprocess                0.6.0                   py_1001    conda-forge
py-opencv                 4.2.0            py38h23f93f0_3    conda-forge
pyasn1                    0.4.8                    pypi_0    pypi
pyasn1-modules            0.2.8                    pypi_0    pypi
pycocotools               2.0                      pypi_0    pypi
pydot                     1.4.1                    pypi_0    pypi
pygments                  2.6.1                      py_0    conda-forge
pyparsing                 2.4.7                    pypi_0    pypi
pyrsistent                0.16.0           py38h1e0a361_0    conda-forge
python                    3.8.2                hcf32534_0  
python-dateutil           2.8.1                      py_0    conda-forge
python_abi                3.8                      1_cp38    conda-forge
pytorch                   1.4.0           py3.8_cuda10.1.243_cudnn7.6.3_0    pytorch
pyyaml                    5.3.1                    pypi_0    pypi
pyzmq                     19.0.0           py38ha71036d_1    conda-forge
qt                        5.12.5               hd8c4c69_1    conda-forge
readline                  8.0                  h7b6447c_0  
requests                  2.23.0                   pypi_0    pypi
requests-oauthlib         1.3.0                    pypi_0    pypi
rsa                       4.0                      pypi_0    pypi
send2trash                1.5.0                      py_0    conda-forge
setuptools                46.1.3                   py38_0  
six                       1.14.0                   py38_0  
sqlite                    3.31.1               h7b6447c_0  
tabulate                  0.8.7                    pypi_0    pypi
tensorboard               2.2.0                    pypi_0    pypi
tensorboard-plugin-wit    1.6.0.post3              pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
terminado                 0.8.3            py38h32f6830_1    conda-forge
testpath                  0.4.4                      py_0    conda-forge
tk                        8.6.8                hbc83047_0  
torchvision               0.5.0                py38_cu101    pytorch
tornado                   6.0.4            py38h1e0a361_1    conda-forge
tqdm                      4.45.0                   pypi_0    pypi
traitlets                 4.3.3            py38h32f6830_1    conda-forge
urllib3                   1.25.8                   pypi_0    pypi
wcwidth                   0.1.9              pyh9f0ad1d_0    conda-forge
webencodings              0.5.1                      py_1    conda-forge
werkzeug                  1.0.1                    pypi_0    pypi
wheel                     0.34.2                   py38_0  
x264                      1!152.20180806       h14c3975_0    conda-forge
xorg-kbproto              1.0.7             h14c3975_1002    conda-forge
xorg-libice               1.0.10               h516909a_0    conda-forge
xorg-libsm                1.2.3             h84519dc_1000    conda-forge
xorg-libx11               1.6.9                h516909a_0    conda-forge
xorg-libxau               1.0.9                h14c3975_0    conda-forge
xorg-libxdmcp             1.1.3                h516909a_0    conda-forge
xorg-libxext              1.3.4                h516909a_0    conda-forge
xorg-libxrender           0.9.10            h516909a_1002    conda-forge
xorg-renderproto          0.11.1            h14c3975_1002    conda-forge
xorg-xextproto            7.3.0             h14c3975_1002    conda-forge
xorg-xproto               7.0.31            h14c3975_1007    conda-forge
xz                        5.2.4                h14c3975_4  
yacs                      0.1.6                    pypi_0    pypi
zeromq                    4.3.2                he1b5a44_2    conda-forge
zipp                      3.1.0                      py_0    conda-forge
zlib                      1.2.11               h7b6447c_3  
zstd                      1.3.7                h0b5b093_0  
