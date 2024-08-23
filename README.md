# Workloads
#### `./Workloads/BYOL.ipynb`
BYOL (Bootstrap Your Own Latent) is a self-supervised learning method, specifically an unsupervised learning technique, that enables effective image representation learning without the need for large pre-trained datasets.

#### `./Server/Recommander_system.ipynb`
This notebook is DLRM relate to recommander system.

# NPU Design
#### `./NPU_Design/Amaranth/pakage_ta/inference.py`  
this code instruct ISA using Amaranth. Using model is CNN and QLinear,  QConv2d make amaranth ISA. if you run this code, you check performance between ISA Anaranth and normal model.

#### `./NPU_Design/QAT/main.py`
QAT means Quantization-Aware Training. using mnist data, trains CNN QAT model.

# Cuda_coding
#### `./Cuda_coding/0_exercise/matmul.mu`
this code is matmul logic using cuda C. you have to complie that `nvcc -o matmul.out matmul.cu`

#### `./Cuda_coding/0_exercise/convolution.mu`
this code is convolution logic using cuda C. you have to complie that `nvcc -o convolution.out convolution.cu`