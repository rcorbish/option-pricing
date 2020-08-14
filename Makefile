CUDA_HOME ?= /usr/local/cuda/
NVCC ?= nvcc -O3 -DGPUTIMING # -lineinfo

INCLUDES ?= -I$(CUDA_HOME)/include -I.

LIBS ?= -L$(CUDA_HOME)/lib64 -lcudart -lcurand 

NVFLAGS ?= -std=c++11 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75 
# Compile cuda source codes to objects
cuda_pricing: cuda_pricing.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(LIBS) -o $@ $<

