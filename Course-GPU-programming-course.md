[## Introduction to Concurrent Programming with GPUs](https://www.coursera.org/learn/introduction-to-concurrent-programming)
```
int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
```

[## Introduction to Parallel Programming with CUDA](https://www.coursera.org/learn/introduction-to-parallel-programming-with-cuda)

[## CUDA At Scale for the Enterprise](https://www.coursera.org/learn/cuda-at-scale-for-the-enterprise)

