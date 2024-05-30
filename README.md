# GPU Programming Learning Notes
## [GPU Optimization Workshop](https://www.youtube.com/watch?v=v_q2JTIqE20)
1. Kernel confusion: Point-wise operations like sin(sin(sin(x))), relu(x) is memory-bound, so we should do kernel fusion, e.g. with torch.compile().
- torch.compile is a fusion compiler.
- Triton compiles with tmp var to show the kerel fusion.
2. Tensor cores: mixed-precision.
- torch.set_float32_matmul_precision("high")
3. Overhead reduction
- CUDA kernels are async so queue them up -> CUDA graphs
4. Quantization helps memory-bound workloads
- What about compute-bound workloads?
- Compilers generally perform fusion, but optimizations reqire math rewriting of the same expr, which is generally harder.
5. Custom kernel

- TensorRT-LLM
1. Phases of LLM request
- 2 phases - prefill and generate
  - Prefill: process the prompt, generate the first token, and initialize the KV cache.
  - Generate: from prior state (KV cache), and last generated token, generate next token, and update KV cache.
