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
2. Token concatenation
- <img width="1345" alt="Screenshot 2024-05-31 at 10 38 21 PM" src="https://github.com/JerryPan2718/GPU-programming/assets/37657480/5b8531e5-a8ac-4bdd-8fa5-39cc0627ad74">
3. Paged KV cache
- KV cache as a linked list of block
- Lazy memory allocation leads to minimal waste and more requests in-flight
4. Speculative decoding
- Guess tokens n, n+1, and n+2 while generating seq[n].

- Triton
  - Graph compiler <- CUDA <- Triton
 
- Scaling data process from CPU to distributed GPUs
  - OLTP is not GPU friendly.
  - OLAP is SIMD/GPU friendly.
  - Apache Arrow: columnar format + language-agnostic + CPU/GPU + zero-copy
 
  
