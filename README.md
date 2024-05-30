# GPU Programming Learning Notes
## [GPU Optimization Workshop](https://www.youtube.com/watch?v=v_q2JTIqE20)
1. Point-wise operations like sin(sin(sin(x))), relu(x) is memory-bound, so we should do kernel fusion, e.g. with torch.compile().
- torch.compile is a fusion compiler.
- Triton compiles with tmp var to show the kerel fusion.
