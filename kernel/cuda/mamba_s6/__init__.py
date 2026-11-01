"""kernel/cuda/mamba_s6/ -- OUR Mamba S6 selective scan. Builds to build/kernel/cuda/mamba_s6/.

Written because mamba_ssm 2.2.4+computecanada is a compiled extension linked against
torch~=2.5.0 and the cluster's torch is 2.13.0: the .so fails with `undefined symbol:
c10::cuda::c10_cuda_check_implementation`, which no install order fixes. A benchmark that
cannot run is not a benchmark. See mamba_s6.cu.
"""
