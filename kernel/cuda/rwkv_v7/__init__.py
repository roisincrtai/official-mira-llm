"""kernel/cuda/rwkv_v7/ -- OUR RWKV-7 kernel. Builds to build/kernel/cuda/rwkv_v7/.

v7 differs from v6 by the DELTA RULE: the state is read through a_t and written back through
b_t, a rank-one correction whose size depends on the state itself. See rwkv_v7.cu.
"""
