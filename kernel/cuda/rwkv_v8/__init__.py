"""kernel/cuda/rwkv_v8/ -- RWKV-8's CUDA. ALL OF IT IS UPSTREAM'S; NONE OF IT IS OURS.

Every other family in this directory has a .cu written here and built into a cubin. This one
has no such file and will not grow one, for two reasons that are both facts about RWKV-8:

  RWKV-8's CUDA IS RWKV-7's. upstream/cuda/wkv7_{cuda.cu,op.cpp} is the wind_backstepping
  kernel with `using bf = float` where RWKV-7 has `using bf = __nv_bfloat16` -- four lines of
  difference in total, and both copies are vendored so that diff can be read rather than
  taken on trust. Writing a third transcription of it here would add nothing.

  WHAT IS NEW IN RWKV-8 HAS NO CUDA AT ALL. RWKV-8 is RWKV-7 plus ROSA, the Rapid Online
  Suffix Automaton, and in every file of the archive ROSA is a Python loop on the CPU with a
  forward and no backward. See baselines/rwkv_v8/upstream/PROVENANCE.

So this package holds a loader and a check, and the kernel it loads is BlinkDL's, compiled by
BlinkDL's own cpp_extension.load from the sources vendored at upstream/cuda/.
"""
