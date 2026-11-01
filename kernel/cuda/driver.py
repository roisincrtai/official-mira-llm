"""kernel/cuda/driver.py -- load our cubins through the CUDA driver API, one per device.

Shared by mira_scan.py, mira_decay.py and mira_causal_cov1d.py. Every one of them is OURS: this
project ships no third-party CUDA kernel in its hot path.

    ok, detail = status()                       is the CUDA path usable, and which cubin
    fns = module_for("mira_scan.cubin", 0)       (name -> CUfunction) for that device
    ok, why = launch(fn, ctx, n_threads, ptrs, ints)

WHY THE DRIVER API. A cubin is loaded with cuModuleLoad and needs no C++ compiler on the
compute node, no PyTorch ABI, and no rebuild when torch is upgraded. The cost is that a cubin
is ARCHITECTURE SPECIFIC; see kernel/cuda/mira-sdpa/legacies/howto.txt.

ONE MODULE PER DEVICE, AND THE CONTEXT SET PER LAUNCH. A CUmodule belongs to a CUDA CONTEXT, a
context belongs to ONE DEVICE, and it is CURRENT PER THREAD. Under nn.DataParallel the replicas
run in separate threads on cuda:0..3; under gradient checkpointing the recomputed forward runs
on the autograd engine's thread. Neither has made the context current, and neither is on the
device the module was first loaded into -- which is why a kernel that passed `make check` on one
GPU failed on the first training step of a four-GPU node.
"""
import ctypes
import os
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
# build/kernel/cuda/ is where the Makefile writes; kernel/cuda/ is searched after it so a cubin
# built by hand beside the source still works.
# THE BASE OF EVERY FAMILY, not one family's directory. A cubin is named with its family --
# "mira/mira_scan.cubin", "mamba_s6/....cubin" -- so one loader, one context cache and one lock
# serve all of them and MIRA_CUDA_KERNEL names a single path however many families exist.
_DIRS = (os.path.join(_ROOT, "build", "kernel", "cuda"), _HERE)
# A FALLBACK, NOT A CHOICE. Every elementwise kernel in this repository was launched with 256
# threads per block regardless of what it does, because one number had to be written and 256
# is the number everyone writes. It is not right for all of them: the best block size depends
# on a kernel's REGISTER COUNT and shared-memory use, which differ between mira_fd_local (a
# 64-step register-resident scan), mira_fd_fixup (three loads and a multiply-add) and
# mira_fd_bwd_point (the heaviest of the seven).
#
# block_for() below asks the DRIVER instead, per kernel, and this constant is what it falls
# back to when the query is unavailable.
THREADS = 256

# The driver error codes worth naming, so a failure says what happened rather than a number.
CU_ERR = {1: "INVALID_VALUE", 201: "INVALID_CONTEXT", 209: "NO_BINARY_FOR_GPU",
          400: "INVALID_HANDLE", 700: "ILLEGAL_ADDRESS", 701: "LAUNCH_OUT_OF_RESOURCES",
          702: "LAUNCH_TIMEOUT", 719: "LAUNCH_FAILED"}

_S = {"tried": False, "ok": False, "lib": None, "mods": {}, "why": None,
      "warned": set(), "occ": {}, "carve": set(), "reported": set()}

# (cubin, device, id(names)) -> (names, {name: CUfunction}), the lock-free hit path of
# module_for(). The name tuple is kept in the value and identity-checked on read.
_RESOLVED = {}

# cuFuncSetAttribute's two shared-memory attributes, by their CUfunction_attribute values.
#
#   MAX_DYNAMIC_SHARED_SIZE_BYTES (8)  the opt-in a kernel needs before it may be launched with
#                                      more than 48 KB of dynamic shared memory. Below 48 KB it
#                                      is a no-op, which is why setting it unconditionally is
#                                      safe.
#   PREFERRED_SHARED_MEMORY_CARVEOUT (9)  a PERCENTAGE of the SM's unified L1/shared that the
#                                      driver should hand to shared. -1 is "let the driver
#                                      decide", 100 is CU_SHAREDMEM_CARVEOUT_MAX_SHARED. On an
#                                      Ada SM the unified block is 128 KB and shared tops out at
#                                      100 KB, so this is the difference between fitting two
#                                      36.1 KB blocks on an SM and fitting one.
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100

# THE FIVE READ-ONLY CUfunction_attribute VALUES, which report() asks for and nothing else does.
# The enumerator is shared with the two settable attributes above -- 8 and 9 are the same enum --
# so these are its lower members and not a second numbering:
#
#   SHARED_SIZE_BYTES (1)      STATIC shared memory, the __shared__ arrays the .cu declares. It
#                              is NOT the dynamic request made at launch, and the two add.
#   LOCAL_SIZE_BYTES (3)       the per-thread stack frame. NONZERO IS A SPILL: an array nvcc
#                              could not keep in registers, or a register allocation that did
#                              not fit. This is the number that says whether a 256 B frame and
#                              544 LDL per thread per token came back; see the coefficient_bottleneck note in
#                              kernel/cuda/mira-vector/__init__.py.
#   NUM_REGS (4)               registers per thread, which with the block size decides how many
#                              blocks the 65,536-register file of an SM can hold.
#   BINARY_VERSION (6)         the SM version of the cubin image actually loaded, as 10*major +
#                              minor: 86 is an Ampere build running on Ada through binary
#                              compatibility, 89 is an Ada build. _cubin() prefers a
#                              .sm_XX. file and falls back to the plain one, so which was opened
#                              is not derivable from the launch.
#   PREFERRED_SHARED_MEMORY_CARVEOUT (9)  readable as well as settable, and read AFTER the set
#                              is the only way to see whether the request was kept.
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6

# DEVICE ATTRIBUTES, WHICH ANSWER WHAT A CARD ALLOWS RATHER THAN WHAT A KERNEL ASKED FOR.
#
#   MULTIPROCESSOR_COUNT (16)  the SM count; the divisor for "does this grid fill the card".
#   MAX_SHARED_MEMORY_PER_BLOCK_OPTIN (97)  the LARGEST per-block dynamic shared allocation the
#                              device will grant AFTER cuFuncSetAttribute has opted in --
#                              99 KiB on Ada (8.9), 163 KiB on Hopper, 48 KiB where there is no
#                              opt-in at all. IT IS NOT 48 KiB ON ANY MODERN CARD, and 48 KiB is
#                              only the default a function gets when nobody asks for more.
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
#   COMPUTE_CAPABILITY_MAJOR (75) / MINOR (76)  the card's own architecture, which report()
#                              compares against the loaded image's BINARY_VERSION. An older image
#                              runs by binary compatibility and says nothing, so the comparison is
#                              the only thing that notices it was never built for this card.
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

# The per-block dynamic shared memory a function gets without opting in. The floor this module
# reports when the driver cannot be asked, and the threshold prefer_shared() tests before it
# spends a cuFuncSetAttribute on the size opt-in.
CU_DEFAULT_SHARED_PER_BLOCK = 48 * 1024

# THE LOCK, AND WHY EVERY ENTRY POINT TAKES IT.
#
# nn.DataParallel runs one PYTHON THREAD PER REPLICA and the autograd engine runs one per
# device, so _init() and module_for() are called CONCURRENTLY, N ways, on the first forward of
# a multi-GPU run. Both were check-then-act on plain dict entries:
#
#   _init()      set _S["tried"] = True BEFORE doing the slow work (torch.cuda.init, dlopen
#                libcuda, cuInit). A second thread arriving in that window saw tried=True and
#                returned _S["ok"], STILL FALSE -- so module_for returned None, cuda_scan
#                returned None, and THAT REPLICA SILENTLY RAN pairwise_scan while cuda:0 ran
#                the kernel. Same maths, ~10-24x slower, and the four devices were not running
#                the same code on the step where they were meant to be identical.
#
#   module_for() missed the cache, and two threads for the same device could both reach
#                cuModuleLoad -- two CUmodule handles for one device, one of them leaked, the
#                launches using whichever won the write.
#
# RLock, not Lock: module_for() calls _init(), and a plain Lock would deadlock on that.
# Uncontended acquisition is tens of nanoseconds against a kernel launch of microseconds, and
# the cached path takes it once per call; there is no measurable cost to being correct here.
_LOCK = threading.RLock()


def fail(tag, msg):
    """Report once per tag, keep the reason. A fallback is not an error."""
    _S["why"] = msg
    if tag not in _S["warned"]:
        _S["warned"].add(tag)
        print(f"[{tag}] {msg}; falling back to the PyTorch reference", flush=True)
    return None


def main_process():
    """Rank 0, or any single-process run. The gate every routine line in this repository takes.

    THE GATE IS model.parallel.is_main() AND IT IS ASKED FOR HERE, NOT REIMPLEMENTED -- but it
    is asked for INSIDE THE FUNCTION and with a fallback, because this module cannot depend on
    it at import time. model/mira.py imports the wrappers that import this file, so a top-level
    `from model.parallel import is_main` here is a cycle: `import model` would be part-way
    through executing when it reached us. By the time a cubin is loaded -- the first forward --
    the model package has finished importing and the call is ordinary.

    THE FALLBACK IS THE SAME TEST, NOT A DIFFERENT ONE. model.parallel.is_main() is
    `dist_env()[0] == 0`, and dist_env() reads RANK from torchrun's environment and answers 0
    when it is absent. That is what the `except` branch does, so the two cannot disagree; it
    exists for the case where this file is used without the model package around it, which is
    every `make check` invocation and every unit test.

    helpers/common._say is the gate for the STAGES, and it is not usable here for the same
    reason and one more: it imports model, config and the tokenizer.
    """
    try:
        from model.parallel import is_main
        return is_main()
    except Exception:                                          # noqa: BLE001
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except ValueError:
            return True


def cubin_dir():
    """Where the cubins are, with the former environment names as read-only aliases."""
    for var in ("MIRA_CUDA_KERNEL", "MIRALLM_MIRA_CUBIN",
                "CDSA_CUDA_KERNEL", "NULLGPT_CDSA_CUBIN"):
        env = os.environ.get(var)
        if env is None:
            continue
        if not env.strip():
            return None                                       # explicitly disabled
        p = env if os.path.isabs(env) else os.path.join(_ROOT, env)
        return os.path.dirname(p) if not os.path.isdir(p) else p
    for d in _DIRS:
        if os.path.isdir(d):
            return d
    return None


def _cubin(name):
    """Full path of `name`, preferring a build for this GPU's architecture."""
    d = cubin_dir()
    if d is None:
        return None
    stem = name[:-len(".cubin")] if name.endswith(".cubin") else name
    cands = []
    # The configured directory is normally the family root (build/kernel/cuda), but older
    # deployments sometimes point it directly at build/kernel/cuda/cdsa. Try both the
    # family-qualified path and its basename so those already-built cubins remain loadable.
    base = os.path.basename(stem)
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        cands.append(f"{stem}.sm_{major}{minor}.cubin")        # this GPU's own build wins
        if base != stem:
            cands.append(f"{base}.sm_{major}{minor}.cubin")
    except Exception:                                          # noqa: BLE001
        pass
    cands.append(f"{stem}.cubin")
    if base != stem:
        cands.append(f"{base}.cubin")
    for c in cands:
        p = os.path.join(d, c)
        if os.path.isfile(p):
            return p
    return None


def legacy_cubin_name(name):
    """The pre-rename cubin path corresponding to a canonical MIRA path, if any."""
    if name.startswith("mira/"):
        return "cdsa/" + name[len("mira/"):].replace("mira_", "cdsa_", 1)
    return None


def _init():
    """Open libcuda once. True when CUDA is usable at all.

    THE WHOLE BODY IS UNDER THE LOCK, not just the flag. Setting tried=True and then doing the
    slow work let a second thread see "already tried" and read ok=False."""
    with _LOCK:
        return _init_locked()


def _init_locked():
    if _S["tried"]:
        return bool(_S["ok"])
    _S["tried"] = True
    enabled = os.environ.get("MIRALLM_MIRA_CUDA")
    enabled_by = "MIRALLM_MIRA_CUDA"
    if enabled is None:
        enabled = os.environ.get("NULLGPT_CDSA_CUDA", "1")
        enabled_by = "NULLGPT_CDSA_CUDA"
    if enabled == "0":
        _S["why"] = f"disabled by {enabled_by}=0"
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            _S["why"] = "no CUDA device"
            return False
        torch.cuda.init()
        torch.zeros(1, device="cuda")
    except Exception as e:                                     # noqa: BLE001
        _S["why"] = f"no CUDA ({type(e).__name__}: {e})"
        return False
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            lib = ctypes.CDLL("libcuda.so")
        except OSError as e:
            _S["why"] = f"libcuda not loadable ({e})"
            return False
    lib.cuInit(0)
    lib.cuLaunchKernel.restype = ctypes.c_int
    # cuLaunchKernel(CUfunction f, unsigned gridX, gridY, gridZ, unsigned blockX, blockY, blockZ,
    #                unsigned sharedBytes, CUstream stream, void** params, void** extra)
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    # cuOccupancyMaxPotentialBlockSize(int* minGrid, int* block, CUfunction f,
    #                                  CUoccupancyB2DSize b2d, size_t dynSmem, int limit)
    # Present in every CUDA driver since 6.5. Declared defensively: a driver without it leaves
    # block_for() on the THREADS fallback rather than raising at import.
    try:
        lib.cuOccupancyMaxPotentialBlockSize.restype = ctypes.c_int
        lib.cuOccupancyMaxPotentialBlockSize.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    except AttributeError:
        pass
    # cuOccupancyMaxActiveBlocksPerMultiprocessor(int* n, CUfunction f, int blockSize,
    #                                             size_t dynSmem)
    # and cuLaunchCooperativeKernel, both needed by resident_blocks()/launch_coop() and by
    # nothing else. Declared defensively for the same reason as the query above: a driver
    # without either leaves resident_blocks() answering 0, which its one caller reads as
    # "refuse to launch" rather than as a number to guess around.
    for _n, _rt in (("cuOccupancyMaxActiveBlocksPerMultiprocessor", ctypes.c_int),
                    ("cuLaunchCooperativeKernel", ctypes.c_int),
                    ("cuFuncSetAttribute", ctypes.c_int),
                    ("cuDeviceGetAttribute", ctypes.c_int)):
        try:
            getattr(lib, _n).restype = _rt
        except AttributeError:
            pass
    # cuFuncGetAttribute(int *pi, CUfunction_attribute attrib, CUfunction hfunc)
    #
    # THE ARGUMENT ORDER IS THE REVERSE OF cuFuncSetAttribute(hfunc, attrib, value), declared in
    # the loop above: the OUT POINTER IS FIRST and the function handle is LAST. The two are not
    # interchangeable and swapping them hands a 64-bit function handle to a parameter the driver
    # reads as an int attribute index, which is not a diagnosable failure -- it is a number.
    # argtypes are therefore stated rather than left to ctypes' default conversion, so a caller
    # that passes them in the setter's order raises here instead of returning garbage.
    #
    # Declared defensively for the same reason as the occupancy queries above: a driver without
    # the symbol leaves report() printing "unavailable" per field rather than raising at import.
    try:
        lib.cuFuncGetAttribute.restype = ctypes.c_int
        lib.cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                           ctypes.c_void_p]
    except AttributeError:
        pass
    _S.update(lib=lib, ok=True)
    return True


def module_for(cubin, index, names):
    """{name: CUfunction} plus the context, for `cubin` on device `index`. None on failure.

    Cached per (cubin, device). Loaded INSIDE torch.cuda.device(index) so that device's primary
    context is current; the context is kept so launch() can restore it on any thread.

    A (cubin, device, names) triple already resolved returns without the lock. The cached entry
    keeps a reference to the name tuple it was resolved for and is used only when the caller
    passes that same object, so a second wrapper with its own names resolves under the lock."""
    cached = _RESOLVED.get((cubin, index, id(names)))
    if cached is not None and cached[0] is names:
        return cached[1]
    with _LOCK:
        got = _module_for_locked(cubin, index, names)
    if got:
        _RESOLVED[(cubin, index, id(names))] = (names, got)
    return got


def _module_for_locked(cubin, index, names):
    if not _init_locked():
        return None
    key = (cubin, index)
    got = _S["mods"].get(key)
    # A REFUSAL IS REMEMBERED. `False` means this cubin was tried on this device and could not
    # be used; the answer cannot change within a process, so nothing below is worth repeating.
    # Every failure path sets it, because the work being skipped is not small: a device
    # allocation, a cuModuleLoad of the entire multi-target fatbin from disk, and up to 128
    # symbol lookups. Each entry point asks twice per invocation -- once through status(), once
    # through the launch -- so without the sentinel one forward pass over a cubin that cannot
    # load costs hundreds of full fatbin loads AND still runs the Python reference, while the
    # warning that explains it is printed once and scrolls away.
    if got is False:
        return None
    if got is not None:
        missing = [n for n in names if n not in got]
        if missing and got.get("__mod__") is not None:
            for n in missing:
                fn = ctypes.c_void_p()
                rc = _S["lib"].cuModuleGetFunction(ctypes.byref(fn), got["__mod__"], n.encode())
                if rc != 0 and n.startswith("mira_"):
                    rc = _S["lib"].cuModuleGetFunction(
                        ctypes.byref(fn), got["__mod__"], ("cdsa_" + n[len("mira_"):]).encode())
                if rc != 0:
                    return fail(cubin.split(".")[0],
                                f"{n} not found in {os.path.basename(got['__path__'])}")
                got[n] = fn
        return got
    path = _cubin(cubin)
    legacy_cubin = legacy_cubin_name(cubin)
    if path is None and legacy_cubin is not None:
        path = _cubin(legacy_cubin)
    if path is None:
        _S["mods"][key] = False
        return fail(cubin.split(".")[0],
                    f"{cubin} not found -- build it: make -f kernel/cuda/<family>/Makefile universal")
    import torch
    lib = _S["lib"]
    try:
        with torch.cuda.device(index):
            torch.zeros(1, device=f"cuda:{index}")             # force the primary context
            ctx = ctypes.c_void_p()
            lib.cuCtxGetCurrent(ctypes.byref(ctx))
            mod = ctypes.c_void_p()
            rc = lib.cuModuleLoad(ctypes.byref(mod), path.encode())
            if rc != 0:
                extra = " (built for a different -arch?)" if rc == 209 else ""
                _S["mods"][key] = False
                return fail(cubin.split(".")[0],
                            f"cuModuleLoad {os.path.basename(path)} on cuda:{index} rc={rc} "
                            f"{CU_ERR.get(rc, 'UNKNOWN')}{extra}")
            fns = {}
            for n in names:
                fn = ctypes.c_void_p()
                rc = lib.cuModuleGetFunction(ctypes.byref(fn), mod, n.encode())
                # A cubin built before the rename exports cdsa_* symbols. The tensor ABI and
                # the state_dict are unchanged, so resolve the former symbol when the loader
                # selected the former file; new builds export the canonical mira_* name.
                if rc != 0 and n.startswith("mira_"):
                    old_name = "cdsa_" + n[len("mira_"):]
                    rc = lib.cuModuleGetFunction(ctypes.byref(fn), mod, old_name.encode())
                if rc != 0:
                    # THE MODULE IS RELEASED BEFORE GIVING UP. A stale cubin fails on one name
                    # out of many; without this unload the device kept a live CUmodule for every
                    # attempt, so the leak grew with the call count rather than staying at one.
                    try:
                        lib.cuModuleUnload(mod)
                    except Exception:                          # noqa: BLE001
                        pass                                   # the report below is what matters
                    _S["mods"][key] = False
                    return fail(cubin.split(".")[0],
                                f"{n} not found in {os.path.basename(path)} -- the cubin is "
                                f"stale; rebuild it: make -f kernel/cuda/<family>/Makefile "
                                f"clean universal")
                fns[n] = fn
    except Exception as e:                                     # noqa: BLE001
        _S["mods"][key] = False
        return fail(cubin.split(".")[0],
                    f"loading {cubin} on cuda:{index} failed ({type(e).__name__}: {e})")
    # __mod__ IS KEPT so a later caller asking for other symbols in the same file can
    # resolve them against the module that is already loaded. See the cache hit above.
    fns["__ctx__"], fns["__path__"], fns["__mod__"] = ctx, path, mod
    _S["mods"][key] = fns
    if main_process():
        print(f"[{cubin.split('.')[0]}] loaded {path} successfully", flush=True)
    return fns


def _occupancy(fn, ctx, index):
    """(block, fill) from ONE cuOccupancyMaxPotentialBlockSize call. Cached per (kernel, device).

    The driver answers two questions at once and this file only ever used the first:

        blockSize    the block that fills an SM best for this kernel -- block_for()
        minGridSize  how many such blocks it takes to fill EVERY SM -- fill_for()

    Their product is the thread count at which THIS kernel is fully resident on THIS card. It
    already accounts for the compiled register count and static shared memory, so it is well
    below SMs * 2048 for a register-heavy kernel and there is no way to derive it from device
    attributes alone. 0 when the query is unavailable, which callers must read as "unknown"
    rather than as a number.
    """
    key = (getattr(fn, "value", fn), index)
    hit = _S["occ"].get(key)
    if hit is not None:
        return hit
    n, fill = THREADS, 0
    try:
        lib = _S["lib"]
        lib.cuCtxSetCurrent(ctx)
        mg, bs = ctypes.c_int(0), ctypes.c_int(0)
        rc = lib.cuOccupancyMaxPotentialBlockSize(
            ctypes.byref(mg), ctypes.byref(bs), fn, None, ctypes.c_size_t(0), 0)
        # A WARP MULTIPLE AND AT LEAST ONE WARP. The driver already returns a multiple of 32,
        # but a 0 or a stray value from an old driver would make the grid arithmetic in
        # launch() divide by zero, and this is on every launch.
        if rc == 0 and bs.value >= 32 and bs.value % 32 == 0:
            n = min(bs.value, 1024)
            if mg.value > 0:
                fill = mg.value * bs.value
    except Exception:                                          # noqa: BLE001
        pass
    _S["occ"][key] = (n, fill)
    return n, fill


def fill_for(fn, ctx, index):
    """Threads needed to fill this device with THIS kernel, or 0 if the driver would not say.

    minGridSize * blockSize from the same query block_for() makes -- see _occupancy(). It is
    what a caller needs when it is deciding how much work to give one thread: giving each
    thread N times as much divides the thread count by N, and this is the floor that division
    must not go under. mira_fused_iir uses it to pick the conv backward's
    chunks-per-thread rather than hard-coding one, for the reason block_for exists at all --
    the cluster allocates whatever card is free and a constant is wrong on the next one.
    """
    return _occupancy(fn, ctx, index)[1]


def block_for(fn, ctx, index):
    """The block size the DRIVER says maximises occupancy for THIS kernel. Cached.

    cuOccupancyMaxPotentialBlockSize reads the compiled function's register count and static
    shared memory and returns the block size that fills an SM best. That is a property of the
    kernel and the card, and it is exactly the thing a hard-coded 256 cannot know: a kernel
    holding 64 registers for a chunk of state and one holding 12 do not want the same block.

    QUERIED ONCE PER (KERNEL, DEVICE) AND CACHED. It is a driver call, not a benchmark, so it
    costs microseconds -- but it is on the path of every launch, and every launch in this
    repository is already fighting for microseconds.

    IT IS AN OCCUPANCY ANSWER, NOT A SPEED ANSWER. Maximum occupancy is not always maximum
    throughput -- a memory-bound kernel can do better with fewer, fatter blocks. It is,
    however, a far better default than one number shared by seven kernels, and it moves with
    the card instead of being re-tuned by hand for each one. Override with MIRALLM_CUDA_BLOCK.

    AND THE CARD IS NOT FIXED HERE. This cluster allocates whatever is free -- A100, H100,
    H200, L40S -- and those differ in registers per SM, maximum warps per SM and SM count.
    A block size tuned on one is wrong on the next, and a table of them would have to be
    maintained by hand for every card the scheduler might hand out. The driver knows which
    card it is on; asking it costs one call per kernel per process.

    THE QUERY ITSELF MOVED to _occupancy(), which keeps both halves of the driver's answer;
    the override, the fallback and the value returned here are unchanged.
    """
    forced = os.environ.get("MIRALLM_CUDA_BLOCK",
                            os.environ.get("NULLGPT_CUDA_BLOCK", ""))
    if forced.isdigit() and int(forced) > 0:
        return int(forced)
    return _occupancy(fn, ctx, index)[0]


def scalar(x):
    """One non-pointer kernel argument: c_float for a Python float, c_int otherwise.

    THE C TYPE IS THE KERNEL'S, NOT PYTHON'S. A Python float is a C double; the kernels declare
    `const float`, and c_float is what matches. Every existing caller passes ints and is
    unaffected.
    """
    return ctypes.c_float(x) if isinstance(x, float) else ctypes.c_int(x)


def launch(fn, ctx, index, n_threads, pointers, ints, block=None):
    """cuLaunchKernel on device `index`. Returns (ok, message)."""
    import torch
    lib = _S["lib"]
    lib.cuCtxSetCurrent(ctx)                                   # current on THIS thread
    holders = [ctypes.c_void_p(p) for p in pointers] + [scalar(i) for i in ints]
    argv = (ctypes.c_void_p * len(holders))(
        *[ctypes.addressof(h) for h in holders])
    # Most kernels use the occupancy-selected block. A cross-channel reduction may require one
    # resident thread per channel; such a caller can request that exact block without changing
    # any existing launch, all of which leave ``block`` as None.
    nt = block_for(fn, ctx, index) if block is None else int(block)
    if nt < 1 or nt > 1024:
        return False, f"invalid explicit block size {nt}"
    blocks = (n_threads + nt - 1) // nt
    stream = ctypes.c_void_p(torch.cuda.current_stream(index).cuda_stream)
    rc = lib.cuLaunchKernel(fn, blocks, 1, 1, nt, 1, 1, 0, stream, argv, None)
    if rc == 0:
        return True, ""
    return False, (f"rc={rc} {CU_ERR.get(rc, 'UNKNOWN')} on cuda:{index} "
                   f"(grid {blocks}, block {nt})")


def prefer_shared(fn, ctx, shared=0, percent=CU_SHAREDMEM_CARVEOUT_MAX_SHARED):
    """Give THIS kernel as much of the SM's unified L1/shared as it will take. True if asked.

    WHY IT IS NOT AUTOMATIC. `__launch_bounds__(min_blocks)` is a compiler hint about registers;
    it does not reserve shared memory, and the number of blocks an SM actually holds is decided
    at launch from the carveout in force for that function. A kernel that asks for 36,096 B of
    dynamic shared runs perfectly well with one block per SM, so nothing fails and nothing warns
    -- it simply runs at a quarter of the warps it was written for, and on a kernel whose inner
    loop is a chain of dependent FFMAs that is most of the runtime. mira-lattice's fused forward
    is the caller; see the note beside the call in kernel/cuda/mira-vector/__init__.py.

    TWO ATTRIBUTES, ANSWERING DIFFERENT QUESTIONS, AND ONLY ONE OF THEM IS USUALLY NEEDED. The
    size opt-in is what makes a per-block request ABOVE 48 KB legal at all; the carveout is what
    makes an SM willing to host SEVERAL blocks that are each under it. THE SIZE OPT-IN IS SET
    ONLY WHEN IT IS THE ONE BEING ASKED FOR -- `shared` over 48 KiB -- because the attribute is a
    ceiling and setting it to a smaller number on a function that would otherwise get the
    architecture's default is a way to make a later, larger launch of the same function fail. A
    caller under 48 KB wants the carveout and nothing else.

    IT NEVER RAISES AND ITS RESULT IS ADVISORY. A driver without cuFuncSetAttribute, or a device
    that refuses the split, leaves the kernel launchable exactly as before -- so the one caller
    ignores the answer, and this returns a bool only so a test can see whether the ask happened.

    ONCE PER (KERNEL, SHARED SIZE). The attribute is a property of the CUfunction and persists,
    and the caller is on the path of every launch.
    """
    key = (getattr(fn, "value", fn), int(shared), int(percent))
    if key in _S["carve"]:
        return True
    try:
        lib = _S["lib"]
        if lib is None:
            return False
        lib.cuCtxSetCurrent(ctx)
        asked = False
        if int(shared) > 48 * 1024:
            asked |= lib.cuFuncSetAttribute(
                fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                ctypes.c_int(int(shared))) == 0
        asked |= lib.cuFuncSetAttribute(
            fn, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
            ctypes.c_int(int(percent))) == 0
    except Exception:                                          # noqa: BLE001
        return False
    if asked:
        _S["carve"].add(key)
    return asked


def _arch_note(binary_version, index):
    """" -- BUILT FOR sm_86, RUNNING ON sm_89" when the loaded image is older than the card.

    A CUBIN BUILT FOR AN OLDER ARCHITECTURE RUNS, AND THAT IS THE PROBLEM. Binary compatibility
    means an sm_86 image executes on Ada without complaint, so nothing fails and nothing warns --
    the kernel simply does not get the architecture's own SASS. Ada raises FP32 throughput over
    Ampere, and this family's inner loop is a chain of FP32 FMA, so a register count or a stall
    profile read off the Ampere image is being read off the wrong kernel.

    THE CAUSE IS ALMOST ALWAYS THE TOOLKIT, NOT THE BUILD RULE. kernel/cuda/arch.mk already asks
    for compute_89 and intersects the request with `nvcc --list-gpu-arch`; a toolkit older than
    CUDA 11.8 does not list it, so 89 is dropped and the PTX target falls back. `nvcc --version`
    is the thing to look at when this line appears.

    SILENT WHEN THERE IS NOTHING TO SAY, including when either number is unavailable: a note that
    appears on every launch is a note nobody reads."""
    try:
        if not binary_version:
            return ""
        major = device_attr(index, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0)
        minor = device_attr(index, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0)
        if not major:
            return ""
        device = 10 * major + minor
        if int(binary_version) >= device:
            return ""
        return (f"   -- BUILT FOR sm_{int(binary_version)}, RUNNING ON sm_{device}; "
                f"this card's SASS was not compiled, check nvcc --version")
    except Exception:                                          # noqa: BLE001
        return ""


def device_attr(index, attr, default=0):
    """One CUdevice_attribute of device `index`, or `default` when the driver will not say.

    NO CONTEXT IS NEEDED and none is set. cuDeviceGetAttribute reads a property of the DEVICE,
    not of a context or a loaded module, so this answers before any cubin has been opened -- and
    a launcher that must decide whether a tile is legal has to know that before it builds one.
    """
    try:
        lib = _S["lib"]
        if lib is None:
            return default
        out = ctypes.c_int(0)
        rc = lib.cuDeviceGetAttribute(
            ctypes.byref(out), ctypes.c_int(int(attr)), ctypes.c_int(int(index)))
        return default if rc != 0 else int(out.value)
    except Exception:                                          # noqa: BLE001
        return default


def max_optin_shared(index):
    """The largest per-block DYNAMIC shared allocation this device will grant after opting in.

    THE NUMBER A LAUNCHER MUST TEST ITS TILE AGAINST, and it is NOT 48 KiB. 48 KiB is the
    default a function is given when nobody calls cuFuncSetAttribute; the opt-in raises it to
    this, which is 99 KiB on Ada and 163 KiB on Hopper. A hard `if shared > 48 * 1024: refuse`
    therefore rejects tiles the card would have run, and rejects them on every card equally --
    which is how the fused forward came to be refused at 70,912 B on a device with 101,376 B
    available and the whole benchmark ran on the fallback instead.

    FALLS BACK TO THE DEFAULT, NOT TO THE MAXIMUM. A driver that cannot be queried is one whose
    opt-in cannot be relied on either, and 48 KiB is the size that needs no opt-in to work.
    """
    got = device_attr(index, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                      CU_DEFAULT_SHARED_PER_BLOCK)
    return got if got > 0 else CU_DEFAULT_SHARED_PER_BLOCK


_SM_COUNT = {}


def sm_count(index):
    """The device's SM count, or 0 for unknown. The divisor for "does this grid fill the card".

    MEMOISED PER DEVICE, ONCE PER PROCESS, BECAUSE THE LAUNCH PATH READS IT. A device's SM count
    cannot change while the process runs, and _inference_chunk calls this on EVERY forward to
    size the token chunk. Rules.md's no-search rule names this case exactly -- "no per-shape
    driver interrogation, however well cached" -- and an uncached one is the worse version of
    it: a cuDeviceGetAttribute on the critical path of every prefill.

    THE ZERO IS NOT CACHED. device_attr returns the default when the driver will not answer, and
    a transient refusal must not pin this device at "unknown" for the life of the process --
    where the caller's `max(1, ...)` would turn it into a one-chunk scan for every later call.
    """
    cached = _SM_COUNT.get(index)
    if cached:
        return cached
    value = device_attr(index, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0)
    if value:
        _SM_COUNT[index] = value
    return value


def function_attr(fn, ctx, attr, default=None):
    try:
        lib = _S["lib"]
        if lib is None:
            return default
        lib.cuCtxSetCurrent(ctx)
        value = ctypes.c_int(0)
        rc = lib.cuFuncGetAttribute(
            ctypes.byref(value), ctypes.c_int(int(attr)), fn)
        return default if rc != 0 else int(value.value)
    except Exception:
        return default


def resident_blocks(fn, ctx, index, block, shared=0):
    """How many blocks of `block` threads of THIS kernel are resident on THIS card AT ONCE.

    cuOccupancyMaxActiveBlocksPerMultiprocessor x the device's SM count. **0 WHEN THE DRIVER
    WILL NOT SAY, AND 0 MEANS REFUSE, NOT GUESS.**

    THIS EXISTS FOR ONE CALLER AND ONE REASON: a COOPERATIVE launch may not exceed the resident
    grid. cudaLaunchCooperativeKernel's contract is that every block of the grid is co-resident,
    because a grid-wide barrier waits for blocks that have not been scheduled yet -- so a grid
    one block too large does not run slowly, it HANGS. There is no device attribute from which
    that number can be derived by hand: it depends on the compiled kernel's register count and
    its static shared memory, which is precisely what this query reads.

    NOT CACHED, unlike _occupancy(). It is called once per Function, not once per launch, and
    the answer depends on `block` and `shared` as well as on the kernel.
    """
    try:
        lib = _S["lib"]
        lib.cuCtxSetCurrent(ctx)
        nb = ctypes.c_int(0)
        rc = lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(nb), fn, ctypes.c_int(int(block)), ctypes.c_size_t(int(shared)))
        if rc != 0 or nb.value <= 0:
            return 0
        sm = ctypes.c_int(0)
        # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
        rc = lib.cuDeviceGetAttribute(ctypes.byref(sm), 16, ctypes.c_int(int(index)))
        if rc != 0 or sm.value <= 0:
            return 0
        return nb.value * sm.value
    except Exception:                                          # noqa: BLE001
        return 0


def _blocks_per_sm(fn, ctx, block, shared=0):
    """cuOccupancyMaxActiveBlocksPerMultiprocessor ALONE. Blocks on ONE SM, or 0 for unknown.

    SEPARATE FROM resident_blocks() BECAUSE THAT ONE MULTIPLIES BY THE SM COUNT and answers a
    different question -- how large a cooperative grid may be -- so its number cannot be read as
    blocks per SM without knowing the divisor, and a report that divided would be reconstructing
    what this call already returns. resident_blocks() is on the precondition path of a
    cooperative launch that HANGS the device when the grid is one block too large; it is left
    exactly as it is rather than refactored onto this helper.

    IT DEPENDS ON THE FUNCTION'S CURRENT CARVEOUT, which is why report() asks twice.
    """
    try:
        lib = _S["lib"]
        if lib is None:
            return 0
        lib.cuCtxSetCurrent(ctx)
        nb = ctypes.c_int(0)
        rc = lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(nb), fn, ctypes.c_int(int(block)), ctypes.c_size_t(int(shared)))
        return nb.value if rc == 0 and nb.value > 0 else 0
    except Exception:                                          # noqa: BLE001
        return 0


def report(tag, fn, ctx, index, block, shared, name=""):
    """ONE BLOCK, ONCE PER PROCESS, ON RANK 0: what the driver GRANTED this kernel. True if said.

    EVERY PREDICTION ABOUT THE FUSED LATTICE KERNEL WAS MADE WITHOUT THESE NUMBERS. The launcher
    asks for a carveout and discards the answer, `__launch_bounds__` is a compiler hint and not a
    grant, and nothing in this tree has ever called cuFuncGetAttribute -- so the register count,
    the stack frame and the blocks per SM were all argued from the source rather than read off
    the loaded image. Two of them decide the kernel's speed outright:

        stack frame > 0     A SPILL. The dominant cost of the previous fused forward was 544 LDL
                            per thread per token out of a 256 B frame; the fix was compiling
                            coefficient_bottleneck in, and this field is what says whether it stayed fixed.
        blocks/SM 1 vs 2    4 warps per SM against 8, on an inner loop that is a chain of
                            dependent FFMAs. At FUSED_ROWS = 8 one block is already 8 warps, so
                            the second block is 16 against 8 -- but only if the carveout landed.

    THE BEFORE/AFTER PAIR IS THE POINT, and it reads:

        1 -> 2   the carveout was the constraint and the request worked
        2 -> 2   the carveout was never the constraint; the occupancy is register- or
                 block-limited and shared memory is not what to change
        1 -> 1   the request was ignored or the carveout level does not exist on this card, and
                 the kernel is running at the block shape's guaranteed residency and no more

    IT CALLS prefer_shared ITSELF, BECAUSE THAT FUNCTION IS CACHED ON (fn, shared, percent) and
    returns True from the cache without touching the driver on every call after the first. A
    report that measured, let the caller ask, and measured again would therefore print an
    identical pair -- the attribute would already have been set by the launch before it. So this
    runs BEFORE the caller's own prefer_shared, which then hits the cache and is a no-op; the
    grant is made exactly once either way.

    NOTHING HERE RAISES. Each field is an independent guarded call and prints "unavailable" on a
    driver that lacks the symbol, exactly as block_for() falls back to THREADS; a diagnostic that
    propagated would take down the training step or the profiler run it was measuring.

    ONCE PER PROCESS, KEYED ON `tag` alone rather than on the CUfunction, so a family whose
    symbols are instantiated per (order, coefficient_bottleneck, dtype) prints one block and not one per variant.
    `name` names the symbol that block describes. MIRALLM_CUDA_REPORT=0 suppresses it.
    """
    if os.environ.get("MIRALLM_CUDA_REPORT",
                      os.environ.get("NULLGPT_CUDA_REPORT", "1")) == "0":
        return False
    if not main_process():
        return False
    with _LOCK:
        if str(tag) in _S["reported"]:
            return False
        _S["reported"].add(str(tag))
        lib = _S["lib"]
    if lib is None:
        return False

    def attr(code):
        """One cuFuncGetAttribute, or None when the driver would not answer.

        THE POINTER IS FIRST AND THE FUNCTION LAST -- see the declaration in _init_locked. Every
        failure mode is one branch: no symbol raises AttributeError, a bad handle or an attribute
        this driver does not know returns nonzero, and both become None."""
        try:
            v = ctypes.c_int(0)
            rc = lib.cuFuncGetAttribute(ctypes.byref(v), ctypes.c_int(int(code)), fn)
            return v.value if rc == 0 else None
        except Exception:                                      # noqa: BLE001
            return None

    def show(v, unit=""):
        return "unavailable" if v is None else f"{v}{unit}"

    def carveout():
        """The percentage held, or what -1 means. A negative value is not a percentage: it is
        CU_SHAREDMEM_CARVEOUT_DEFAULT, the driver deciding for itself, and printing it as -1%
        would read as a refused request rather than as a request never made."""
        v = attr(CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT)
        if v is None:
            return "unavailable"
        return f"{v}%" if v >= 0 else "-1, the driver's own choice"

    try:
        lib.cuCtxSetCurrent(ctx)
    except Exception:                                          # noqa: BLE001
        pass
    block, shared = int(block), int(shared)
    before = _blocks_per_sm(fn, ctx, block, shared)
    try:
        granted = prefer_shared(fn, ctx, shared)
    except Exception:                                          # noqa: BLE001
        granted = False
    after = _blocks_per_sm(fn, ctx, block, shared)
    warps = f"{after * block // 32} warps/SM" if after and block >= 32 else "warps/SM unknown"
    head = f"{name} " if name else ""
    print("\n".join((
        f"[{tag}] driver report -- what the GPU granted {head}on cuda:{index}",
        f"[{tag}]   registers/thread          {show(attr(CU_FUNC_ATTRIBUTE_NUM_REGS))}",
        f"[{tag}]   stack frame               "
        f"{show(attr(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES), ' B')}   (nonzero is a SPILL)",
        f"[{tag}]   static shared             "
        f"{show(attr(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES), ' B')}",
        f"[{tag}]   dynamic shared requested  {shared} B   at {block} threads/block",
        f"[{tag}]   binary version            "
        f"{show(attr(CU_FUNC_ATTRIBUTE_BINARY_VERSION))}   (10*major + minor of the loaded "
        f"image){_arch_note(attr(CU_FUNC_ATTRIBUTE_BINARY_VERSION), index)}",
        f"[{tag}]   carveout held             {carveout()}",
        f"[{tag}]   blocks/SM                 {show(before) if before else 'unavailable'} -> "
        f"{show(after) if after else 'unavailable'}   ({warps}; prefer_shared returned "
        f"{granted})",
    )), flush=True)
    return True


def launch_coop(fn, ctx, index, grid, block, pointers, ints, shared=0):
    """cuLaunchCooperativeKernel -- the launch a grid-wide barrier requires. (ok, message).

    IDENTICAL TO launch_dims EXCEPT FOR THE ENTRY POINT IT CALLS, and it is a separate function
    rather than a flag on that one because the two have different PRECONDITIONS, not different
    parameters: this one demands that the whole grid be co-resident and the caller has to have
    established that with resident_blocks() before calling. A cooperative launch of an
    over-sized grid hangs the device; an ordinary launch of the same grid merely runs.

    A DRIVER WITHOUT cuLaunchCooperativeKernel RETURNS (False, ...) rather than falling back to
    cuLaunchKernel. Falling back would launch a kernel whose grid barrier cannot be satisfied,
    which is the hang this function exists to avoid.
    """
    import torch
    g = (grid, 1, 1) if isinstance(grid, int) else tuple(grid) + (1,) * (3 - len(grid))
    b = (block, 1, 1) if isinstance(block, int) else tuple(block) + (1,) * (3 - len(block))
    lib = _S["lib"]
    if not hasattr(lib, "cuLaunchCooperativeKernel"):
        return False, "driver has no cuLaunchCooperativeKernel"
    lib.cuCtxSetCurrent(ctx)                                   # current on THIS thread
    holders = [ctypes.c_void_p(p) for p in pointers] + [scalar(i) for i in ints]
    argv = (ctypes.c_void_p * len(holders))(
        *[ctypes.addressof(h) for h in holders])
    stream = ctypes.c_void_p(torch.cuda.current_stream(index).cuda_stream)
    rc = lib.cuLaunchCooperativeKernel(fn, g[0], g[1], g[2], b[0], b[1], b[2],
                                       shared, stream, argv)
    if rc == 0:
        return True, ""
    return False, (f"rc={rc} {CU_ERR.get(rc, 'UNKNOWN')} on cuda:{index} "
                   f"(grid {g}, block {b}, shared {shared}, cooperative)")


def launch_dims(fn, ctx, index, grid, block, pointers, ints, shared=0):
    """cuLaunchKernel with the grid and block SPECIFIED, not derived. Returns (ok, message).

    WHY THIS EXISTS ALONGSIDE launch(). `launch` takes a thread count and slices it into fixed
    blocks of THREADS, which is right for an elementwise kernel where a block means nothing.
    It is wrong for the RWKV kernels, where the BLOCK IS THE UNIT OF THE ALGORITHM: one block
    per (batch, head), one thread per channel within the head, and the threads of a block
    cooperate through shared memory and __syncthreads(). Reshaping that grid changes what the
    kernel computes, so the caller states it.

    `grid` and `block` are (x, y, z) tuples or plain ints meaning (x, 1, 1)."""
    import torch
    g = (grid, 1, 1) if isinstance(grid, int) else tuple(grid) + (1,) * (3 - len(grid))
    b = (block, 1, 1) if isinstance(block, int) else tuple(block) + (1,) * (3 - len(block))
    lib = _S["lib"]
    lib.cuCtxSetCurrent(ctx)                                   # current on THIS thread
    holders = [ctypes.c_void_p(p) for p in pointers] + [scalar(i) for i in ints]
    argv = (ctypes.c_void_p * len(holders))(
        *[ctypes.addressof(h) for h in holders])
    stream = ctypes.c_void_p(torch.cuda.current_stream(index).cuda_stream)
    rc = lib.cuLaunchKernel(fn, g[0], g[1], g[2], b[0], b[1], b[2], shared, stream, argv, None)
    if rc == 0:
        return True, ""
    return False, (f"rc={rc} {CU_ERR.get(rc, 'UNKNOWN')} on cuda:{index} "
                   f"(grid {g}, block {b}, shared {shared})")


def why():
    return _S.get("why") or "no CUDA device"
