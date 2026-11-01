"""kernel/cuda/upstream_extension.py -- build a VENDORED THIRD-PARTY kernel THEIR WAY.

    m = load("rwkv_v7", HERE, "kernel.cuda.rwkv_v7.upstream.op_x070",
             namespace="wind_backstepping")
    if m is None: ...                                  # why() says what happened

THIS FILE EXISTS BECAUSE THIS REPOSITORY AND UPSTREAM BUILD KERNELS DIFFERENTLY, and the
whole point of vendoring was to stop rewriting anyone's code to fit our habits.

    ours       nvcc -cubin ahead of time, into build/kernel/cuda/<family>/, loaded with
               cuModuleLoad through kernel/cuda/driver.py. No C++ compiler on the compute
               node, no PyTorch ABI, no rebuild when torch moves.
    theirs     torch.utils.cpp_extension.load() at import, which runs nvcc AND a host
               compiler right there, links against the running torch, and registers a
               TORCH_LIBRARY namespace.

THE CHOICE MADE HERE IS: LET THEIR LOADER RUN. Not because it is nicer -- it is worse for
this repository in every way our Makefiles were built to fix -- but because the alternative
is not "build their .cu into a cubin", it is "rewrite their kernel". Their entry points are
C++ templates instantiated inside host functions that take at::Tensor and compute their own
grid; `wkv6_op.cpp` is torch C++ end to end. A cubin needs extern "C" __global__ symbols with
raw pointers and a grid decided by the caller, which is a launcher we would have to write, in
their file, over their names. The comparison would then rest on OUR launcher again, which is
exactly the unfairness the vendoring was for.

WHAT THAT COSTS, said plainly so nobody discovers it in a queue: nvcc and a C++ toolchain must
be present AT RUN TIME on the node that trains an upstream family, the first forward pays a
one-off compile of a minute or two, and TORCH_EXTENSIONS_DIR must be writable. `make -C
kernel/cuda/<family> upstream` does that compile ahead of time so a job script can pay it on
a login node instead.

A MISSING KERNEL IS NOT AN ERROR HERE, WHICH IS THIS REPOSITORY'S RULE. Every failure is
caught, recorded and reported once; load() returns None. What the CALLER does with None is the
caller's business, and for the upstream RWKV families it is to raise, because upstream ships
no PyTorch fallback for those recurrences and a silent one would have to be written by us.

CWD IS SET FOR THE DURATION OF THE IMPORT, AND THAT IS ON PURPOSE. Upstream calls
`load(sources=['cuda/wkv7_cuda.cu', ...])` with paths relative to the working directory. The
vendored sources were therefore placed at <family>/upstream/cuda/ -- upstream's own layout --
so the only thing needed to make their line work is to be standing in the right place.
Rewriting the paths to absolute ones would have been two characters and would have stopped
their file diffing against the archive.

    THE CHDIR IS PROCESS-WIDE AND IS HELD UNDER THE LOCK. It lasts for one import and is
    restored in a finally. Under DDP each rank is its own process and construction and the
    first forward are single-threaded, so nothing else is looking; under nn.DataParallel the
    lock is what keeps two replica threads from interleaving two chdirs. A thread doing
    unrelated relative-path I/O during that window would still see the wrong directory, which
    is the honest limitation of doing it upstream's way.

ONE NAMESPACE, ONE CLAIMANT. TORCH_LIBRARY(name, m) may be executed ONCE per namespace per
process; a second registration is a hard error inside torch. RWKV-7's op and RWKV-8's op both
register `wind_backstepping` -- they are the same kernel in bf16 and in fp32 -- so those two
families cannot both hold a kernel in one interpreter. That is detected here and reported as
a sentence, rather than left to surface as a C++ exception halfway through a benchmark.
"""
import os
import threading

# The repository root: this file is kernel/cuda/upstream_extension.py, so three levels up.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# The architectures this project ships images for, kept identical to ARCHS in kernel/cuda/arch.mk
# so the two build paths fall back to the same card. Ascending; see _arch_flags.
#
#     80  A100                        the cluster
#     86  A10, RTX 3090               and the image Ada falls back to on an older toolkit
#     89  L40S, RTX 6000 Ada          native Ada, CUDA 11.8 and later
#     90  H100
_SHIP_ARCHES = (80, 86, 89, 90)

_ARCHES = {}                      # nvcc path -> the compute capabilities it lists
_HOSTS = {}                       # (nvcc, preferred) -> a host compiler nvcc can drive
_RELEASE = {}                     # nvcc path -> (major, minor) it reports, or None
_CHAIN = {}                       # (requested nvcc, preferred host) -> (nvcc, host)
_MEMFDS = []                      # in-memory copies held open for the process
_LOCK = threading.RLock()
_MODS = {}                        # key -> module, or None when it could not be built
_WHY = {}                         # key -> why not, or where it came from
_NS = {}                          # torch library namespace -> the key that claimed it
_SAID = set()


def _say(key, msg):
    """Report once per key, keep the reason. Mirrors driver.fail, deliberately: a reader
    should not have to learn two shapes of kernel-unavailable message."""
    _WHY[key] = msg
    if key not in _SAID:
        _SAID.add(key)
        print(f"[{key}] {msg}", flush=True)
    return None


def why(key):
    return _WHY.get(key, "not attempted")


def status(key):
    """(is it loaded, one line about it). Never builds anything -- see load()."""
    m = _MODS.get(key)
    if m is None:
        return False, why(key)
    return True, _WHY.get(key, "loaded")


def namespace_owner(ns):
    """Which key holds the TORCH_LIBRARY namespace `ns` in this process, or None."""
    return _NS.get(ns)


def build_torch_library(family, name, sources, cwd, extra_cuda_flags=(), python_module=False):
    """Compile upstream's TORCH_LIBRARY sources to a .so with nvcc and load it. No Ninja.

    torch.utils.cpp_extension.load() has no build backend but Ninja: _jit_compile writes a
    build.ninja and shells out to it, so a node without Ninja cannot load any upstream kernel
    even when nvcc and a host compiler are both present. That is the failure this replaces:

        RuntimeError: Ninja is required to load C++ extensions

    The sources register their ops with TORCH_LIBRARY, which means the resulting shared object
    registers exactly the same torch.ops.<ns>.<fn> entries whether Ninja built it or this did,
    and torch.ops.load_library binds it either way.

    COMPILE, THEN LINK -- THE SAME TWO STEPS NINJA WOULD RUN, and the reason is not tidiness.
    Building in ONE nvcc call means every flag intended for the host toolchain has to be
    smuggled through nvcc's forwarding syntax, and that syntax has its own rules:

        nvcc fatal : Unknown option '-Wl,-rpath,...'          -Wl is the linker's, not nvcc's
        gcc-11: error: unrecognized command-line option '-Wl' -Xcompiler SPLITS ON COMMAS, so
                                                             -Wl,-rpath,DIR arrives at gcc as
                                                             three separate words

    Each source therefore becomes an object -- .cu through nvcc, .cpp through the host compiler
    -- and the objects are linked by the HOST COMPILER, where -Wl,-rpath,DIR is ordinary and
    needs no forwarding at all. Upstream's nvcc flags reach only the .cu compile, which is where
    they were meant to go and where cpp_extension also puts them.

    The kernel is upstream's, unchanged. Only the build commands differ.

    Returns the path loaded. Raises on failure, and the caller reports it.
    """
    import sysconfig
    import torch
    from torch.utils import cpp_extension as _cpp

    # EVERY BUILD ARTEFACT STAYS INSIDE THE PROJECT. cpp_extension.load() writes to
    # TORCH_EXTENSIONS_DIR, which defaults to ~/.cache/torch_extensions -- outside the tree, per
    # user, and invisible to anyone reading the repository.
    out_dir = os.path.join(_ROOT, "build", "kernel", "cuda", family, "upstream")
    os.makedirs(out_dir, exist_ok=True)
    # IT STAYS HERE EVEN WHERE THE MOUNT REFUSES TO EXECUTE IT. See _load_shared: the object is
    # built and kept in the tree, and only its MAPPING is taken from memory when it has to be.
    target = os.path.join(out_dir, f"lib{name}.so")

    absolute = [s if os.path.isabs(s) else os.path.join(cwd, s) for s in sources]
    newest = max(os.path.getmtime(s) for s in absolute)
    # THE BUILD DIRECTORY IS INSIDE THE PROJECT AND THE PROJECT IS ON SHARED STORAGE, so the .so
    # a previous run left may have been built on a DIFFERENT NODE, for a different card, by a
    # different toolkit. Mtimes cannot see that. The stamp records what the artefact was built
    # from, and a mismatch rebuilds.
    stamp = target + ".built-with"
    requested_host = os.environ.get("CXX") or "c++"
    nvcc, host = _pick_toolchain(os.environ.get("NVCC"), requested_host, out_dir)
    # THE TORCH VERSION AND THE ABI ARE PART OF THE RECIPE, not decoration. This .so links
    # against libtorch by soname; upgrade the wheel and the cached library is built against
    # symbols the new one may not export, which surfaces as an undefined symbol at dlopen --
    # after nvcc reported success, on a cache hit, with the sources untouched so no mtime moved.
    recipe = repr((_arch_flags(nvcc), sorted(_split_flags(extra_cuda_flags)), sorted(absolute),
                   getattr(torch, "__version__", "?"),
                   int(bool(getattr(getattr(torch, "_C", None), "_GLIBCXX_USE_CXX11_ABI", True))),
                   os.path.realpath(nvcc), os.path.realpath(host)))
    fresh = os.path.isfile(target) and os.path.getmtime(target) > newest
    if fresh:
        try:
            with open(stamp) as handle:
                fresh = handle.read() == recipe
        except OSError:
            fresh = False
    if not fresh:
        # PYTHON'S OWN HEADERS ARE ON THIS LIST, and they are not optional. Upstream's op files
        # open with `#include <torch/extension.h>`, which reaches pybind11 and therefore
        # Python.h. torch's include_paths() does not return it; cpp_extension appends
        # sysconfig's separately, and so must this.
        includes = []
        for d in _cuda_include_paths(_cpp) + [sysconfig.get_path("include")]:
            if d:
                includes += ["-I", d]
        libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
        defines = _torch_defines(torch, _cpp, name)
        # EVERY INTERMEDIATE IS NAMED FOR THIS PROCESS, AND THE RESULT IS MOVED INTO PLACE AT THE
        # END. Under torchrun every rank is a separate process running this same code against
        # this same path; without that, four ranks write one .so at once and some of them load a
        # half-written file. os.replace is atomic within a filesystem, so a rank either sees the
        # old library or the complete new one, and the duplicated compiles cost seconds once.
        mine = f".{os.getpid()}"
        staged = target + mine
        objects = []
        arch = _arch_flags(nvcc)
        try:
            for source in absolute:
                obj = os.path.join(out_dir, os.path.basename(source) + mine + ".o")
                objects.append(obj)
                if source.endswith(".cu"):
                    step, tool = [nvcc, *arch, "-ccbin", host, "-c", "-O3", "-std=c++17",
                                  "--compiler-options", "-fPIC", *defines,
                                  *_split_flags(extra_cuda_flags), *includes,
                                  source, "-o", obj], nvcc
                else:
                    # UPSTREAM'S CUDA FLAGS ARE NOT PASSED HERE. -res-usage and --use_fast_math
                    # are nvcc's; handing them to g++ is the mirror image of the -Wl mistake
                    # above. Only the -D defines cross over.
                    step, tool = [host, "-c", "-O3", "-std=c++17", "-fPIC",
                                  *[d for d in defines if d.startswith("-D")],
                                  *includes, source, "-o", obj], host
                _run(step, tool, name, nvcc)

            # LINKED BY THE HOST COMPILER, so -Wl,-rpath,DIR is just a linker flag. The rpath is
            # what lets the finished .so find libtorch without LD_LIBRARY_PATH set by whoever
            # launches the job.
            #
            # -ltorch_python IS ON THIS LINE BECAUSE cpp_extension PUTS IT THERE, for every
            # extension that is not standalone. Upstream's op file opens with
            # `#include <torch/extension.h>`, which reaches torch/python.h; leaving the library
            # out links cleanly and then fails at dlopen with an undefined symbol, which is the
            # worst place to discover it.
            link = [host, "-shared", *objects, "-L", libdir,
                    "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
                    "-ltorch_python", f"-Wl,-rpath,{libdir}"]
            # -lcudart ONLY WHEN A TOOLKIT LIBRARY DIRECTORY WAS FOUND. torch's own lib/ ships
            # libcudart as a versioned file with no development symlink, so asking the linker
            # for -lcudart with only that directory on the path fails to find it. When torch
            # cannot name a CUDA_HOME the link goes without, and the runtime it resolves against
            # is torch's own.
            cudalib = _cuda_lib_dir(_cpp)
            if cudalib:
                link += ["-L", cudalib, f"-Wl,-rpath,{cudalib}", "-lcudart"]
            link += ["-o", staged]
            _run(link, host, name, nvcc)

            # A LINK THAT RETURNED ZERO CAN STILL HAVE PRODUCED A FILE THAT WILL NOT LOAD --
            # a full disk truncates without an error anywhere. Checked here, where the tool
            # that wrote it can still be named, rather than at dlopen as "failed to map
            # segment", which says nothing about which of the two causes it was.
            with open(staged, "rb") as handle:
                magic = handle.read(4)
            if magic != b"\x7fELF":
                raise RuntimeError(f"{host} produced {staged}, which is not an ELF object "
                                   f"({len(magic)} bytes read). The filesystem holding "
                                   f"{out_dir} is the first thing to check.")
            os.replace(staged, target)
            with open(stamp + mine, "w") as handle:
                handle.write(recipe)
            os.replace(stamp + mine, stamp)
        finally:
            for leftover in objects + [staged, stamp + mine]:
                try:
                    os.remove(leftover)
                except OSError:
                    pass
    return _load_shared(target, name, python_module)


def _load_shared(target, name, python_module):
    """Load the built object, FROM MEMORY when the filesystem refuses to execute it.

    A .so is not merely read at load, it is mmap'd WITH PROT_EXEC, and a filesystem mounted
    `noexec` refuses that. glibc reports it in terms that name neither the mount nor the reason:

        OSError: /data/.../build/kernel/cuda/rwkv_v6/upstream/libwkv6.so: failed to map segment
        from shared object

    Shared project and scratch filesystems are very often mounted noexec, and nothing before
    this point notices: nvcc compiles, the linker links, the ELF check passes, and only dlopen
    fails. Python itself keeps working from such a mount because the interpreter READS .py files
    rather than mapping them executable, so the whole run looks healthy until the first kernel.

    THE ARTEFACT STILL LIVES IN THE PROJECT, WHICH IS NOT NEGOTIABLE. The previous answer to a
    noexec mount was to rebuild somewhere else -- ~/.cache, /tmp -- and that is forbidden:

    So nothing moves. The object stays where it was built, where it can be read, cleaned and
    inspected, and only the MAPPING comes from somewhere else: the bytes are copied into an
    anonymous in-memory file and dlopen'd through /proc/self/fd/N. memfd has no mount and
    therefore no noexec, nothing is written outside the tree -- nothing is written outside this
    process -- and the file on disk is untouched.

    VERIFIED AGAINST A REAL noexec MOUNT, both paths, 2026-09-14: a bind mount remounted noexec
    reproduced "failed to map segment from shared object" verbatim for ctypes.CDLL and for an
    extension import, and both loaded from the memfd copy afterwards.

    torch.ops.load_library IS NOT USED ON THE FALLBACK and cannot be: it calls realpath on what
    it is given, and the realpath of /proc/self/fd/N is "/memfd:<name> (deleted)", which does not
    exist. It does nothing else that matters -- the TORCH_LIBRARY registrations happen in the
    object's own static initialisers as it is mapped -- so the fallback maps it with ctypes
    directly, under torch's dl_open_guard so RTLD_GLOBAL is set exactly as load_library sets it.

    importlib.util.spec_from_file_location IS NOT USED ON THE FALLBACK either, for a smaller
    reason: it picks a loader by the path's SUFFIX, and /proc/self/fd/3 has none, so it returns
    None. The ExtensionFileLoader it would have chosen is named directly instead.
    """
    try:
        return _load_from(target, target, name, python_module)
    except (OSError, ImportError) as e:
        if "failed to map segment" not in str(e) and "cannot map" not in str(e):
            raise
        fd = getattr(os, "memfd_create", None)
        if fd is None:                        # not Linux, or a Python without memfd
            raise OSError(
                f"{e}. The filesystem holding {os.path.dirname(target)} is mounted noexec, so a "
                f"shared object cannot be mapped executable from it, and this interpreter has no "
                f"memfd to map it from memory instead. Mount the project without noexec.")
        handle = fd(os.path.basename(target), 0)
        with open(target, "rb") as source:
            while True:
                block = source.read(1 << 20)
                if not block:
                    break
                os.write(handle, block)
        _MEMFDS.append(handle)                # held open for the process: the module's __file__
        os.lseek(handle, 0, os.SEEK_SET)      # points at it, and closing would invalidate that
        print(f"[upstream] {os.path.dirname(target)} is on a filesystem that refuses to map a "
              f"shared object executable (a noexec mount); {os.path.basename(target)} stays "
              f"there and is loaded from memory instead", flush=True)
        return _load_from(f"/proc/self/fd/{handle}", target, name, python_module)


def _load_from(path, target, name, python_module):
    """One load, from wherever `path` says. `target` is only ever used in messages."""
    import torch
    if python_module:
        import importlib.machinery
        import importlib.util
        import sys
        loader = importlib.machinery.ExtensionFileLoader(name, path)
        spec = importlib.util.spec_from_loader(name, loader, origin=path)
        if spec is None:
            raise ImportError(f"cannot load Python extension {name} from {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module
    if path == target:
        torch.ops.load_library(target)
    else:
        import ctypes
        guard = getattr(getattr(torch, "_ops", None), "dl_open_guard", None)
        if guard is None:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        else:
            with guard():
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    return target


def _pick_host(nvcc, preferred, scratch):
    """A host compiler THIS nvcc can actually drive, found by trying one.

    A CUDA toolkit older than the system's GCC does not merely warn -- its front end fails to
    parse that GCC's libstdc++, and the message names neither compiler:

        /usr/include/c++/11/bits/std_function.h:435: error: parameter packs not expanded
        with '...':   'ArgTypes'

    That is nvcc 11.5 reading GCC 11's <functional>, reached through torch's headers. No flag
    fixes it: the fix is a host compiler the toolkit predates, or a newer toolkit.

    THE TEST IS A COMPILE, NOT A VERSION TABLE. A table of which CUDA tolerates which GCC is
    wrong somewhere and goes stale; a two-line translation unit that includes <functional>
    reproduces the failure exactly and costs a second. The candidates are tried newest first,
    so a working default is kept and an older compiler is reached for only when needed.

    Returns the compiler to use. Raises when none of them works, saying what to install.
    """
    import shutil
    import subprocess
    if (nvcc, preferred) in _HOSTS:
        return _HOSTS[(nvcc, preferred)]

    candidates = [preferred]
    candidates += [f"g++-{major}" for major in range(13, 6, -1)]
    candidates += ["g++", "clang++"]
    seen, ordered = set(), []
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved and resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)

    preferred_path = shutil.which(preferred)
    probe = os.path.join(scratch, f"hostprobe.{os.getpid()}.cu")
    with open(probe, "w") as handle:
        handle.write("#include <functional>\n"
                     "__global__ void k(){}\n"
                     "int probe(){ std::function<int(int)> f=[](int x){return x;}; "
                     "return f(0); }\n")
    tried = []
    try:
        for candidate in ordered:
            try:
                done = subprocess.run(
                    [nvcc, "-ccbin", candidate, "-std=c++17", "-c", probe,
                     "-o", probe + ".o"],
                    capture_output=True, text=True, timeout=60)
            except (OSError, subprocess.TimeoutExpired):
                tried.append(candidate)
                continue
            if done.returncode == 0:
                _HOSTS[(nvcc, preferred)] = candidate
                if candidate != preferred_path:
                    print(f"[upstream] {os.path.basename(nvcc)} cannot compile the C++17 "
                          f"standard library with {preferred}; building with {candidate} "
                          f"instead", flush=True)
                return candidate
            tried.append(candidate)
    finally:
        for leftover in (probe, probe + ".o"):
            try:
                os.remove(leftover)
            except OSError:
                pass
    raise RuntimeError(
        "no host compiler on this node can compile the required CUDA/C++17 source with " +
        nvcc + " (tried " +
        ", ".join(tried or ["none found"]) + "). Install a g++ the toolkit supports, or "
        "point NVCC at a newer toolkit" + _nvcc_note(nvcc))


def _nvcc_release(nvcc):
    """(major, minor) this nvcc reports, or None when it cannot be run. Cached per path."""
    import re
    import subprocess
    if nvcc in _RELEASE:
        return _RELEASE[nvcc]
    found = None
    try:
        done = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60)
        match = re.search(r"release (\d+)\.(\d+)", done.stdout or "")
        if done.returncode == 0 and match:
            found = (int(match.group(1)), int(match.group(2)))
    except Exception:                                             # noqa: BLE001
        found = None
    _RELEASE[nvcc] = found
    return found


def _nvcc_candidates():
    """Every nvcc on this node worth trying, best first.

    WHY PATH IS NOT ENOUGH, AND WHY THIS FUNCTION EXISTS. The toolkit that compiles and the
    runtime that is linked were chosen by two different mechanisms: nvcc came from `which nvcc`
    while the -L came from cpp_extension.CUDA_HOME (see _cuda_lib_dir). On a node where the
    distribution ships its own nvcc those are DIFFERENT TOOLKITS, and the compile is the one
    that loses:

        [rwkv_v6_upstream] building upstream's kernel failed (RuntimeError: no host compiler on
        this node can compile the required CUDA/C++17 source with /usr/bin/nvcc (tried
        /usr/bin/c++, /usr/bin/g++-11, /usr/bin/g++) ... release 11.5, targets
        compute_35..compute_87, card is sm_89)

    That node had a cu12 torch wheel and an apt nvcc 11.5. 11.5 predates both the host GCC and
    the card, so every host compiler present failed its probe -- while the 12.x toolkit torch
    itself was built against sat unused, one directory away, supporting both.

    THE RULE IS THE TOOLKIT torch WAS BUILT WITH, FIRST. That is the one whose runtime the .so
    links against, so matching it is not merely newer, it is consistent: torch.version.cuda
    names it and cpp_extension.CUDA_HOME usually holds it. Everything else follows in descending
    release order, because a newer toolkit understands a newer host compiler and a newer card,
    and PATH is simply one more candidate rather than the only one.

    NVCC IN THE ENVIRONMENT IS NOT RANKED, it is obeyed -- _pick_toolchain handles that before
    calling here. The note every build failure carries already tells the reader to set it.
    """
    import glob
    import shutil
    seen, found = set(), []

    def offer(path):
        if not path:
            return
        resolved = shutil.which(path) if not os.path.isabs(path) else path
        if not resolved or not os.path.isfile(resolved) or resolved in seen:
            return
        seen.add(resolved)
        found.append(resolved)

    homes = []
    try:
        from torch.utils import cpp_extension as _cpp
        homes.append(getattr(_cpp, "CUDA_HOME", None))
    except Exception:                                             # noqa: BLE001
        pass
    homes += [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    for home in homes:
        if home:
            offer(os.path.join(home, "bin", "nvcc"))
    # /usr/local/cuda-12.4 before /usr/local/cuda-11.8, and the unversioned symlink last of the
    # three because it is whichever of them someone pointed it at.
    versioned = sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"),
                       key=lambda p: _release_key(p), reverse=True)
    for path in versioned:
        offer(path)
    offer("/usr/local/cuda/bin/nvcc")
    offer("nvcc")
    # RANKED, NOT ORDERED BY WHERE THEY WERE FOUND. The toolkit torch names comes first; the
    # rest descend by release. A candidate that will not run at all sorts last and is still
    # tried, because a failure to run --version is not proof it cannot compile.
    want = None
    try:
        import torch
        version = getattr(getattr(torch, "version", None), "cuda", None)
        if version:
            parts = str(version).split(".")
            want = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except Exception:                                             # noqa: BLE001
        want = None

    def rank(path):
        release = _nvcc_release(path)
        if release is None:
            return (0, (0, 0))
        if want is not None and release == want:
            return (3, release)
        if want is not None and release[0] == want[0]:
            return (2, release)
        return (1, release)

    found.sort(key=rank, reverse=True)
    return found


def _release_key(path):
    """(major, minor) parsed from a /usr/local/cuda-X.Y path, for ordering the glob."""
    import re
    match = re.search(r"cuda-(\d+)(?:\.(\d+))?", path)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2) or 0))


def _pick_toolchain(requested_nvcc, preferred_host, scratch):
    """(nvcc, host compiler) -- a PAIR that works, rather than an nvcc and then a host.

    THE TWO CHOICES ARE NOT INDEPENDENT AND WERE MADE AS THOUGH THEY WERE. An nvcc was taken
    from PATH, and only then was a host compiler looked for that it could drive; when none
    existed the build failed there, with a toolkit that was never the only one on the node. The
    host probe is what decides whether a toolkit is usable, so it belongs inside the search for
    one: a toolkit no host compiler can drive is a toolkit this function moves past.

    An explicit NVCC is obeyed and not searched around. Someone who set it wants that toolkit,
    and silently building with another would be the same defect in the other direction -- so a
    host compiler that cannot drive it is reported against it, exactly as before.
    """
    import shutil
    key = (requested_nvcc, preferred_host)
    if key in _CHAIN:
        return _CHAIN[key]
    if requested_nvcc:
        nvcc = shutil.which(requested_nvcc) or requested_nvcc
        chain = (nvcc, _pick_host(nvcc, preferred_host, scratch))
        _CHAIN[key] = chain
        return chain
    candidates = _nvcc_candidates() or ["nvcc"]
    failures = []
    for nvcc in candidates:
        try:
            host = _pick_host(nvcc, preferred_host, scratch)
        except RuntimeError as e:                                 # noqa: BLE001
            failures.append(str(e))
            continue
        # THE TOOLCHAIN IS NAMED IN THE LOG, ALWAYS. Which nvcc built a kernel decides whether
        # it runs as SASS or as JIT'd PTX, and it was previously nowhere in the output -- the
        # first anyone heard of the toolkit was in the failure message.
        device = _detected_arch()
        known = _nvcc_arches(nvcc)
        native = device is None or not known or device in known
        print(f"[upstream] building with {nvcc} and {host}"
              + ("" if native else f"; it cannot target sm_{device}, so PTX is emitted and the "
                                   f"driver JITs it"), flush=True)
        _CHAIN[key] = (nvcc, host)
        return _CHAIN[key]
    raise RuntimeError(
        "no CUDA toolkit on this node can be driven by any host compiler here. Tried "
        + ", ".join(candidates) + ". " + (failures[0] if failures else "")
        + " Set NVCC=/path/to/nvcc to name one, or install a g++ a present toolkit supports.")


def _run(command, tool, name, nvcc):
    """Run one build step. On failure, raise with the tail of what the tool actually said.

    THE TOOL THAT FAILED IS NAMED, because the two in play reject each other's flags and the
    message is the only way to tell which one is complaining.
    """
    import subprocess
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-6:]
        raise RuntimeError(f"{os.path.basename(tool)} failed building {name}: "
                           + " | ".join(tail) + _nvcc_note(nvcc))
    return done


def _cuda_lib_dir(cpp_extension):
    """The CUDA toolkit's library directory, or "" when it cannot be located.

    torch knows where its toolkit is; asking it avoids a guess about lib vs lib64 and about
    whether CUDA_HOME is set in this shell. Empty means the link simply omits -L for it, which
    is right when the runtime is the one bundled in torch's own lib directory.
    """
    home = getattr(cpp_extension, "CUDA_HOME", None)
    if not home:
        return ""
    for leaf in ("lib64", "lib"):
        candidate = os.path.join(home, leaf)
        if os.path.isdir(candidate):
            return candidate
    return ""


def _torch_defines(torch, cpp_extension, name):
    """The defines torch's own builder puts on every CUDA extension compile.

    UPSTREAM'S KERNEL WAS WRITTEN AGAINST THIS EXACT SET, because upstream builds through
    cpp_extension, which adds them unconditionally. Leaving them out does not produce a purer
    build of their code -- it produces a DIFFERENT one, compiled against half and bfloat16
    headers that behave differently from the ones they tested with.

        COMMON_NVCC_FLAGS           the __CUDA_NO_*_OPERATORS__ family. They suppress the
                                    implicit conversions and operators in cuda_fp16.h and
                                    cuda_bf16.h that clash with c10's own, which is why every
                                    torch extension in existence is built with them.
        _GLIBCXX_USE_CXX11_ABI      MUST MATCH THE WHEEL. libtorch is compiled with one value of
                                    it; a .so compiled with the other links against
                                    std::__cxx11::basic_string where libtorch exports
                                    std::basic_string, and the mismatch surfaces as undefined
                                    symbols at load, long after nvcc has reported success.
        TORCH_API_INCLUDE_EXTENSION_H
        TORCH_EXTENSION_NAME        what torch/extension.h expects to have been defined.
        PYBIND11_*                  compiler, standard-library and ABI tags embedded in the
                                    running torch build, which the Python module must match.

    The list is READ FROM torch WHERE POSSIBLE rather than copied, so a future release that
    changes it changes this too. The literal is the fallback for a torch that does not expose it.
    """
    common = list(getattr(cpp_extension, "COMMON_NVCC_FLAGS", None) or (
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
    ))
    torch_c = getattr(torch, "_C", None)
    abi = getattr(torch_c, "_GLIBCXX_USE_CXX11_ABI", True)
    defines = common + [f"-D_GLIBCXX_USE_CXX11_ABI={int(bool(abi))}",
                        "-DTORCH_API_INCLUDE_EXTENSION_H",
                        f"-DTORCH_EXTENSION_NAME={name}"]
    for suffix in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI"):
        value = getattr(torch_c, f"_PYBIND11_{suffix}", None)
        if value is not None:
            defines.append(f'-DPYBIND11_{suffix}="{value}"')
    return defines


def _split_flags(flags):
    """Upstream's compiler flags, split the way a shell would split them.

    THEIR LISTS CONTAIN ARGUMENT PAIRS AS ONE STRING, because that is what their build does with
    them. Every RWKV op module here passes

        extra_cuda_cflags=[..., "-O3", "-Xptxas -O3", "--extra-device-vectorization", ...]

    and cpp_extension writes those into a build.ninja command line, where the shell splits
    "-Xptxas -O3" into two words. subprocess.run with a list does not: it hands nvcc a single
    argument with a space inside it, and nvcc rejects the whole thing.

        nvcc fatal   : Unknown option '-Xptxas -O3'

    shlex.split reproduces the shell's word-splitting, including its treatment of quotes, so
    -Xptxas "-O3,-v" would still arrive as the two arguments upstream intended. Nothing is
    dropped, reordered or rewritten -- the flags are theirs.
    """
    import shlex
    out = []
    for flag in flags:
        out += shlex.split(str(flag))
    return out


def _cuda_include_paths(cpp_extension):
    """Torch's include directories for a CUDA build, across the signature change.

    The parameter that asks for the CUDA headers has been spelled three ways:

        include_paths(cuda=True)                 torch <= 2.5
        include_paths(device_type="cuda")        torch >= 2.6
        include_paths()                          neither keyword accepted

    Calling the wrong one raises TypeError at build time, on the compute node, on the first
    forward -- which is how a run gets to step 200 and then dies with a message about nvcc.
    Each is tried in turn, and the last resort still returns the CPU include set, which carries
    torch's own headers; only the CUDA toolkit's own would be missing, and nvcc supplies those.
    """
    for attempt in (lambda: cpp_extension.include_paths(device_type="cuda"),
                    lambda: cpp_extension.include_paths(cuda=True),
                    lambda: cpp_extension.include_paths()):
        try:
            return list(attempt())
        except TypeError:
            continue
    return []


def _detected_arch():
    """The compute capability of the card in this process as an integer, e.g. 89, or None.

    89 is an RTX 6000 Ada; 90 is a Hopper H100. The number is what nvcc spells sm_89 and
    compute_89, and what is compared against the architectures nvcc actually knows.
    """
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            return major * 10 + minor
    except Exception:                                             # noqa: BLE001
        pass
    return None


def _nvcc_arches(nvcc):
    """Every compute capability THIS nvcc can target, ascending. Empty if it cannot be asked.

    THE TOOLKIT IS OLDER THAN THE CARD MORE OFTEN THAN ANYONE EXPECTS, because torch ships its
    own CUDA runtime in the wheel while nvcc comes from the distribution. A cu118 torch on an
    Ubuntu 22.04 box gets nvcc 11.5 from apt, which has never heard of sm_89:

        nvcc fatal   : Value 'sm_89' is not defined for option 'gpu-architecture'

    Asking is the only reliable test -- the capability a toolkit supports is not derivable from
    its version number without a table that would go stale.
    """
    import re
    import subprocess
    if nvcc in _ARCHES:                       # asked once per toolkit, not once per source file
        return _ARCHES[nvcc]
    answer = []
    for argument in ("--list-gpu-arch", "--help"):
        try:
            done = subprocess.run([nvcc, argument], capture_output=True, text=True, timeout=60)
        except Exception:                                         # noqa: BLE001
            continue
        found = sorted({int(n) for n in re.findall(r"compute_(\d+)", done.stdout or "")})
        if found:
            answer = found
            break
    _ARCHES[nvcc] = answer
    return answer


def _arch_flags(nvcc):
    """The -gencode flags to build for the card in this process with the nvcc at hand.

    Three cases, and the middle one is the reason this function exists:

        nvcc knows the card       compile to SASS for it, and embed PTX as well so the same .so
                                  keeps working on a later card
        nvcc is too old for it    compile the highest architecture nvcc DOES know, and emit ONLY
                                  PTX -- the driver JIT-compiles that to the real card at load.
                                  Ada runs Ampere PTX; a newer driver with an older toolkit is
                                  exactly the case PTX forward compatibility was designed for.
                                  The one-off JIT costs a moment on first load and is cached by
                                  the driver.
        nothing can be determined leave the flags off and let nvcc pick its own default

    ASKING nvcc RATHER THAN ASSUMING is the whole point: passing -arch=sm_89 to a toolkit that
    predates Ada is a fatal error at build time, on the compute node, at the first forward.
    """
    device = _detected_arch()
    if device is None:
        return []
    known = _nvcc_arches(nvcc)
    if not known or device in known:
        return ["-gencode", f"arch=compute_{device},code=sm_{device}",
                "-gencode", f"arch=compute_{device},code=compute_{device}"]
    shipped = [a for a in _SHIP_ARCHES
               if a in known and a <= device and a // 10 == device // 10]
    if shipped:
        best = shipped[-1]
    else:
        # NOTHING OF THE CARD'S OWN MAJOR VERSION IS AVAILABLE -- an old toolkit in front of a
        # much newer card. PTX still crosses that gap where SASS cannot, so the highest
        # architecture nvcc knows is emitted as PTX and the driver JITs it to the real card.
        usable = [a for a in known if a <= device]
        if not usable:
            return []
        best = usable[-1]
    return ["-gencode", f"arch=compute_{best},code=compute_{best}"]


def _nvcc_note(nvcc):
    """A sentence naming the toolkit, appended to a build failure. Never raises."""
    import re
    import subprocess
    try:
        done = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60)
        version = re.search(r"release ([0-9.]+)", done.stdout or "")
        release = version.group(1) if version else "unknown version"
    except Exception:                                             # noqa: BLE001
        release = "not runnable"
    device = _detected_arch()
    known = _nvcc_arches(nvcc)
    where = f" ({nvcc}, release {release}"
    if known:
        where += f", targets compute_{known[0]}..compute_{known[-1]}"
    if device is not None:
        where += f", card is sm_{device}"
    return where + "). Set NVCC=/path/to/newer/nvcc to build with a different toolkit."


def load(key, pkgdir, dotted, namespace=None, env=None):
    """Import `dotted` -- an upstream op module -- with the working directory at `pkgdir`.

    `key`        the name failures are reported under, e.g. "rwkv_v7_upstream"
    `pkgdir`     the directory upstream's relative source paths resolve against; the vendored
                 <family>/upstream/, which holds cuda/
    `dotted`     the module that carries upstream's load() call and their autograd.Function
    `namespace`  the TORCH_LIBRARY namespace that module registers, if it registers one
    `env`        environment upstream reads at import; applied before the import, not after

    Returns the module, or None. Cached per key, INCLUDING the failure: a compile that failed
    once will fail the same way on every micro-batch, and re-running nvcc 30,000 times to find
    that out is not a diagnostic, it is a hang.
    """
    with _LOCK:
        if key in _MODS:
            return _MODS[key]
        try:
            import torch                                          # noqa: F401
        except Exception as e:                                    # noqa: BLE001
            _MODS[key] = None
            return _say(key, f"no torch ({type(e).__name__}: {e}); upstream kernel not built")
        if namespace is not None:
            owner = _NS.get(namespace)
            if owner is not None and owner != key:
                _MODS[key] = None
                return _say(key, f"the TORCH_LIBRARY namespace {namespace!r} is already held "
                                 f"by {owner!r} in this process and torch allows one holder "
                                 f"per namespace. Run these families in separate processes "
                                 f"-- pretrain_benchmark.py --only {key.split('_upstream')[0]}")
        if not os.path.isdir(pkgdir):
            _MODS[key] = None
            return _say(key, f"vendored sources missing at {pkgdir}")
        # NOTHING IS WRITTEN OUTSIDE THE PROJECT ROOT, including by any upstream path that
        # still reaches cpp_extension. Its scratch directory is redirected into the tree before
        # the import that would otherwise create ~/.cache/torch_extensions/<name>/.
        scratch = os.path.join(_ROOT, "build", "kernel", "cuda",
                               key.replace("_upstream", ""), "upstream")
        os.makedirs(scratch, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = scratch
        for k, v in (env or {}).items():
            os.environ[k] = str(v)
        cwd = os.getcwd()
        try:
            os.chdir(pkgdir)
            import importlib
            m = importlib.import_module(dotted)
        except Exception as e:                                    # noqa: BLE001
            _MODS[key] = None
            return _say(key, f"building upstream's kernel failed ({type(e).__name__}: {e}). "
                             f"It needs nvcc and a C++ compiler on THIS node at run time; the "
                             f"build directory is inside the project and Ninja is not used.")
        finally:
            os.chdir(cwd)
        _MODS[key] = m
        if namespace is not None:
            _NS[namespace] = key
        _WHY[key] = f"upstream kernel built from {os.path.join(pkgdir, 'cuda')}"
        return m
