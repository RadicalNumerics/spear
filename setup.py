# Copyright 2025 Radical Numerics Inc.
#
# This source code is licensed under the Apache License, Version 2.0, found in the
# LICENSE file in the root directory of this source tree.

import ast
import os
import re
import subprocess
import sys
import sysconfig

from pathlib import Path
from shutil import which

import torch

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))


# Entry points that can be imported from spear
# See CMakeLists.txt for the actual targets
KERNELS = [
    "spear._btp",
]


def _probe_cutlass_include(root: Path) -> str | None:
    # Common layouts used by the official repo and python wheels
    candidates = [
        root / "include",  # cutlass/include/...
        root / "source" / "include",  # some packages use source/include
        root.parent / "include",  # fallback if module file sits under subdir
    ]
    for c in candidates:
        if (c / "cutlass" / "cutlass.h").exists():
            return str(c)
    return None


def find_cutlass_headers() -> str | None:
    import importlib.util

    for name in ("cutlass", "nvidia_cutlass", "cutlass_library"):
        spec = importlib.util.find_spec(name)
        if not spec or not spec.origin:
            continue
        root = Path(spec.origin).parent
        hit = _probe_cutlass_include(root)
        if hit:
            return hit
    return None


def get_cutlass_include_path() -> str | None:
    """Get the CUTLASS include path in the environment."""
    # If CUTLASS_INCLUDE_PATH is set, return it
    if os.environ.get("CUTLASS_INCLUDE_PATH", None) is not None:
        return os.environ.get("CUTLASS_INCLUDE_PATH")

    return find_cutlass_headers()


cutlass_include = get_cutlass_include_path()
if not cutlass_include:
    raise RuntimeError(
        "CUTLASS headers not found. Ensure 'nvidia-cutlass' is listed under "
        "[build-system].requires in pyproject.toml or set CUTLASS_INCLUDE_PATH to the "
        "CUTLASS 'include' directory (the one containing cutlass/cutlass.h)."
    )


def get_package_version():
    with open(Path(this_dir) / "spear" / "__init__.py") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("SPEAR_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", optional: bool = False) -> None:
        super().__init__(name, sources=[], optional=optional)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


def _is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(int(os.getenv("SPEAR_DISABLE_SCCACHE", "0")))


def _is_ccache_available() -> bool:
    return which("ccache") is not None


def _is_ninja_available() -> bool:
    return which("ninja") is not None


class CMakeBuildExt(build_ext):
    """CMake-driven build_ext with caching and parallel builds."""

    did_config: dict[str, bool] = {}

    def _compute_jobs(self) -> tuple[int, str | None]:
        # Respect MAX_JOBS if set; otherwise compute based on cores/memory
        num_jobs_env = os.environ.get("MAX_JOBS")
        if num_jobs_env is not None:
            try:
                return max(1, int(num_jobs_env)), os.environ.get("NVCC_THREADS")
            except ValueError:
                pass

        import psutil

        max_num_jobs_cores = max(1, (os.cpu_count() or 1) // 2)
        free_memory_gb = psutil.virtual_memory().available / (1024**3)
        max_num_jobs_memory = int(free_memory_gb / 9)  # ~9GB peak per job when threads=4
        max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
        return max_jobs, os.environ.get("NVCC_THREADS")

    def _target_from_ext(self, ext_name: str) -> str:
        base = ext_name.split(".")[-1]
        # if base.startswith("_"):
        #     base = base[1:]
        return base

    def _install_prefix_for_ext(self, ext_name: str) -> Path:
        outdir = Path(self.get_ext_fullpath(ext_name)).parent.absolute()
        # Remove package prefix(s) so cmake --install places into correct dir
        prefix = outdir
        for _ in range(ext_name.count(".")):
            prefix = prefix.parent
        return prefix

    def _configure(self, ext: CMakeExtension) -> None:
        if self.did_config.get(ext.cmake_lists_dir):
            return
        self.did_config[ext.cmake_lists_dir] = True

        cfg = "Debug" if self.debug else "RelWithDebInfo"
        cmake_args = [f"-DCMAKE_BUILD_TYPE={cfg}"]

        # Cache launchers
        if _is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
            ]
        elif _is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ]

        # Python info for extension suffix
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        cmake_args += [
            f"-DSPEAR_PYTHON_EXECUTABLE={sys.executable}",
            f"-DSPEAR_PYTHON_EXTENSION_SUFFIX={ext_suffix}",
        ]

        # Help CMake find Torch by providing torch's cmake prefix path
        if hasattr(torch.utils, "cmake_prefix_path"):
            cmake_args += [f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}"]

        # CUDA compiler path (use CUDA_HOME) to avoid wrong nvcc
        if CUDA_HOME is not None:
            nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
            if os.path.exists(nvcc):
                cmake_args += [f"-DCMAKE_CUDA_COMPILER={nvcc}"]

        # CUTLASS include directory from earlier probe
        if cutlass_include:
            cmake_args += [f"-DCUTLASS_INCLUDE_DIR={cutlass_include}"]

        # CUDA arch list and NVCC threads
        if compute_capabilities:
            cmake_args += [f"-DSPEAR_CUDA_ARCH_LIST={';'.join(compute_capabilities)}"]

        _, nvcc_threads = self._compute_jobs()
        if nvcc_threads:
            cmake_args += [f"-DNVCC_THREADS={nvcc_threads}"]

        # Generator / job pools
        build_tool = []
        num_jobs, _ = self._compute_jobs()
        if _is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
            ]

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            [
                "cmake",
                ext.cmake_lists_dir,
                *build_tool,
                *cmake_args,
            ],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])  # ensure cmake is present
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        os.makedirs(self.build_temp, exist_ok=True)

        # Configure once (covers all extensions sharing the same CMakeLists.txt)
        for ext in self.extensions:
            assert isinstance(ext, CMakeExtension)
            self._configure(ext)

        # Build all targets in one go
        num_jobs, _ = self._compute_jobs()
        targets = [self._target_from_ext(ext.name) for ext in self.extensions]

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={t}" for t in targets],
        ]
        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Install each extension component into wheel/editable lib dir
        for ext in self.extensions:
            component = ext.name.split(".")[-1]  # e.g., _btp_unfused
            prefix = self._install_prefix_for_ext(ext.name)
            install_args = [
                "cmake",
                "--install",
                ".",
                "--prefix",
                str(prefix),
                "--component",
                component,
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self) -> None:
        super().run()


def get_cuda_version():
    """Get CUDA version from nvcc."""
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set. Please install CUDA.")

    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    if not os.path.exists(nvcc):
        raise RuntimeError(f"nvcc not found at {nvcc}")

    result = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get CUDA version: {result.stderr}")

    for line in result.stdout.split("\n"):
        if "release" in line:
            version = line.split("release")[1].split(",")[0].strip()
            return version

    raise RuntimeError("Could not parse CUDA version")


# TODO: check -- we might want to enable overrides for multiple archs
def detect_gpu_architecture():
    if not torch.cuda.is_available():
        # Safer default: compile only SM90 (tweak if your build host is different)
        print("No GPU detected; compiling only SM90")
        return ["90"]
    major, minor = torch.cuda.get_device_capability()
    cc = f"{major}{minor}"
    print(f"Detected SM{cc}; compiling only SM{cc}")
    return [cc]


# Detect architecture
compute_capabilities = detect_gpu_architecture()


ext_modules = [CMakeExtension(name=kernel) for kernel in KERNELS]

setup(
    version=get_package_version(),
    packages=find_packages(where=".", include=["spear*"]),
    package_dir={"": "."},
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuildExt},
    install_requires=[
        "torch>=2.0.0",
        "einops>=0.6.0",
        "numpy>=1.21.0",
        "nvidia-cutlass",
        "ninja",
        "pybind11",
        "psutil",
        "einops>=0.8.0",
    ],
)
