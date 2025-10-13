cmake_minimum_required(VERSION 3.26)

project(spear_extensions LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options passed from setup.py
set(SPEAR_PYTHON_EXECUTABLE "" CACHE STRING "Path to python executable")
set(SPEAR_PYTHON_EXTENSION_SUFFIX ".so" CACHE STRING "Python extension suffix (from sysconfig)")
set(SPEAR_CUDA_ARCH_LIST "" CACHE STRING "Semicolon-separated list of CUDA arch numbers, e.g., 90;89")
set(NVCC_THREADS "" CACHE STRING "Number of threads to pass to NVCC --threads")
set(CUTLASS_INCLUDE_DIR "" CACHE PATH "Path to CUTLASS include directory")

if(NOT SPEAR_PYTHON_EXECUTABLE)
  message(FATAL_ERROR "SPEAR_PYTHON_EXECUTABLE must be set")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)

# Torch variables (for some environments imported target Torch::Torch is missing)
set(SPEAR_TORCH_LIBS ${TORCH_LIBRARIES})
set(SPEAR_TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
find_library(TORCH_PYTHON_LIBRARY NAMES torch_python libtorch_python.so PATHS ${TORCH_LIBRARY_DIRS} ${TORCH_LIBRARIES} ${TORCH_INSTALL_PREFIX}/lib ${TORCH_INSTALL_PREFIX}/lib64 ${Python3_LIBRARY_DIRS} ${Python3_RUNTIME_LIBRARY_DIRS} ${Python3_STDLIB} NO_DEFAULT_PATH)
if(NOT TORCH_PYTHON_LIBRARY)
  # fallback to typical torch lib dir
  get_filename_component(_torch_libdir ${TORCH_LIBRARIES} DIRECTORY)
  find_library(TORCH_PYTHON_LIBRARY NAMES torch_python libtorch_python.so PATHS ${_torch_libdir})
endif()
find_package(CUDAToolkit REQUIRED)

# Ensure we use the CUDA compiler from environment when set by setup.py
if(DEFINED CMAKE_CUDA_COMPILER)
  message(STATUS "Using CUDA compiler: ${CMAKE_CUDA_COMPILER}")
endif()

# Include dirs
include_directories(
  ${PROJECT_SOURCE_DIR}/csrc
)
if(CUTLASS_INCLUDE_DIR)
  include_directories(${CUTLASS_INCLUDE_DIR})
endif()

# Helper to apply NVCC arch flags per-target
function(spear_apply_cuda_arch_flags target)
  if(SPEAR_CUDA_ARCH_LIST)
    foreach(arch IN LISTS SPEAR_CUDA_ARCH_LIST)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${arch},code=sm_${arch}>)
    endforeach()
  endif()
endfunction()

# Helper to apply common CUDA flags
function(spear_apply_common_cuda_flags target)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
  if(NVCC_THREADS)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--threads=${NVCC_THREADS}>)
  endif()
endfunction()

# Helper to mark a library as Python extension and install it under spear/
function(spear_mark_python_extension target out_name)
  set_target_properties(${target} PROPERTIES
    PREFIX ""
    OUTPUT_NAME "${out_name}"
    SUFFIX "${SPEAR_PYTHON_EXTENSION_SUFFIX}"
  )
  install(TARGETS ${target}
    LIBRARY DESTINATION spear COMPONENT ${out_name})
endfunction()

# Helper to apply Torch C++ flags and required defines
function(spear_apply_torch_flags target)
  if (DEFINED TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND ${TORCH_CXX_FLAGS})
    target_compile_options(${target} PRIVATE ${TORCH_CXX_FLAGS_LIST})
  endif()
  target_compile_definitions(${target} PRIVATE TORCH_API_INCLUDE_EXTENSION_H=1)
endfunction()

# ------------------------------
# _btp extension
# ------------------------------
add_library(_btp MODULE
  csrc/btp/_bindings.cu
  csrc/btp/btp-forward.cu
  csrc/btp/btp-backwards.cu
)
target_compile_definitions(_btp PRIVATE TORCH_EXTENSION_NAME=_btp)
target_include_directories(_btp PRIVATE ${SPEAR_TORCH_INCLUDE_DIRS})
target_include_directories(_btp PRIVATE ${Python3_INCLUDE_DIRS})
spear_apply_torch_flags(_btp)
if(TORCH_PYTHON_LIBRARY)
  target_link_libraries(_btp PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()
target_link_libraries(_btp PRIVATE ${SPEAR_TORCH_LIBS} Python3::Python CUDA::cudart CUDA::cublas)
spear_apply_common_cuda_flags(_btp)
spear_apply_cuda_arch_flags(_btp)
spear_mark_python_extension(_btp "_btp")

