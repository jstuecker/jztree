cmake_minimum_required(VERSION 3.24)

# ================== Make sure we have a valid python interpreter =================================
# (We need it to define some paths)
if(DEFINED ENV{PYTHON_EXECUTABLE} AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "$ENV{PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable to use")
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Using Python: ${Python_EXECUTABLE}")

if (NOT Python_Development.Module_FOUND)
  message(FATAL_ERROR
    "Python headers not found for ${Python_EXECUTABLE}.\n"
    "Recommended: Use a uv-managed Python:\n"
    "  uv python install 3.12 && uv python pin 3.12 && uv pip install -e . --no-build-isolation\n"
  )
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "from nvidia import cu13; print(next(iter(cu13.__path__)))"
  OUTPUT_VARIABLE CUDA_ROOT
  ERROR_VARIABLE CUDA_PATH_ERR
  RESULT_VARIABLE CUDA_PATH_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using a python installed cuda directory: ${CUDA_ROOT}")
if(NOT CUDA_PATH_RESULT EQUAL 0)
  message(WARNING "Could not find python cuda directory - I will assume a system or conda/miniforge" 
          "-managed installation ") # ${CUDA_PATH_ERR}
else()
  # CUDA_ROOT defined via python we will try to set up CUDA paths from a python environment 
  message(STATUS "Using python/CUDA setup from: $ENV{CUDA_ROOT}")

  set(_cuda_root "${CUDA_ROOT}")
  set(_cuda_bin   "${_cuda_root}/bin")
  set(_cuda_lib   "${_cuda_root}/lib")
  set(_cuda_inc   "${_cuda_root}/include")

  message(STATUS "Using CUDA bin path: ${_cuda_bin}")
  message(STATUS "Using CUDA lib path: ${_cuda_lib}")
  message(STATUS "Using CUDA inc path: ${_cuda_inc}")  

  if(DEFINED ENV{PATH})
    set(ENV{PATH} "${_cuda_bin}:$ENV{PATH}")
  else()
    set(ENV{PATH} "${_cuda_bin}")
  endif()

  # Make sure the linker & runtime see the wheel libs during compiler detection
  if(DEFINED ENV{LIBRARY_PATH})
    set(ENV{LIBRARY_PATH} "${_cuda_lib}:$ENV{LIBRARY_PATH}")
  else()
    set(ENV{LIBRARY_PATH} "${_cuda_lib}")
  endif()

  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)   # modern CMake
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)     # legacy toggle that some logic still checks

  include_directories(SYSTEM "${_cuda_root}/include" "${_cuda_root}/include/cccl")
  # Need to define the link directory explicitly here so that ninja finds it
  add_link_options(-L${_cuda_root}/lib)
  add_link_options(-L${_cuda_root}/lib64)
endif()

# Somehow some conda environments like to put the env. variable NVCC_PREPEND_FLAGS and/or
# NVCC_PREPEND_FLAGS_BACKUP to "UNSET" which confuses nvcc. 
# Let's undefine these explicitly in that case:
foreach(var IN ITEMS NVCC_PREPEND_FLAGS NVCC_PREPEND_FLAGS_BACKUP)
  if(DEFINED ENV{${var}} AND "$ENV{${var}}" STREQUAL "UNSET")
    message(STATUS "Clearing environment ${var} which was 'UNSET'")
    # CMake can't "unset" ENV vars, but setting to empty is equivalent for nvcc
    set(ENV{${var}} "")
  endif()
endforeach()

# Now let Cmake detect CUDA
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(nanobind CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # To simplify finding paths in vscode

if (NOT CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "Defaulting to native CUDA artchitecture")
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Find FFI Paths
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_DIR
  ERROR_VARIABLE XLA_ERR
  RESULT_VARIABLE XLA_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT XLA_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to get XLA include directory: ${XLA_ERR}")
endif()

message(STATUS "XLA include directory: ${XLA_DIR}")

# The following flag is debug output, helping to diagnose local memory spilling in cuda kernels:
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -L${_cuda_lib} -Xlinker -rpath,${_cuda_lib}")

# Define a list of modules and their source files
set(MODULES nb_forces nb_tree nb_multipoles nb_knn)
set(MODULE_PATHS src/custom_jax/forces.cu
    src/custom_jax/tree.cu
    src/custom_jax/multipoles.cu
    src/custom_jax/knn.cu
)

include_directories(SYSTEM ${XLA_DIR})

foreach(var IN ZIP_LISTS MODULES MODULE_PATHS)
    set(MODULE_NAME ${var_0})
    set(MODULE_SOURCE ${var_1})

    nanobind_add_module(${MODULE_NAME} ${MODULE_SOURCE})

    target_compile_options(${MODULE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_features(${MODULE_NAME} PRIVATE cxx_std_17)

    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endforeach()
