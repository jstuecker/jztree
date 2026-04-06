cmake_minimum_required(VERSION 3.24)

# Shared backend packaging knobs used by packaging/jztree-cu12 and packaging/jztree-cu13
set(JZTREE_CUDA_MAJOR "13" CACHE STRING "CUDA major version baked into jztree backend wheel")
set(JZTREE_BACKEND_PACKAGE_NAME "jztree_cuda" CACHE STRING "Python package path where backend modules are installed")

if(NOT JZTREE_CUDA_MAJOR MATCHES "^[0-9]+$")
  message(FATAL_ERROR "JZTREE_CUDA_MAJOR must be a numeric major version (e.g. 12 or 13)")
endif()

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
  OUTPUT_VARIABLE CUDA_PATH
  ERROR_VARIABLE CUDA_PATH_ERR
  RESULT_VARIABLE CUDA_PATH_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Using a python installed cuda directory: ${CUDA_PATH}")

# set(CUDAToolkit_ROOT "${CUDA_PATH}")

if(NOT CUDA_PATH_RESULT EQUAL 0)
  message(WARNING "Could not find python cuda directory - I will assume a system or conda/miniforge" 
          "-managed installation ") # ${CUDA_PATH_ERR}
else()
  # CUDA_ROOT defined via python we will try to set up CUDA paths from a python environment 
  message(STATUS "Using python/CUDA setup from: $ENV{CUDA_PATH}")

  set(_cuda_root "${CUDA_PATH}")

  
  # set(_cuda_bin   "${_cuda_root}/bin")
  set(_cuda_lib   "${_cuda_root}/lib")
  # set(_cuda_inc   "${_cuda_root}/include")

  # message(STATUS "Using CUDA bin path: ${_cuda_bin}")
  # message(STATUS "Using CUDA lib path: ${_cuda_lib}")
  # message(STATUS "Using CUDA inc path: ${_cuda_inc}")  

  # if(DEFINED ENV{PATH})
  #   set(ENV{PATH} "${_cuda_bin}:$ENV{PATH}")
  # else()
  #   set(ENV{PATH} "${_cuda_bin}")
  # endif()

  # Make sure the linker & runtime see the wheel libs during compiler detection
  if(DEFINED ENV{LIBRARY_PATH})
    set(ENV{LIBRARY_PATH} "${_cuda_lib}:$ENV{LIBRARY_PATH}")
  else()
    set(ENV{LIBRARY_PATH} "${_cuda_lib}")
  endif()

  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)

  # include_directories(SYSTEM "${_cuda_root}/include" "${_cuda_root}/include/cccl")
  # Need to define the link directory explicitly here so that ninja finds it
  add_link_options(-L${_cuda_root}/lib)
  # add_link_options(-L${_cuda_root}/lib64)

  set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc")
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

# set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc")

# Now let Cmake detect CUDA
# find_package(CUDAToolkit CONFIG REQUIRED)

if(DEFINED ENV{CUDAARCHS})
  message(STATUS "User provided CUDAARCHS env: '$ENV{CUDAARCHS}'")
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: '${CMAKE_CUDA_ARCHITECTURES}'")
else()
  message(STATUS "Using all CUDA artchitecture")
  set(CMAKE_CUDA_ARCHITECTURES all)
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

message(STATUS "Using")

find_package(nanobind CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # To simplify finding paths in vscode

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

include_directories(SYSTEM ${XLA_DIR})

# ----------------------------------------------------------------------------------------------- #
# Auto-add all modules from src/jztree_cuda/generated
# ----------------------------------------------------------------------------------------------- #

# Build-time toggle: only generate sources when explicitly requested
# option(CJ_GENERATE "Generate FFI source files at configure time" OFF)
# Allow enabling via environment variable too: CJ_GENERATE=1/ON/TRUE
# if(DEFINED ENV{CJ_GENERATE})
#   if("$ENV{CJ_GENERATE}" STREQUAL "1" OR "$ENV{CJ_GENERATE}" STREQUAL "ON" OR "$ENV{CJ_GENERATE}" STREQUAL "TRUE")
#     set(CJ_GENERATE ON CACHE BOOL "Generate FFI source files at configure time" FORCE)
#   endif()
# else()
#   set(CJ_GENERATE OFF CACHE BOOL "Generate FFI source files at configure time" FORCE)
# endif()

# Optionally run the generator so files exist at configure time (opt-in)
if(DEFINED ENV{CJ_GENERATE} AND ( "$ENV{CJ_GENERATE}" STREQUAL "1" OR "$ENV{CJ_GENERATE}" STREQUAL "ON" OR "$ENV{CJ_GENERATE}" STREQUAL "TRUE") )
  message(STATUS "CJ_GENERATE=ON: Running CUDA FFI generator script")
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/src/jztree_cuda/_generate_ffi.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
else()
  message(STATUS "CJ_GENERATE=OFF: Skipping CUDA FFI generation (enable with -DCJ_GENERATE=ON or CJ_GENERATE=1)")
endif()

# Pick up all generated .cu files; re-scan on re-configure
file(GLOB GEN_FFI_CU CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/jztree_cuda/generated/*.cu"
)

foreach(GEN_SRC IN LISTS GEN_FFI_CU)
  message(STATUS "Compiling generated FFI module from source: ${GEN_SRC}")
  get_filename_component(MOD_NAME "${GEN_SRC}" NAME_WE)  # e.g. ffi_fmm.cu -> ffi_fmm

  nanobind_add_module(${MOD_NAME} "${GEN_SRC}")
  target_compile_definitions(${MOD_NAME} PRIVATE JZTREE_CUDA_MAJOR=${JZTREE_CUDA_MAJOR})
  target_compile_options(${MOD_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  target_compile_features(${MOD_NAME} PRIVATE cxx_std_17)

  install(TARGETS ${MOD_NAME} LIBRARY DESTINATION ${JZTREE_BACKEND_PACKAGE_NAME})
endforeach()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/jztree_cuda/__init__.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/jztree_cuda/__init__.py"
  @ONLY
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/jztree_cuda/__init__.py"
  DESTINATION ${JZTREE_BACKEND_PACKAGE_NAME}
)

install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/jztree_cuda/THIRD_PARTY_NOTICES"
  DESTINATION ${JZTREE_BACKEND_PACKAGE_NAME}
)
