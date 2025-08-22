cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # To simplify finding paths in vscode

if (NOT CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "Defaulting to native CUDA artchitecture")
  set(CMAKE_CUDA_ARCHITECTURES 86) # For now we fix this to 86, can improve this later
endif()

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

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

# The following flag is debug output, helping to diagnose local memory spilling in cuda kernels
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=-v")

# Define a list of modules and their source files
set(MODULES nb_forces nb_tree nb_multipoles)
set(MODULE_PATHS src/custom_jax/forces.cu
    src/custom_jax/tree.cu
    src/custom_jax/multipoles.cu
)

foreach(var IN ZIP_LISTS MODULES MODULE_PATHS)
    set(MODULE_NAME ${var_0})
    set(MODULE_SOURCE ${var_1})


    nanobind_add_module(${MODULE_NAME} ${MODULE_SOURCE})
    set_target_properties(${MODULE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
    target_include_directories(${MODULE_NAME} SYSTEM PRIVATE ${XLA_DIR})
    target_compile_options(${MODULE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_features(${MODULE_NAME} PRIVATE cxx_std_17)
    target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
    target_compile_options(${MODULE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)

    install(TARGETS ${MODULE_NAME} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
endforeach()