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

nanobind_add_module(nb_forces src/custom_jax/forces.cu)
set_target_properties(nb_forces PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
target_include_directories(nb_forces SYSTEM PRIVATE ${XLA_DIR})
target_compile_features(nb_forces PRIVATE cxx_std_17)
install(TARGETS nb_forces LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
