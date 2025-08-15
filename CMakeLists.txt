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

nanobind_add_module(nb_forces src/custom_jax/forces.cu)
set_target_properties(nb_forces PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
target_include_directories(nb_forces SYSTEM PRIVATE ${XLA_DIR})
target_compile_options(nb_forces PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_compile_features(nb_forces PRIVATE cxx_std_17)
target_link_libraries(nb_forces PRIVATE CUDA::cudart)
# Add NVCC flags for extended lambdas and relaxed constexpr
target_compile_options(nb_forces PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)

nanobind_add_module(nb_tree src/custom_jax/tree.cu)
set_target_properties(nb_tree PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
target_include_directories(nb_tree SYSTEM PRIVATE ${XLA_DIR})
target_compile_options(nb_tree PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_compile_features(nb_tree PRIVATE cxx_std_17)
target_link_libraries(nb_tree PRIVATE CUDA::cudart)
# Add NVCC flags for extended lambdas and relaxed constexpr
target_compile_options(nb_tree PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)

nanobind_add_module(nb_multipoles src/custom_jax/multipoles.cu)
set_target_properties(nb_multipoles PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_STANDARD 17)
target_include_directories(nb_multipoles SYSTEM PRIVATE ${XLA_DIR})
target_compile_options(nb_multipoles PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3> $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_compile_features(nb_multipoles PRIVATE cxx_std_17)
target_link_libraries(nb_multipoles PRIVATE CUDA::cudart)
# Add NVCC flags for extended lambdas and relaxed constexpr
target_compile_options(nb_multipoles PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)

install(TARGETS nb_forces LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
install(TARGETS nb_tree LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
install(TARGETS nb_multipoles LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})