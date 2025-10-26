from pathlib import Path
import os

import cj_codetools.parse as parse
import cj_codetools.generator as gen

HERE = Path(__file__).resolve().parent

p_instance_values = (1, 2, 3)

# ------------------------------------------------------------------------------------------------ #
#                                            ffi_example                                           #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "ffi_example.cuh"), 
    only_kernels=True
)

kernels["SimpleArange"].grid_size_expression = "div_ceil(output->element_count(), block_size)"
kernels["SimpleArange"].template_par["p"].instances = p_instance_values

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_example.cu"), 
    functions = kernels, 
    includes = ["../ffi_example.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                              fmm.cuh                                             #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "fmm.cuh"), 
    only_kernels=True, 
    names=["EvaluateTreePlane"]
)

kernels["EvaluateTreePlane"].template_par["p"].instances = p_instance_values
kernels["EvaluateTreePlane"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["EvaluateTreePlane"].init_outputs_zero = True

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_fmm.cu"), 
    functions = kernels, 
    includes = ["../fmm.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                          multipoles.cuh                                          #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "multipoles.cuh"), 
    only_kernels=True
)

print(list(kernels.keys()))
for kernel in kernels.values():
    kernel.template_par["p"].instances = p_instance_values
    kernel.init_outputs_zero = True

kernels["IlistM2L"].grid_size_expression = "div_ceil(interactions.element_count() / 2, block_size*interactions_per_block)"
kernels["IlistLeaf2NodeM2L"].grid_size_expression = "div_ceil(interactions.element_count() / 2, interactions_per_block)"
kernels["MultipolesFromParticles"].grid_size_expression = "isplit.element_count() - 1"
kernels["CoarsenMultipoles"].grid_size_expression = "isplit.element_count() - 1"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_multipoles.cu"), 
    functions = kernels, 
    includes = ["../multipoles.cuh"]
)