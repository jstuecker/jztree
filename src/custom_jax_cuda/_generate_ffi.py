from pathlib import Path
import os

import cj_codetools.parse as parse
import cj_codetools.generator as gen

HERE = Path(__file__).resolve().parent

p_instance_values = (1, 2, 3)

# ------------------------------------------------------------------------------------------------ #
#                                            ffi_example                                           #
# ------------------------------------------------------------------------------------------------ #

funcs = parse.get_functions_from_file(
    str(HERE / "ffi_example.cuh"),
    only_kernels=False,
    names=["SimpleArange", "SetToConstantCall"]
)

funcs["SimpleArange"].grid_size_expression = "div_ceil(output->element_count(), block_size)"
funcs["SimpleArange"].template_par["p"].instances = p_instance_values
funcs["SetToConstantCall"].template_par["tpar"].instances = (16, 32, 64)
funcs["SetToConstantCall"].par["size"].expression = "output->element_count()"

for kernel in funcs.values():
    print(kernel.name, kernel.is_kernel)

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_example.cu"), 
    functions = funcs, 
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

# ------------------------------------------------------------------------------------------------ #
#                                             tree.cuh                                             #
# ------------------------------------------------------------------------------------------------ #


functions = parse.get_functions_from_file(
    str(HERE / "tree_new.cuh"),
    names=["PosZorderSort", "BuildZTree", "SummarizeLeaves"],
    only_kernels=False
)

print(list(functions.keys()))

functions["PosZorderSort"].par["size"].expression = "pos_in.element_count()/3"
functions["PosZorderSort"].par["tmp_bytes"].expression = "tmp_buffer->size_bytes()"

functions["BuildZTree"].par["size"].expression = "pos_in.element_count()/3"

functions["SummarizeLeaves"].par["n_leaves"].expression = "xnleaf.element_count()/4"
functions["SummarizeLeaves"].grid_size_expression = "div_ceil(n_leaves+1, block_size)"
functions["SummarizeLeaves"].smem_size_expression = "(block_size + 2*scan_size + 1) * (sizeof(PosN) + sizeof(int32_t))"

print(functions["PosZorderSort"].type)

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_tree_new.cu"), 
    functions = functions, 
    includes = ["../tree_new.cuh"]
)