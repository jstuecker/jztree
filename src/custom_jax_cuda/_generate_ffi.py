from pathlib import Path
import os

import cj_codetools.parse as parse
import cj_codetools.generator as gen

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------ #
#                                            ffi_example                                           #
# ------------------------------------------------------------------------------------------------ #

kernels = parse.get_functions_from_file(
    str(HERE / "ffi_example.cuh"), 
    only_kernels=True
)

kernels["SimpleArange"].grid_size_expression = "div_ceil(output->element_count(), block_size)"
kernels["SimpleArange"].template_par["p"].instances = [0,1,2,3]

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

kernels["EvaluateTreePlane"].template_par["p"].instances = [1,2,3]
kernels["EvaluateTreePlane"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["EvaluateTreePlane"].init_outputs_zero = True

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_fmm.cu"), 
    functions = kernels, 
    includes = ["../fmm.cuh"]
)