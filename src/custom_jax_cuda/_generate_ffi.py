import cj_codetools.parse as parse
import cj_codetools.generator as gen

kernels = parse.get_functions_from_file(
    "fmm.cuh", 
    only_kernels=True, 
    names=["TestPositions", "SimpleArange", "EvaluateTreePlane"]
)

kernels["TestPositions"].template_par["p"].instances = [0,1,2,3]
kernels["TestPositions"].block_size_expression = "64"

kernels["SimpleArange"].grid_size_expression = "div_ceil(output->element_count(), block_size)"
kernels["SimpleArange"].par["size"].expression = "output->element_count()"

kernels["EvaluateTreePlane"].template_par["p"].instances = [1,2,3]
kernels["EvaluateTreePlane"].grid_size_expression = "spl_nodes.element_count() - 1"
kernels["EvaluateTreePlane"].init_outputs_zero = True

gen.generate_ffi_module_file(
    output_file="generated/ffi_fmm.cu", 
    functions=kernels, 
    includes = ["../fmm.cuh"]
)