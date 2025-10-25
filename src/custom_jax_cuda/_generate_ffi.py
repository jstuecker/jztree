import cj_codetools.parse as parse
import cj_codetools.generator as gen

kernels = parse.get_functions_from_file(
    "fmm.cuh", 
    only_kernels=True, 
    names=["TestPositions", "SimpleArange"]
)

kernels["TestPositions"].template_par[0].instances = [0,1,2,3]
kernels["TestPositions"].block_size_expression = "64"
kernels["SimpleArange"].grid_size_expression = "div_ceil(output->dimensions()[0], block_size)"

gen.generate_ffi_module_file(
    output_file="generated/ffi_fmm.cu", 
    functions=kernels, 
    includes = ["../fmm.cuh"]
)