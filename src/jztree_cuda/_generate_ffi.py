from pathlib import Path
import os

import fmdj_utils.parse as parse
import fmdj_utils.generator as gen

HERE = Path(__file__).resolve().parent

p_instance_values = (1, 2, 3, 4, 5)
default_includes = ["../common/math.cuh"]

# ------------------------------------------------------------------------------------------------ #
#                                             tree.cuh                                             #
# ------------------------------------------------------------------------------------------------ #


functions = parse.get_functions_from_file(
    str(HERE / "tree.cuh"),
    names=["PosZorderSort", "BuildZTree", "SummarizeLeaves", "SearchSortedZ"],
    only_kernels=False
)

print(list(functions.keys()))

functions["PosZorderSort"].par["size"].expression = "pos_in.element_count()/3"
functions["PosZorderSort"].par["tmp_bytes"].expression = "tmp_buffer->size_bytes()"

functions["BuildZTree"].par["size"].expression = "pos_in.element_count()/3"

functions["SummarizeLeaves"].par["n_leaves"].expression = "xnleaf.element_count()/4"
functions["SummarizeLeaves"].grid_size_expression = "div_ceil(n_leaves+1, block_size)"
functions["SummarizeLeaves"].smem_size_expression = "(block_size + 2*scan_size + 1) * (sizeof(PosN) + sizeof(int32_t))"

functions["SearchSortedZ"].par["n_have"].expression = "posz_have.element_count()/3"
functions["SearchSortedZ"].par["n_query"].expression = "posz_query.element_count()/3"
functions["SearchSortedZ"].grid_size_expression = "div_ceil(n_query, block_size)"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_tree.cu"), 
    functions = functions, 
    includes = default_includes + ["../tree.cuh"]
)