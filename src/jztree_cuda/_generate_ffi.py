from pathlib import Path
import os

import fmdj_utils.parse as parse
import fmdj_utils.generator as gen

HERE = Path(__file__).resolve().parent

k_instance_values = (4, 8, 12, 16, 32, 64)
default_includes = ["../common/math.cuh"]


# ------------------------------------------------------------------------------------------------ #
#                                              knn.cuh                                             #
# ------------------------------------------------------------------------------------------------ #

functions = parse.get_functions_from_file(
    str(HERE / "knn.cuh"),
    names=["IlistKNN", "ConstructIlist", "SegmentSort"],
    only_kernels=False
)

functions["IlistKNN"].template_par["k"].instances = k_instance_values
functions["IlistKNN"].block_size_expression = 32
functions["IlistKNN"].smem_size_expression = "blockDim.x * sizeof(PosId)"
functions["IlistKNN"].grid_size_expression = "isplitQ.element_count() - 1"

functions["ConstructIlist"].par["nnodes"].expression = "isplit.element_count() - 1"
functions["ConstructIlist"].par["nleaves"].expression = "leaves_npart.element_count()"
functions["ConstructIlist"].par["leaf_ilist_size"].expression = "leaf_ilist->element_count()"

functions["SegmentSort"].par["nkeys"].expression = "key.element_count()"
functions["SegmentSort"].par["nsegs"].expression = "isplit.element_count() - 1"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_knn.cu"), 
    functions = functions, 
    includes = default_includes + ["../knn.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                              fof.cuh                                             #
# ------------------------------------------------------------------------------------------------ #

functions = parse.get_functions_from_file(
    str(HERE / "fof.cuh"),
    names=["NodeFofAndIlist", "ParticleFof"],
    only_kernels=False
)

functions["NodeFofAndIlist"].par["nnodes"].expression = "isplit.element_count() - 1"
functions["NodeFofAndIlist"].par["nleaves"].expression = "leaf_igroup->element_count()"
functions["NodeFofAndIlist"].par["ilist_out_size"].expression = "ilist_out->element_count()"

functions["ParticleFof"].par["nnodes"].expression = "isplit.element_count() - 1"
functions["ParticleFof"].par["npart"].expression = "particle_igroup->element_count()"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_fof.cu"), 
    functions = functions, 
    includes = default_includes + ["../fof.cuh"]
)