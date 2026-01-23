from pathlib import Path
import os

from jax_ffi_gen import parse, generator as gen

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
    names=["NodeFofAndIlist", "ParticleFof", "InsertLinks", "NodeToChildLabel"],
    only_kernels=False
)

functions["NodeFofAndIlist"].par["nnodes"].expression = "isplit.element_count() - 1"
functions["NodeFofAndIlist"].par["nleaves"].expression = "leaf_igroup_in.element_count()"
functions["NodeFofAndIlist"].par["ilist_out_size"].expression = "ilist_out->element_count()"

functions["ParticleFof"].par["nnodes"].expression = "isplit.element_count() - 1"
functions["ParticleFof"].par["npart"].expression = "particle_igroup->element_count()"

functions["InsertLinks"].par["size_links"].expression = "igroupLinkA.element_count()"
functions["InsertLinks"].par["size_groups"].expression = "igroup_in.element_count()"

functions["NodeToChildLabel"].init_outputs_zero = True
functions["NodeToChildLabel"].grid_size_expression = "node_igroup.element_count()"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_fof.cu"),
    functions = functions,
    includes = default_includes + ["../fof.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                             sort.cuh                                             #
# ------------------------------------------------------------------------------------------------ #

functions = parse.get_functions_from_file(
    str(HERE / "sort.cuh"),
    names=["PosZorderSort", "SearchSortedZ"],
    only_kernels=False
)

functions["PosZorderSort"].par["size"].expression = "pos_in.element_count()/3"
functions["PosZorderSort"].par["tmp_bytes"].expression = "tmp_buffer->size_bytes()"

functions["SearchSortedZ"].par["n_have"].expression = "posz_have.element_count()/3"
functions["SearchSortedZ"].par["n_query"].expression = "posz_query.element_count()/3"
functions["SearchSortedZ"].grid_size_expression = "div_ceil(n_query, block_size)"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_sort.cu"), 
    functions = functions, 
    includes = default_includes + ["../sort.cuh"]
)

# ------------------------------------------------------------------------------------------------ #
#                                             tree.cuh                                             #
# ------------------------------------------------------------------------------------------------ #

functions = parse.get_functions_from_file(
    str(HERE / "tree.cuh"),
    names=["FlagLeafBoundaries", "FindNodeBoundaries", "GetNodeGeometry", "CenterOfMass",
           "GetBoundaryExtendPerLevel"],
    only_kernels=False
)

functions["FlagLeafBoundaries"].par["size_part"].expression = "posz.element_count()/3"
functions["FlagLeafBoundaries"].grid_size_expression = "div_ceil(size_part+1, block_size)"
functions["FlagLeafBoundaries"].smem_size_expression = "(block_size + 2*scan_size + 1) * sizeof(int32_t)"

functions["FindNodeBoundaries"].par["size_nodes"].expression = "nodes_levels->element_count()"
functions["FindNodeBoundaries"].grid_size_expression = "div_ceil(size_nodes, block_size)"

functions["GetNodeGeometry"].par["size_nodes"].expression = "level->element_count()"
functions["GetNodeGeometry"].par["size_part"].expression = "pos.element_count()/3"
functions["GetNodeGeometry"].grid_size_expression = "div_ceil(size_nodes, block_size)"

functions["GetBoundaryExtendPerLevel"].par["size"].expression = "posz.element_count()/3"
functions["GetBoundaryExtendPerLevel"].template_par["left"].instances = ["true", "false"]

functions["CenterOfMass"].grid_size_expression = "div_ceil(isplit.element_count() - 1, block_size)"
functions["CenterOfMass"].par["nnodes"].expression = "isplit.element_count() - 1"


gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_tree.cu"), 
    functions = functions, 
    includes = default_includes + ["../tree.cuh"]
)