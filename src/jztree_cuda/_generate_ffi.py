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
    names=["KnnLeaf2Leaf", "KnnNode2Node", "SegmentSort"],
    only_kernels=False
)

functions["KnnLeaf2Leaf"].template_par["k"].instances = k_instance_values
functions["KnnLeaf2Leaf"].block_size_expression = 32
functions["KnnLeaf2Leaf"].smem_size_expression = "blockDim.x * sizeof(PosId<3>)"
functions["KnnLeaf2Leaf"].grid_size_expression = "splQ.element_count() - 1"

functions["KnnNode2Node"].par["size_parents"].expression = "parent_spl.element_count() - 1"
functions["KnnNode2Node"].par["size_nodes"].expression = "nodes_npart.element_count()"
functions["KnnNode2Node"].par["node_ilist_size"].expression = "node_ilist_ioth->element_count()"

functions["SegmentSort"].par["size_segs"].expression = "spl.element_count() - 1"
functions["SegmentSort"].par["size_keys"].expression = "key.element_count()"

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
    names=["FofNode2Node", "FofLeaf2Leaf", "InsertLinks", "NodeToChildLabel"],
    only_kernels=False
)

functions["FofNode2Node"].par["size_parent"].expression = "parent_spl.element_count() - 1"
functions["FofNode2Node"].par["size_node"].expression = "node_igroup->element_count()"
functions["FofNode2Node"].par["size_node_ilist"].expression = "node_ilist->element_count()"

functions["FofLeaf2Leaf"].par["size_leaves"].expression = "spl.element_count() - 1"
functions["FofLeaf2Leaf"].par["size_part"].expression = "part_igroup->element_count()"

functions["InsertLinks"].par["size_links"].expression = "igroupLinkA.element_count()"
functions["InsertLinks"].par["size_groups"].expression = "igroup_in.element_count()"

functions["NodeToChildLabel"].init_outputs_zero = True
functions["NodeToChildLabel"].grid_size_expression = "parent_igroup.element_count()"
functions["NodeToChildLabel"].par["size_parent"].expression = "parent_igroup.element_count()"

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

functions["PosZorderSort"].template_par["dim"].instances = (2,3)
functions["PosZorderSort"].template_par["dim"].expression = "pos_in.dimensions()[1]"
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

# ------------------------------------------------------------------------------------------------ #
#                                             tools.cuh                                            #
# ------------------------------------------------------------------------------------------------ #

functions = parse.get_functions_from_file(
    str(HERE / "tools.cuh"),
    names=["RearangeSegments"],
    only_kernels=False
)

functions["RearangeSegments"].par["size_seg"].expression = "seg_spl_out.element_count()-1"

gen.generate_ffi_module_file(
    output_file = str(HERE / "generated/ffi_tools.cu"), 
    functions = functions, 
    includes = default_includes + ["../tools.cuh"]
)