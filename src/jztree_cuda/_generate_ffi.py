from pathlib import Path
import os

from jax_ffi_gen import parse, generator as gen

HERE = Path(__file__).resolve().parent

dimensions = (2,3)
pos_types = ("float", "double")
pos_types_sort = ("float", "double", "int32_t", "int64_t")

k_instance_values = (4, 8, 12, 16, 32, 64)
default_includes = ["../common/math.cuh"]

def add_dim_dtype_templates(func, buf_from, dimensions=dimensions, pos_types=pos_types):
    func.template_par["dim"].instances = dimensions
    func.template_par["dim"].expression = f"{buf_from}.dimensions()[1]"
    func.template_par["tvec"].instances = pos_types
    func.template_par["tvec"].expression = f"{buf_from}.element_type()"

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
functions["KnnLeaf2Leaf"].smem_size_expression = "blockDim.x * sizeof(PosId<3,float>)"
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
    names=["DtypeTest", "PosZorderSort", "SearchSortedZ"],
    only_kernels=False
)

# For DtypeTest: float_type is a dtype parameter that maps types to ffi::DataType enum values
functions["DtypeTest"].template_par["in_type"].instances = ("float", "double")
functions["DtypeTest"].template_par["in_type"].expression = "in.element_type()"
functions["DtypeTest"].template_par["out_type"].instances = ("float", "double")
functions["DtypeTest"].template_par["out_type"].expression = "out->element_type()"
functions["DtypeTest"].template_par["offset"].instances = (0, 10)

# functions["PosZorderSort"].template_par["dim"].instances = dimensions
# functions["PosZorderSort"].template_par["dim"].expression = "pos_in.dimensions()[1]"
# functions["PosZorderSort"].template_par["tvec"].instances = pos_types_sort
# functions["PosZorderSort"].template_par["tvec"].expression = "pos_in.element_type()"
add_dim_dtype_templates(functions["PosZorderSort"], "pos_in")
functions["PosZorderSort"].par["size"].expression = "pos_in.dimensions()[0]"
functions["PosZorderSort"].par["tmp_bytes"].expression = "tmp_buffer->size_bytes()"

# functions["SearchSortedZ"].template_par["dim"].instances = dimensions
# functions["SearchSortedZ"].template_par["dim"].expression = "posz_have.dimensions()[1]"
# functions["SearchSortedZ"].template_par["tvec"].instances = pos_types
# functions["SearchSortedZ"].template_par["tvec"].expression = "posz_have.element_type()"
add_dim_dtype_templates(functions["SearchSortedZ"], "posz_have", pos_types=pos_types_sort)
functions["SearchSortedZ"].par["n_have"].expression = "posz_have.dimensions()[0]"
functions["SearchSortedZ"].par["n_query"].expression = "posz_query.dimensions()[0]"
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

# functions["FlagLeafBoundaries"].template_par["dim"].instances = dimensions
# functions["FlagLeafBoundaries"].template_par["dim"].expression = "posz.dimensions()[1]"
# functions["FlagLeafBoundaries"].template_par["tpos"].instances = pos_types
# functions["FlagLeafBoundaries"].template_par["tpos"].expression = "posz.element_type()"
add_dim_dtype_templates(functions["FlagLeafBoundaries"], "posz")
functions["FlagLeafBoundaries"].par["size_part"].expression = "posz.dimensions()[0]"
functions["FlagLeafBoundaries"].grid_size_expression = "div_ceil(size_part+1, block_size)"
functions["FlagLeafBoundaries"].smem_size_expression = "(block_size + 2*scan_size + 1) * sizeof(int32_t)"

add_dim_dtype_templates(functions["FindNodeBoundaries"], "pos_in")
functions["FindNodeBoundaries"].par["size_nodes"].expression = "nodes_levels->element_count()"
functions["FindNodeBoundaries"].grid_size_expression = "div_ceil(size_nodes, block_size)"

add_dim_dtype_templates(functions["GetNodeGeometry"], "pos")
functions["GetNodeGeometry"].par["size_nodes"].expression = "level->element_count()"
functions["GetNodeGeometry"].par["size_part"].expression = "pos.dimensions()[0]"
functions["GetNodeGeometry"].grid_size_expression = "div_ceil(size_nodes, block_size)"

add_dim_dtype_templates(functions["GetBoundaryExtendPerLevel"], "posz")
functions["GetBoundaryExtendPerLevel"].par["size"].expression = "posz.dimensions()[0]"
functions["GetBoundaryExtendPerLevel"].template_par["left"].instances = ["true", "false"]

# add_dim_dtype_templates(functions["CenterOfMass"], "posz")
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