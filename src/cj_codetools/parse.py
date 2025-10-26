try:
    import tree_sitter_cuda
    from tree_sitter import Language, Parser, Query, QueryCursor, Node, Tree
except ImportError as e:
    raise ImportError("Please install tree-sitter and tree-sitter-cuda packages to use this module.") from e

from dataclasses import dataclass, field

@dataclass
class ParamInfo():
    type : str = ""
    name : str = ""
    is_ptr : bool = False
    is_const : bool = False
    expression : str = ""
    init_zero : bool = False

@dataclass
class TemplateParamInfo():
    type : str = ""
    name : str = ""
    instances : list[str] = ()

@dataclass
class FunctionInfo():
    name : str
    par : dict[str, ParamInfo]
    type : str = "void"
    is_kernel : bool = False
    template_par : dict[str, ParamInfo] = field(default_factory=dict)
    block_size_expression : str = ""
    grid_size_expression : str = ""
    init_outputs_zero : bool = False

CUDA = Language(tree_sitter_cuda.language())
parser = Parser(CUDA)

def node_text(node: Node, txt: str) -> str:
    return txt[node.start_byte:node.end_byte]

def query(node: Node, query_src: str) -> dict:
    q = QueryCursor(Query(CUDA, query_src))
    caps = q.captures(node)

    return caps

def interprete_parameter_list(node_param: Node, txt: str) -> dict[str, ParamInfo]:
    assert node_param.type == "parameter_list"

    res = {}
    for c in node_param.named_children:
        if c.type == "comment": continue

        assert c.type == "parameter_declaration"

        pinfo = ParamInfo()

        pinfo.is_const = len(query(c, '(type_qualifier)? @tq (#eq? @tq "const")')) > 0

        pinfo.type = node_text(c.child_by_field_name("type"), txt)

        decl = c.child_by_field_name("declarator")
        if decl.type == "identifier":
            pinfo.name = node_text(decl, txt)
        elif decl.type == "pointer_declarator":
            pinfo.is_ptr = True
            pinfo.name = node_text(decl.child_by_field_name("declarator"), txt)
        else:
            raise ValueError("Unknown type %s" % decl.type)
        
        res[pinfo.name] = pinfo

    return res

def interprete_template_list(node_param: Node, txt: str) -> dict[str, TemplateParamInfo]:
    assert node_param.type == "template_parameter_list"

    res = {}
    for c in node_param.named_children:
        assert c.type == "parameter_declaration"

        pinfo = TemplateParamInfo()
        pinfo.type = node_text(c.child_by_field_name("type"), txt)
        pinfo.name = node_text(c.child_by_field_name("declarator"), txt)
        pinfo.instances = []
        
        res[pinfo.name] = pinfo

    return res

def get_functions(node: Node, txt: str) -> dict[str, FunctionInfo]:
    query_func = f"""
        (function_definition
            ("__global__")? @fglobal
            type: (_) @ftype
            declarator: (function_declarator
                declarator: (identifier) @fname
                parameters: (parameter_list) @fparam
            )
        ) @node
    """

    cursor = QueryCursor(Query(CUDA, query_func))
    res = {}
    for i,match in cursor.matches(node):
        if (not "ftemp" in match) and () :
            print("template function")
        new_func = FunctionInfo(
            name = node_text(match["fname"][0], txt),
            par = interprete_parameter_list(match["fparam"][0], txt),
            type = node_text(match["ftype"][0], txt),
            is_kernel = match.get("fglobal") is not None
        )
        parent = match["node"][0].parent
        if parent.type == "template_declaration":
            tpar_list = parent.child_by_field_name("parameters")
            new_func.template_par = interprete_template_list(tpar_list, txt)

        res[new_func.name] = new_func
        
    return res

def build_tree_sitter(file_path: str) -> tuple[Tree, str]:
    with open(file_path, 'r') as f:
        txt = f.read()
    tree = parser.parse(txt.encode())
    return tree, txt

def get_functions_from_file(file_path: str, only_kernels: bool = True, names: tuple[str] = None
                            ) -> dict[str, FunctionInfo]:
    tree, txt = build_tree_sitter(file_path)
    funcs = get_functions(tree.root_node, txt)
    
    if only_kernels:
        funcs = {k:v for k,v in funcs.items() if v.is_kernel}

    if names is not None:
        funcs = {k:v for k,v in funcs.items() if k in names}
        for name in names:
            if name not in funcs:
                raise ValueError(f"Function {name} not found in file {file_path}")

    return funcs
