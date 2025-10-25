try:
    import tree_sitter_cuda
    from tree_sitter import Language, Parser, Query, QueryCursor, Node
except ImportError as e:
    raise ImportError("Please install tree-sitter and tree-sitter-cuda packages to use this module.") from e

from dataclasses import dataclass

@dataclass
class ParamInfo():
    type : str = ""
    name : str = ""
    is_ptr : bool = False
    is_const : bool = False

@dataclass
class TemplateParamInfo():
    type : str = ""
    name : str = ""
    instances : list[str] = ()

@dataclass
class FunctionInfo():
    name : str
    par : list[ParamInfo]
    type : str = "void"
    is_kernel : bool = False
    template_par : list[ParamInfo] = ()

CUDA = Language(tree_sitter_cuda.language())
parser = Parser(CUDA)

def node_text(node: Node, txt: str) -> str:
    return txt[node.start_byte:node.end_byte]

def query(node: Node, query_src: str) -> dict:
    q = QueryCursor(Query(CUDA, query_src))
    caps = q.captures(node)

    return caps

def interprete_parameter_list(node_param: Node, txt: str) -> list[ParamInfo]:
    assert node_param.type == "parameter_list"

    res = []
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
        
        res.append(pinfo)

    return res

def interprete_template_list(node_param: Node, txt: str) -> list[TemplateParamInfo]:
    assert node_param.type == "template_parameter_list"

    res = []
    for c in node_param.named_children:
        assert c.type == "parameter_declaration"

        pinfo = TemplateParamInfo()
        pinfo.type = node_text(c.child_by_field_name("type"), txt)
        pinfo.name = node_text(c.child_by_field_name("declarator"), txt)
        pinfo.instances = []
        
        res.append(pinfo)

    return res

def get_functions(node: Node, txt: str, name: str | None = None) -> dict[str, FunctionInfo]:
    if name is not None:
        name_match = f'(#eq? @fname "{name}")'
    else:
        name_match=""

    query_func = f"""
        (function_definition
            ("__global__")? @fglobal
            type: (_) @ftype
            declarator: (function_declarator
                declarator: (identifier) @fname {name_match}
                parameters: (parameter_list) @fparam
            )
        ) @node
    """

    cursor = QueryCursor(Query(CUDA, query_func))
    # res = {}
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