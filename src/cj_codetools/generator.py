from .parse import FunctionInfo
from jinja2 import Environment, PackageLoader, select_autoescape
from dataclasses import dataclass

env = Environment(
    loader=PackageLoader("cj_codetools", "templates"),
    autoescape=select_autoescape()
)

def validate_function_info(func: FunctionInfo):
    for p in func.template_par:
        if len(p.instances) == 0:
            raise ValueError(f"Please define instances for template parameter {p.name} "
                             f"in function {func.name}.")

def create_ffi_call(func: FunctionInfo) -> str:
    validate_function_info(func)

    template = env.get_template("template_ffi_call.j2")
    return template.render(f=func)

def create_ffi_module_code(funcs: list[FunctionInfo], includes: tuple[str] = ()) -> str:
    if type(funcs) is dict:
        funcs = list(funcs.values())

    for f in funcs:
        validate_function_info(f)

    template = env.get_template("template_ffi_module.j2")
    return template.render(functions=funcs, includes=includes)

def generate_ffi_module_file(output_file: str, 
                             functions: list[FunctionInfo],
                             includes: tuple[str] = ()) -> None:
    code = create_ffi_module_code(functions, includes)

    with open(output_file, 'r') as f:
        txt = f.read()
    if txt != code:
        print(f"Updated generated file at {output_file}")
        with open(output_file, 'w') as f:
            f.write(code)
    else:
        print(f"No changes to generated file at {output_file}")