from .parse import FunctionInfo
from jinja2 import Environment, PackageLoader, select_autoescape
from dataclasses import dataclass, replace
import os

env = Environment(
    loader=PackageLoader("cj_codetools", "templates"),
    autoescape=select_autoescape()
)

def simplify_and_validate(func: FunctionInfo) -> FunctionInfo:
    if not func.is_kernel:
        assert tuple(func.par.keys())[0] == "stream", "All Host functions must use stream as first parameter"
        del func.par["stream"] # will be added automatically

    # convert pars to lists for easier templating
    func = replace(
        func, 
        par = list(func.par.values())
    )

    if func.template_instances is None:
        for name,p in func.template_par.items():
            if len(p.instances) == 0:
                raise ValueError(f"Please define instances for template parameter {p.name} "
                                f"in function {func.name}.")

    return func

def create_ffi_call(func: FunctionInfo) -> str:
    func = simplify_and_validate(func)

    template = env.get_template("template_ffi_call.j2")
    return template.render(f=func)

def create_ffi_module_code(funcs: list[FunctionInfo], 
                           includes: tuple[str] = (), 
                           module_name: str = "ffi_module") -> str:
    if type(funcs) is dict:
        funcs = list(funcs.values())

    new_funcs = []
    for f in funcs:
        new_funcs.append(simplify_and_validate(f))

    template = env.get_template("template_ffi_module.j2")
    return template.render(functions=new_funcs, includes=includes, module_name=module_name)

def generate_ffi_module_file(output_file: str, 
                             functions: list[FunctionInfo],
                             includes: tuple[str] = (),
                             module_name: str | None = None) -> None:
    if module_name is None:
        module_name = output_file.split("/")[-1].split(".")[0]

    code = create_ffi_module_code(functions, includes, module_name)

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            txt = f.read()
        if txt == code:
            print(f"No changes to generated file at {output_file}")
            return
        else:
            print(f"Updating generated file at {output_file}")
    else: 
        print(f"Generating new file file at {output_file}")
    
    with open(output_file, 'w') as f:
        f.write(code)