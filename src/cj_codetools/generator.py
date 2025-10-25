from .parse import FunctionInfo
from jinja2 import Environment, PackageLoader, select_autoescape
from dataclasses import dataclass

env = Environment(
    loader=PackageLoader("cj_codetools", "templates"),
    autoescape=select_autoescape()
)

def create_ffi_call(func: FunctionInfo) -> str:
    template = env.get_template("template_ffi_call.j2")
    return template.render(f=func)

def create_ffi_module(funcs: list[FunctionInfo], includes: tuple[str] = ()) -> str:
    template = env.get_template("template_ffi_module.j2")
    return template.render(functions=funcs, includes=includes)