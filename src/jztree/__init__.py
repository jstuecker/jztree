from ._backend import load_backend as _load_backend

_jztree_cuda = _load_backend()

from . import jax_ext
from . import config
from . import stats
from . import data
from . import tools
from . import comm
from . import tree
from . import knn
from . import fof

del _load_backend