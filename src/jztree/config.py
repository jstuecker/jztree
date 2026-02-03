from dataclasses import dataclass

@dataclass(unsafe_hash=True)
class LoggingConfig():
    level : int = 1
    show_loc : bool = True

@dataclass(unsafe_hash=True)
class TreeConfig():
    # structure:
    max_leaf_size: int = 32
    coarse_fac: float = 6.0
    stop_coarsen: int = 1024

    # memory usage:
    alloc_fac_nodes: float = 1.0

    # distributed sort:
    nsamp: int = 1024

    # other:
    mass_centered: bool = True

@dataclass(unsafe_hash=True)
class TreeConfig():
    # structure:
    max_leaf_size: int = 32
    coarse_fac: float = 6.0
    stop_coarsen: int = 1024

    # memory usage:
    alloc_fac_nodes: float = 1.0

    # distributed sort:
    nsamp: int = 1024

    # other:
    mass_centered: bool = True

@dataclass(unsafe_hash=True)
class FofCatalogueConfig():
    npart_min: int = 20

@dataclass(frozen=True)
class FofConfig:
    alloc_fac_ilist: float = 32.
    alloc_fac_distr_links: float = 0.01

    tree: TreeConfig = TreeConfig(
        max_leaf_size = 48,
        coarse_fac = 8.,       
        alloc_fac_nodes = 1.,
        stop_coarsen = 2048,
        mass_centered = False
    )
    catalogue: FofCatalogueConfig = FofCatalogueConfig()

@dataclass(frozen=True)
class KNNConfig:
    alloc_fac_ilist: float = 256.

    tree: TreeConfig = TreeConfig(
        max_leaf_size = 48,
        coarse_fac = 8.,       
        alloc_fac_nodes = 1.,
        stop_coarsen = 2048,
        mass_centered = False
    )