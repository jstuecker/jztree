from dataclasses import dataclass

@dataclass(unsafe_hash=True)
class LoggingConfig():
    level : int = 1
    show_loc : bool = True

@dataclass(unsafe_hash=True)
class RegularizationConfig():
    regularize_percentile: float = 90.
    max_volume_fac: float = 20.

@dataclass(unsafe_hash=True)
class TreeConfig():
    # structure:
    max_leaf_size: int = 32
    coarse_fac: float = 6.0
    stop_coarsen: int = 1024
    regularization: RegularizationConfig | None = None

    # memory usage:
    alloc_fac_nodes: float = 1.0

    # distributed sort:
    nsamp: int = 1024

    # other:
    mass_centered: bool = True

@dataclass(unsafe_hash=True)
class FofCatalogueConfig():
    npart_min: int = 20

@dataclass(unsafe_hash=True)
class FofConfig:
    alloc_fac_ilist: float = 32.
    alloc_fac_distr_links: float = 0.01

    tree: TreeConfig = TreeConfig(
        max_leaf_size = 48,
        coarse_fac = 8.,
        alloc_fac_nodes = 1.1,
        stop_coarsen = 2048,
        mass_centered = False
    )
    catalogue: FofCatalogueConfig = FofCatalogueConfig()

@dataclass(unsafe_hash=True)
class KNNConfig:
    alloc_fac_ilist: float = 256.

    tree: TreeConfig = TreeConfig(
        max_leaf_size = 48,
        coarse_fac = 8.,
        alloc_fac_nodes = 1.,
        stop_coarsen = 2048,
        mass_centered = False,
        regularization=RegularizationConfig()
    )