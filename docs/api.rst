API Reference
=============

The API of jz-tree. Only functions that may be relevant for interfacing and calling the code
are listed. For developers we recommend to directly consider the source code.

jztree.config
-------------
This module contains config dataclasses that bundle arguments that modify low-level details
of the code. Relevant configs are always passed through function arguments and trigger
jit-recompilation if modified. The main reasons to pass non-default configuartions 
is if some allocation turned out too small, or to optimize memory usage.

.. automodule:: jztree.config
   :members: RegularizationConfig, TreeConfig, FofCatalogueConfig, FofConfig, KNNConfig
   :member-order: bysource

jztree.data
-------------
This module contains some dataclasses that define interfaces between different parts of the code.
In particular relevant are the particle data classes, like :class:`Pos` that are needed in
multi-GPU setups, to keep track of particle counts.

.. automodule:: jztree.data
   :members: Pos, PosMass, ParticleData, RankIdx, TreeHierarchy, FofCatalogue, InteractionList, Label,
      LevelInfo, PackedArray
   :member-order: bysource

.. Some helper functions:
.. currentmodule:: jztree.data
.. autofunction:: expand_particles
.. autofunction:: flatten_particles
.. autofunction:: pad_particles
.. autofunction:: squeeze_catalogue

jztree.tree
------------
This module contains functions for sorting particles into z-order, for building a plane-based
tree hierarchy and for defining interaction lists.

.. automodule:: jztree.tree
   :members: build_tree_hierarchy, grouped_dense_interaction_list, simplify_interaction_list,
      zsort, search_sorted_z, zsort_and_tree, zsort_and_tree_multi_type
   :member-order: bysource

jztree.knn
-----------
This module contains functions for doing k-nearest neighbour search.

.. automodule:: jztree.knn
   :members: knn

jztree.fof
-----------
This module contains functions to calculate friends-of-friends (FoF) group labels, for
bringing particles into group order and for calculating a group catalogue.

.. automodule:: jztree.fof
   :members: fof_labels, distr_fof_labels, fof_and_catalogue, fof_is_superset
   :member-order: bysource