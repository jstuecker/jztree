API Reference
=============

The API of jz-tree. Only functions that may be relevant for interfacing and calling the code
are listed. For developers we recommend to directly consider the source code.

jztree.config
-------------
.. automodule:: jztree.config
   :members:
   :member-order: bysource

jztree.data
-------------
.. automodule:: jztree.data
   :members: Pos, PosMass, ParticleData, RankIdx, TreeHierarchy, FofCatalogue, InteractionList, 
      LevelInfo, squeeze_catalogue, PackedArray
   :member-order: bysource

jztree.tree
------------
.. automodule:: jztree.tree
   :members: build_tree_hierarchy, grouped_dense_interaction_list, simplify_interaction_list,
      zsort, search_sorted_z, zsort_and_tree, zsort_and_tree_multi_type
   :member-order: bysource

jztree.knn
-----------
.. automodule:: jztree.knn
   :members: knn

jztree.fof
-----------
.. automodule:: jztree.fof
   :members: