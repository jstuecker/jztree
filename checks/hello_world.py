import jax
import jax.numpy as jnp
import jztree as jz

pos0 = jax.random.uniform(jax.random.PRNGKey(1), (1024*128,3), dtype=jnp.float32, minval=0.1, maxval=0.4)
rnn, inn = jz.knn.knn.jit(pos0, boxsize=None, k=8)

print("rnn:", rnn[0:3])

print("""Should be:
[[0.         0.00327169 0.00362817 0.00418469 0.00620413 0.00629311
  0.00657683 0.0065819 ]
 [0.         0.0018539  0.00218362 0.00325193 0.00418783 0.00457177
  0.00464585 0.00483315]
 [0.         0.00392929 0.00410999 0.00522736 0.00623543 0.00679859
  0.006818   0.00706907]]
""")