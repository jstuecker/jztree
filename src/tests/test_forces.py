import pytest
import custom_jax as cj
import jax
import jax.numpy as jnp
import numpy as np

def test_potential():
    x = jax.random.normal(jax.random.PRNGKey(0), (8*1024, 3)).block_until_ready()

    phi0 = cj.forces.potential_pure_jax_jit(x, eps=1e-2)

    for block_size in (32, 64, 128):
        phi1 = cj.forces.potential_jit(x, block_size=block_size, eps=1e-2)

        assert np.allclose(phi0, phi1, rtol=1e-4)

def test_force():
    x = jax.random.normal(jax.random.PRNGKey(0), (8*1024, 3)).block_until_ready()

    phi0 = cj.forces.potential_pure_jax_jit(x, eps=1e-2)
    acc0 = cj.forces.force_pure_jax_jit(x, eps=1e-2)

    for block_size in (32, 64, 128):
        acc1 = cj.forces.force_jit(x, block_size=block_size, eps=1e-2)
        acc2, phi2 =  cj.forces.force_jit(x, block_size=32, eps=1e-2, get_potential=True)

        assert np.allclose(acc0, acc1, rtol=1e-2)
        assert np.allclose(acc0, acc2, rtol=1e-2)
        assert np.allclose(phi0, phi2, rtol=1e-2)

def test_ilist_force():
    for n in 1024, 3333, 13231:
        x = jax.random.normal(jax.random.PRNGKey(0), (n, 3)).block_until_ready()

        phi0 = cj.forces.potential_pure_jax_jit(x, eps=1e-2)
        acc0 = cj.forces.force_pure_jax_jit(x, eps=1e-2)

        isplit = jnp.append(jnp.arange(0, n, 32), jnp.array([n]))
        # Add some random shifts so that we don't always have the same number of interactions
        isplit = isplit.at[1:-1].add(jax.random.randint(jax.random.PRNGKey(0), len(isplit) - 2, -16, 16))
        iarange = jnp.arange(0, len(isplit)-1)
        interactions = jnp.stack(jnp.meshgrid(iarange, iarange, indexing='ij'), axis=-1).reshape(-1, 2)

        for block_size, interactions_per_block in ((32, None), (32, 7), (64, 16), (128, 99)): 
            # Any combination of block size and interactions per block should be valid.
            # These are just parameters that may affect execution speed
            acc1, phi1 = cj.forces.ilist_force_jit(x, isplit, interactions, block_size=block_size, interactions_per_block=interactions_per_block, eps=1e-2, get_potential=True)

            assert np.allclose(acc0, acc1, atol=1e-2)
            assert np.allclose(phi0, phi1, rtol=1e-2)

def test_ilist_bwd_force():
    # Test the backward pass of the force and potential calculation
    for n in 1024, 3333:
        xm = jax.random.normal(jax.random.PRNGKey(0), (n, 4)).block_until_ready()

        def loss_fphi(xm):
            force = cj.forces.force_pure_jax_jit(xm[...,0:3], xm[...,3], eps=1e-2)
            phi = cj.forces.potential_pure_jax_jit(xm[...,0:3], xm[...,3], eps=1e-2)
            return jnp.sum(jnp.abs(phi)) + jnp.sum(force**2)
        
        loss0 = loss_fphi(xm)
        xmgrad0 = jax.grad(loss_fphi)(xm)

        isplit = jnp.append(jnp.arange(0, n, 32), jnp.array([n]))
        # # Add some random shifts so that we don't always have the same number of interactions
        isplit = isplit.at[1:-1].add(jax.random.randint(jax.random.PRNGKey(0), len(isplit) - 2, -16, 16))
        iarange = jnp.arange(0, len(isplit)-1)
        interactions = jnp.stack(jnp.meshgrid(iarange, iarange, indexing='ij'), axis=-1).reshape(-1, 2)

        for block_size, interactions_per_block in ((32, None), (32, 7), (64, 16), (128, 99)): 
            def loss_fphi_ilist(xm):
                fphi = cj.forces.ilist_fphi(xm, isplit, interactions, eps=1e-2, block_size=block_size, interactions_per_block=interactions_per_block)
                return jnp.sum(jnp.abs(fphi[...,3])) + jnp.sum(fphi[...,0:3]**2)
            
            loss1 = loss_fphi_ilist(xm)

            assert np.allclose(loss0, loss1, rtol=1e-4)
            xmgrad1 = jax.grad(loss_fphi_ilist)(xm)

            assert np.allclose(xmgrad0, xmgrad1, rtol=1e-2)
