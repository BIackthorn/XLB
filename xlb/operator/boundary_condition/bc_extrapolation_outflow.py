"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax
from functools import partial
import warp as wp
from typing import Any
from collections import Counter
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.boundary_condition.boundary_condition import (
    ImplementationStep,
    BoundaryCondition,
)
from xlb.operator.boundary_condition.boundary_condition_registry import (
    boundary_condition_registry,
)


class ExtrapolationOutflowBC(BoundaryCondition):
    """
    Extrapolation outflow boundary condition for a lattice Boltzmann method simulation.

    This class implements the extrapolation outflow boundary condition, which is a type of outflow boundary condition
    that uses extrapolation to avoid strong wave reflections.

    References
    ----------
    Geier, M., Schönherr, M., Pasquali, A., & Krafczyk, M. (2015). The cumulant lattice Boltzmann equation in three
    dimensions: Theory and validation. Computers & Mathematics with Applications, 70(4), 507-547.
    doi:10.1016/j.camwa.2015.05.001.
    """

    id = boundary_condition_registry.register_boundary_condition(__qualname__)

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy: PrecisionPolicy = None,
        compute_backend: ComputeBackend = None,
        indices=None,
        mesh_vertices=None,
    ):
        # Call the parent constructor
        super().__init__(
            ImplementationStep.STREAMING,
            velocity_set,
            precision_policy,
            compute_backend,
            indices,
            mesh_vertices,
        )

        # find and store the normal vector using indices
        self._get_normal_vec(indices)

        # Unpack the two warp functionals needed for this BC!
        if self.compute_backend == ComputeBackend.WARP:
            self.warp_functional, self.prepare_bc_auxilary_data = self.warp_functional

    def _get_normal_vec(self, indices):
        # Get the frequency count and most common element directly
        freq_counts = [Counter(coord).most_common(1)[0] for coord in indices]

        # Extract counts and elements
        counts = np.array([count for _, count in freq_counts])
        elements = np.array([element for element, _ in freq_counts])

        # Normalize the counts
        self.normal = counts // counts.max()

        # Reverse the normal vector if the most frequent element is 0
        if elements[np.argmax(counts)] == 0:
            self.normal *= -1
        return

    @partial(jit, static_argnums=(0,), inline=True)
    def _roll(self, fld, vec):
        """
        Perform rolling operation of a field with dimentions [q, nx, ny, nz] in a direction
        given by vec. All q-directions are rolled at the same time.
        # TODO: how to improve this for multi-gpu runs?
        """
        if self.velocity_set.d == 2:
            return jnp.roll(fld, (vec[0], vec[1]), axis=(1, 2))
        elif self.velocity_set.d == 3:
            return jnp.roll(fld, (vec[0], vec[1], vec[2]), axis=(1, 2, 3))

    @partial(jit, static_argnums=(0,), inline=True)
    def prepare_bc_auxilary_data(self, f_pre, f_post, bc_mask, missing_mask):
        """
        Prepare the auxilary distribution functions for the boundary condition.
        Since this function is called post-collisiotn: f_pre = f_post_stream and f_post = f_post_collision
        """
        sound_speed = 1.0 / jnp.sqrt(3.0)
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))

        # Roll boundary mask in the opposite of the normal vector to mask its next immediate neighbour
        neighbour = self._roll(boundary, -self.normal)

        # gather post-streaming values associated with previous time-step to construct the auxilary data for BC
        fpop = jnp.where(boundary, f_pre, f_post)
        fpop_neighbour = jnp.where(neighbour, f_pre, f_post)

        # With fpop_neighbour isolated, now roll it back to be positioned at the boundary for subsequent operations
        fpop_neighbour = self._roll(fpop_neighbour, self.normal)
        fpop_extrapolated = sound_speed * fpop_neighbour + (1.0 - sound_speed) * fpop

        # Use the iknown directions of f_postcollision that leave the domain during streaming to store the BC data
        opp = self.velocity_set.opp_indices
        known_mask = missing_mask[opp]
        f_post = jnp.where(jnp.logical_and(boundary, known_mask), fpop_extrapolated[opp], f_post)
        return f_post

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def apply_jax(self, f_pre, f_post, bc_mask, missing_mask):
        boundary = bc_mask == self.id
        new_shape = (self.velocity_set.q,) + boundary.shape[1:]
        boundary = lax.broadcast_in_dim(boundary, new_shape, tuple(range(self.velocity_set.d + 1)))
        return jnp.where(
            jnp.logical_and(missing_mask, boundary),
            f_pre[self.velocity_set.opp_indices],
            f_post,
        )

    def _construct_warp(self):
        # Set local constants
        sound_speed = self.compute_dtype(1.0 / wp.sqrt(3.0))
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)
        _c = self.velocity_set.c
        _q = self.velocity_set.q
        _opp_indices = self.velocity_set.opp_indices

        @wp.func
        def get_normal_vectors_2d(
            missing_mask: Any,
        ):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) == 1:
                    return -wp.vec2i(_c[0, l], _c[1, l])

        @wp.func
        def get_normal_vectors_3d(
            missing_mask: Any,
        ):
            for l in range(_q):
                if missing_mask[l] == wp.uint8(1) and wp.abs(_c[0, l]) + wp.abs(_c[1, l]) + wp.abs(_c[2, l]) == 1:
                    return -wp.vec3i(_c[0, l], _c[1, l], _c[2, l])

        # Construct the functionals for this BC
        @wp.func
        def functional(
            f_pre: Any,
            f_post: Any,
            f_aux: Any,
            missing_mask: Any,
        ):
            # Post-streaming values are only modified at missing direction
            _f = f_post
            for l in range(self.velocity_set.q):
                # If the mask is missing then take the opposite index
                if missing_mask[l] == wp.uint8(1):
                    _f[l] = f_pre[_opp_indices[l]]

            return _f

        @wp.func
        def prepare_bc_auxilary_data(
            f_pre: Any,
            f_post: Any,
            f_aux: Any,
            missing_mask: Any,
        ):
            # Preparing the formulation for this BC using the neighbour's populations stored in f_aux and
            # f_pre (posti-streaming values of the current voxel). We use directions that leave the domain
            # for storing this prepared data.
            _f = f_post
            for l in range(self.velocity_set.q):
                if missing_mask[l] == wp.uint8(1):
                    _f[_opp_indices[l]] = (self.compute_dtype(1.0) - sound_speed) * f_pre[l] + sound_speed * f_aux[l]
            return _f

        # Construct the warp kernel
        @wp.kernel
        def kernel2d(
            f_pre: wp.array3d(dtype=Any),
            f_post: wp.array3d(dtype=Any),
            bc_mask: wp.array3d(dtype=wp.uint8),
            missing_mask: wp.array3d(dtype=wp.bool),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data_2d(f_pre, f_post, bc_mask, missing_mask, index)
            _f_aux = _f_vec()

            # special preparation of auxiliary data
            if _boundary_id == wp.uint8(ExtrapolationOutflowBC.id):
                nv = get_normal_vectors_2d(_missing_mask)
                for l in range(self.velocity_set.q):
                    if _missing_mask[l] == wp.uint8(1):
                        # f_0 is the post-collision values of the current time-step
                        # Get pull index associated with the "neighbours" pull_index
                        pull_index = type(index)()
                        for d in range(self.velocity_set.d):
                            pull_index[d] = index[d] - (_c[d, l] + nv[d])
                        # The following is the post-streaming values of the neighbor cell
                        _f_aux[l] = _f_pre[l, pull_index[0], pull_index[1]]

            # Apply the boundary condition
            if _boundary_id == wp.uint8(ExtrapolationOutflowBC.id):
                # TODO: is there any way for this BC to have a meaningful kernel given that it has two steps after both
                # collision and streaming?
                _f = functional(_f_pre, _f_post, _f_aux, _missing_mask)
            else:
                _f = _f_post

            # Write the distribution function
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1]] = _f[l]

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            f_pre: wp.array4d(dtype=Any),
            f_post: wp.array4d(dtype=Any),
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)

            # read tid data
            _f_pre, _f_post, _boundary_id, _missing_mask = self._get_thread_data_3d(f_pre, f_post, bc_mask, missing_mask, index)
            _f_aux = _f_vec()

            # special preparation of auxiliary data
            if _boundary_id == wp.uint8(ExtrapolationOutflowBC.id):
                nv = get_normal_vectors_3d(_missing_mask)
                for l in range(self.velocity_set.q):
                    if _missing_mask[l] == wp.uint8(1):
                        # f_0 is the post-collision values of the current time-step
                        # Get pull index associated with the "neighbours" pull_index
                        pull_index = type(index)()
                        for d in range(self.velocity_set.d):
                            pull_index[d] = index[d] - (_c[d, l] + nv[d])
                        # The following is the post-streaming values of the neighbor cell
                        _f_aux[l] = _f_pre[l, pull_index[0], pull_index[1], pull_index[2]]

            # Apply the boundary condition
            if _boundary_id == wp.uint8(ExtrapolationOutflowBC.id):
                # TODO: is there any way for this BC to have a meaninful kernel given that it has two steps after both
                # collision and streaming?
                _f = functional(_f_pre, _f_post, _f_aux, _missing_mask)
            else:
                _f = _f_post

            # Write the distribution function
            for l in range(self.velocity_set.q):
                f_post[l, index[0], index[1], index[2]] = _f[l]

        kernel = kernel3d if self.velocity_set.d == 3 else kernel2d

        return (functional, prepare_bc_auxilary_data), kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f_pre, f_post, bc_mask, missing_mask):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[f_pre, f_post, bc_mask, missing_mask],
            dim=f_pre.shape[1:],
        )
        return f_post
