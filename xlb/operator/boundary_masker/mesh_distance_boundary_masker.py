# Base class for all equilibriums

import numpy as np
import warp as wp
import jax
from typing import Any
from xlb.velocity_set.velocity_set import VelocitySet
from xlb.precision_policy import PrecisionPolicy
from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator


class MeshDistanceBoundaryMasker(Operator):
    """
    Operator for creating a boundary missing_mask from an STL file
    """

    def __init__(
        self,
        velocity_set: VelocitySet,
        precision_policy: PrecisionPolicy,
        compute_backend: ComputeBackend.WARP,
    ):
        # Call super
        super().__init__(velocity_set, precision_policy, compute_backend)

        # Raise error if used for 2d examples:
        if self.velocity_set.d == 2:
            raise NotImplementedError("This Operator is not implemented in 2D!")

        # Also using Warp kernels for JAX implementation
        if self.compute_backend == ComputeBackend.JAX:
            self.warp_functional, self.warp_kernel = self._construct_warp()

    @Operator.register_backend(ComputeBackend.JAX)
    def jax_implementation(
        self,
        bc,
        origin,
        spacing,
        id_number,
        bc_mask,
        missing_mask,
        f_field,
        start_index=(0, 0, 0),
    ):
        raise NotImplementedError(f"Operation {self.__class__.__name} not implemented in JAX!")

    def _construct_warp(self):
        # Make constants for warp
        _c = self.velocity_set.c
        _q = wp.constant(self.velocity_set.q)
        _opp_indices = self.velocity_set.opp_indices

        @wp.func
        def check_index_bounds(index: wp.vec3i, shape: wp.vec3i):
            is_in_bounds = index[0] >= 0 and index[0] < shape[0] and index[1] >= 0 and index[1] < shape[1] and index[2] >= 0 and index[2] < shape[2]
            return is_in_bounds

        @wp.func
        def index_to_position(index: wp.vec3i, origin: wp.vec3, spacing: wp.vec3):
            # position of the point
            ijk = wp.vec3(wp.float32(index[0]), wp.float32(index[1]), wp.float32(index[2]))
            ijk = ijk + wp.vec3(0.5, 0.5, 0.5)  # cell center
            pos = wp.cw_mul(ijk, spacing) + origin
            return pos

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            mesh_id: wp.uint64,
            origin: wp.vec3,
            spacing: wp.vec3,
            id_number: wp.int32,
            bc_mask: wp.array4d(dtype=wp.uint8),
            missing_mask: wp.array4d(dtype=wp.bool),
            f_field: wp.array4d(dtype=Any),
            start_index: wp.vec3i,
        ):
            # get index
            i, j, k = wp.tid()

            # Get local indices
            index = wp.vec3i()
            index[0] = i - start_index[0]
            index[1] = j - start_index[1]
            index[2] = k - start_index[2]

            # position of the point
            pos_bc_cell = index_to_position(index, origin, spacing)

            # Find the fractional distance to the mesh in each direction
            for l in range(1, _q):
                dir = wp.vec3f(float(_c[0, l]),float(_c[1, l]), float(_c[2, l]))
                len = wp.length(dir)
                # Max length depends on ray direction  (diagonals are longer)
                max_length = wp.sqrt(
                    (spacing[0] * wp.float32(dir[0])) ** 2.0
                    + (spacing[1] * wp.float32(dir[1])) ** 2.0
                    + (spacing[2] * wp.float32(dir[2])) ** 2.0
                )
                query = wp.mesh_query_ray(mesh_id, pos_bc_cell, dir / len, max_length)
                # if query.result and query.sign > 0:
                if query.result:
                    # Set the boundary id and missing_mask
                    bc_mask[0, index[0], index[1], index[2]] = wp.uint8(id_number)
                    missing_mask[_opp_indices[l], index[0], index[1], index[2]] = True

                    # get position of the mesh triangle that intersects with the ray
                    pos_mesh = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
                    dist = wp.length(pos_mesh - pos_bc_cell)
                    # wp.printf('Dist: %f, Max_length: %f\n', dist, max_length)
                    f_field[l, index[0], index[1], index[2]] = self.store_dtype(dist/max_length)
                    if (dist > max_length or dist <= 0 or f_field[l, index[0], index[1], index[2]] >= 1.0):
                        wp.printf('Dist: %f, Max_length: %f\n', dist, max_length)


        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        bc,
        origin,
        spacing,
        bc_mask,
        missing_mask,
        f_field,
        start_index=(0, 0, 0),
    ):
        assert bc.mesh_vertices is not None, f'Please provide the mesh vertices for {bc.__class__.__name__} BC using keyword "mesh_vertices"!'
        assert bc.indices is None, f"Cannot find the implicit distance to the boundary for {bc.__class__.__name__} without a mesh!"
        assert (
            bc.mesh_vertices.shape[1] == self.velocity_set.d
        ), "Mesh points must be reshaped into an array (N, 3) where N indicates number of points!"
        assert (
            f_field is not None and f_field.shape == missing_mask.shape
        ), "To compute and store the implicit distance to the boundary for this BC, use a population field!"
        mesh_vertices = bc.mesh_vertices
        id_number = bc.id

        # We are done with bc.mesh_vertices. Remove them from BC objects
        bc.__dict__.pop("mesh_vertices", None)

        # Ensure this masker is called only for BCs that need implicit distance to the mesh
        assert bc.needs_mesh_distance, 'Please use "MeshBoundaryMasker" if this BC does NOT need mesh distance!'

        mesh_indices = np.arange(mesh_vertices.shape[0])
        mesh = wp.Mesh(
            points=wp.array(mesh_vertices, dtype=wp.vec3),
            indices=wp.array(mesh_indices, dtype=int),
        )

        # Convert input tuples to warp vectors
        origin = wp.vec3(origin[0], origin[1], origin[2])
        spacing = wp.vec3(spacing[0], spacing[1], spacing[2])
        start_index = wp.vec3i(start_index[0], start_index[1], start_index[2])
        mesh_id = wp.uint64(mesh.id)

        print(["Setting up mesh distance boundary masker on mesh with ", len(mesh.points), " vertices, ", len(mesh.indices), " faces"])
        print(mesh.points)
        print(mesh.indices)
        print(mesh_vertices.shape)
        print(mesh_indices.shape)

        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                mesh_id,
                origin,
                spacing,
                id_number,
                bc_mask,
                missing_mask,
                f_field,
                start_index,
            ],
            dim=missing_mask.shape[1:],
        )

        return bc_mask, missing_mask, f_field
