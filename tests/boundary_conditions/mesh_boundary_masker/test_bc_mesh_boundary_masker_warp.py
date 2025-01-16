import pytest
import numpy as np
import xlb
import trimesh
import warp as wp
from xlb.utils import save_fields_vtk, save_image
from xlb.compute_backend import ComputeBackend
from xlb import DefaultConfig
from xlb.grid import grid_factory
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    FullwayBounceBackBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    HybridBC,
)

def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, backend=ComputeBackend.WARP)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.WARP,
        velocity_set=vel_set,
    )


# @pytest.mark.parametrize(
#     "dim,velocity_set,grid_shape",
#     [
#         (2, xlb.velocity_set.D2Q9, (50, 50)),
#         (2, xlb.velocity_set.D2Q9, (100, 100)),
#         (3, xlb.velocity_set.D3Q19, (50, 50, 50)),
#         (3, xlb.velocity_set.D3Q19, (100, 100, 100)),
#         (3, xlb.velocity_set.D3Q27, (50, 50, 50)),
#         (3, xlb.velocity_set.D3Q27, (100, 100, 100)),
#     ],
# )

def define_boundary_indices(grid, grid_shape, velocity_set):
    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)
    inlet = box_no_edge["left"]
    outlet = box_no_edge["right"]
    walls = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
    walls = np.unique(np.array(walls), axis=-1).tolist()

    # Load the mesh (replace with your own mesh)
    stl_filename = "stl-files/DrivAer-Notchback.stl"
    mesh = trimesh.load_mesh(stl_filename, process=False)
    mesh_vertices = mesh.vertices

    # Transform the mesh points to be located in the right position in the wind tunnel
    mesh_vertices -= mesh_vertices.min(axis=0)
    mesh_extents = mesh_vertices.max(axis=0)
    length_phys_unit = mesh_extents.max()
    length_lbm_unit = grid_shape[0] / 4
    dx = length_phys_unit / length_lbm_unit
    mesh_vertices = mesh_vertices / dx
    shift = np.array([grid_shape[0] / 4, (grid_shape[1] - mesh_extents[1] / dx) / 2, 0.0])
    car = mesh_vertices + shift
    # self.car_cross_section = np.prod(mesh_extents[1:]) / dx**2

    return inlet, outlet, walls, car

# Converts a population array to a grid where each cell has a 3x3x3 layout of populations
def convertPopToGrid(pop, velocity_set, grid_shape):
    new_grid_shape = (3*grid_shape[0], 3*grid_shape[1], 3*grid_shape[2])
    print('New grid shape: ', new_grid_shape)
    pop_grid = np.zeros(new_grid_shape)
    for i in range(velocity_set.q):
        c = velocity_set._c[:,i] + 1
        pop_grid[c[0]:new_grid_shape[0]:3,c[1]:new_grid_shape[1]:3,c[2]:new_grid_shape[2]:3] = pop[i,:,:,:]
    return pop_grid

# Converts a population array to a grid where each cell has a 3x3x3 layout of populations
def clearNonWeightPops(pop, missing_mask, velocity_set):
    for i in range(velocity_set.q):
        # c = velocity_set._c[:,i] + 1
        o = velocity_set._opp_indices[i]
        pop[o,:,:,:] = pop[o,:,:,:] * missing_mask[i,:,:,:]
    return pop

def test_indices_masker_warp(dim, velocity_set, grid_shape):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)
    velocity_set = DefaultConfig.velocity_set

    missing_mask = my_grid.create_field(cardinality=velocity_set.q, dtype=xlb.Precision.BOOL)

    bc_mask = my_grid.create_field(cardinality=1, dtype=xlb.Precision.UINT8)

    indices_boundary_masker = IndicesBoundaryMasker()

    # Make indices for boundary conditions (sphere)
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0 / np.sqrt(3))
    mesh.export("sphere.stl")
    mesh = trimesh.load_mesh("sphere.stl", process=False)
    mesh_vertices = mesh.vertices
    
    # Transform the mesh points to be located in the right position in the wind tunnel
    mesh_vertices -= mesh_vertices.min(axis=0)
    mesh_extents = mesh_vertices.max(axis=0)
    length_phys_unit = mesh_extents.max()
    length_lbm_unit = grid_shape[0] / 2
    dx = length_phys_unit / length_lbm_unit
    mesh_vertices = mesh_vertices / dx
    shift = np.array([(grid_shape[0] - mesh_extents[0] / dx) / 2, (grid_shape[1] - mesh_extents[1] / dx) / 2, grid_shape[2] / 4])

    verts = mesh_vertices + shift
    
    # Update the mesh itself
    mesh.vertices = verts
    # Write out the mesh as stl file
    mesh.export("sphereScaled.stl")

    if dim == 2:
        grid_size_x, grid_size_y = grid_shape
    if dim == 3:
        grid_size_x, grid_size_y, grid_size_z = grid_shape
    
    wind_speed = 0.02
    # Set up Reynolds number and deduce relaxation time (omega)
    Re = 500000000.0
    clength = grid_size_x - 1
    visc = wind_speed * clength / Re
    omega = 1.0 / (3.0 * visc + 0.5)
    backend = ComputeBackend.WARP

    grid = grid_factory(grid_shape, compute_backend=backend)

    test_bc = HybridBC(bc_method="dorschner_localized", mesh_vertices=verts)
    inlet, outlet, walls, car = define_boundary_indices(grid, grid_shape,velocity_set)
    bc_left = RegularizedBC("velocity", prescribed_value=(wind_speed, 0.0, 0.0), indices=inlet)
    bc_walls = FullwayBounceBackBC(indices=walls)
    bc_do_nothing = ExtrapolationOutflowBC(indices=outlet)
    # bc_car = HybridBC(bc_method="dorschner_localized", mesh_vertices=car)
    # bc_car = HybridBC(bc_method="bounceback_regularized", mesh_vertices=car, use_mesh_distance=True)
    boundary_conditions = [bc_walls, bc_left, bc_do_nothing, test_bc]


    stepper = IncompressibleNavierStokesStepper(
        omega=omega,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type="KBC",
    )

    print('Velocity set dirs: ', velocity_set._c)
    print('Velocity set dirs shape: ', velocity_set._c.shape)
    print('Velocity set dir 1: ', velocity_set._c[:,1] + 1)

    # Initialize fields using the stepper
    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    # Make a copy of the initial populations
    f_1_init = f_1.numpy()
    print('Grid shape: ', grid_shape)
    f_1_temp = convertPopToGrid(f_1_init, velocity_set, grid_shape)
    
    num_steps = 2000
    save_iterval = 100
    # start_time = time.time()
    for i in range(num_steps):
        if (i % save_iterval == 0):
            f_temp = f_1.numpy()
            # f_temp = clearNonWeightPops(f_temp, missing_mask.numpy(), velocity_set)
            f_temp_fields = {"f": convertPopToGrid(f_temp, velocity_set, grid_shape)}
            save_fields_vtk(f_temp_fields, timestep=i, output_dir='output', prefix='test_bc_mesh_boundary_masker_warp_temp', spacing=(1.0/3,1.0/3,1.0/3))

        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, i)
        f_0, f_1 = f_1, f_0
        

    f_1_final = f_1.numpy()

    print('Shape of bc_mask:', bc_mask.shape)
    print('Shape of pop:', f_1_init.shape)

    
    # Compare the final populations at the non-zero entries of the initial populations
    # assert np.all(f_1_final[f_1_init != 0] == f_1_init[f_1_init != 0])

    # Zero out the populations at the zero entries of the initial populations
    # f_1_final[f_1_init == 0] = 0
    # Zero out the populations except for those opposite to the missing mask
    # f_1_init[missing_mask.numpy()] = 0

    # Clear the non-weight populations
    f_1_init = clearNonWeightPops(f_1_init, missing_mask.numpy(), velocity_set)
    f_1_final = clearNonWeightPops(f_1_final, missing_mask.numpy(), velocity_set)

    # Save the initial populations and final populations to vtk files
    fields = {"initial f": convertPopToGrid(f_1_init, velocity_set, grid_shape), "final f": convertPopToGrid(f_1_final, velocity_set, grid_shape)}

    save_fields_vtk(fields, timestep=0, prefix='test_bc_mesh_boundary_masker_warp', spacing=(1.0/3,1.0/3,1.0/3))


    bc_fields = {"bc mask": bc_mask.numpy().squeeze()}

    save_fields_vtk(bc_fields, timestep=0, prefix='test_bc_mesh_boundary_masker_warp_bc_mask')

if __name__ == "__main__":
    wp.clear_kernel_cache()
    test_indices_masker_warp(3, xlb.velocity_set.D3Q27, (30, 30, 30))
    # pytest.main()