"""
This script performs a Lattice Boltzmann Method (LBM) simulation of fluid flow over a car model. Here are the main concepts and steps in the simulation:

Here are the main concepts introduced simulation:

1. Lattice: Given the usually high Reynolds number required for these simulations, a D3Q27 lattice is used, which is a three-dimensional lattice model with 27 discrete velocity directions.

2. Loading geometry and voxelization: The geometry of the car is loaded from a STL file. 
This is a file format commonly used for 3D models. The model is then voxelized to a binary matrix which represents the presence or absence of the object in the lattice. We use the DrivAer model, which is a common car model used for aerodynamic simulations.

3. Output: After each specified number of iterations, the script outputs the state of the simulation. This includes the error (difference between consecutive velocity fields), lift and drag coefficients, and visualization files in the VTK format.
"""


import os
import jax
import trimesh
from time import time
import numpy as np
import jax.numpy as jnp
from jax import config
import json, getopt, sys
from scipy.interpolate import RegularGridInterpolator

from src.utils import *
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD3Q19, LatticeD3Q27
from src.boundary_conditions import *

# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

# disable JIt compilation

#jax.config.update('jax_array', True)

class Car(KBCSim):
    def __init__(self, **kwargs):
        self.project = kwargs['project']
        self.prescribed_velocity = kwargs['prescribed_vel']
        self.origin = kwargs['origin']
        super().__init__(**kwargs)

    def voxelize_stl(self, stl_filename, length_lbm_unit):
        mesh = trimesh.load_mesh(stl_filename, process=False)
        length_phys_unit = mesh.extents.max()
        pitch = length_phys_unit/length_lbm_unit
        mesh_voxelized = mesh.voxelized(pitch=pitch)
        mesh_matrix = mesh_voxelized.matrix
        return mesh_matrix, pitch

    def set_boundary_conditions(self):
        print('Voxelizing meshes...')
        time_start = time()
        proj_path = self.project['projPath']
        lsto_params = self.project['lsTOParams']
        voxSize = lsto_params['voxSize']

        first = True
        ko_indices = []
        for ko in self.project['keepOuts']:
            ko_filename = os.path.join(proj_path, ko)
            #car_length_lbm_unit = self.nx / 4
            #car_voxelized, pitch = voxelize_stl(ko_filename, car_length_lbm_unit)
            komesh = trimesh.load_mesh(ko_filename, process=False)
            ko_origin = komesh.bounds[0]

            # print(komesh.bounds[0])
            # #verts = komesh.vertices
            # print(komesh.vertices.shape)
            # print(self.origin)
            # newverts = np.append(komesh.vertices, self.origin.reshape(1,3), axis=0)
            # print(newverts.shape)
            # komesh.vertices = newverts
            # #komesh.vertices.append(self.origin)
            # #komesh.vertices = np.concatenate(komesh.vertices,self.origin)
            # mask = np.ones((1, komesh.vertices.shape[0]), dtype=bool)
            # komesh.update_vertices(mask)
            # ko_origin = komesh.bounds[0]
            # print(ko_origin)
            shift = np.asarray((ko_origin - self.origin)/voxSize).astype(int)
            #shift = np.asarray(-self.origin)
            #komesh.apply_transform(trimesh.transformations.translation_matrix(shift))            
            komesh_voxelized = komesh.voxelized(pitch=voxSize)
            #komesh_voxelized = trimesh.voxel.creation.local_voxelized(mesh=komesh, point=self.origin, pitch=voxSize, )
            ko_indices.append(np.argwhere(komesh_voxelized.matrix)+shift)
            #ko_indices.append(np.argwhere(komesh_voxelized.matrix))
            if (first):
                car_matrix = komesh_voxelized.matrix
                first = False
        
        print('Voxelization time for pitch={}: {} seconds'.format(voxSize, time() - time_start))
        print("Car matrix shape: ", car_matrix.shape)

        self.car_area = np.prod(car_matrix.shape[1:])
        # tx, ty, tz = np.array([self.nx, self.ny, self.nz]) - car_matrix.shape
        # shift = [tx//4, ty//2, 0]
        # car_indices = np.argwhere(car_matrix) + shift
        # self.BCs.append(BounceBackHalfway(tuple(car_indices.T), self.gridInfo, self.precisionPolicy))
        
        for k_inds in ko_indices:
           self.BCs.append(BounceBackHalfway(tuple(k_inds.T), self.gridInfo, self.precisionPolicy))

        wall = np.concatenate((self.boundingBoxIndices['bottom'], self.boundingBoxIndices['top'],
                               self.boundingBoxIndices['front'], self.boundingBoxIndices['back']))
        self.BCs.append(BounceBack(tuple(wall.T), self.gridInfo, self.precisionPolicy))

        outlet = self.boundingBoxIndices['left']
        self.BCs.append(DoNothing(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        # self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        self.BCs[-1].implementationStep = 'PostCollision'
        # rho_outlet = np.ones(doNothing.shape[0], dtype=self.precisionPolicy.compute_dtype)
        # self.BCs.append(ZouHe(tuple(doNothing.T),
        #                                          self.gridInfo,
        #                                          self.precisionPolicy,
        #                                          'pressure', rho_outlet))

        inlet = self.boundingBoxIndices['right']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)

        vel_inlet[:, 0] = self.prescribed_velocity
        self.BCs.append(EquilibriumBC(tuple(inlet.T), self.gridInfo, self.precisionPolicy, rho_inlet, vel_inlet))
        # self.BCs.append(ZouHe(tuple(inlet.T),
        #                                          self.gridInfo,
        #                                          self.precisionPolicy,
        #                                          'velocity', vel_inlet))

    def output_data(self, **kwargs):
        # 1:-1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs['rho'][..., 1:-1, 1:-1, :])
        u = np.array(kwargs['u'][..., 1:-1, 1:-1, :])
        timestep = kwargs['timestep']
        u_prev = kwargs['u_prev'][..., 1:-1, 1:-1, :]

        # compute lift and drag over the car
        car = self.BCs[0]
        boundary_force = car.momentum_exchange_force(kwargs['f_poststreaming'], kwargs['f_postcollision'])
        boundary_force = np.sum(boundary_force, axis=0)
        drag = np.sqrt(boundary_force[0]**2 + boundary_force[1]**2)     #xy-plane
        lift = boundary_force[2]                                        #z-direction
        cd = 2. * drag / (self.prescribed_velocity ** 2 * self.car_area)
        cl = 2. * lift / (self.prescribed_velocity ** 2 * self.car_area)

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        # Slices
        lsto_params = self.project['lsTOParams']
        voxSize = lsto_params['voxSize']
        output_params = lsto_params['outputParams']
        output_slices = output_params['outputSlices']
        x = np.linspace(self.origin[0], self.origin[0] + (self.nx+1)*voxSize, self.nx)
        y = np.linspace(self.origin[1], self.origin[1] + (self.ny+1)*voxSize, self.ny)
        z = np.linspace(self.origin[2], self.origin[2] + (self.nz+1)*voxSize, self.nz)
        
        # for sliceSet in output_slices:
        #     if (sliceSet['field'] == 'velocityMag'):
        #         sliceField = u
        #     height_vec = [sliceSet['heightVec']['x'], sliceSet['heightVec']['y'], sliceSet['heightVec']['z']]
        #     width_vec = [sliceSet['widthVec']['x'], sliceSet['widthVec']['y'], sliceSet['widthVec']['z']]
        #     height = sliceSet['height']
        #     width = sliceSet['width']
        #     for orig in sliceSet['origin']:
        #         img_pts = np.mgrid(orig[0]:width_vec[0]:width_vec[0]*width, )

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}, CL = {:07.6f}, CD = {:07.6f}'.format(err, cl, cd))
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        save_fields_vtk(timestep, fields, self.project['lsTOParams']['outputName'])

def parseJSON(inputfile):
    f = open(inputfile)
    project = json.load(f)
    proj_path = os.path.dirname(os.path.abspath(inputfile))
    project['projPath'] = proj_path

    domain_filename = project['seedGeom']
    lsto_params = project['lsTOParams']
    fluid_proj = project['fluidProj']
    fluid_cases = fluid_proj['fluidCases']
    voxSize = lsto_params['voxSize']
    settings = fluid_proj['poseidonSettings']

    outputDir = lsto_params['outputName']
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    print(settings)

    if settings['doublePrecision']:
        precision = 'f64/f64'
    else:
        precision = 'f32/f32'
    lattice = LatticeD3Q27(precision)

    print(os.path.join(proj_path, domain_filename))
    domain_mesh = trimesh.load_mesh(os.path.join(proj_path, domain_filename), process=False)
    domain_origin = domain_mesh.bounds[0]
    
    nx = (int)(domain_mesh.extents[0]/voxSize)
    ny = (int)(domain_mesh.extents[1]/voxSize)
    nz = (int)(domain_mesh.extents[2]/voxSize)

    Re = 50000.0
    #prescribed_vel = 
    clength = nx - 1

    # Extract the inlet velocity from the json dict
    fluid_bcs = fluid_cases[0]['fluidBCs']
    for bc in fluid_bcs:
        fluidDef = bc['fluidDef']
        if fluidDef['bcType'] == "velocity":
            prescribed_vel = fluidDef['velocity']['x']

    print(prescribed_vel)
    visc = abs(prescribed_vel) * clength / Re
    omega = 1.0 / (3. * visc + 0.5)

    kwargs = {
        'project': project,
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'origin': domain_origin,
        'precision': precision,
        'prescribed_vel': prescribed_vel,
        'io_rate': settings['solutionPrintFreq'],
        'print_info_rate': settings['solutionPrintFreq'],
        'return_fpost': True  # Need to retain fpost-collision for computation of lift and drag
    }
    sim = Car(**kwargs)
    sim.run(settings['maxFwdIterations'])

def main(argv):
   inputfile = ''
   usage = 'windtunnel_json.py -i <inputjson>'
   print('Welcome to Studio Wind Tunnel Solver')
   os.system('rm -rf ./*.vtk && rm -rf ./*.png')
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
   except getopt.GetoptError:
      print(usage)
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print(usage)
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
         parseJSON(inputfile)

#%% Main run
if __name__ == "__main__":
   main(sys.argv[1:])