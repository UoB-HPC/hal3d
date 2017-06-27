#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "hale_interface.h"
#include "hale_data.h"
#include "../mesh.h"
#if 0
#include "../shared_data.h"
#endif // if 0
#include "../comms.h"
#include "../params.h"

int main(int argc, char** argv)
{
  if(argc != 2) {
    TERMINATE("usage: ./hale <parameter_filename>\n");
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  const char* hale_params = argv[1];
  mesh.global_nx = get_int_parameter("nx", hale_params);
  mesh.global_ny = get_int_parameter("ny", hale_params);
  mesh.pad = 0;
  mesh.local_nx = mesh.global_nx+2*mesh.pad;
  mesh.local_ny = mesh.global_ny+2*mesh.pad;
  mesh.niters = get_int_parameter("iterations", hale_params);
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.max_dt = get_double_parameter("max_dt", ARCH_ROOT_PARAMS);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.dt = get_double_parameter("dt", hale_params);
  mesh.dt_h = mesh.dt;
  mesh.rank = MASTER;
  mesh.nranks = 1;
  const int visit_dump = get_int_parameter("visit_dump", hale_params);

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_comms(&mesh);
  initialise_devices(mesh.rank);
  initialise_mesh_2d(&mesh);

  UnstructuredMesh umesh;
  umesh.node_filename = "tri.1.node";
  umesh.ele_filename = "tri.1.ele";

#if 0
  size_t allocated = initialise_unstructured_mesh(&mesh, &umesh);
#endif // if 0

  // Reads an unstructured mesh from an input file
  size_t allocated = read_unstructured_mesh(
      &mesh, &umesh);

  int nthreads = 0;
#pragma omp parallel 
  {
    nthreads = omp_get_num_threads();
  }

  if(mesh.rank == MASTER) {
    printf("Number of ranks: %d\n", mesh.nranks);
    printf("Number of threads: %d\n", nthreads);
  }

  HaleData hale_data = {0};
  hale_data.visc_coeff1 = get_double_parameter("visc_coeff1", hale_params);
  hale_data.visc_coeff2 = get_double_parameter("visc_coeff2", hale_params);
  allocated += initialise_hale_data_2d(
      mesh.local_nx, mesh.local_ny, &hale_data, &umesh);
  printf("Allocated %.3fGB bytes of data\n", allocated/(double)GB);

  write_unstructured_tris_to_visit( 
      umesh.nnodes, umesh.ncells, 0, umesh.nodes_x0, 
      umesh.nodes_y0, umesh.cells_to_nodes, hale_data.density0, 0);

  // Prepare for solve
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;

  set_timestep(
      umesh.ncells, umesh.cells_to_nodes, 
      umesh.cells_to_nodes_off, umesh.nodes_x0, 
      umesh.nodes_y0, hale_data.energy0, &mesh.dt);

  // Main timestep loop
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {

    if(mesh.rank == MASTER) {
      printf("\nIteration %d\n", tt+1);
    }

    double w0 = omp_get_wtime();

    solve_unstructured_hydro_2d(
        &mesh, umesh.ncells, umesh.nnodes, hale_data.visc_coeff1, 
        hale_data.visc_coeff2, umesh.cell_centroids_x, 
        umesh.cell_centroids_y, umesh.cells_to_nodes, 
        umesh.cells_to_nodes_off, umesh.nodes_x0, 
        umesh.nodes_y0, umesh.nodes_x1, 
        umesh.nodes_y1, umesh.boundary_index, 
        umesh.boundary_type, umesh.boundary_normal_x, 
        umesh.boundary_normal_y, hale_data.energy0, hale_data.energy1, 
        hale_data.density0, hale_data.density1, hale_data.pressure0, hale_data.pressure1, 
        hale_data.velocity_x0, hale_data.velocity_y0, hale_data.velocity_x1, 
        hale_data.velocity_y1, hale_data.cell_force_x, hale_data.cell_force_y, 
        hale_data.node_force_x, hale_data.node_force_y, hale_data.cell_mass, 
        hale_data.nodal_mass, hale_data.nodal_volumes, hale_data.nodal_soundspeed, 
        hale_data.limiter);

    wallclock += omp_get_wtime()-w0;
    elapsed_sim_time += mesh.dt;
    if(elapsed_sim_time >= mesh.sim_end) {
      if(mesh.rank == MASTER) {
        printf("reached end of simulation time\n");
      }
      break;
    }

    if(mesh.rank == MASTER) {
      printf("simulation time: %.4lfs\nwallclock: %.4lfs\n", 
          elapsed_sim_time, wallclock);
    }

    if(visit_dump) {
      write_unstructured_tris_to_visit( 
          umesh.nnodes, umesh.ncells, tt, umesh.nodes_x0, 
          umesh.nodes_y0, umesh.cells_to_nodes, hale_data.density0, 0);
    }
  }

  barrier();

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    PRINT_PROFILING_RESULTS(&comms_profile);
    printf("Wallclock %.4fs, Elapsed Simulation Time %.4fs\n", 
        wallclock, elapsed_sim_time);
  }

  finalise_mesh(&mesh);

  return 0;
}

