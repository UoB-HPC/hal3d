#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "hale_interface.h"
#include "hale_data.h"
#include "../mesh.h"
#include "../shared_data.h"
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
  mesh.pad = 1;
  mesh.local_nx = mesh.global_nx+2*mesh.pad;
  mesh.local_ny = mesh.global_ny+2*mesh.pad;
  mesh.niters = get_int_parameter("iterations", hale_params);
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.max_dt = get_double_parameter("max_dt", ARCH_ROOT_PARAMS);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.dt = C_T*get_double_parameter("dt", hale_params);
  mesh.dt_h = mesh.dt;
  mesh.rank = MASTER;
  mesh.nranks = 1;

  const int visit_dump = get_int_parameter("visit_dump", hale_params);

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_comms(&mesh);
  initialise_devices(mesh.rank);
  initialise_mesh_2d(&mesh);

  UnstructuredMesh unstructured_mesh;
  size_t allocated = initialise_unstructured_mesh(&mesh, &unstructured_mesh);

  int nthreads = 0;
#pragma omp parallel 
  {
    nthreads = omp_get_num_threads();
  }

  if(mesh.rank == MASTER) {
    printf("Number of ranks: %d\n", mesh.nranks);
    printf("Number of threads: %d\n", nthreads);
  }

  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.pad, mesh.x_off, mesh.y_off, mesh.width, mesh.height,
      hale_params, mesh.edgex, mesh.edgey, &shared_data);

  handle_boundary_2d(
      mesh.local_nx, mesh.local_ny, &mesh, shared_data.rho, NO_INVERT, PACK);
  handle_boundary_2d(
      mesh.local_nx, mesh.local_ny, &mesh, shared_data.e, NO_INVERT, PACK);

  HaleData hale_data = {0};
  allocated += initialise_hale_data_2d(mesh.local_nx, mesh.local_ny, &hale_data, &unstructured_mesh);
  printf("Allocated %.3fGB bytes of data\n", allocated/(double)GB);

  if(visit_dump) {
    write_quad_data_to_visit(
        mesh.local_nx, mesh.local_ny, 0, unstructured_mesh.nodes_x0, 
        unstructured_mesh.nodes_y0, shared_data.rho, 0);
  }

  // Prepare for solve
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;

  // Main timestep loop
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {

    if(mesh.rank == MASTER) {
      printf("\nIteration %d\n", tt+1);
    }

    double w0 = omp_get_wtime();

    solve_unstructured_hydro_2d(
        &mesh, unstructured_mesh.ncells, unstructured_mesh.nnodes, mesh.dt, 
        unstructured_mesh.cell_centroids_x, unstructured_mesh.cell_centroids_y, 
        unstructured_mesh.cells_to_nodes, unstructured_mesh.nodes_to_cells, 
        unstructured_mesh.nodes_to_cells_off, unstructured_mesh.cells_to_nodes_off, 
        unstructured_mesh.nodes_x0, unstructured_mesh.nodes_y0, 
        unstructured_mesh.nodes_x1, unstructured_mesh.nodes_y1,
        unstructured_mesh.halo_cell, unstructured_mesh.halo_node, 
        shared_data.e, hale_data.energy1, 
        shared_data.rho, shared_data.rho_old, shared_data.P, hale_data.pressure1, 
        shared_data.u, shared_data.v, hale_data.velocity_x1, 
        hale_data.velocity_y1, hale_data.cell_force_x, hale_data.cell_force_y, 
        hale_data.node_force_x, hale_data.node_force_y, hale_data.cell_volumes, 
        hale_data.cell_mass, hale_data.nodal_mass, hale_data.nodal_volumes, 
        hale_data.nodal_soundspeed, hale_data.limiter);

#if 0
    print_conservation(
        mesh.local_nx, mesh.local_ny, shared_data.rho, shared_data.e, 
        shared_data.reduce_array0, &mesh);
#endif // if 0

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
      write_quad_data_to_visit(
          mesh.local_nx, mesh.local_ny, tt, unstructured_mesh.nodes_x0, 
          unstructured_mesh.nodes_y0, shared_data.rho, 0);
    }
  }

  barrier();

  validate(
      mesh.local_nx, mesh.local_ny, hale_params, 
      mesh.rank, shared_data.rho, shared_data.e);

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    PRINT_PROFILING_RESULTS(&comms_profile);
    printf("Wallclock %.4fs, Elapsed Simulation Time %.4fs\n", 
        wallclock, elapsed_sim_time);
  }

  finalise_shared_data(&shared_data);
  finalise_mesh(&mesh);

  return 0;
}

// Validates the results of the simulation
void validate(
    const int nx, const int ny, const char* params_filename, 
    const int rank, double* density, double* energy)
{
  double* h_energy;
  double* h_density;
  allocate_host_data(&h_energy, nx*ny);
  allocate_host_data(&h_density, nx*ny);
  copy_buffer(nx*ny, &energy, &h_energy, RECV);
  copy_buffer(nx*ny, &density, &h_density, RECV);

  double local_density_total = 0.0;
  double local_energy_total = 0.0;

#pragma omp parallel for reduction(+: local_density_total, local_energy_total)
  for(int ii = 0; ii < nx*ny; ++ii) {
    local_density_total += h_density[ii];
    local_energy_total += h_energy[ii];
  }

  double global_density_total = reduce_all_sum(local_density_total);
  double global_energy_total = reduce_all_sum(local_energy_total);

  if(rank != MASTER) {
    return;
  }

  int nresults = 0;
  char* keys = (char*)malloc(sizeof(char)*MAX_KEYS*(MAX_STR_LEN+1));
  double* values = (double*)malloc(sizeof(double)*MAX_KEYS);
  if(!get_key_value_parameter(
        params_filename, HALE_TESTS, keys, values, &nresults)) {
    printf("Warning. Test entry was not found, could NOT validate.\n");
    return;
  }

  double expected_energy;
  double expected_density;
  if(strmatch(&(keys[0]), "energy")) {
    expected_energy = values[0];
    expected_density = values[1];
  }
  else {
    expected_energy = values[1];
    expected_density = values[0];
  }

  printf("\nExpected energy %.12e, result was %.12e.\n", expected_energy, global_energy_total);
  printf("Expected density %.12e, result was %.12e.\n", expected_density, global_density_total);

  const int pass = 
    within_tolerance(expected_energy, global_energy_total, VALIDATE_TOLERANCE) &&
    within_tolerance(expected_density, global_density_total, VALIDATE_TOLERANCE);

  if(pass) {
    printf("PASSED validation.\n");
  }
  else {
    printf("FAILED validation.\n");
  }

  free(keys);
  free(values);
  deallocate_host_data(h_energy);
  deallocate_host_data(h_density);
}

