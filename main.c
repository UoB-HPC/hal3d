#include "../comms.h"
#include "../mesh.h"
#include "../params.h"
#include "../umesh.h"
#include "hale_data.h"
#include "hale_interface.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Validates the results of the simulation
void validate(const int ncells, const char* params_filename, const int rank,
              double* density, double* energy);

int main(int argc, char** argv) {
  if (argc != 2) {
    TERMINATE("usage: ./hale <parameter_filename>\n");
  }

  // Store some of the generic mesh meta data
  Mesh mesh = {0};
  const char* hale_params = argv[1];
  mesh.niters = get_int_parameter("iterations", hale_params);
  mesh.max_dt = get_double_parameter("max_dt", ARCH_ROOT_PARAMS);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.dt = get_double_parameter("dt", hale_params);
  mesh.dt_h = mesh.dt;
  mesh.rank = MASTER;
  mesh.nranks = 1;
  const int visit_dump = get_int_parameter("visit_dump", hale_params);
  const int read_umesh = get_int_parameter("read_umesh", hale_params);

  // Perform initialisation routines
  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_comms(&mesh);
  initialise_devices(mesh.rank);
  initialise_mesh_2d(&mesh);

  size_t allocated = 0;

  // Fetch the size of the unstructured mesh
  UnstructuredMesh umesh = {0};
  HaleData hale_data = {0};

  if (read_umesh) {
    umesh.node_filename = get_parameter("node_file", hale_params);
    umesh.ele_filename = get_parameter("ele_file", hale_params);

    double** cell_variables[2] = {&hale_data.density0, &hale_data.energy0};
    allocated += read_unstructured_mesh(&umesh, cell_variables, 2);
  } else {
    allocated += convert_mesh_to_umesh(&umesh, &mesh);
  }

  // Initialise the hale-specific data arrays
  hale_data.visc_coeff1 = get_double_parameter("visc_coeff1", hale_params);
  hale_data.visc_coeff2 = get_double_parameter("visc_coeff2", hale_params);
  allocated += initialise_hale_data_2d(&hale_data, &umesh);

  printf("Allocated %.3fGB bytes of data\n", allocated / (double)GB);

  int nthreads = 0;
#pragma omp parallel
  { nthreads = omp_get_num_threads(); }

  if (mesh.rank == MASTER) {
    printf("Number of ranks: %d\n", mesh.nranks);
    printf("Number of threads: %d\n", nthreads);
  }

  if (visit_dump) {
    write_unstructured_to_visit(
        umesh.nnodes, umesh.ncells, 0, umesh.nodes_x0, umesh.nodes_y0,
        umesh.cells_to_nodes, hale_data.density0, 0, umesh.nnodes_by_cell == 4);
  }

  // Prepare for solve
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;

  set_timestep(umesh.ncells, umesh.cells_to_nodes, umesh.cells_offsets,
               umesh.nodes_x0, umesh.nodes_y0, hale_data.energy0, &mesh.dt);

  // Main timestep loop
  int tt;
  for (tt = 0; tt < mesh.niters; ++tt) {

    if (mesh.rank == MASTER) {
      printf("\nIteration %d\n", tt + 1);
    }

    double w0 = omp_get_wtime();

    solve_unstructured_hydro_2d(
        &mesh, umesh.ncells, umesh.nnodes, umesh.nsub_cell_edges,
        hale_data.visc_coeff1, hale_data.visc_coeff2, umesh.cell_centroids_x,
        umesh.cell_centroids_y, umesh.cells_to_nodes, umesh.cells_offsets,
        umesh.nodes_to_cells, umesh.cells_to_cells, umesh.nodes_offsets,
        umesh.nodes_x0, umesh.nodes_y0, umesh.nodes_x1, umesh.nodes_y1,
        umesh.boundary_index, umesh.boundary_type, hale_data.original_nodes_x,
        hale_data.original_nodes_y, umesh.boundary_normal_x,
        umesh.boundary_normal_y, hale_data.energy0, hale_data.energy1,
        hale_data.density0, hale_data.density1, hale_data.pressure0,
        hale_data.pressure1, hale_data.velocity_x0, hale_data.velocity_y0,
        hale_data.velocity_x1, hale_data.velocity_y1,
        hale_data.sub_cell_force_x, hale_data.sub_cell_force_y,
        hale_data.node_force_x, hale_data.node_force_y, hale_data.node_force_x2,
        hale_data.node_force_y2, hale_data.cell_mass, hale_data.nodal_mass,
        hale_data.nodal_volumes, hale_data.nodal_soundspeed, hale_data.limiter,
        hale_data.sub_cell_volume, hale_data.sub_cell_energy,
        hale_data.sub_cell_mass, hale_data.sub_cell_velocity_x,
        hale_data.sub_cell_velocity_y, hale_data.sub_cell_kinetic_energy,
        hale_data.sub_cell_centroids_x, hale_data.sub_cell_centroids_y,
        hale_data.sub_cell_grad_x, hale_data.sub_cell_grad_y);

    wallclock += omp_get_wtime() - w0;
    elapsed_sim_time += mesh.dt;
    if (elapsed_sim_time >= mesh.sim_end) {
      if (mesh.rank == MASTER) {
        printf("reached end of simulation time\n");
      }
      break;
    }

    if (mesh.rank == MASTER) {
      printf("simulation time: %.4lfs\nwallclock: %.4lfs\n", elapsed_sim_time,
             wallclock);
    }

    if (visit_dump) {
      write_unstructured_to_visit(umesh.nnodes, umesh.ncells, tt + 1,
                                  umesh.nodes_x0, umesh.nodes_y0,
                                  umesh.cells_to_nodes, hale_data.density0, 0,
                                  umesh.nnodes_by_cell == 4);
    }
  }

  barrier();

  validate(umesh.ncells, hale_params, mesh.rank, hale_data.density0,
           hale_data.energy0);

  if (mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);
    PRINT_PROFILING_RESULTS(&comms_profile);
    printf("Wallclock %.4fs, Elapsed Simulation Time %.4fs\n", wallclock,
           elapsed_sim_time);
  }

  finalise_mesh(&mesh);

  return 0;
}

// Validates the results of the simulation
void validate(const int ncells, const char* params_filename, const int rank,
              double* density, double* energy) {
  double* h_energy;
  double* h_density;
  allocate_host_data(&h_energy, ncells);
  allocate_host_data(&h_density, ncells);
  copy_buffer(ncells, &energy, &h_energy, RECV);
  copy_buffer(ncells, &density, &h_density, RECV);

  double local_density_total = 0.0;
  double local_energy_total = 0.0;

#pragma omp parallel for reduction(+ : local_density_total, local_energy_total)
  for (int ii = 0; ii < ncells; ++ii) {
    local_density_total += h_density[ii];
    local_energy_total += h_energy[ii];
  }

  double global_density_total = reduce_all_sum(local_density_total);
  double global_energy_total = reduce_all_sum(local_energy_total);

  if (rank != MASTER) {
    return;
  }

  int nresults = 0;
  char* keys = (char*)malloc(sizeof(char) * MAX_KEYS * (MAX_STR_LEN + 1));
  double* values = (double*)malloc(sizeof(double) * MAX_KEYS);
  if (!get_key_value_parameter(params_filename, HALE_TESTS, keys, values,
                               &nresults)) {
    printf("Warning. Test entry was not found, could NOT validate.\n");
    return;
  }

  double expected_energy;
  double expected_density;
  if (strmatch(&(keys[0]), "energy")) {
    expected_energy = values[0];
    expected_density = values[1];
  } else {
    expected_energy = values[1];
    expected_density = values[0];
  }

  printf("\nExpected energy %.12e, result was %.12e.\n", expected_energy,
         global_energy_total);
  printf("Expected density %.12e, result was %.12e.\n", expected_density,
         global_density_total);

  const int pass = within_tolerance(expected_energy, global_energy_total,
                                    VALIDATE_TOLERANCE) &&
                   within_tolerance(expected_density, global_density_total,
                                    VALIDATE_TOLERANCE);

  if (pass) {
    printf("PASSED validation.\n");
  } else {
    printf("FAILED validation.\n");
  }

  free(keys);
  free(values);
  deallocate_host_data(h_energy);
  deallocate_host_data(h_density);
}
