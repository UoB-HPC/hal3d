#include "../../comms.h"
#include "../../shared.h"
#include "../../cuda/shared.h"
#include "hale.h"
#include "lagrange.h"
#include "lagrange.k"
#include <float.h>
#include <math.h>

// Performs the Lagrangian step of the hydro solve
void lagrangian_phase(Mesh* mesh, UnstructuredMesh* umesh,
                      HaleData* hale_data) {

  predictor(mesh, umesh, hale_data);

  corrector(mesh, umesh, hale_data);
}

// Performs the predictor step of the Lagrangian phase
void predictor(Mesh* mesh, UnstructuredMesh* umesh, HaleData* hale_data) {

  const int nblocks_cells = ceil(umesh->ncells / NTHREADS);
  const int nblocks_nodes = ceil(umesh->nnodes / NTHREADS);

  // Update the pressure
  START_PROFILING(&compute_profile);
  equation_of_state<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, hale_data->energy0, hale_data->density0,
      hale_data->pressure0);
  STOP_PROFILING(&compute_profile, "equation_of_state");

  // Calculate the nodal volume and sound speed
  START_PROFILING(&compute_profile);
  calc_nodal_vol_and_c<<<nblocks_nodes, NTHREADS>>>(
      umesh->nnodes, umesh->nodes_to_faces_offsets, umesh->nodes_to_faces,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->faces_to_cells0, umesh->faces_to_cells1, umesh->nodes_x0,
      umesh->nodes_y0, umesh->nodes_z0, umesh->cell_centroids_x,
      umesh->cell_centroids_y, umesh->cell_centroids_z, hale_data->energy0,
      hale_data->nodal_volumes, hale_data->nodal_soundspeed);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_nodal_vol_and_c");

  // Sets all of the subcell forces to 0
  START_PROFILING(&compute_profile);
  zero_subcell_forces<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, umesh->cells_to_nodes_offsets, hale_data->subcell_force_x,
      hale_data->subcell_force_y, hale_data->subcell_force_z);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "zero_subcell_forces");

  START_PROFILING(&compute_profile);
  calc_subcell_force_from_pressure<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, umesh->cells_to_faces_offsets,
      umesh->cells_to_nodes_offsets, umesh->cells_to_faces,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->cells_to_nodes, umesh->faces_cclockwise_cell, umesh->nodes_x0,
      umesh->nodes_y0, umesh->nodes_z0, hale_data->pressure0,
      hale_data->subcell_force_x, hale_data->subcell_force_y,
      hale_data->subcell_force_z);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_subcell_force_from_pressure");

  START_PROFILING(&compute_profile);
  scale_soundspeed<<<nblocks_nodes, NTHREADS>>>(
      umesh->nnodes, hale_data->nodal_volumes, hale_data->nodal_soundspeed);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "scale_soundspeed");

  START_PROFILING(&compute_profile);
  calc_artificial_viscosity<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, hale_data->visc_coeff1, hale_data->visc_coeff2,
      umesh->cells_to_nodes_offsets, umesh->cells_to_nodes,
      umesh->faces_cclockwise_cell, umesh->nodes_x0, umesh->nodes_y0,
      umesh->nodes_z0, umesh->cell_centroids_x, umesh->cell_centroids_y,
      umesh->cell_centroids_z, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0, hale_data->nodal_soundspeed,
      hale_data->nodal_mass, hale_data->nodal_volumes, hale_data->limiter,
      hale_data->subcell_force_x, hale_data->subcell_force_y,
      hale_data->subcell_force_z, umesh->faces_to_nodes_offsets,
      umesh->faces_to_nodes, umesh->cells_to_faces_offsets,
      umesh->cells_to_faces);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_artificial_viscosity");

  START_PROFILING(&compute_profile);
  calc_new_velocity<<<nblocks_nodes, NTHREADS>>>(
      umesh->nnodes, mesh->dt, umesh->nodes_to_cells_offsets,
      umesh->nodes_to_cells, umesh->cells_to_nodes_offsets,
      umesh->cells_to_nodes, hale_data->subcell_force_x,
      hale_data->subcell_force_y, hale_data->subcell_force_z,
      hale_data->nodal_mass, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0, hale_data->velocity_x1, hale_data->velocity_y1,
      hale_data->velocity_z1);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_new_velocity");

  handle_unstructured_reflect_3d(
      umesh->nnodes, umesh->boundary_index, umesh->boundary_type,
      umesh->boundary_normal_x, umesh->boundary_normal_y,
      umesh->boundary_normal_z, hale_data->velocity_x1, hale_data->velocity_y1,
      hale_data->velocity_z1);

  // Move the nodes by the predicted velocity
  START_PROFILING(&compute_profile);
  move_nodes<<<nblocks_cells, NTHREADS>>>(
      umesh->nnodes, mesh->dt, umesh->nodes_x0, umesh->nodes_y0,
      umesh->nodes_z0, hale_data->velocity_x1, hale_data->velocity_y1,
      hale_data->velocity_z1, umesh->nodes_x1, umesh->nodes_y1,
      umesh->nodes_z1);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "move_nodes");

  init_cell_centroids(umesh->ncells, umesh->cells_to_nodes_offsets,
                      umesh->cells_to_nodes, umesh->nodes_x1, umesh->nodes_y1,
                      umesh->nodes_z1, umesh->cell_centroids_x,
                      umesh->cell_centroids_y, umesh->cell_centroids_z);

  set_timestep(umesh->ncells, umesh->nodes_x1, umesh->nodes_y1, umesh->nodes_z1,
               hale_data->energy0, &mesh->dt, umesh->cells_to_faces_offsets,
               umesh->cells_to_faces, umesh->faces_to_nodes_offsets,
               umesh->faces_to_nodes, hale_data->reduce_array);

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
  calc_predicted_energy<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, mesh->dt, umesh->cells_to_nodes_offsets,
      umesh->cells_to_nodes, hale_data->velocity_x1, hale_data->velocity_y1,
      hale_data->velocity_z1, hale_data->subcell_force_x,
      hale_data->subcell_force_y, hale_data->subcell_force_z,
      hale_data->energy0, hale_data->cell_mass, hale_data->energy1);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_predicted_energy");

  // Using the new volume, calculate the predicted density
  START_PROFILING(&compute_profile);
  calc_predicted_density<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, umesh->cells_to_faces_offsets, umesh->cells_to_faces,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes, umesh->nodes_x1,
      umesh->nodes_y1, umesh->nodes_z1, umesh->cell_centroids_x,
      umesh->cell_centroids_y, umesh->cell_centroids_z, hale_data->cell_mass,
      hale_data->density1);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_predicted_density");

  // Calculate the time centered pressure from mid point between rezoned and
  // predicted pressures
  START_PROFILING(&compute_profile);
  time_center_pressure<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, hale_data->energy1, hale_data->density1,
      hale_data->pressure0, hale_data->pressure1);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "time_center_pressure");

  // Prepare time centered variables for the corrector step
  START_PROFILING(&compute_profile);
  time_center_nodes<<<nblocks_nodes, NTHREADS>>>(
      umesh->nnodes, umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
      umesh->nodes_x1, umesh->nodes_y1, umesh->nodes_z1);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "time_center_nodes");
}

// Performs the corrector step of the Lagrangian phase
void corrector(Mesh* mesh, UnstructuredMesh* umesh, HaleData* hale_data) {

  const int nblocks_cells = ceil(umesh->ncells / NTHREADS);
  const int nblocks_nodes = ceil(umesh->nnodes / NTHREADS);

  // Sets all of the subcell forces to 0
  START_PROFILING(&compute_profile);
  zero_subcell_forces<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, umesh->cells_to_nodes_offsets, hale_data->subcell_force_x,
      hale_data->subcell_force_y, hale_data->subcell_force_z);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");

  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
  calc_nodal_vol_and_c<<<nblocks_nodes, NTHREADS>>>(
      umesh->nnodes, umesh->nodes_to_faces_offsets, umesh->nodes_to_faces,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->faces_to_cells0, umesh->faces_to_cells1, umesh->nodes_x1,
      umesh->nodes_y1, umesh->nodes_z1, umesh->cell_centroids_x,
      umesh->cell_centroids_y, umesh->cell_centroids_z, hale_data->energy1,
      hale_data->nodal_volumes, hale_data->nodal_soundspeed);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_nodal_vol_and_c");

  START_PROFILING(&compute_profile);
  scale_soundspeed<<<nblocks_nodes, NTHREADS>>>(
      umesh->nnodes, hale_data->nodal_volumes, hale_data->nodal_soundspeed);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "scale_soundspeed");

  // Calculate the pressure gradients
  START_PROFILING(&compute_profile);
  calc_subcell_force_from_pressure<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, umesh->cells_to_faces_offsets,
      umesh->cells_to_nodes_offsets, umesh->cells_to_faces,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->cells_to_nodes, umesh->faces_cclockwise_cell, umesh->nodes_x1,
      umesh->nodes_y1, umesh->nodes_z1, hale_data->pressure1,
      hale_data->subcell_force_x, hale_data->subcell_force_y,
      hale_data->subcell_force_z);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "node_force_from_pressure");

  calc_artificial_viscosity<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, hale_data->visc_coeff1, hale_data->visc_coeff2,
      umesh->cells_to_nodes_offsets, umesh->cells_to_nodes,
      umesh->faces_cclockwise_cell, umesh->nodes_x1, umesh->nodes_y1,
      umesh->nodes_z1, umesh->cell_centroids_x, umesh->cell_centroids_y,
      umesh->cell_centroids_z, hale_data->velocity_x1, hale_data->velocity_y1,
      hale_data->velocity_z1, hale_data->nodal_soundspeed,
      hale_data->nodal_mass, hale_data->nodal_volumes, hale_data->limiter,
      hale_data->subcell_force_x, hale_data->subcell_force_y,
      hale_data->subcell_force_z, umesh->faces_to_nodes_offsets,
      umesh->faces_to_nodes, umesh->cells_to_faces_offsets,
      umesh->cells_to_faces);

  START_PROFILING(&compute_profile);
  // Updates and time center velocity in the corrector step
  update_and_time_center_velocity<<<nblocks_nodes, NTHREADS>>>(
      umesh->nnodes, mesh->dt, umesh->nodes_to_cells_offsets,
      umesh->nodes_to_cells, umesh->cells_to_nodes_offsets,
      umesh->cells_to_nodes, hale_data->nodal_mass, hale_data->subcell_force_x,
      hale_data->subcell_force_y, hale_data->subcell_force_z,
      hale_data->velocity_x0, hale_data->velocity_y0, hale_data->velocity_z0,
      hale_data->velocity_x1, hale_data->velocity_y1, hale_data->velocity_z1);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_new_velocity");

  handle_unstructured_reflect_3d(
      umesh->nnodes, umesh->boundary_index, umesh->boundary_type,
      umesh->boundary_normal_x, umesh->boundary_normal_y,
      umesh->boundary_normal_z, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0);

  // Advances the nodes using the corrected velocity
  START_PROFILING(&compute_profile);
  advance_nodes_corrected<<<nblocks_nodes, NTHREADS>>>(
      umesh->nnodes, mesh->dt, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0, umesh->nodes_x0, umesh->nodes_y0,
      umesh->nodes_z0);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "advance_nodes_corrected");

  set_timestep(umesh->ncells, umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
               hale_data->energy1, &mesh->dt, umesh->cells_to_faces_offsets,
               umesh->cells_to_faces, umesh->faces_to_nodes_offsets,
               umesh->faces_to_nodes);

  // Calculate the corrected energy
  START_PROFILING(&compute_profile);
  calc_corrected_energy<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, mesh->dt, umesh->cells_to_nodes_offsets,
      umesh->cells_to_nodes, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0, hale_data->subcell_force_x,
      hale_data->subcell_force_y, hale_data->subcell_force_z,
      hale_data->cell_mass, hale_data->energy0);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_corrected_energy");

  init_cell_centroids(umesh->ncells, umesh->cells_to_nodes_offsets,
                      umesh->cells_to_nodes, umesh->nodes_x0, umesh->nodes_y0,
                      umesh->nodes_z0, umesh->cell_centroids_x,
                      umesh->cell_centroids_y, umesh->cell_centroids_z);

  // Using the new corrected volume, calculate the density
  START_PROFILING(&compute_profile);
  calc_corrected_density<<<nblocks_cells, NTHREADS>>>(
      umesh->ncells, umesh->cells_to_faces_offsets, umesh->cells_to_faces,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes, umesh->nodes_x0,
      umesh->nodes_y0, umesh->nodes_z0, umesh->cell_centroids_x,
      umesh->cell_centroids_y, umesh->cell_centroids_z, hale_data->cell_mass,
      hale_data->cell_volume, hale_data->density0);
  gpu_check(cudaDeviceSynchronize());
  STOP_PROFILING(&compute_profile, "calc_corrected_density");
}

// Controls the timestep for the simulation
void set_timestep(const int ncells, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes, double* reduce_array) {

  START_PROFILING(&compute_profile);

  const int nblocks_cells = ceil(ncells / NTHREADS);
  calc_timestep<<<nblocks_cells, NTHREADS>>>(
      ncells, nodes_x, nodes_y, nodes_z, energy, dt, cells_to_faces_offsets,
      cells_to_faces, faces_to_nodes_offsets, faces_to_nodes, reduce_array);
  gpu_check(cudaDeviceSynchronize());

  double local_min_dt;
  finish_min_reduce(nblocks_cells, reduce_array, &local_min_dt);
  gpu_check(cudaDeviceSynchronize());

  *dt = CFL * local_min_dt;
  printf("Timestep %.8fs\n", *dt);
  STOP_PROFILING(&compute_profile, __func__);
}
