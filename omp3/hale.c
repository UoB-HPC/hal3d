#include "hale.h"
#include "../../comms.h"
#include "../../params.h"
#include "../../shared.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// TODO: At this stage, there are so many additional fields required
// to handle the sub-cell data for the remapping phase, there will be some use
// in considering whether some of the fields could be shared or whether
// adaptations to the algorithm are even necessary for this particular point

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_2d(
    Mesh* mesh, const int ncells, const int nnodes,
    const int nsub_cell_neighbours, const double visc_coeff1,
    const double visc_coeff2, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_to_nodes,
    int* cells_offsets, int* nodes_to_cells, int* cells_to_cells,
    int* nodes_offsets, double* nodes_x0, double* nodes_y0, double* nodes_z0,
    double* nodes_x1, double* nodes_y1, double* nodes_z1, int* boundary_index,
    int* boundary_type, const double* original_nodes_x,
    const double* original_nodes_y, const double* original_nodes_z,
    double* boundary_normal_x, double* boundary_normal_y,
    double* boundary_normal_z, double* energy0, double* energy1,
    double* density0, double* density1, double* pressure0, double* pressure1,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* velocity_x1, double* velocity_y1, double* velocity_z1,
    double* sub_cell_force_x, double* sub_cell_force_y,
    double* sub_cell_force_z, double* node_force_x, double* node_force_y,
    double* node_force_z, double* node_force_x2, double* node_force_y2,
    double* node_force_z2, double* cell_mass, double* nodal_mass,
    double* nodal_volumes, double* nodal_soundspeed, double* limiter,
    double* sub_cell_volume, double* sub_cell_energy, double* sub_cell_mass,
    double* sub_cell_velocity_x, double* sub_cell_velocity_y,
    double* sub_cell_velocity_z, double* sub_cell_kinetic_energy,
    double* sub_cell_centroids_x, double* sub_cell_centroids_y,
    double* sub_cell_centroids_z, double* sub_cell_grad_x,
    double* sub_cell_grad_y, double* sub_cell_grad_z) {

  double total_mass = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
  }
  printf("total mass %.12f\n", total_mass);

  /*
   *    PREDICTOR
   */

  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int nn = 0; nn < nnodes; ++nn) {
    nodal_mass[(nn)] = 0.0;
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }
  STOP_PROFILING(&compute_profile, "zero_nodal_arrays");

  // Calculate the pressure using the ideal gas equation of state
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    pressure0[(cc)] = (GAM - 1.0) * energy0[(cc)] * density0[(cc)];
  }
  STOP_PROFILING(&compute_profile, "equation_of_state");

  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int nodes_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - nodes_off;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(nodes_off + cc)];
      const int cell_offset = cells_offsets[(cell_index)];
      const int nnodes_by_cell = cells_offsets[(cell_index + 1)] - cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_offset + nn2)] == nn) {
          break;
        }
      }

      // Fetch the centroid of the cell
      const double cell_c_x = cell_centroids_x[(cell_index)];
      const double cell_c_y = cell_centroids_y[(cell_index)];
      const double cell_c_z = cell_centroids_z[(cell_index)];

      // TODO: Determine the sub-cell face area vectors

      // Calculate the sub-cell volume
      const double sub_cell_volume = 0.0;

      // TODO: this should be updated to fix the issues with hourglassing...
      if (sub_cell_volume <= 0.0) {
        TERMINATE("Encountered cell with unphysical volume %.12f in cell %d.",
                  sub_cell_volume, cc);
      }

      nodal_mass[(nn)] += density0[(cell_index)] * sub_cell_volume;

      // Calculate the volume and soundspeed at the node
      nodal_soundspeed[(nn)] +=
          sqrt(GAM * (GAM - 1.0) * energy0[(cell_index)]) * sub_cell_volume;
      nodal_volumes[(nn)] += sub_cell_volume;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    node_force_x[(nn)] = 0.0;
    node_force_y[(nn)] = 0.0;
    node_force_z[(nn)] = 0.0;
    node_force_x2[(nn)] = 0.0;
    node_force_y2[(nn)] = 0.0;
    node_force_z2[(nn)] = 0.0;
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }
  STOP_PROFILING(&compute_profile, "calc_soundspeed");

  // Calculate the force contributions for pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int nodes_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - nodes_off;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(nodes_off + cc)];
      const int cell_offset = cells_offsets[(cell_index)];
      const int nnodes_by_cell = cells_offsets[(cell_index + 1)] - cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_offset + nn2)] == nn) {
          break;
        }
      }

      // TODO: Determine the area vectors
      double S_x = 0.0;
      double S_y = 0.0;
      double S_z = 0.0;

      // Calculate the force vector due to pressure at the node
      node_force_x[(nn)] += pressure0[(cell_index)] * S_x;
      node_force_y[(nn)] += pressure0[(cell_index)] * S_y;
      node_force_z[(nn)] += pressure0[(cell_index)] * S_z;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_force");

  // Calculate the force contributions for pressure gradients
  // TODO: COULD WE SOMEHOW USE LESS CAPACITY THEN BOTH node_force and
  // sub_cell_force
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_c_index = cells_to_nodes[(cells_off) + (nn)];

      // TODO: determine the area vectors
      double S_x = 0.0;
      double S_y = 0.0;
      double S_z = 0.0;

      // Store the sub-cell force due to pressure
      sub_cell_force_x[(cells_off + nn)] = pressure0[(cc)] * S_x;
      sub_cell_force_y[(cells_off + nn)] = pressure0[(cc)] * S_y;
      sub_cell_force_z[(cells_off + nn)] = pressure0[(cc)] * S_z;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_cell_forces");

  calculate_artificial_viscosity(
      nnodes, visc_coeff1, visc_coeff2, cells_offsets, cells_to_nodes,
      nodes_offsets, nodes_to_cells, nodes_x0, nodes_y0, nodes_z0,
      cell_centroids_x, cell_centroids_y, cell_centroids_z, velocity_x0,
      velocity_y0, velocity_z0, nodal_soundspeed, nodal_mass, nodal_volumes,
      limiter, node_force_x, node_force_y, node_force_z, node_force_x2,
      node_force_y2, node_force_z2);

  // Calculate the time centered evolved velocities, by first calculating the
  // predicted values at the new timestep and then averaging with current
  // velocity
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int nn = 0; nn < nnodes; ++nn) {
    // Determine the predicted velocity
    velocity_x1[(nn)] =
        velocity_x0[(nn)] + mesh->dt * node_force_x[(nn)] / nodal_mass[(nn)];
    velocity_y1[(nn)] =
        velocity_y0[(nn)] + mesh->dt * node_force_y[(nn)] / nodal_mass[(nn)];
    velocity_z1[(nn)] =
        velocity_z0[(nn)] + mesh->dt * node_force_z[(nn)] / nodal_mass[(nn)];

    // Calculate the time centered velocity
    velocity_x1[(nn)] = 0.5 * (velocity_x0[(nn)] + velocity_x1[(nn)]);
    velocity_y1[(nn)] = 0.5 * (velocity_y0[(nn)] + velocity_y1[(nn)]);
    velocity_z1[(nn)] = 0.5 * (velocity_z0[(nn)] + velocity_z1[(nn)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_velocity");

  handle_unstructured_reflect_3d(nnodes, boundary_index, boundary_type,
                                 boundary_normal_x, boundary_normal_y,
                                 boundary_normal_z, velocity_x1, velocity_y1,
                                 velocity_z1);

  // Move the nodes by the predicted velocity
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x1[(nn)] = nodes_x0[(nn)] + mesh->dt * velocity_x1[(nn)];
    nodes_y1[(nn)] = nodes_y0[(nn)] + mesh->dt * velocity_y1[(nn)];
    nodes_z1[(nn)] = nodes_z0[(nn)] + mesh->dt * velocity_z1[(nn)];
  }
  STOP_PROFILING(&compute_profile, "move_nodes");

  set_timestep(ncells, cells_to_nodes, cells_offsets, nodes_x1, nodes_y1,
               nodes_z1, energy0, &mesh->dt);

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    // Sum the time centered velocity by the sub-cell forces
    double force = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cells_off) + (nn)];
      force +=
          (velocity_x1[(node_index)] * sub_cell_force_x[(cells_off) + (nn)] +
           velocity_y1[(node_index)] * sub_cell_force_y[(cells_off) + (nn)] +
           velocity_z1[(node_index)] * sub_cell_force_z[(cells_off) + (nn)]);
    }

    energy1[(cc)] = energy0[(cc)] - mesh->dt * force / cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  // Using the new volume, calculate the predicted density
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    double cell_volume = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {

      // TODO: NEED TO CALCULATE THE VOLUME OF THE CELL
    }

    density1[(cc)] = cell_mass[(cc)] / cell_volume;
  }
  STOP_PROFILING(&compute_profile, "calc_new_density");

  // Calculate the time centered pressure from mid point between rezoned and
  // predicted pressures
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    // Calculate the predicted pressure from the equation of state
    pressure1[(cc)] = (GAM - 1.0) * energy1[(cc)] * density1[(cc)];

    // Determine the time centered pressure
    pressure1[(cc)] = 0.5 * (pressure0[(cc)] + pressure1[(cc)]);
  }
  STOP_PROFILING(&compute_profile, "equation_of_state_time_center");

  // Prepare time centered variables for the corrector step
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x1[(nn)] = 0.5 * (nodes_x1[(nn)] + nodes_x0[(nn)]);
    nodes_y1[(nn)] = 0.5 * (nodes_y1[(nn)] + nodes_y0[(nn)]);
    nodes_z1[(nn)] = 0.5 * (nodes_z1[(nn)] + nodes_z0[(nn)]);
    node_force_x[(nn)] = 0.0;
    node_force_y[(nn)] = 0.0;
    node_force_z[(nn)] = 0.0;
    node_force_x2[(nn)] = 0.0;
    node_force_y2[(nn)] = 0.0;
    node_force_z2[(nn)] = 0.0;
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }
  STOP_PROFILING(&compute_profile, "move_nodes2");

  /*
   *    CORRECTOR
   */

  initialise_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x1,
                            nodes_y1, nodes_z1, cell_centroids_x,
                            cell_centroids_y, cell_centroids_z);

  // Calculate the new nodal soundspeed and volumes
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int nodes_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - nodes_off;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(nodes_off + cc)];
      const int cell_offset = cells_offsets[(cell_index)];
      const int nnodes_by_cell = cells_offsets[(cell_index + 1)] - cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_offset + nn2)] == nn) {
          break;
        }
      }

      // Fetch the cell centroid position
      const double cell_c_x = cell_centroids_x[(cell_index)];
      const double cell_c_y = cell_centroids_y[(cell_index)];
      const double cell_c_z = cell_centroids_z[(cell_index)];

      // TODO: CALCULATE THE SUB-CELL VOLUME
      const double sub_cell_volume = 0.0;

      // Add contributions to the nodal mass from adjacent sub-cells
      nodal_soundspeed[(nn)] +=
          sqrt(GAM * (GAM - 1.0) * energy1[(cell_index)]) * sub_cell_volume;

      // Calculate the volume at the node
      nodal_volumes[(nn)] += sub_cell_volume;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_volume");

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_soundspeed");

  // Calculate the force contributions for pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int nodes_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - nodes_off;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(nodes_off + cc)];
      const int cell_offset = cells_offsets[(cell_index)];
      const int nnodes_by_cell = cells_offsets[(cell_index + 1)] - cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_offset + nn2)] == nn) {
          break;
        }
      }

      // TODO: CALCULATE THE AREA VECTORS
      double S_x = 0.0;
      double S_y = 0.0;
      double S_z = 0.0;

      node_force_x[(nn)] += pressure1[(cell_index)] * S_x;
      node_force_y[(nn)] += pressure1[(cell_index)] * S_y;
      node_force_z[(nn)] += pressure1[(cell_index)] * S_z;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_force");

  // Calculate the force contributions for pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_c_index = cells_to_nodes[(cells_off) + (nn)];

      // TODO: CALCULATE THE AREA VECTORS
      double S_x = 0.0;
      double S_y = 0.0;
      double S_z = 0.0;

      sub_cell_force_x[(cells_off + nn)] = pressure1[(cc)] * S_x;
      sub_cell_force_y[(cells_off + nn)] = pressure1[(cc)] * S_y;
      sub_cell_force_z[(cells_off + nn)] = pressure1[(cc)] * S_z;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_cell_forces");

  calculate_artificial_viscosity(
      nnodes, visc_coeff1, visc_coeff2, cells_offsets, cells_to_nodes,
      nodes_offsets, nodes_to_cells, nodes_x1, nodes_y1, nodes_z1,
      cell_centroids_x, cell_centroids_y, cell_centroids_z, velocity_x1,
      velocity_y1, velocity_z1, nodal_soundspeed, nodal_mass, nodal_volumes,
      limiter, node_force_x, node_force_y, node_force_z, node_force_x2,
      node_force_y2, node_force_z2);

  update_velocity(nnodes, mesh->dt, node_force_x, node_force_y, node_force_z,
                  nodal_mass, velocity_x0, velocity_y0, velocity_z0,
                  velocity_x1, velocity_y1, velocity_z1);

  handle_unstructured_reflect_3d(nnodes, boundary_index, boundary_type,
                                 boundary_normal_x, boundary_normal_y,
                                 boundary_normal_z, velocity_x0, velocity_y0,
                                 velocity_z0);

  // Calculate the corrected node movements
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x0[(nn)] += mesh->dt * velocity_x0[(nn)];
    nodes_y0[(nn)] += mesh->dt * velocity_y0[(nn)];
    nodes_z0[(nn)] += mesh->dt * velocity_z0[(nn)];
  }
  STOP_PROFILING(&compute_profile, "move_nodes");

  initialise_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0,
                            nodes_y0, nodes_z0, cell_centroids_x,
                            cell_centroids_y, cell_centroids_z);

  set_timestep(ncells, cells_to_nodes, cells_offsets, nodes_x0, nodes_y0,
               nodes_z0, energy1, &mesh->dt);

  // Calculate the final energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    // Sum the time centered velocity by the sub-cell forces
    double force = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cells_off) + (nn)];
      force +=
          (velocity_x0[(node_index)] * sub_cell_force_x[(cells_off) + (nn)] +
           velocity_y0[(node_index)] * sub_cell_force_y[(cells_off) + (nn)] +
           velocity_z0[(node_index)] * sub_cell_force_z[(cells_off) + (nn)]);
    }

    energy0[(cc)] -= mesh->dt * force / cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  // Using the new corrected volume, calculate the density
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    double cell_volume = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      // TODO: CALCULATE THE AREA OF THE CELL
    }

    // Update the density using the new volume
    density0[(cc)] = cell_mass[(cc)] / cell_volume;
  }
  STOP_PROFILING(&compute_profile, "calc_new_density");
}

// Calculates the artificial viscous forces for momentum acceleration
void calculate_artificial_viscosity(
    const int nnodes, const double visc_coeff1, const double visc_coeff2,
    const int* cells_offsets, const int* cells_to_nodes,
    const int* nodes_offsets, const int* nodes_to_cells, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* velocity_x,
    const double* velocity_y, const double* velocity_z,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter, double* node_force_x,
    double* node_force_y, double* node_force_z, double* node_force_x2,
    double* node_force_y2, double* node_force_z2) {
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int nodes_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - nodes_off;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(nodes_off + cc)];
      const int cell_offset = cells_offsets[(cell_index)];
      const int nnodes_by_cell = cells_offsets[(cell_index + 1)] - cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_offset + nn2)] == nn) {
          break;
        }
      }

      // TODO NEED TO CALCULATE THE CORRECT NODES THAT WE NEED TO BE LOOKING AT
      // HERE??? LOOK BACK AT THE PAPER...
      int node_r_index = 0;

      // Calculate the velocity gradients
      const double grad_velocity_x =
          velocity_x[(node_r_index)] - velocity_x[(nn)];
      const double grad_velocity_y =
          velocity_y[(node_r_index)] - velocity_y[(nn)];
      const double grad_velocity_z =
          velocity_z[(node_r_index)] - velocity_z[(nn)];
      const double grad_velocity_mag = sqrt(grad_velocity_x * grad_velocity_x +
                                            grad_velocity_y * grad_velocity_y +
                                            grad_velocity_z * grad_velocity_z);

      // Calculate the unit vectors of the velocity gradients
      const double grad_velocity_unit_x =
          (grad_velocity_x != 0.0) ? grad_velocity_x / grad_velocity_mag : 0.0;
      const double grad_velocity_unit_y =
          (grad_velocity_y != 0.0) ? grad_velocity_y / grad_velocity_mag : 0.0;
      const double grad_velocity_unit_z =
          (grad_velocity_z != 0.0) ? grad_velocity_z / grad_velocity_mag : 0.0;

      // TODO: WE NEED TO CALCULATE THE FACE CENTERED DENSITY HERE, WHICH IS
      // PROBABLY ACHIEVED BY CALCULATING THE HARMONIC MEAN OF ALL FOUR NODES
      double nodal_density0 = nodal_mass[(nn)] / nodal_volumes[(nn)];
      double nodal_density1 = 0.0;
      double nodal_density2 = 0.0;
      double nodal_density3 = 0.0;
      const double density_edge =
          (4.0 * nodal_density0 * nodal_density1 * nodal_density2 *
           nodal_density3) /
          (nodal_density0 + nodal_density1 + nodal_density2 + nodal_density3);

      // TODO: CALCULATE THE AREA VECTORS
      double S_x = 0.0;
      double S_y = 0.0;
      double S_z = 0.0;

      // Calculate the artificial viscous force term for the edge
      const double t = 0.25 * (GAM + 1.0);
      double expansion_term = (grad_velocity_x * S_x + grad_velocity_y * S_y +
                               grad_velocity_z * S_z);

      // Calculate the minimum soundspeed
      const double cs =
          min(nodal_soundspeed[(nn)], nodal_soundspeed[(node_r_index)]);

      // If the cell is compressing, calculate the edge forces and add their
      // contributions to the node forces
      if (expansion_term <= 0.0) {
        const double edge_visc_force_x =
            density_edge * (visc_coeff2 * t * fabs(grad_velocity_x) +
                            sqrt(visc_coeff2 * visc_coeff2 * t * t *
                                     grad_velocity_x * grad_velocity_x +
                                 visc_coeff1 * visc_coeff1 * cs * cs)) *
            (1.0 - limiter[(nn)]) * (grad_velocity_x * S_x) *
            grad_velocity_unit_x;
        const double edge_visc_force_y =
            density_edge * (visc_coeff2 * t * fabs(grad_velocity_y) +
                            sqrt(visc_coeff2 * visc_coeff2 * t * t *
                                     grad_velocity_y * grad_velocity_y +
                                 visc_coeff1 * visc_coeff1 * cs * cs)) *
            (1.0 - limiter[(nn)]) * (grad_velocity_y * S_y) *
            grad_velocity_unit_y;
        const double edge_visc_force_z =
            density_edge * (visc_coeff2 * t * fabs(grad_velocity_z) +
                            sqrt(visc_coeff2 * visc_coeff2 * t * t *
                                     grad_velocity_z * grad_velocity_z +
                                 visc_coeff1 * visc_coeff1 * cs * cs)) *
            (1.0 - limiter[(nn)]) * (grad_velocity_z * S_y) *
            grad_velocity_unit_z;

        // Add the contributions of the edge based artifical viscous terms
        // to the main force terms
        node_force_x[(nn)] -= edge_visc_force_x;
        node_force_y[(nn)] -= edge_visc_force_y;
        node_force_z[(nn)] -= edge_visc_force_z;

        //
        //
        //
        //
        //
        // TODO : There is a race condition here...
        node_force_x[(node_r_index)] += edge_visc_force_x;
        node_force_y[(node_r_index)] += edge_visc_force_y;
        node_force_z[(node_r_index)] += edge_visc_force_z;
      }
    }
  }

  STOP_PROFILING(&compute_profile, "artificial_viscosity");
}

// Controls the timestep for the simulation
void set_timestep(const int ncells, const int* cells_to_nodes,
                  const int* cells_offsets, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt) {

  // TODO: THIS COMPUTATION IS VERY INEFFICIENT, IT REPEAT MANY CALCULATIONS
  // UNNECESSARILY

  // Calculate the timestep based on the computational mesh and CFL condition
  double local_dt = DBL_MAX;
  START_PROFILING(&compute_profile);
#pragma omp parallel for reduction(min : local_dt)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    double shortest_edge = DBL_MAX;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {

      // TODO: NEED TO FIND A WAY TO STEP THROUGH, FINDING THE EDGES THAT BELONG
      // TO THE CELL...
      int node_c_index = 0;
      int node_r_index = 0;

      // Find the shortest edge of this cell
      const double x_component =
          nodes_x[(node_c_index)] - nodes_x[(node_r_index)];
      const double y_component =
          nodes_y[(node_c_index)] - nodes_y[(node_r_index)];
      const double z_component =
          nodes_z[(node_c_index)] - nodes_z[(node_r_index)];
      shortest_edge = min(shortest_edge, sqrt(x_component * x_component +
                                              y_component * y_component +
                                              z_component * z_component));
    }

    const double soundspeed = sqrt(GAM * (GAM - 1.0) * energy[(cc)]);
    local_dt = min(local_dt, shortest_edge / soundspeed);
  }
  STOP_PROFILING(&compute_profile, __func__);

  *dt = CFL * local_dt;
  printf("Timestep %.8fs\n", *dt);
}

// Uodates the velocity due to the pressure gradients
void update_velocity(const int nnodes, const double dt,
                     const double* node_force_x, const double* node_force_y,
                     const double* node_force_z, const double* nodal_mass,
                     double* velocity_x0, double* velocity_y0,
                     double* velocity_z0, double* velocity_x1,
                     double* velocity_y1, double* velocity_z1) {
  // Calculate the corrected time centered velocities
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    // Calculate the new velocities
    velocity_x1[(nn)] += dt * node_force_x[(nn)] / nodal_mass[(nn)];
    velocity_y1[(nn)] += dt * node_force_y[(nn)] / nodal_mass[(nn)];
    velocity_z1[(nn)] += dt * node_force_z[(nn)] / nodal_mass[(nn)];

    // Calculate the corrected time centered velocities
    velocity_x0[(nn)] = 0.5 * (velocity_x1[(nn)] + velocity_x0[(nn)]);
    velocity_y0[(nn)] = 0.5 * (velocity_y1[(nn)] + velocity_y0[(nn)]);
    velocity_z0[(nn)] = 0.5 * (velocity_z1[(nn)] + velocity_z0[(nn)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_velocity_time_center");
}

// Initialises the cell mass, sub-cell mass and sub-cell volume
void initialise_mesh_mass(const int ncells, const int* cells_offsets,
                          const double* cell_centroids_x,
                          const double* cell_centroids_y,
                          const double* cell_centroids_z,
                          const int* cells_to_nodes, const double* density,
                          const double* nodes_x, const double* nodes_y,
                          const double* nodes_z, double* cell_mass,
                          double* sub_cell_volume, double* sub_cell_mass) {
  // Calculate the cell mass
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    // Fetch the center of the cell
    const double cell_c_x = cell_centroids_x[(cc)];
    const double cell_c_y = cell_centroids_y[(cc)];
    const double cell_c_z = cell_centroids_z[(cc)];

    double cell_volume = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {

      // TODO: NEED TO FIND THE SUB-CELL VOLUMe
      sub_cell_volume[(cells_off + nn)] = 0.0;

      // TODO: this should be updated to fix the issues with hourglassing...
      if (sub_cell_volume[(cells_off + nn)] <= 0.0) {
        TERMINATE("Encountered cell with unphysical volume %.12f in cell %d.",
                  sub_cell_volume[(cells_off + nn)], cc);
      }

      // Reduce the total cell volume for later calculation
      cell_volume += sub_cell_volume[(cells_off + nn)];

      // TODO: This should be an initialisation step right?
      sub_cell_mass[(cells_off + nn)] =
          density[(cc)] * sub_cell_volume[(cells_off + nn)];
    }

    // Calculate the mass and store volume for the whole cell
    cell_mass[(cc)] = density[(cc)] * cell_volume;
    total_mass += cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, __func__);

  printf("Initial total mesh mash: %.15f\n", total_mass);
}

// Initialises the centroids for each cell
void initialise_cell_centroids(const int ncells, const int* cells_offsets,
                               const int* cells_to_nodes, const double* nodes_x,
                               const double* nodes_y, const double* nodes_z,
                               double* cell_centroids_x,
                               double* cell_centroids_y,
                               double* cell_centroids_z) {
  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cells_off + nn)];
      cx += nodes_x[(node_index)];
      cy += nodes_y[(node_index)];
      cz += nodes_z[(node_index)];
    }
    cell_centroids_x[(cc)] = cx / (double)nnodes_by_cell;
    cell_centroids_y[(cc)] = cy / (double)nnodes_by_cell;
    cell_centroids_z[(cc)] = cz / (double)nnodes_by_cell;
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Initialises the centroids for each cell
void initialise_sub_cell_centroids(
    const int ncells, const int* cells_offsets, const int* cells_to_nodes,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, double* sub_cell_centroids_x,
    double* sub_cell_centroids_y, double* sub_cell_centroids_z) {
  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nsub_cells = cells_offsets[(cc + 1)] - cells_off;

    const double cell_c_x = cell_centroids_x[(cc)];
    const double cell_c_y = cell_centroids_y[(cc)];
    const double cell_c_z = cell_centroids_z[(cc)];

    for (int ss = 0; ss < nsub_cells; ++ss) {
      // TODO: GET THE NODES AROUND A SUB-CELL

      sub_cell_centroids_x[(cc)] = 0.0;
      sub_cell_centroids_y[(cc)] = 0.0;
      sub_cell_centroids_z[(cc)] = 0.0;
    }
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Stores the rezoned mesh specification as the original mesh. Until we
// determine a reasonable rezoning algorithm, this makes us Eulerian
void store_rezoned_mesh(const int nnodes, const double* nodes_x,
                        const double* nodes_y, const double* nodes_z,
                        double* rezoned_nodes_x, double* rezoned_nodes_y,
                        double* rezoned_nodes_z) {

// Store the rezoned nodes
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    rezoned_nodes_x[(nn)] = nodes_x[(nn)];
    rezoned_nodes_y[(nn)] = nodes_y[(nn)];
    rezoned_nodes_z[(nn)] = nodes_z[(nn)];
  }
}

#if 0
/*
 * REMAP STEP PROTOTYPE
 */

// Calculate the sub-cell energies
for (int cc = 0; cc < ncells; ++cc) {
  const int cells_off = cells_offsets[(cc)];
  const int nedges = cells_offsets[(cc + 1)] - cells_off;
  const int nsub_cells = nedges;
  const int nnodes_by_cell = nedges;

  // Fetch the cell centroids position
  const double cell_c_x = cell_centroids_x[(cc)];
  const double cell_c_y = cell_centroids_y[(cc)];

  // TODO: Does this least squares implementation still work if we break
  // some of the conditions that fall naturally from having even sides
  // on all of our shapes?

  /* Least squares regression taken from nodes in order to determine
   * the gradients that exist across the internal energy */

  // Calculate the coefficents to matrix M
  double MTM[3] = {0.0}; // Describes the three unique quantities in (M^T.M)
  double MT_del_phi[2] = {0.0};

  // Calculate the coefficients for all edges
  for (int ee = 0; ee < nedges; ++ee) {
    const int neighbour_index = cells_to_cells[(cells_off + ee)];

    // TODO: NOT SURE IF THIS IS THE CORRECT THING TO DO WITH BOUNDARY
    // CONDITION
    if (neighbour_index == IS_BOUNDARY) {
      continue;
    }

    // Calculate the vector pointing between the cell centroids
    double es_x = (cell_centroids_x[(neighbour_index)] - cell_c_x);
    double es_y = (cell_centroids_y[(neighbour_index)] - cell_c_y);
    const double centroid_distance = sqrt(es_x * es_x + es_y * es_y);
    es_x /= centroid_distance;
    es_y /= centroid_distance;

    // The edge relating to our current neighbour
    const int node_c_index = cells_to_nodes[(cells_off + ee)];
    const int node_l_index =
      (ee - 1 >= 0) ? cells_to_nodes[(cells_off + ee - 1)]
      : cells_to_nodes[(cells_off + nnodes_by_cell - 1)];

    // Calculate the area vector for the face that we are looking at
    double A_x = (nodes_y0[(node_l_index)] - nodes_y0[(node_c_index)]);
    double A_y = -(nodes_x0[(node_l_index)] - nodes_x0[(node_c_index)]);

    // Fix the direction that the area vector is pointing in
    if ((A_x * es_x + A_y * es_y) < 0.0) {
      A_x = -A_x;
      A_y = -A_y;
    }

    // Calculate the gradient matrix
    const double phi0 = energy0[(cc)];
    const double phi_ff = energy0[(neighbour_index)];
    MTM[0] += es_x * es_x;
    MTM[1] += es_x * es_y;
    MTM[2] += es_y * es_y;
    MT_del_phi[0] += es_x * (phi_ff - phi0);
    MT_del_phi[1] += es_y * (phi_ff - phi0);
  }

  // Solve the equation for the temperature gradients
  const double MTM_det = (1.0 / (MTM[0] * MTM[2] - MTM[1] * MTM[1]));
  const double grad_e_x =
    MTM_det * (MT_del_phi[0] * MTM[2] - MT_del_phi[1] * MTM[1]);
  const double grad_e_y =
    MTM_det * (MT_del_phi[1] * MTM[0] - MT_del_phi[0] * MTM[1]);

  // Calculate the energy density in the cell
  double energy_density = energy0[(cc)] * density0[(cc)];

  /* Can now determine the sub cell internal energy */

  // Loop over all sub-cells to calculate integrals
  for (int ss = 0; ss < nsub_cells; ++ss) {
    // Determine the three point stencil of nodes around anchor node
    const int node_l_index =
      (ss == 0) ? cells_to_nodes[(cells_off + nnodes_by_cell - 1)]
      : cells_to_nodes[(cells_off) + (ss - 1)];
    const int node_c_index = cells_to_nodes[(cells_off) + (ss)];
    const int node_r_index = (ss == nnodes_by_cell - 1)
      ? cells_to_nodes[(cells_off)]
      : cells_to_nodes[(cells_off) + (ss + 1)];

    // TODO: refactor the routines here so that they operate in loops...

    // Get the anchor node position
    const double node_c_x = nodes_x0[(node_c_index)];
    const double node_c_y = nodes_y0[(node_c_index)];

    // Get the midpoints between l and r nodes and current node
    const double node_l_x = 0.5 * (nodes_x0[(node_l_index)] + node_c_x);
    const double node_l_y = 0.5 * (nodes_y0[(node_l_index)] + node_c_y);
    const double node_r_x = 0.5 * (node_c_x + nodes_x0[(node_r_index)]);
    const double node_r_y = 0.5 * (node_c_y + nodes_y0[(node_r_index)]);

    // Calculate the volume integral weighted by x and y
    const double sub_cell_x_volume =
      (1.0 / 6.0) *
      ((node_c_x * node_c_x + node_c_x * node_r_x + node_r_x * node_r_x) *
       (node_r_y - node_c_y) +
       (node_r_x * node_r_x + node_r_x * cell_c_x + cell_c_x * cell_c_x) *
       (cell_c_y - node_r_y) +
       (cell_c_x * cell_c_x + cell_c_x * node_l_x + node_l_x * node_l_x) *
       (node_l_y - cell_c_y) +
       (node_l_x * node_l_x + node_l_x * node_c_x + node_c_x * node_c_x) *
       (node_c_y - node_l_y));
    const double sub_cell_y_volume =
      (1.0 / 6.0) *
      ((node_c_y * node_c_y + node_c_y * node_r_y + node_r_y * node_r_y) *
       (node_r_x - node_c_x) +
       (node_r_y * node_r_y + node_r_y * cell_c_y + cell_c_y * cell_c_y) *
       (cell_c_x - node_r_x) +
       (cell_c_y * cell_c_y + cell_c_y * node_l_y + node_l_y * node_l_y) *
       (node_l_x - cell_c_x) +
       (node_l_y * node_l_y + node_l_y * node_c_y + node_c_y * node_c_y) *
       (node_c_x - node_l_x));

    // Calculate the sub cell energy mass
    double sub_cell_e_mass =
      energy_density * sub_cell_volume[(cells_off + ss)] +
      grad_e_x * (sub_cell_x_volume -
          sub_cell_volume[(cells_off + ss)] * cell_c_x) +
      grad_e_y * (sub_cell_y_volume -
          sub_cell_volume[(cells_off + ss)] * cell_c_y);

    sub_cell_energy[(cells_off + ss)] = sub_cell_e_mass;
  }
}

// Calculate the sub-cell velocities and kinetic energy
for (int cc = 0; cc < ncells; ++cc) {
  const int cells_off = cells_offsets[(cc)];
  const int nsub_cells = cells_offsets[(cc + 1)] - cells_off;
  const int nnodes_by_cell = nsub_cells;

  double velocities_x[nnodes_by_cell];
  double velocities_y[nnodes_by_cell];
  double masses[nsub_cells];

  // Privatise all of the data we will need
  for (int ss = 0; ss < nsub_cells; ++ss) {
    const int node_index = cells_to_nodes[(cells_off) + (ss)];
    velocities_x[(ss)] = velocity_x0[(node_index)];
    velocities_y[(ss)] = velocity_y0[(node_index)];
    masses[(ss)] = sub_cell_mass[(cells_off + ss)];
  }

  // Calculate the constant coefficients for determining the sub-cell
  // velocities and kinetic energies
  double vel_coeff_x = 0.0;
  double vel_coeff_y = 0.0;
  double ke_coeff = 0.0;
  for (int ss = 0; ss < nsub_cells; ++ss) {
    const int p_index = (ss == 0) ? (nsub_cells - 1) : (ss - 1);
    const int n_index = (ss == nsub_cells - 1) ? (0) : (ss + 1);
    vel_coeff_x +=
      velocities_x[(ss)] *
      (4.0 * masses[(ss)] - masses[(p_index)] - masses[(n_index)]) /
      (8.0 * cell_mass[(cc)]);
    vel_coeff_y +=
      velocities_y[(ss)] *
      (4.0 * masses[(ss)] - masses[(p_index)] - masses[(n_index)]) /
      (8.0 * cell_mass[(cc)]);
    ke_coeff += 0.5 * (velocities_x[(ss)] * velocities_x[(ss)] +
        velocities_y[(ss)] * velocities_y[(ss)]) *
      (4.0 * masses[(ss)] - masses[(p_index)] - masses[(n_index)]) /
      (8.0 * cell_mass[(cc)]);
  }

  for (int ss = 0; ss < nsub_cells; ++ss) {
    const int p_index = (ss == 0) ? (nsub_cells - 1) : (ss - 1);
    const int n_index = (ss == nsub_cells - 1) ? (0) : (ss + 1);

    // Store the sub-cell velocities
    sub_cell_velocity_x[(cells_off + ss)] =
      vel_coeff_x +
      0.25 * (2.0 * velocities_x[(ss)] + 0.5 * velocities_x[(n_index)] +
          0.5 * velocities_x[(p_index)]);
    sub_cell_velocity_y[(cells_off + ss)] =
      vel_coeff_y +
      0.25 * (2.0 * velocities_y[(ss)] + 0.5 * velocities_y[(n_index)] +
          0.5 * velocities_y[(p_index)]);

    // Calculate the surrounding values of the kinetic energy
    const double ke_c = 0.5 * (velocities_x[(ss)] * velocities_x[(ss)] +
        velocities_y[(ss)] * velocities_y[(ss)]);
    const double ke_p =
      0.5 * (velocities_x[(p_index)] * velocities_x[(p_index)] +
          velocities_y[(p_index)] * velocities_y[(p_index)]);
    const double ke_n =
      0.5 * (velocities_x[(n_index)] * velocities_x[(n_index)] +
          velocities_y[(n_index)] * velocities_y[(n_index)]);

    sub_cell_kinetic_energy[(cells_off + ss)] =
      ke_coeff + 0.25 * (2.0 * ke_c + 0.5 * ke_n + 0.5 * ke_p);
  }
}

initialise_sub_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0,
    nodes_y0, cell_centroids_x, cell_centroids_y,
    sub_cell_centroids_x, sub_cell_centroids_y);

// TODO: Need to handle the boundary conditions here... in the other routine
// we simply ignore the boundaries in the gradient calculation, still not sure
// if this a reasonable approach but it's definitely the easiest

// Here we calculate the gradient for the quantity in each sub-cell
for (int cc = 0; cc < ncells; ++cc) {
  const int cells_off = cells_offsets[(cc)];
  const int nsub_cells = cells_offsets[(cc + 1)] - cells_off;

  // Calculate the gradient of the change in density between the sub-cells
  for (int ss = 0; ss < nsub_cells; ++ss) {
    const int sub_cell_c_index = (cells_off + ss);
    const int node_c_index = cells_to_nodes[(cells_off) + (ss)];

    const int nodes_off = nodes_offsets[(node_c_index)];
    const int ncells_by_node = nodes_offsets[(node_c_index + 1)] - nodes_off;

    // Scan through the cells attached to the sub-cell's external node, to
    // fetch the two cells that neighbour the sub-cell
    int out_cell_index[2] = {-1, -1};
    for (int cc2 = 0; cc2 < ncells_by_node; ++cc2) {
      const int cell_index = nodes_to_cells[(nodes_off + cc2)];
      if (cell_index == cc) {
        // We have found our current cell, so get the neighbours
        out_cell_index[0] =
          (cc2 == 0) ? nodes_to_cells[(nodes_off + ncells_by_node - 1)]
          : nodes_to_cells[(nodes_off) + (cc2 - 1)];
        out_cell_index[1] = (cc2 == ncells_by_node - 1)
          ? nodes_to_cells[(nodes_off)]
          : nodes_to_cells[(nodes_off) + (cc2 + 1)];
        break;
      }
    }

    if (out_cell_index[0] == -1 || out_cell_index[1] == -1) {
      TERMINATE(
          "We were not able to find the cell neighbouring this sub-cell.");
    }

    // The indices for sub-cells in neighbourhood of current sub-cell
    int sub_cell_neighbour_indices[nsub_cell_edges];

    for (int oo = 0; oo < 2; ++oo) {
      const int out_cells_off = cells_offsets[(out_cell_index[oo])];
      const int out_nnodes =
        cells_offsets[(out_cell_index[oo] + 1)] - cells_off;

      for (int nn = 0; nn < out_nnodes; ++nn) {
        const int out_node_c_index = cells_to_nodes[(out_cells_off + nn)];
        if (node_c_index == node_c_index) {
          // We have found the neighbouring sub-cell
          sub_cell_neighbour_indices[oo] = (out_cells_off + nn);
          break;
        }
      }
    }

    sub_cell_neighbour_indices[2] =
      (ss == nsub_cells - 1) ? (cells_off + 0) : (cells_off + ss + 1);
    sub_cell_neighbour_indices[3] =
      (ss == 0) ? (cells_off + nsub_cells - 1) : (cells_off + ss - 1);

    // Loop over all of the neighbours
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    double e = 0.0;

    // Fetch the density for the sub-cell
    double sub_cell_c_density = sub_cell_mass[(sub_cell_c_index)] /
      sub_cell_volume[(sub_cell_c_index)];

    for (int nn = 0; nn < nsub_cell_edges; ++nn) {
      const int sub_cell_n_index = sub_cell_neighbour_indices[(nn)];

      // Fetch the density for the neighbour
      double sub_cell_n_density = sub_cell_mass[(sub_cell_c_index)] /
        sub_cell_volume[(sub_cell_c_index)];

      // Calculate the differential quantities
      const double dx = sub_cell_centroids_x[(sub_cell_n_index)] -
        sub_cell_centroids_x[(sub_cell_c_index)];
      const double dy = sub_cell_centroids_y[(sub_cell_n_index)] -
        sub_cell_centroids_y[(sub_cell_c_index)];
      const double drho = sub_cell_n_density - sub_cell_c_density;

      double omega2 = 1.0 / (dx * dx + dy * dy);

      // Calculate the coefficients for the minimisation
      a += omega2 * dx * dx;
      b += omega2 * dx * dy;
      c += omega2 * dy * dy;
      d += omega2 * drho * dx;
      e += omega2 * drho * dy;
    }

    // Solve the minimisation problem to get the gradients for the sub-cell
    sub_cell_grad_x[(sub_cell_c_index)] = (c * d - b * e) / (a * c - b * b);
    sub_cell_grad_y[(sub_cell_c_index)] = (a * e - b * d) / (a * c - b * b);
  }
}

// Here we are going to update the mass using the linear function that we have
// now been able to successfully describe for all of the sub-cells
for (int cc = 0; cc < ncells; ++cc) {
  const int cells_off = cells_offsets[(cc)];
  const int nsub_cells = cells_offsets[(cc + 1)] - cells_off;

  // Store the cell centroids as these are the same for all of the
  // sub-cells that are within this cell
  const double cell_c_x = cell_centroids_x[(cell_index)];
  const double cell_c_y = cell_centroids_y[(cell_index)];
  const double rezoned_cell_c_x = cell_centroids_x[(cell_index)];
  const double rezoned_cell_c_y = cell_centroids_y[(cell_index)];

  // Consider all of the sub-cells that are in the new grid, assuming that we
  // have maintained identical connectivity
  for (int ss = 0; ss < nsub_cells; ++ss) {
    const int sub_cell_c_index = (cells_off + ss);

    // Determine the three point stencil of nodes around anchor node
    const int node_l_index =
      (ss == 0) ? cells_to_nodes[(cells_off + nnodes_by_cell - 1)]
      : cells_to_nodes[(cells_off) + (ss - 1)];
    const int node_c_index = cells_to_nodes[(cells_off) + (ss)];
    const int node_r_index = (ss == nnodes_by_cell - 1)
      ? cells_to_nodes[(cells_off)]
      : cells_to_nodes[(cells_off) + (ss + 1)];

    // Calculate the nodes surrounding the sub-cell
    double sub_cell_nodes_x[nsub_cell_edges];
    double sub_cell_nodes_y[nsub_cell_edges];
    double new_sub_cell_nodes_x[nsub_cell_edges];
    double new_sub_cell_nodes_y[nsub_cell_edges];

    // The sub-cell is surrounded by the anchor node, midpoints on the outer
    // edges and the center of the cell
    sub_cell_nodes_x[0] = nodes_x0[(node_c_index)];
    sub_cell_nodes_y[0] = nodes_y0[(node_c_index)];
    sub_cell_nodes_x[1] = 0.5 * (node_c_x + nodes_x0[(node_r_index)]);
    sub_cell_nodes_y[1] = 0.5 * (node_c_y + nodes_y0[(node_r_index)]);
    sub_cell_nodes_x[2] = cell_c_x;
    sub_cell_nodes_y[2] = cell_c_y;
    sub_cell_nodes_x[3] = 0.5 * (nodes_x0[(node_l_index)] + node_c_x);
    sub_cell_nodes_y[3] = 0.5 * (nodes_y0[(node_l_index)] + node_c_y);

    // Get the same data for the new sub-cell after rezoning
    new_sub_cell_nodes_x[0] = rezoned_nodes_x0[(node_c_index)];
    new_sub_cell_nodes_y[0] = rezoned_nodes_y0[(node_c_index)];
    new_sub_cell_nodes_x[1] =
      0.5 * (node_c_x + rezoned_nodes_x0[(node_r_index)]);
    new_sub_cell_nodes_y[1] =
      0.5 * (node_c_y + rezoned_nodes_y0[(node_r_index)]);
    new_sub_cell_nodes_x[2] = cell_c_x;
    new_sub_cell_nodes_y[2] = cell_c_y;
    new_sub_cell_nodes_x[3] =
      0.5 * (rezoned_nodes_x0[(node_l_index)] + node_c_x);
    new_sub_cell_nodes_y[3] =
      0.5 * (rezoned_nodes_y0[(node_l_index)] + node_c_y);

    for (int ee = 0; ee < nsub_cell_edges; ++ee) {
      const int node0_off = (cells_off + ss);
      const int node1_off =
        (ss == nsub_cells - 1) ? (cells_off) : (cells_off + ss);

      const int node0_index = cells_to_nodes[(node0_off)];
      const int node1_index = cells_to_nodes[(node1_off)];

      // This ordering of nodes encompasses the swept region for the edge
      double nodes_x[nsub_cell_indices];
      double nodes_y[nsub_cell_indices];
      nodes_x[0] = nodes_x0[(node0_index)];
      nodes_x[1] = nodes_x0[(node1_index)];
      nodes_x[2] = rezoned_nodes_x[(node1_index)];
      nodes_x[3] = rezoned_nodes_x[(node0_index)];
      nodes_y[0] = nodes_y0[(node0_index)];
      nodes_y[1] = nodes_y0[(node1_index)];
      nodes_y[2] = rezoned_nodes_y[(node1_index)];
      nodes_y[3] = rezoned_nodes_y[(node0_index)];

      // We want to calculate the volume integral
      double volume = 0.0;
      for (int nn = 0; nn < nsub_cell_indices; ++nn) {
        volume += 0.5 * (nodes_x[0] * nodes_x[1]) * (nodes_y[1] - nodes_y[0]);
      }

      // The sign of the volume determines the donor sub-cell
    }
  }
}
#endif // if 0
