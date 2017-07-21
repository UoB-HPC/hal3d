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
    double* sub_cell_grad_y, double* sub_cell_grad_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces) {

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

  // TODO: SOOO MUCH POTENTIAL FOR OPTIMISATION HERE...!
  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#if 0
#pragma omp parallel for
#endif // if 0
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;
    const double node_c_x = nodes_x0[(nn)];
    const double node_c_y = nodes_y0[(nn)];
    const double node_c_z = nodes_z0[(nn)];

    // Consider all faces attached to node
    for (int ff = 0; ff < nfaces_by_node; ++ff) {
      const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
      if (face_index == -1) {
        continue;
      }

      // Determine the offset into the list of nodes
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Find node center and location of current node on face
      int node_in_face_c;
      double face_c_x = 0.0;
      double face_c_y = 0.0;
      double face_c_z = 0.0;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c_x += nodes_x0[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y0[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z0[(node_index)] / nnodes_by_face;

        // Choose the node in the list of nodes attached to the face
        if (nn == node_index) {
          node_in_face_c = nn2;
        }
      }

      // Fetch the nodes attached to our current node on the current face
      int nodes[2];
      nodes[0] = (node_in_face_c - 1 >= 0)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c - 1)]
                     : faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)];
      nodes[1] = (node_in_face_c + 1 < nnodes_by_face)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c + 1)]
                     : faces_to_nodes[(face_to_nodes_off)];

      // Fetch the cells attached to our current face
      int cells[2];
      cells[0] = faces_to_cells0[(face_index)];
      cells[1] = faces_to_cells1[(face_index)];

      // Add contributions from all of the cells attached to the face
      for (int cc = 0; cc < 2; ++cc) {
        if (cells[(cc)] == -1) {
          continue;
        }

        // Add contributions for both edges attached to our current node
        for (int nn2 = 0; nn2 < 2; ++nn2) {
          // Get the halfway point on the right edge
          const double half_edge_x =
              0.5 * (nodes_x0[(nodes[(nn2)])] + nodes_x0[(nn)]);
          const double half_edge_y =
              0.5 * (nodes_y0[(nodes[(nn2)])] + nodes_y0[(nn)]);
          const double half_edge_z =
              0.5 * (nodes_z0[(nodes[(nn2)])] + nodes_z0[(nn)]);

          // Setup basis on plane of tetrahedron
          const double a_x = (half_edge_x - face_c_x);
          const double a_y = (half_edge_y - face_c_y);
          const double a_z = (half_edge_z - face_c_z);
          const double b_x = (cell_centroids_x[(cells[cc])] - face_c_x);
          const double b_y = (cell_centroids_y[(cells[cc])] - face_c_y);
          const double b_z = (cell_centroids_z[(cells[cc])] - face_c_z);

          // Calculate the area vector S using cross product
          const double S_x = 0.5 * (a_y * b_z - a_z * b_y);
          const double S_y = -0.5 * (a_x * b_z - a_z * b_x);
          const double S_z = 0.5 * (a_x * b_y - a_y * b_x);

          double sub_cell_volume =
              ((half_edge_x - node_c_x) * S_x + (half_edge_y - node_c_y) * S_y +
               (half_edge_z - node_c_z) * S_z) /
              3.0;

          nodal_mass[(nn)] += density0[(cells[(cc)])] * sub_cell_volume;
          nodal_soundspeed[(nn)] +=
              sqrt(GAM * (GAM - 1.0) * energy0[(cells[(cc)])]) *
              sub_cell_volume;
          nodal_volumes[(nn)] += sub_cell_volume;

          // Calculate the force vector due to pressure at the node
          node_force_x[(nn)] += pressure0[(cells[(cc)])] * S_x;
          node_force_y[(nn)] += pressure0[(cells[(cc)])] * S_y;
          node_force_z[(nn)] += pressure0[(cells[(cc)])] * S_z;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");

#if 0
  calculate_artificial_viscosity(
      nnodes, visc_coeff1, visc_coeff2, cells_offsets, cells_to_nodes,
      nodes_offsets, nodes_to_cells, nodes_x0, nodes_y0, nodes_z0,
      cell_centroids_x, cell_centroids_y, cell_centroids_z, velocity_x0,
      velocity_y0, velocity_z0, nodal_soundspeed, nodal_mass, nodal_volumes,
      limiter, node_force_x, node_force_y, node_force_z, node_force_x2,
      node_force_y2, node_force_z2);
#endif // if 0

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
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double cell_force = 0.0;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Calculate the face center... SHOULD WE PRECOMPUTE?
      double face_c_x = 0.0;
      double face_c_y = 0.0;
      double face_c_z = 0.0;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c_x += nodes_x0[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y0[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z0[(node_index)] / nnodes_by_face;
      }

      // Now we will sum the contributions at each of the nodes
      // TODO: THERE IS SOME SYMMETRY HERE THAT MEANS WE MIGHT BE ABLE TO
      // OPTIMISE
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        // Fetch the nodes attached to our current node on the current face
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn2)];
        const int next_node =
            (nn2 + 1 < nnodes_by_face)
                ? faces_to_nodes[(face_to_nodes_off + nn2 + 1)]
                : faces_to_nodes[(face_to_nodes_off)];

        // Get the halfway point on the right edge
        const double half_edge_x =
            0.5 * (nodes_x0[(current_node)] + nodes_x0[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y0[(current_node)] + nodes_y0[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z0[(current_node)] + nodes_z0[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (half_edge_x - face_c_x);
        const double a_y = (half_edge_y - face_c_y);
        const double a_z = (half_edge_z - face_c_z);
        const double b_x = (cell_centroids_x[(cc)] - face_c_x);
        const double b_y = (cell_centroids_y[(cc)] - face_c_y);
        const double b_z = (cell_centroids_z[(cc)] - face_c_z);

        // Calculate the area vector S using cross product
        const double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        const double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        const double S_z = 0.5 * (a_x * b_y - a_y * b_x);

        cell_force += (velocity_x1[(current_node)] * pressure0[(cc)] * S_x +
                       velocity_y1[(current_node)] * pressure0[(cc)] * S_y +
                       velocity_z1[(current_node)] * pressure0[(cc)] * S_z);
      }
    }

    energy1[(cc)] = energy0[(cc)] - mesh->dt * cell_force / cell_mass[(cc)];
    printf("energy1 %.4f\n energy0 %.4f", energy1[(cc)], energy0[(cc)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  TERMINATE("");

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

      // TODO NEED TO CALCULATE THE CORRECT NODES THAT WE NEED TO BE LOOKING
      // AT
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

      // TODO: NEED TO FIND A WAY TO STEP THROUGH, FINDING THE EDGES THAT
      // BELONG
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
void initialise_mesh_mass(
    const int ncells, const int* cells_offsets, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const int* cells_to_nodes, const double* density, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* cell_mass,
    double* sub_cell_volume, double* sub_cell_mass, int* cells_to_faces_offsets,
    int* cells_to_faces, int* faces_to_nodes_offsets, int* faces_to_nodes) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Calculate the face center... SHOULD WE PRECOMPUTE?
      double face_c_x = 0.0;
      double face_c_y = 0.0;
      double face_c_z = 0.0;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c_x += nodes_x[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z[(node_index)] / nnodes_by_face;
      }

      // Now we will sum the contributions at each of the nodes
      // TODO: THERE IS SOME SYMMETRY HERE THAT MEANS WE MIGHT BE ABLE TO
      // OPTIMISE
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        // Fetch the nodes attached to our current node on the current face
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn2)];
        const int next_node =
            (nn2 + 1 < nnodes_by_face)
                ? faces_to_nodes[(face_to_nodes_off + nn2 + 1)]
                : faces_to_nodes[(face_to_nodes_off)];

        // Get the halfway point on the right edge
        const double half_edge_x =
            0.5 * (nodes_x[(current_node)] + nodes_x[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y[(current_node)] + nodes_y[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z[(current_node)] + nodes_z[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (half_edge_x - face_c_x);
        const double a_y = (half_edge_y - face_c_y);
        const double a_z = (half_edge_z - face_c_z);
        const double b_x = (cell_centroids_x[(cc)] - face_c_x);
        const double b_y = (cell_centroids_y[(cc)] - face_c_y);
        const double b_z = (cell_centroids_z[(cc)] - face_c_z);

        // Calculate the area vector S using cross product
        const double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        const double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        const double S_z = 0.5 * (a_x * b_y - a_y * b_x);
        double sub_cell_volume =
            ((half_edge_x - nodes_x[(current_node)]) * S_x +
             (half_edge_y - nodes_y[(current_node)]) * S_y +
             (half_edge_z - nodes_z[(current_node)]) * S_z) /
            3.0;

        cell_mass[(cc)] += sub_cell_volume;
        total_mass += cell_mass[(cc)];
      }
    }
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
