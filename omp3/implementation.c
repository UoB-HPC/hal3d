#include "../../comms.h"
#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>

// Performs the Lagrangian step of the hydro solve
void lagrangian_phase(
    Mesh* mesh, const int ncells, const int nnodes, const double visc_coeff1,
    const double visc_coeff2, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_to_nodes,
    int* cells_offsets, int* nodes_to_cells, int* nodes_offsets,
    double* nodes_x0, double* nodes_y0, double* nodes_z0, double* nodes_x1,
    double* nodes_y1, double* nodes_z1, int* boundary_index, int* boundary_type,
    double* boundary_normal_x, double* boundary_normal_y,
    double* boundary_normal_z, double* energy0, double* energy1,
    double* density0, double* density1, double* pressure0, double* pressure1,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* velocity_x1, double* velocity_y1, double* velocity_z1,
    double* subcell_force_x, double* subcell_force_y, double* subcell_force_z,
    double* cell_mass, double* nodal_mass, double* nodal_volumes,
    double* nodal_soundspeed, double* limiter, int* nodes_to_faces_offsets,
    int* nodes_to_faces, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* faces_to_cells0, int* faces_to_cells1, int* cells_to_faces_offsets,
    int* cells_to_faces) {

  /*
   *    PREDICTOR
   */

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodal_mass[(nn)] = 0.0;
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }
  STOP_PROFILING(&compute_profile, "zero_node_data");

  // Equation of state, ideal gas law
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    pressure0[(cc)] = (GAM - 1.0) * energy0[(cc)] * density0[(cc)];
  }
  STOP_PROFILING(&compute_profile, "equation_of_state");

  // TODO: SOOO MUCH POTENTIAL FOR OPTIMISATION HERE...!
  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#pragma omp parallel for
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
          const double a_x = (face_c_x - node_c_x);
          const double a_y = (face_c_y - node_c_y);
          const double a_z = (face_c_z - node_c_z);
          const double b_x = (face_c_x - half_edge_x);
          const double b_y = (face_c_y - half_edge_y);
          const double b_z = (face_c_z - half_edge_z);
          const double ab_x = (cell_centroids_x[(cells[cc])] - face_c_x);
          const double ab_y = (cell_centroids_y[(cells[cc])] - face_c_y);
          const double ab_z = (cell_centroids_z[(cells[cc])] - face_c_z);

          // Calculate the area vector S using cross product
          double A_x = 0.5 * (a_y * b_z - a_z * b_y);
          double A_y = -0.5 * (a_x * b_z - a_z * b_x);
          double A_z = 0.5 * (a_x * b_y - a_y * b_x);

          // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES
          // SO
          // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
          // CALCULATION
          const double subcell_volume =
              fabs((ab_x * A_x + ab_y * A_y + ab_z * A_z) / 3.0);

          nodal_mass[(nn)] += density0[(cells[(cc)])] * subcell_volume;
          nodal_soundspeed[(nn)] +=
              sqrt(GAM * (GAM - 1.0) * energy0[(cells[(cc)])]) * subcell_volume;
          nodal_volumes[(nn)] += subcell_volume;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      subcell_force_x[(cell_to_nodes_off + nn)] = 0.0;
      subcell_force_y[(cell_to_nodes_off + nn)] = 0.0;
      subcell_force_z[(cell_to_nodes_off + nn)] = 0.0;
    }
  }

  // Calculate the pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

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
        const double a_x = (face_c_x - nodes_x0[(current_node)]);
        const double a_y = (face_c_y - nodes_y0[(current_node)]);
        const double a_z = (face_c_z - nodes_z0[(current_node)]);
        const double b_x = (face_c_x - half_edge_x);
        const double b_y = (face_c_y - half_edge_y);
        const double b_z = (face_c_z - half_edge_z);
        const double ab_x = (cell_centroids_x[(cc)] - face_c_x);
        const double ab_y = (cell_centroids_y[(cc)] - face_c_y);
        const double ab_z = (cell_centroids_z[(cc)] - face_c_z);

        // Calculate the area vector S using cross product
        double A_x = 0.5 * (a_y * b_z - a_z * b_y);
        double A_y = -0.5 * (a_x * b_z - a_z * b_x);
        double A_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER CLOSED
        // FORM SOLUTION?
        int node_off;
        int next_node_off;
        for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
          if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
            node_off = nn3;
          } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == next_node) {
            next_node_off = nn3;
          }
        }

        // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES SO
        // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
        // CALCULATION
        const int flip = (ab_x * A_x + ab_y * A_y + ab_z * A_z > 0.0);
        subcell_force_x[(cell_to_nodes_off + node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_x : A_x);
        subcell_force_y[(cell_to_nodes_off + node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_y : A_y);
        subcell_force_z[(cell_to_nodes_off + node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_z : A_z);
        subcell_force_x[(cell_to_nodes_off + next_node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_x : A_x);
        subcell_force_y[(cell_to_nodes_off + next_node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_y : A_y);
        subcell_force_z[(cell_to_nodes_off + next_node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_z : A_z);
      }
    }
  }
  STOP_PROFILING(&compute_profile, "node_force_from_pressure");

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }
  STOP_PROFILING(&compute_profile, "scale_soundspeed");

  calc_artificial_viscosity(
      ncells, visc_coeff1, visc_coeff2, cells_offsets, cells_to_nodes, nodes_x0,
      nodes_y0, nodes_z0, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      velocity_x0, velocity_y0, velocity_z0, nodal_soundspeed, nodal_mass,
      nodal_volumes, limiter, subcell_force_x, subcell_force_y, subcell_force_z,
      faces_to_nodes_offsets, faces_to_nodes, cells_to_faces_offsets,
      cells_to_faces);

  // Calculate the time centered evolved velocities, by first calculating the
  // predicted values at the new timestep and then averaging with current
  // velocity
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - node_to_cells_off;

    // Accumulate the force at this node
    double node_force_x0 = 0.0;
    double node_force_y0 = 0.0;
    double node_force_z0 = 0.0;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_offsets[(cell_index + 1)] - cell_to_nodes_off;

      // ARRGHHHH
      int node_off;
      for (node_off = 0; node_off < nnodes_by_cell; ++node_off) {
        if (cells_to_nodes[(cell_to_nodes_off + node_off)] == nn) {
          break;
        }
      }

      node_force_x0 += subcell_force_x[(cell_to_nodes_off + node_off)];
      node_force_y0 += subcell_force_y[(cell_to_nodes_off + node_off)];
      node_force_z0 += subcell_force_z[(cell_to_nodes_off + node_off)];
    }

    // Determine the predicted velocity
    velocity_x1[(nn)] =
        velocity_x0[(nn)] + mesh->dt * node_force_x0 / nodal_mass[(nn)];
    velocity_y1[(nn)] =
        velocity_y0[(nn)] + mesh->dt * node_force_y0 / nodal_mass[(nn)];
    velocity_z1[(nn)] =
        velocity_z0[(nn)] + mesh->dt * node_force_z0 / nodal_mass[(nn)];

    // Calculate the time centered velocity
    velocity_x1[(nn)] = 0.5 * (velocity_x0[(nn)] + velocity_x1[(nn)]);
    velocity_y1[(nn)] = 0.5 * (velocity_y0[(nn)] + velocity_y1[(nn)]);
    velocity_z1[(nn)] = 0.5 * (velocity_z0[(nn)] + velocity_z1[(nn)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_velocity");

  // TODO: NEED TO WORK OUT HOW TO HANDLE BOUNDARY CONDITIONS REASONABLY
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

  init_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x1, nodes_y1,
                      nodes_z1, cell_centroids_x, cell_centroids_y,
                      cell_centroids_z);

  set_timestep(ncells, nodes_x1, nodes_y1, nodes_z1, energy0, &mesh->dt,
               cells_to_faces_offsets, cells_to_faces, faces_to_nodes_offsets,
               faces_to_nodes);

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    double cell_force = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      cell_force += (velocity_x1[(node_index)] *
                         subcell_force_x[(cell_to_nodes_off + nn)] +
                     velocity_y1[(node_index)] *
                         subcell_force_y[(cell_to_nodes_off + nn)] +
                     velocity_z1[(node_index)] *
                         subcell_force_z[(cell_to_nodes_off + nn)]);
    }
    energy1[(cc)] = energy0[(cc)] - mesh->dt * cell_force / cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  // Using the new volume, calculate the predicted density
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double cell_volume = 0.0;

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
        face_c_x += nodes_x1[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y1[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z1[(node_index)] / nnodes_by_face;
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
            0.5 * (nodes_x1[(current_node)] + nodes_x1[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y1[(current_node)] + nodes_y1[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z1[(current_node)] + nodes_z1[(next_node)]);

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

        // TODO: WE MULTIPLY BY 2 HERE BECAUSE WE ARE ADDING THE VOLUME TO BOTH
        // THE CURRENT AND NEXT NODE, OTHERWISE WE ONLY ACCOUNT FOR HALF OF THE
        // 'HALF' TETRAHEDRONS
        cell_volume +=
            fabs(2.0 * ((half_edge_x - nodes_x1[(current_node)]) * S_x +
                        (half_edge_y - nodes_y1[(current_node)]) * S_y +
                        (half_edge_z - nodes_z1[(current_node)]) * S_z) /
                 3.0);
      }
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
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }
  STOP_PROFILING(&compute_profile, "move_nodes2");

/*
 *    CORRECTOR
 */

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      subcell_force_x[(cell_to_nodes_off + nn)] = 0.0;
      subcell_force_y[(cell_to_nodes_off + nn)] = 0.0;
      subcell_force_z[(cell_to_nodes_off + nn)] = 0.0;
    }
  }

  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;
    const double node_c_x = nodes_x1[(nn)];
    const double node_c_y = nodes_y1[(nn)];
    const double node_c_z = nodes_z1[(nn)];

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
        face_c_x += nodes_x1[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y1[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z1[(node_index)] / nnodes_by_face;

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
              0.5 * (nodes_x1[(nodes[(nn2)])] + nodes_x1[(nn)]);
          const double half_edge_y =
              0.5 * (nodes_y1[(nodes[(nn2)])] + nodes_y1[(nn)]);
          const double half_edge_z =
              0.5 * (nodes_z1[(nodes[(nn2)])] + nodes_z1[(nn)]);

          // Setup basis on plane of tetrahedron
          const double a_x = (face_c_x - node_c_x);
          const double a_y = (face_c_y - node_c_y);
          const double a_z = (face_c_z - node_c_z);
          const double b_x = (face_c_x - half_edge_x);
          const double b_y = (face_c_y - half_edge_y);
          const double b_z = (face_c_z - half_edge_z);
          const double ab_x = (cell_centroids_x[(cells[cc])] - face_c_x);
          const double ab_y = (cell_centroids_y[(cells[cc])] - face_c_y);
          const double ab_z = (cell_centroids_z[(cells[cc])] - face_c_z);

          // Calculate the area vector S using cross product
          double A_x = 0.5 * (a_y * b_z - a_z * b_y);
          double A_y = -0.5 * (a_x * b_z - a_z * b_x);
          double A_z = 0.5 * (a_x * b_y - a_y * b_x);

          const double subcell_volume =
              fabs((ab_x * A_x + ab_y * A_y + ab_z * A_z) / 3.0);

          nodal_soundspeed[(nn)] +=
              sqrt(GAM * (GAM - 1.0) * energy1[(cells[(cc)])]) * subcell_volume;
          nodal_volumes[(nn)] += subcell_volume;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_soundspeed");

  // Calculate the pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

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
        face_c_x += nodes_x1[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y1[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z1[(node_index)] / nnodes_by_face;
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
            0.5 * (nodes_x1[(current_node)] + nodes_x1[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y1[(current_node)] + nodes_y1[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z1[(current_node)] + nodes_z1[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (face_c_x - nodes_x1[(current_node)]);
        const double a_y = (face_c_y - nodes_y1[(current_node)]);
        const double a_z = (face_c_z - nodes_z1[(current_node)]);
        const double b_x = (face_c_x - half_edge_x);
        const double b_y = (face_c_y - half_edge_y);
        const double b_z = (face_c_z - half_edge_z);
        const double ab_x = (cell_centroids_x[(cc)] - face_c_x);
        const double ab_y = (cell_centroids_y[(cc)] - face_c_y);
        const double ab_z = (cell_centroids_z[(cc)] - face_c_z);

        // Calculate the area vector S using cross product
        double A_x = 0.5 * (a_y * b_z - a_z * b_y);
        double A_y = -0.5 * (a_x * b_z - a_z * b_x);
        double A_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER CLOSED
        // FORM SOLUTION?
        int node_off;
        int next_node_off;
        for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
          if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
            node_off = nn3;
          } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == next_node) {
            next_node_off = nn3;
          }
        }

        // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES SO
        // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
        // CALCULATION
        const int flip = (ab_x * A_x + ab_y * A_y + ab_z * A_z > 0.0);
        subcell_force_x[(cell_to_nodes_off + node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_x : A_x);
        subcell_force_y[(cell_to_nodes_off + node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_y : A_y);
        subcell_force_z[(cell_to_nodes_off + node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_z : A_z);
        subcell_force_x[(cell_to_nodes_off + next_node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_x : A_x);
        subcell_force_y[(cell_to_nodes_off + next_node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_y : A_y);
        subcell_force_z[(cell_to_nodes_off + next_node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_z : A_z);
      }
    }
  }
  STOP_PROFILING(&compute_profile, "node_force_from_pressure");

  calc_artificial_viscosity(
      ncells, visc_coeff1, visc_coeff2, cells_offsets, cells_to_nodes, nodes_x1,
      nodes_y1, nodes_z1, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      velocity_x1, velocity_y1, velocity_z1, nodal_soundspeed, nodal_mass,
      nodal_volumes, limiter, subcell_force_x, subcell_force_y, subcell_force_z,
      faces_to_nodes_offsets, faces_to_nodes, cells_to_faces_offsets,
      cells_to_faces);

  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - node_to_cells_off;

    // Consider all faces attached to node
    double node_force_x0 = 0.0;
    double node_force_y0 = 0.0;
    double node_force_z0 = 0.0;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_offsets[(cell_index + 1)] - cell_to_nodes_off;

      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
          break;
        }
      }

      node_force_x0 += subcell_force_x[(cell_to_nodes_off + nn2)];
      node_force_y0 += subcell_force_y[(cell_to_nodes_off + nn2)];
      node_force_z0 += subcell_force_z[(cell_to_nodes_off + nn2)];
    }

    // Calculate the new velocities
    velocity_x1[(nn)] += mesh->dt * node_force_x0 / nodal_mass[(nn)];
    velocity_y1[(nn)] += mesh->dt * node_force_y0 / nodal_mass[(nn)];
    velocity_z1[(nn)] += mesh->dt * node_force_z0 / nodal_mass[(nn)];

    // Calculate the corrected time centered velocities
    velocity_x0[(nn)] = 0.5 * (velocity_x1[(nn)] + velocity_x0[(nn)]);
    velocity_y0[(nn)] = 0.5 * (velocity_y1[(nn)] + velocity_y0[(nn)]);
    velocity_z0[(nn)] = 0.5 * (velocity_z1[(nn)] + velocity_z0[(nn)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_velocity");

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

  set_timestep(ncells, nodes_x0, nodes_y0, nodes_z0, energy1, &mesh->dt,
               cells_to_faces_offsets, cells_to_faces, faces_to_nodes_offsets,
               faces_to_nodes);

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    double cell_force = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      cell_force += (velocity_x0[(node_index)] *
                         subcell_force_x[(cell_to_nodes_off + nn)] +
                     velocity_y0[(node_index)] *
                         subcell_force_y[(cell_to_nodes_off + nn)] +
                     velocity_z0[(node_index)] *
                         subcell_force_z[(cell_to_nodes_off + nn)]);
    }

    energy0[(cc)] -= mesh->dt * cell_force / cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  init_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0, nodes_y0,
                      nodes_z0, cell_centroids_x, cell_centroids_y,
                      cell_centroids_z);

  // Using the new corrected volume, calculate the density
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double cell_volume = 0.0;

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

        // TODO: WE MULTIPLY BY 2 HERE BECAUSE WE ARE ADDING THE VOLUME TO
        // BOTH
        // THE CURRENT AND NEXT NODE, OTHERWISE WE ONLY ACCOUNT FOR HALF OF
        // THE
        // 'HALF' TETRAHEDRONS
        cell_volume +=
            fabs(2.0 * ((half_edge_x - nodes_x0[(current_node)]) * S_x +
                        (half_edge_y - nodes_y0[(current_node)]) * S_y +
                        (half_edge_z - nodes_z0[(current_node)]) * S_z) /
                 3.0);
      }
    }

    // Update the density using the new volume
    density0[(cc)] = cell_mass[(cc)] / cell_volume;
  }
  STOP_PROFILING(&compute_profile, "calc_new_density");
}

// Controls the timestep for the simulation
void set_timestep(const int ncells, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes) {

  // TODO: THIS IS SOO BAD, WE NEED TO CORRECTLY CALCULATE THE CHARACTERISTIC
  // LENGTHS AND DETERMINE THE FULL CONDITION

  // Calculate the timestep based on the computational mesh and CFL
  // condition
  double local_dt = DBL_MAX;
  START_PROFILING(&compute_profile);
#pragma omp parallel for reduction(min : local_dt)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double shortest_edge = DBL_MAX;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // Fetch the nodes attached to our current node on the current face
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn)];

        const int next_node = (nn + 1 < nnodes_by_face)
                                  ? faces_to_nodes[(face_to_nodes_off + nn + 1)]
                                  : faces_to_nodes[(face_to_nodes_off)];
        const double x_component =
            nodes_x[(current_node)] - nodes_x[(next_node)];
        const double y_component =
            nodes_y[(current_node)] - nodes_y[(next_node)];
        const double z_component =
            nodes_z[(current_node)] - nodes_z[(next_node)];

        // Find the shortest edge of this cell
        shortest_edge = min(shortest_edge, sqrt(x_component * x_component +
                                                y_component * y_component +
                                                z_component * z_component));
      }
    }

    const double soundspeed = sqrt(GAM * (GAM - 1.0) * energy[(cc)]);
    local_dt = min(local_dt, shortest_edge / soundspeed);
  }
  STOP_PROFILING(&compute_profile, __func__);

  *dt = CFL * local_dt;
  printf("Timestep %.8fs\n", *dt);
}

// Gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(
    const int ncells, double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_to_nodes, int* cells_offsets,
    double* nodes_x0, double* nodes_y0, double* nodes_z0, double* energy0,
    double* density0, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* cell_mass, double* subcell_volume,
    double* subcell_ie_density, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_integrals_x,
    double* subcell_integrals_y, double* subcell_integrals_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces) {

  // Collect the sub-cell centered velocities
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // Calculate the weighted velocity at the sub-cell center
    double uc_x = 0.0;
    double uc_y = 0.0;
    double uc_z = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = (cells_to_nodes[(cell_to_nodes_off + nn)]);
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;

      int Sn = 0;
      double b_x = 0.0;
      double b_y = 0.0;
      double b_z = 0.0;
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if ((faces_to_cells0[(face_index)] != cc &&
             faces_to_cells1[(face_index)] != cc) ||
            face_index == -1) {
          continue;
        }

        // We have encountered a true face
        Sn += 2;

        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Look at all of the nodes around a face
        int node;
        double f_x = 0.0;
        double f_y = 0.0;
        double f_z = 0.0;
        double face_mass = 0.0;
        for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          const int node_index0 = faces_to_nodes[(face_to_nodes_off + nn2)];
          const int node_l_index =
              (nn2 == 0)
                  ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                  : faces_to_nodes[(face_to_nodes_off + nn2 - 1)];
          const int node_r_index =
              (nn2 == nnodes_by_face - 1)
                  ? faces_to_nodes[(face_to_nodes_off)]
                  : faces_to_nodes[(face_to_nodes_off + nn2 + 1)];

          // Add the face center contributions
          double mass = subcell_mass[(cell_to_nodes_off + nn2)];
          f_x += mass * (2.0 * velocity_x0[(node_index0)] -
                         0.5 * velocity_x0[(node_l_index)] -
                         0.5 * velocity_x0[(node_r_index)]);
          f_y += mass * (2.0 * velocity_y0[(node_index0)] -
                         0.5 * velocity_y0[(node_l_index)] -
                         0.5 * velocity_y0[(node_r_index)]);
          f_z += mass * (2.0 * velocity_z0[(node_index0)] -
                         0.5 * velocity_z0[(node_l_index)] -
                         0.5 * velocity_z0[(node_r_index)]);
          face_mass += mass;
          if (node_index0 == node_index) {
            node = nn2;
          }
        }

        const int node_l_index =
            (node == 0)
                ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                : faces_to_nodes[(face_to_nodes_off + node - 1)];
        const int node_r_index =
            (node == nnodes_by_face - 1)
                ? faces_to_nodes[(face_to_nodes_off)]
                : faces_to_nodes[(face_to_nodes_off + node + 1)];

        // Add contributions for right, left, and face center
        b_x += 0.5 * velocity_x0[(node_l_index)] +
               0.5 * velocity_x0[(node_r_index)] + 2.0 * f_x / face_mass;
        b_y += 0.5 * velocity_y0[(node_l_index)] +
               0.5 * velocity_y0[(node_r_index)] + 2.0 * f_y / face_mass;
        b_z += 0.5 * velocity_z0[(node_l_index)] +
               0.5 * velocity_z0[(node_r_index)] + 2.0 * f_z / face_mass;
      }

      double mass = subcell_mass[(cell_to_nodes_off + nn)];
      uc_x += mass * (2.5 * velocity_x0[(node_index)] - (b_x / Sn)) /
              cell_mass[(cc)];
      uc_y += mass * (2.5 * velocity_y0[(node_index)] - (b_y / Sn)) /
              cell_mass[(cc)];
      uc_z += mass * (2.5 * velocity_z0[(node_index)] - (b_z / Sn)) /
              cell_mass[(cc)];
    }

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = (cells_to_nodes[(cell_to_nodes_off + nn)]);
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;

      int Sn = 0;
      double b_x = 0.0;
      double b_y = 0.0;
      double b_z = 0.0;
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if ((faces_to_cells0[(face_index)] != cc &&
             faces_to_cells1[(face_index)] != cc) ||
            face_index == -1) {
          continue;
        }

        // We have encountered a true face
        Sn += 2;

        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        int node;
        double f_x = 0.0;
        double f_y = 0.0;
        double f_z = 0.0;
        double face_mass = 0.0;
        for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {

          const int node_index0 = faces_to_nodes[(face_to_nodes_off + nn2)];
          const int node_l_index =
              (nn2 == 0)
                  ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                  : faces_to_nodes[(face_to_nodes_off + nn2 - 1)];
          const int node_r_index =
              (nn2 == nnodes_by_face - 1)
                  ? faces_to_nodes[(face_to_nodes_off)]
                  : faces_to_nodes[(face_to_nodes_off + nn2 + 1)];

          // Add the face center contributions
          double mass = subcell_mass[(cell_to_nodes_off + nn2)];
          f_x += mass * (2.0 * velocity_x0[(node_index0)] -
                         0.5 * velocity_x0[(node_l_index)] -
                         0.5 * velocity_x0[(node_r_index)]);
          f_y += mass * (2.0 * velocity_y0[(node_index0)] -
                         0.5 * velocity_y0[(node_l_index)] -
                         0.5 * velocity_y0[(node_r_index)]);
          f_z += mass * (2.0 * velocity_z0[(node_index0)] -
                         0.5 * velocity_z0[(node_l_index)] -
                         0.5 * velocity_z0[(node_r_index)]);
          face_mass += mass;

          if (node_index0 == node_index) {
            node = nn2;
          }
        }

        const int node_l_index =
            (node == 0)
                ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                : faces_to_nodes[(face_to_nodes_off + node - 1)];
        const int node_r_index =
            (node == nnodes_by_face - 1)
                ? faces_to_nodes[(face_to_nodes_off)]
                : faces_to_nodes[(face_to_nodes_off + node + 1)];

        // Add right and left node contributions
        b_x += 0.5 * velocity_x0[(node_l_index)] +
               0.5 * velocity_x0[(node_r_index)] + 2.0 * f_x / face_mass;
        b_y += 0.5 * velocity_y0[(node_l_index)] +
               0.5 * velocity_y0[(node_r_index)] + 2.0 * f_y / face_mass;
        b_z += 0.5 * velocity_z0[(node_l_index)] +
               0.5 * velocity_z0[(node_r_index)] + 2.0 * f_z / face_mass;
      }

      // Calculate the final sub-cell velocities
      subcell_velocity_x[(cell_to_nodes_off + nn)] =
          0.25 * (1.5 * velocity_x0[(node_index)] + uc_x + b_x / Sn);
      subcell_velocity_y[(cell_to_nodes_off + nn)] =
          0.25 * (1.5 * velocity_y0[(node_index)] + uc_y + b_y / Sn);
      subcell_velocity_z[(cell_to_nodes_off + nn)] =
          0.25 * (1.5 * velocity_z0[(node_index)] + uc_z + b_z / Sn);
    }
  }

  /*
  *      GATHERING STAGE OF THE REMAP
  */

  // Calculate the sub-cell internal energies
  for (int cc = 0; cc < ncells; ++cc) {
    // Calculating the volume integrals necessary for the least squares
    // regression
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    // The coefficients of the 3x3 gradient coefficient matrix
    vec_t coeff[3] = {{0.0, 0.0, 0.0}};
    vec_t rhs = {0.0, 0.0, 0.0};

    // Determine the weighted volume integrals for neighbouring cells
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int neighbour_index = (faces_to_cells0[(face_index)] == cc)
                                      ? faces_to_cells1[(face_index)]
                                      : faces_to_cells0[(face_index)];
      // Check if boundary face
      if (neighbour_index == -1) {
        continue;
      }

      const int neighbour_to_faces_off =
          cells_to_faces_offsets[(neighbour_index)];
      const int nfaces_by_neighbour =
          cells_to_faces_offsets[(neighbour_index + 1)] -
          neighbour_to_faces_off;

      // Calculate the weighted volume integral coefficients
      double vol = 0.0;
      vec_t integrals;
      vec_t neighbour_centroid = {0.0, 0.0, 0.0};
      neighbour_centroid.x = cell_centroids_x[(neighbour_index)];
      neighbour_centroid.y = cell_centroids_y[(neighbour_index)];
      neighbour_centroid.z = cell_centroids_z[(neighbour_index)];
      calc_weighted_volume_integrals(
          neighbour_to_faces_off, nfaces_by_neighbour, cells_to_faces,
          faces_to_nodes, faces_to_nodes_offsets, nodes_x0, nodes_y0, nodes_z0,
          &neighbour_centroid, &integrals, &vol);

      // Complete the integral coefficient as a distance
      integrals.x -= cell_centroid.x * vol;
      integrals.y -= cell_centroid.y * vol;
      integrals.z -= cell_centroid.z * vol;

      // Store the neighbouring cell's contribution to the coefficients
      coeff[0].x += (2.0 * integrals.x * integrals.x) / (vol * vol);
      coeff[0].y += (2.0 * integrals.x * integrals.y) / (vol * vol);
      coeff[0].z += (2.0 * integrals.x * integrals.z) / (vol * vol);

      coeff[1].x += (2.0 * integrals.y * integrals.x) / (vol * vol);
      coeff[1].y += (2.0 * integrals.y * integrals.y) / (vol * vol);
      coeff[1].z += (2.0 * integrals.y * integrals.z) / (vol * vol);

      coeff[2].x += (2.0 * integrals.z * integrals.x) / (vol * vol);
      coeff[2].y += (2.0 * integrals.z * integrals.y) / (vol * vol);
      coeff[2].z += (2.0 * integrals.z * integrals.z) / (vol * vol);

      // Prepare the RHS, which includes energy differential
      const double de = (energy0[(neighbour_index)] - energy0[(cc)]);
      rhs.x += (2.0 * integrals.x * de / vol);
      rhs.y += (2.0 * integrals.y * de / vol);
      rhs.z += (2.0 * integrals.z * de / vol);
    }

    // Determine the inverse of the coefficient matrix
    vec_t inv[3];
    calc_3x3_inverse(&coeff, &inv);

    // Solve for the energy gradient
    vec_t grad_energy;
    grad_energy.x = inv[0].x * rhs.x + inv[0].y * rhs.y + inv[0].z * rhs.z;
    grad_energy.y = inv[1].x * rhs.x + inv[1].y * rhs.y + inv[1].z * rhs.z;
    grad_energy.z = inv[2].x * rhs.x + inv[2].y * rhs.y + inv[2].z * rhs.z;

// Describe the connectivity for a simple tetrahedron, the sub-cell shape
#define NSUBCELL_FACES 4
#define NSUBCELL_NODES 4
#define NSUBCELL_NODES_PER_FACE 3
    const int subcell_faces_to_nodes_offsets[NSUBCELL_FACES + 1] = {0, 3, 6, 9,
                                                                    12};
    const int subcell_faces_to_nodes[NSUBCELL_FACES * NSUBCELL_NODES_PER_FACE] =
        {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
    const int subcell_to_faces[NSUBCELL_FACES] = {0, 1, 2, 3};
    double subcell_nodes_x[NSUBCELL_NODES] = {0.0};
    double subcell_nodes_y[NSUBCELL_NODES] = {0.0};
    double subcell_nodes_z[NSUBCELL_NODES] = {0.0};

    // The centroid remains a component of all sub-cells
    subcell_nodes_x[3] = cell_centroid.x;
    subcell_nodes_y[3] = cell_centroid.y;
    subcell_nodes_z[3] = cell_centroid.z;

    // Determine the weighted volume integrals for neighbouring cells
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // TODO: SHOULD WE PRECOMPUTE THE FACE CENTROID???
      // The face centroid is the same for all nodes on the face
      subcell_nodes_x[2] = 0.0;
      subcell_nodes_y[2] = 0.0;
      subcell_nodes_z[2] = 0.0;
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        subcell_nodes_x[2] += nodes_x0[(node_index)] / nnodes_by_face;
        subcell_nodes_y[2] += nodes_y0[(node_index)] / nnodes_by_face;
        subcell_nodes_z[2] += nodes_z0[(node_index)] / nnodes_by_face;
      }

      // Each face/node pair has two sub-cells
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];

        // TODO: HAVE MOVED THIS, CHECK IT WORKS....
        // Find the node offset in the cell
        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
          if (cells_to_nodes[(cell_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

// The left and right nodes on the face for this anchor node
#define NNEIGHBOUR_NODES 2
        int nodes_a[NNEIGHBOUR_NODES];
        nodes_a[0] =
            (nn == 0) ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                      : faces_to_nodes[(face_to_nodes_off + nn - 1)];
        nodes_a[1] = (nn == nnodes_by_face - 1)
                         ? faces_to_nodes[(face_to_nodes_off)]
                         : faces_to_nodes[(face_to_nodes_off + nn + 1)];

        // Loop over both of the neighbour nodes
        for (int ss = 0; ss < NNEIGHBOUR_NODES; ++ss) {
          // Store the right and left nodes
          subcell_nodes_x[1] =
              0.5 * (nodes_x0[nodes_a[ss]] + nodes_x0[(node_index)]);
          subcell_nodes_y[1] =
              0.5 * (nodes_y0[nodes_a[ss]] + nodes_y0[(node_index)]);
          subcell_nodes_z[1] =
              0.5 * (nodes_z0[nodes_a[ss]] + nodes_z0[(node_index)]);

          // Store the anchor node
          subcell_nodes_x[0] = nodes_x0[(node_index)];
          subcell_nodes_y[0] = nodes_y0[(node_index)];
          subcell_nodes_z[0] = nodes_z0[(node_index)];

          // Determine the sub-cell centroid
          vec_t subcell_centroid = {0.0, 0.0, 0.0};
          for (int ii = 0; ii < NSUBCELL_NODES; ++ii) {
            subcell_centroid.x += subcell_nodes_x[ii] / NSUBCELL_NODES;
            subcell_centroid.y += subcell_nodes_y[ii] / NSUBCELL_NODES;
            subcell_centroid.z += subcell_nodes_z[ii] / NSUBCELL_NODES;
          }

          // Calculate the weighted volume integral coefficients
          double vol = 0.0;
          vec_t integrals = {0.0, 0.0, 0.0};
          calc_weighted_volume_integrals(
              0, NSUBCELL_FACES, subcell_to_faces, subcell_faces_to_nodes,
              subcell_faces_to_nodes_offsets, subcell_nodes_x, subcell_nodes_y,
              subcell_nodes_z, &subcell_centroid, &integrals, &vol);

          // TODO: THIS MIGHT BE A STUPID WAY TO DO THIS.
          // WE ARE LOOKING AT ALL OF THE SUBCELL TETRAHEDRONS, WHEN WE COULD BE
          // LOOKING AT A SINGLE CORNER SUBCELL PER NODE

          // Store the weighted integrals
          subcell_integrals_x[(cell_to_nodes_off + nn2)] += integrals.x;
          subcell_integrals_y[(cell_to_nodes_off + nn2)] += integrals.y;
          subcell_integrals_z[(cell_to_nodes_off + nn2)] += integrals.z;
          subcell_volume[(cell_to_nodes_off + nn2)] += vol;

          // Determine subcell energy from linear function at cell
          subcell_ie_density[(cell_to_nodes_off + nn2)] +=
              vol * (density0[(cc)] * energy0[(cc)] -
                     (grad_energy.x * cell_centroid.x +
                      grad_energy.y * cell_centroid.y +
                      grad_energy.z * cell_centroid.z)) +
              grad_energy.x * integrals.x + grad_energy.y * integrals.y +
              grad_energy.z * integrals.z;
        }
      }
    }
  }
}

// Calculates the artificial viscous forces for momentum acceleration
void calc_artificial_viscosity(
    const int ncells, const double visc_coeff1, const double visc_coeff2,
    const int* cells_offsets, const int* cells_to_nodes, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* velocity_x,
    const double* velocity_y, const double* velocity_z,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter, double* subcell_force_x,
    double* subcell_force_y, double* subcell_force_z,
    int* faces_to_nodes_offsets, int* faces_to_nodes,
    int* cells_to_faces_offsets, int* cells_to_faces) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

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
        const double ab_x = (nodes_x[(current_node)] - half_edge_x);
        const double ab_y = (nodes_y[(current_node)] - half_edge_y);
        const double ab_z = (nodes_z[(current_node)] - half_edge_z);

        // Calculate the area vector S using cross product
        double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        double S_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES SO
        // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
        // CALCULATION
        if ((ab_x * S_x + ab_y * S_y + ab_z * S_z) > 0.0) {
          S_x *= -1.0;
          S_y *= -1.0;
          S_z *= -1.0;
        }

        // Calculate the velocity gradients
        const double dvel_x =
            velocity_x[(next_node)] - velocity_x[(current_node)];
        const double dvel_y =
            velocity_y[(next_node)] - velocity_y[(current_node)];
        const double dvel_z =
            velocity_z[(next_node)] - velocity_z[(current_node)];
        const double dvel_mag =
            sqrt(dvel_x * dvel_x + dvel_y * dvel_y + dvel_z * dvel_z);

        // Calculate the unit vectors of the velocity gradients
        const double dvel_unit_x = (dvel_mag != 0.0) ? dvel_x / dvel_mag : 0.0;
        const double dvel_unit_y = (dvel_mag != 0.0) ? dvel_y / dvel_mag : 0.0;
        const double dvel_unit_z = (dvel_mag != 0.0) ? dvel_z / dvel_mag : 0.0;

        // Get the edge-centered density
        double nodal_density0 =
            nodal_mass[(current_node)] / nodal_volumes[(current_node)];
        double nodal_density1 =
            nodal_mass[(next_node)] / nodal_volumes[(next_node)];
        const double density_edge = (2.0 * nodal_density0 * nodal_density1) /
                                    (nodal_density0 + nodal_density1);

        // Calculate the artificial viscous force term for the edge
        double expansion_term = (dvel_x * S_x + dvel_y * S_y + dvel_z * S_z);

        // If the cell is compressing, calculate the edge forces and add
        // their
        // contributions to the node forces
        if (expansion_term <= 0.0) {
          // Calculate the minimum soundspeed
          const double cs = min(nodal_soundspeed[(current_node)],
                                nodal_soundspeed[(next_node)]);
          const double t = 0.25 * (GAM + 1.0);
          const double edge_visc_force_x =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_x) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_x * dvel_x +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_x;
          const double edge_visc_force_y =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_y) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_y * dvel_y +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_y;
          const double edge_visc_force_z =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_z) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_z * dvel_z +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_z;

          // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER
          // CLOSED
          // FORM SOLUTION?
          int node_off;
          int next_node_off;
          for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
            if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
              node_off = nn3;
            } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == next_node) {
              next_node_off = nn3;
            }
          }

          // Add the contributions of the edge based artifical viscous terms
          // to the main force terms
          subcell_force_x[(cell_to_nodes_off + node_off)] -= edge_visc_force_x;
          subcell_force_y[(cell_to_nodes_off + node_off)] -= edge_visc_force_y;
          subcell_force_z[(cell_to_nodes_off + node_off)] -= edge_visc_force_z;
          subcell_force_x[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_x;
          subcell_force_y[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_y;
          subcell_force_z[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_z;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Calculates the face integral for the provided face, projected onto
// the two-dimensional basis
void calc_projections(const int nnodes_by_face, const int face_to_nodes_off,
                      const int* faces_to_nodes, const double* alpha,
                      const double* beta, pi_t* pi) {

  double pione = 0.0;
  double pialpha = 0.0;
  double pialpha2 = 0.0;
  double pibeta = 0.0;
  double pibeta2 = 0.0;
  double pialphabeta = 0.0;

  // Calculate the coefficients for the projected face integral
  for (int nn = 0; nn < nnodes_by_face; ++nn) {
    const int n0 = faces_to_nodes[(face_to_nodes_off + nn)];
    const int n1 = (nn == nnodes_by_face - 1)
                       ? faces_to_nodes[(face_to_nodes_off)]
                       : faces_to_nodes[(face_to_nodes_off + nn + 1)];

    // Calculate all of the coefficients
    const double a0 = alpha[(n0)];
    const double a1 = alpha[(n1)];
    const double b0 = beta[(n0)];
    const double b1 = beta[(n1)];
    const double dalpha = a1 - a0;
    const double dbeta = b1 - b0;
    const double Calpha = a1 * (a1 + a0) + a0 * a0;
    const double Cbeta = b1 * b1 + b1 * b0 + b0 * b0;
    const double Calphabeta = 3.0 * a1 * a1 + 2.0 * a1 * a0 + a0 * a0;
    const double Kalphabeta = a1 * a1 + 2.0 * a1 * a0 + 3.0 * a0 * a0;

    // Accumulate the projection integrals
    pione += dbeta * (a1 + a0) / 2.0;
    pialpha += dbeta * (Calpha) / 6.0;
    pialpha2 += dbeta * (a1 * Calpha + a0 * a0 * a0) / 12.0;
    pibeta -= dalpha * (Cbeta) / 6.0;
    pibeta2 -= dalpha * (b1 * Cbeta + b0 * b0 * b0) / 12.0;
    pialphabeta += dbeta * (b1 * Calphabeta + b0 * Kalphabeta) / 24.0;
  }

  // Store the final coefficients, flipping all results if we went through
  // in a clockwise order and got a negative area
  const double flip = (pione > 0.0 ? 1.0 : -1.0);
  pi->one += flip * pione;
  pi->alpha += flip * pialpha;
  pi->alpha2 += flip * pialpha2;
  pi->beta += flip * pibeta;
  pi->beta2 += flip * pibeta2;
  pi->alpha_beta += flip * pialphabeta;
}

// Resolves the volume integrals in alpha-beta-gamma basis
void calc_face_integrals(const int nnodes_by_face, const int face_to_nodes_off,
                         const int orientation, const int n0,
                         const int* faces_to_nodes, const double* nodes_alpha,
                         const double* nodes_beta, const double* nodes_gamma,
                         vec_t normal, vec_t* T, double* vol) {

  pi_t pi = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  calc_projections(nnodes_by_face, face_to_nodes_off, faces_to_nodes,
                   nodes_alpha, nodes_beta, &pi);

  // The projection of the normal vector onto a point on the face
  double omega = -(normal.x * nodes_alpha[(n0)] + normal.y * nodes_beta[(n0)] +
                   normal.z * nodes_gamma[(n0)]);

  // Finalise the weighted face integrals
  const double Falpha = pi.alpha / fabs(normal.z);
  const double Fbeta = pi.beta / fabs(normal.z);
  const double Fgamma =
      -(normal.x * pi.alpha + normal.y * pi.beta + omega * pi.one) /
      (fabs(normal.z) * normal.z);

  const double Falpha2 = pi.alpha2 / fabs(normal.z);
  const double Fbeta2 = pi.beta2 / fabs(normal.z);
  const double Fgamma2 =
      (normal.x * normal.x * pi.alpha2 +
       2.0 * normal.x * normal.y * pi.alpha_beta +
       normal.y * normal.y * pi.beta2 + 2.0 * normal.x * omega * pi.alpha +
       2.0 * normal.y * omega * pi.beta + omega * omega * pi.one) /
      (fabs(normal.z) * normal.z * normal.z);

  // TODO: STUPID HACK UNTIL I FIND THE CULPRIT!
  // x-y-z and the volumes are in the wrong order..

  // Accumulate the weighted volume integrals
  if (orientation == XYZ) {
    T->y += 0.5 * normal.x * Falpha2;
    T->x += 0.5 * normal.y * Fbeta2;
    T->z += 0.5 * normal.z * Fgamma2;
    *vol += normal.y * Fbeta;
  } else if (orientation == YZX) {
    T->y += 0.5 * normal.y * Fbeta2;
    T->x += 0.5 * normal.z * Fgamma2;
    T->z += 0.5 * normal.x * Falpha2;
    *vol += normal.x * Falpha;
  } else if (orientation == ZXY) {
    T->y += 0.5 * normal.z * Fgamma2;
    T->x += 0.5 * normal.x * Falpha2;
    T->z += 0.5 * normal.y * Fbeta2;
    *vol += normal.z * Fgamma;
  }
}

// Checks if the normal vector is pointing inward or outward
// n0 is just a point on the plane
int check_normal_orientation(const int n0, const double* nodes_x,
                             const double* nodes_y, const double* nodes_z,
                             const vec_t* cell_centroid, vec_t* normal) {

  // Calculate a vector from face to cell centroid
  vec_t ab;
  ab.x = (cell_centroid->x - nodes_x[(n0)]);
  ab.y = (cell_centroid->y - nodes_y[(n0)]);
  ab.z = (cell_centroid->z - nodes_z[(n0)]);

  return (ab.x * normal->x + ab.y * normal->y + ab.z * normal->z > 0.0);
}

// Calculates the surface normal of a vector pointing outwards
void calc_surface_normal(const int n0, const int n1, const int n2,
                         const double* nodes_x, const double* nodes_y,
                         const double* nodes_z, const vec_t* cell_centroid,
                         vec_t* normal) {

  // Calculate the unit normal vector
  calc_unit_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, normal);

  // Determine the orientation of the normal
  const int flip = check_normal_orientation(n0, nodes_x, nodes_y, nodes_z,
                                            cell_centroid, normal);

  // Flip the vector if necessary
  normal->x *= (flip ? -1.0 : 1.0);
  normal->y *= (flip ? -1.0 : 1.0);
  normal->z *= (flip ? -1.0 : 1.0);
}

// Calculate the normal vector from the provided nodes
void calc_unit_normal(const int n0, const int n1, const int n2,
                      const double* nodes_x, const double* nodes_y,
                      const double* nodes_z, vec_t* normal) {

  // Calculate the normal
  calc_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, normal);

  // Normalise the normal
  normalise(normal);
}

// Normalise a vector
void normalise(vec_t* a) {
  const double a_inv_mag = 1.0 / sqrt(a->x * a->x + a->y * a->y + a->z * a->z);
  a->x *= a_inv_mag;
  a->y *= a_inv_mag;
  a->z *= a_inv_mag;
}

// Calculate the normal for a plane
void calc_normal(const int n0, const int n1, const int n2,
                 const double* nodes_x, const double* nodes_y,
                 const double* nodes_z, vec_t* normal) {
  // Get two vectors on the face plane
  vec_t dn0 = {0.0, 0.0, 0.0};
  vec_t dn1 = {0.0, 0.0, 0.0};
  dn0.x = nodes_x[(n2)] - nodes_x[(n1)];
  dn0.y = nodes_y[(n2)] - nodes_y[(n1)];
  dn0.z = nodes_z[(n2)] - nodes_z[(n1)];
  dn1.x = nodes_x[(n1)] - nodes_x[(n0)];
  dn1.y = nodes_y[(n1)] - nodes_y[(n0)];
  dn1.z = nodes_z[(n1)] - nodes_z[(n0)];

  // Cross product to get the normal
  normal->x = (dn0.y * dn1.z - dn0.z * dn1.y);
  normal->y = (dn0.z * dn1.x - dn0.x * dn1.z);
  normal->z = (dn0.x * dn1.y - dn0.y * dn1.x);
}

// Calculates the weighted volume integrals for a provided cell along x-y-z
void calc_weighted_volume_integrals(
    const int cell_to_faces_off, const int nfaces_by_cell,
    const int* cells_to_faces, const int* faces_to_nodes,
    const int* faces_to_nodes_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const vec_t* cell_centroid,
    vec_t* T, double* vol) {

  // Zero as we are reducing into this container
  T->x = 0.0;
  T->y = 0.0;
  T->z = 0.0;
  *vol = 0.0;

  // The weighted volume integrals are calculated over the polyhedral faces
  for (int ff = 0; ff < nfaces_by_cell; ++ff) {
    const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    // Choosing three nodes for calculating the unit normal
    // We can obviously assume there are at least three nodes
    const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
    const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
    const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];

    // Determine the outward facing unit normal vector
    vec_t normal = {0.0, 0.0, 0.0};
    calc_surface_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, cell_centroid,
                        &normal);

    // Select the orientation based on the face area
    int orientation;
    if (fabs(normal.x) > fabs(normal.y)) {
      orientation = (fabs(normal.x) > fabs(normal.z)) ? YZX : XYZ;
    } else {
      orientation = (fabs(normal.z) > fabs(normal.y)) ? XYZ : ZXY;
    }

    // The orientation determines which order we pass the nodes by axes
    // We calculate the individual face integrals and the unit normal to the
    // face in the alpha-beta-gamma basis
    // The weighted integrals essentially provide the center of mass
    // coordinates
    // for the polyhedra
    if (orientation == XYZ) {
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, orientation, n0,
                          faces_to_nodes, nodes_x, nodes_y, nodes_z, normal, T,
                          vol);
    } else if (orientation == YZX) {
      dswap(normal.x, normal.y);
      dswap(normal.y, normal.z);
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, orientation, n0,
                          faces_to_nodes, nodes_y, nodes_z, nodes_x, normal, T,
                          vol);
    } else if (orientation == ZXY) {
      dswap(normal.x, normal.y);
      dswap(normal.x, normal.z);
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, orientation, n0,
                          faces_to_nodes, nodes_z, nodes_x, nodes_y, normal, T,
                          vol);
    }
  }
}

// Calculates the inverse of a 3x3 matrix, out-of-place
void calc_3x3_inverse(vec_t (*a)[3], vec_t (*inv)[3]) {
  // Calculate the determinant of the 3x3
  const double det =
      (*a)[0].x * ((*a)[1].y * (*a)[2].z - (*a)[1].z * (*a)[2].y) -
      (*a)[0].y * ((*a)[1].x * (*a)[2].z - (*a)[1].z * (*a)[2].x) +
      (*a)[0].z * ((*a)[1].x * (*a)[2].y - (*a)[1].y * (*a)[2].x);

  // Check if the matrix is singular
  if (det == 0.0) {
    TERMINATE("singular coefficient matrix");
  } else {
    // Perform the simple and fast 3x3 matrix inverstion
    (*inv)[0].x = ((*a)[1].y * (*a)[2].z - (*a)[1].z * (*a)[2].y) / det;
    (*inv)[0].y = ((*a)[0].z * (*a)[2].y - (*a)[0].y * (*a)[2].z) / det;
    (*inv)[0].z = ((*a)[0].y * (*a)[1].z - (*a)[0].z * (*a)[1].y) / det;

    (*inv)[1].x = ((*a)[1].z * (*a)[2].x - (*a)[1].x * (*a)[2].z) / det;
    (*inv)[1].y = ((*a)[0].x * (*a)[2].z - (*a)[0].z * (*a)[2].x) / det;
    (*inv)[1].z = ((*a)[0].z * (*a)[1].x - (*a)[0].x * (*a)[1].z) / det;

    (*inv)[2].x = ((*a)[1].x * (*a)[2].y - (*a)[1].y * (*a)[2].x) / det;
    (*inv)[2].y = ((*a)[0].x * (*a)[2].x - (*a)[0].x * (*a)[2].y) / det;
    (*inv)[2].z = ((*a)[0].x * (*a)[1].y - (*a)[0].y * (*a)[1].x) / det;
  }
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
