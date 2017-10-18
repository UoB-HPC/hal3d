#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <float.h>
#include <stdio.h>

// Perform the scatter step of the ALE remapping algorithm
void scatter_phase(const int ncells, const int nnodes, vec_t* initial_momentum,
                   const double* rezoned_nodes_x, const double* rezoned_nodes_y,
                   const double* rezoned_nodes_z, double* cell_volume,
                   double* energy, double* density, double* velocity_x,
                   double* velocity_y, double* velocity_z, double* cell_mass,
                   double* nodal_mass, double* subcell_ie_mass,
                   double* subcell_mass, double* subcell_ie_mass_flux,
                   double* subcell_mass_flux, double* subcell_momentum_x,
                   double* subcell_momentum_y, double* subcell_momentum_z,
                   double* subcell_momentum_flux_x,
                   double* subcell_momentum_flux_y,
                   double* subcell_momentum_flux_z, int* faces_to_nodes,
                   int* faces_to_nodes_offsets, int* cells_to_faces_offsets,
                   int* cells_to_faces, int* nodes_to_cells_offsets,
                   int* nodes_to_cells, int* cells_to_nodes_offsets,
                   int* cells_to_nodes, double* total_mass, double* total_ie) {

  // Scatter the subcell energy and mass quantities back to the cell centers
  scatter_energy_and_mass(
      ncells, rezoned_nodes_x, rezoned_nodes_y, rezoned_nodes_z, cell_volume,
      energy, density, cell_mass, subcell_ie_mass, subcell_mass,
      subcell_ie_mass_flux, subcell_mass_flux, faces_to_nodes,
      faces_to_nodes_offsets, cells_to_faces_offsets, cells_to_faces,
      cells_to_nodes_offsets, cells_to_nodes, total_mass, total_ie);

  // Scatter the subcell momentum to the node centered velocities
  scatter_momentum(nnodes, initial_momentum, nodes_to_cells_offsets,
                   nodes_to_cells, cells_to_nodes_offsets, cells_to_nodes,
                   velocity_x, velocity_y, velocity_z, nodal_mass, subcell_mass,
                   subcell_momentum_x, subcell_momentum_y, subcell_momentum_z,
                   subcell_momentum_flux_x, subcell_momentum_flux_y,
                   subcell_momentum_flux_z);
}

// Scatter the subcell energy and mass quantities back to the cell centers
void scatter_energy_and_mass(
    const int ncells, const double* rezoned_nodes_x,
    const double* rezoned_nodes_y, const double* rezoned_nodes_z,
    double* cell_volume, double* energy, double* density, double* cell_mass,
    double* subcell_ie_mass, double* subcell_mass, double* subcell_ie_mass_flux,
    double* subcell_mass_flux, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_offsets,
    int* cells_to_nodes, double* total_mass, double* total_ie) {

  // Scatter energy and density, and print the conservation of mass
  double rz_total_ie = 0.0;
  double rz_total_mass = 0.0;
  double initial_total_mass = 0.0;
  double initial_total_ie = 0.0;
#if 0
#pragma omp parallel for reduction(+ : rz_total_mass, rz_total_ie,             \
                                   initial_total_mass, initial_total_ie)
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    initial_total_mass += cell_mass[(cc)];
    initial_total_ie += cell_mass[(cc)] * energy[(cc)];

    cell_mass[(cc)] = 0.0;

    // Update the volume of the cell to the new rezoned mesh
    vec_t rz_cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, rezoned_nodes_x, rezoned_nodes_y,
                  rezoned_nodes_z, cells_to_nodes, cell_to_nodes_off,
                  &rz_cell_centroid);
    calc_volume(cell_to_faces_off, nfaces_by_cell, cells_to_faces,
                faces_to_nodes, faces_to_nodes_offsets, rezoned_nodes_x,
                rezoned_nodes_y, rezoned_nodes_z, &rz_cell_centroid,
                &cell_volume[(cc)]);

    double total_ie_mass = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int subcell_index = cell_to_nodes_off + nn;

      // Scatter the subcell mass data back to the cell
      subcell_mass[(subcell_index)] -= subcell_mass_flux[(subcell_index)];
      subcell_ie_mass[(subcell_index)] -= subcell_ie_mass_flux[(subcell_index)];
      cell_mass[(cc)] += subcell_mass[(subcell_index)];
      total_ie_mass += subcell_ie_mass[(subcell_index)];

      subcell_mass_flux[(subcell_index)] = 0.0;
      subcell_ie_mass_flux[(subcell_index)] = 0.0;
    }

    // Scatter the energy and density
    density[(cc)] = cell_mass[(cc)] / cell_volume[(cc)];
    energy[(cc)] = total_ie_mass / cell_mass[(cc)];

    // Calculate the conservation data
    rz_total_mass += cell_mass[(cc)];
    rz_total_ie += total_ie_mass;
  }

  *total_mass = initial_total_mass;
  *total_ie = initial_total_ie;

  printf("Rezoned Total Mass %.12f\n", rz_total_mass);
  printf("Initial Total Mass %.12f\n", *total_mass);
  printf("Difference         %.12f\n\n", rz_total_mass - *total_mass);

  printf("Rezoned Total Internal Energy %.12f\n", rz_total_ie);
  printf("Initial Total Energy          %.12f\n", *total_ie);
  printf("Difference                    %.12f\n\n", rz_total_ie - *total_ie);
}

// Scatter the subcell momentum to the node centered velocities
void scatter_momentum(const int nnodes, vec_t* initial_momentum,
                      int* nodes_to_cells_offsets, int* nodes_to_cells,
                      int* cells_to_nodes_offsets, int* cells_to_nodes,
                      double* velocity_x, double* velocity_y,
                      double* velocity_z, double* nodal_mass,
                      double* subcell_mass, double* subcell_momentum_x,
                      double* subcell_momentum_y, double* subcell_momentum_z,
                      double* subcell_momentum_flux_x,
                      double* subcell_momentum_flux_y,
                      double* subcell_momentum_flux_z) {

  double total_node_momentum_x = 0.0;
  double total_node_momentum_y = 0.0;
  double total_node_momentum_z = 0.0;

#if 0
#pragma omp parallel for reduction(                                            \
    + : total_node_momentum_x, total_node_momentum_y, total_node_momentum_z)
#endif // if 0
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
    const int ncells_by_node =
        nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

    velocity_x[(nn)] = 0.0;
    velocity_y[(nn)] = 0.0;
    velocity_z[(nn)] = 0.0;
    nodal_mass[(nn)] = 0.0;

    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_to_nodes_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_to_nodes_offsets[(cell_index + 1)] - cell_to_nodes_off;

      // Determine the position of the node in the cell
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
          break;
        }
      }

      const int subcell_index = cell_to_nodes_off + nn2;

      double new_subcell_momentum_x = subcell_momentum_x[(subcell_index)] -
                                      subcell_momentum_flux_x[(subcell_index)];
      double new_subcell_momentum_y = subcell_momentum_y[(subcell_index)] -
                                      subcell_momentum_flux_y[(subcell_index)];
      double new_subcell_momentum_z = subcell_momentum_z[(subcell_index)] -
                                      subcell_momentum_flux_z[(subcell_index)];

      // Clear the array that we will be reducing into during next timestep
      subcell_momentum_flux_x[(subcell_index)] = 0.0;
      subcell_momentum_flux_y[(subcell_index)] = 0.0;
      subcell_momentum_flux_z[(subcell_index)] = 0.0;

      velocity_x[(nn)] += new_subcell_momentum_x;
      velocity_y[(nn)] += new_subcell_momentum_y;
      velocity_z[(nn)] += new_subcell_momentum_z;

      total_node_momentum_x += new_subcell_momentum_x;
      total_node_momentum_y += new_subcell_momentum_y;
      total_node_momentum_z += new_subcell_momentum_z;

      nodal_mass[(nn)] += subcell_mass[(subcell_index)];
    }

    velocity_x[(nn)] /= nodal_mass[(nn)];
    velocity_y[(nn)] /= nodal_mass[(nn)];
    velocity_z[(nn)] /= nodal_mass[(nn)];
  }

  printf("Rezoned total momentum %.12f %.12f %.12f\n", total_node_momentum_x,
         total_node_momentum_y, total_node_momentum_z);
  printf("Initial total momentum %.12f %.12f %.12f\n", initial_momentum->x,
         initial_momentum->y, initial_momentum->z);
  printf("Difference             %.12f %.12f %.12f\n\n",
         initial_momentum->x - total_node_momentum_x,
         initial_momentum->y - total_node_momentum_y,
         initial_momentum->z - total_node_momentum_z);
}
