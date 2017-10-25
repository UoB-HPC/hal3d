#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <float.h>
#include <stdio.h>

// Scatter the subcell energy and mass quantities back to the cell centers
void scatter_energy_and_mass(
    const int ncells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, double* cell_volume, double* energy, double* density,
    double* kinetic_energy, double* velocity_x, double* velocity_y,
    double* velocity_z, double* cell_mass, double* subcell_mass,
    double* subcell_ie_mass, double* subcell_ke_mass, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* cells_to_faces_offsets,
    int* cells_to_faces, int* cells_offsets, int* cells_to_nodes,
    int* cells_to_nodes_offsets, double initial_mass, double initial_ie_mass,
    double initial_ke_mass);

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
                      double* subcell_momentum_flux_z);

// Perform the scatter step of the ALE remapping algorithm
void scatter_phase(UnstructuredMesh* umesh, HaleData* hale_data,
                   vec_t* initial_momentum, double initial_mass,
                   double initial_ie_mass, double initial_ke_mass) {

  // Scatter the subcell momentum to the node centered velocities
  scatter_momentum(
      umesh->nnodes, initial_momentum, umesh->nodes_offsets,
      umesh->nodes_to_cells, umesh->cells_offsets, umesh->cells_to_nodes,
      hale_data->velocity_x0, hale_data->velocity_y0, hale_data->velocity_z0,
      hale_data->nodal_mass, hale_data->subcell_mass,
      hale_data->subcell_momentum_x, hale_data->subcell_momentum_y,
      hale_data->subcell_momentum_z, hale_data->subcell_momentum_flux_x,
      hale_data->subcell_momentum_flux_y, hale_data->subcell_momentum_flux_z);

  // Scatter the subcell energy and mass quantities back to the cell centers
  scatter_energy_and_mass(
      umesh->ncells, umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
      hale_data->cell_volume, hale_data->energy0, hale_data->density0,
      hale_data->ke_mass, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0, hale_data->cell_mass, hale_data->subcell_mass,
      hale_data->subcell_ie_mass, hale_data->subcell_ke_mass,
      umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
      umesh->cells_to_faces_offsets, umesh->cells_to_faces,
      umesh->cells_offsets, umesh->cells_to_nodes, umesh->cells_offsets,
      initial_mass, initial_ie_mass, initial_ke_mass);
}

// Scatter the subcell energy and mass quantities back to the cell centers
void scatter_energy_and_mass(
    const int ncells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, double* cell_volume, double* energy, double* density,
    double* ke_mass, double* velocity_x, double* velocity_y, double* velocity_z,
    double* cell_mass, double* subcell_mass, double* subcell_ie_mass,
    double* subcell_ke_mass, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_offsets,
    int* cells_to_nodes, int* cells_to_nodes_offsets, double initial_mass,
    double initial_ie_mass, double initial_ke_mass) {

  // Scatter energy and density, and print the conservation of mass
  double rz_total_mass = 0.0;
  double rz_total_e_mass = 0.0;
#pragma omp parallel for reduction(+ : rz_total_mass, rz_total_e_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double new_ke_mass = 0.0;
    double total_mass = 0.0;
    double total_ie_mass = 0.0;
    double total_ke_mass = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      total_mass += subcell_mass[(subcell_index)];
      total_ie_mass += subcell_ie_mass[(subcell_index)];
      total_ke_mass += subcell_ke_mass[(subcell_index)];
      new_ke_mass += subcell_mass[(subcell_index)] *
                     (velocity_x[(node_index)] * velocity_x[(node_index)] +
                      velocity_y[(node_index)] * velocity_y[(node_index)] +
                      velocity_z[(node_index)] * velocity_z[(node_index)]);
    }

    // Update the volume of the cell to the new rezoned mesh
    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_c);
    calc_volume(cell_to_faces_off, nfaces_by_cell, cells_to_faces,
                faces_to_nodes, faces_to_nodes_offsets, nodes_x, nodes_y,
                nodes_z, &cell_c, &cell_volume[(cc)]);

    // Scatter the energy and density
    cell_mass[(cc)] = total_mass;
    density[(cc)] = cell_mass[(cc)] / cell_volume[(cc)];

    double total_e_mass = total_ie_mass; // + (total_ke_mass - new_ke_mass);
    energy[(cc)] = total_e_mass / cell_mass[(cc)];

    // Calculate the conservation data
    rz_total_mass += total_mass;
    rz_total_e_mass += total_e_mass;
  }

  printf("Initial Total Mass %.12f\n", initial_mass);
  printf("Rezoned Total Mass %.12f\n", rz_total_mass);
  printf("Difference         %.12f\n\n", rz_total_mass - initial_mass);

  printf("Initial Total Energy          %.12f\n",
         (initial_ie_mass + initial_ke_mass));
  printf("Rezoned Total Internal Energy %.12f\n", rz_total_e_mass);
  printf("Difference                    %.12f\n\n",
         rz_total_e_mass - (initial_ie_mass + initial_ke_mass));
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

  double total_momentum_x = 0.0;
  double total_momentum_y = 0.0;
  double total_momentum_z = 0.0;

#pragma omp parallel for reduction(+ : total_momentum_x, total_momentum_y,     \
                                   total_momentum_z)
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
    const int ncells_by_node =
        nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

    double mass_at_node = 0.0;
    double node_momentum_x = 0.0;
    double node_momentum_y = 0.0;
    double node_momentum_z = 0.0;

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
      node_momentum_x += subcell_momentum_x[(subcell_index)] -
                         subcell_momentum_flux_x[(subcell_index)];
      node_momentum_y += subcell_momentum_y[(subcell_index)] -
                         subcell_momentum_flux_y[(subcell_index)];
      node_momentum_z += subcell_momentum_z[(subcell_index)] -
                         subcell_momentum_flux_z[(subcell_index)];
      mass_at_node += subcell_mass[(subcell_index)];
    }

    nodal_mass[(nn)] = mass_at_node;

    total_momentum_x += node_momentum_x;
    total_momentum_y += node_momentum_y;
    total_momentum_z += node_momentum_z;

    velocity_x[(nn)] = node_momentum_x / nodal_mass[(nn)];
    velocity_y[(nn)] = node_momentum_y / nodal_mass[(nn)];
    velocity_z[(nn)] = node_momentum_z / nodal_mass[(nn)];
  }

  printf("Initial total momentum %.12f %.12f %.12f\n", initial_momentum->x,
         initial_momentum->y, initial_momentum->z);
  printf("Rezoned total momentum %.12f %.12f %.12f\n", total_momentum_x,
         total_momentum_y, total_momentum_z);
  printf("Difference             %.12f %.12f %.12f\n\n",
         initial_momentum->x - total_momentum_x,
         initial_momentum->y - total_momentum_y,
         initial_momentum->z - total_momentum_z);
}
