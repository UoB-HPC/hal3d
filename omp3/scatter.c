#include "../hale_data.h"
#include "hale.h"
#include <stdio.h>

// Perform the scatter step of the ALE remapping algorithm
void scatter_phase(const int ncells, const int nnodes, const double total_mass,
                   const double total_ie, const double* rezoned_nodes_x,
                   const double* rezoned_nodes_y, const double* rezoned_nodes_z,
                   double* cell_volume, double* energy0, double* energy1,
                   double* density0, double* velocity_x0, double* velocity_y0,
                   double* velocity_z0, double* cell_mass, double* nodal_mass,
                   double* subcell_ie_mass, double* subcell_mass,
                   double* subcell_ie_mass_flux, double* subcell_mass_flux,
                   double* subcell_momentum_x, double* subcell_momentum_y,
                   double* subcell_momentum_z, int* nodes_to_faces_offsets,
                   int* nodes_to_faces, int* faces_to_nodes,
                   int* faces_to_nodes_offsets, int* faces_to_cells0,
                   int* faces_to_cells1, int* cells_to_faces_offsets,
                   int* cells_to_faces, int* cells_offsets,
                   int* cells_to_nodes) {

  // Scatter the subcell energy and mass quantities back to the cell centers
  scatter_energy_and_mass(
      ncells, total_mass, total_ie, rezoned_nodes_x, rezoned_nodes_y,
      rezoned_nodes_z, cell_volume, energy0, energy1, density0, cell_mass,
      subcell_ie_mass, subcell_mass, subcell_ie_mass_flux, subcell_mass_flux,
      faces_to_nodes, faces_to_nodes_offsets, cells_to_faces_offsets,
      cells_to_faces, cells_offsets, cells_to_nodes);

#if 0
  // Scattering the momentum
  double total_vx = 0.0;
  double total_vy = 0.0;
  double total_vz = 0.0;
#pragma omp parallel for reduction(+ : total_vx, total_vy, total_vz)
  for (int nn = 0; nn < nnodes; ++nn) {
    velocity_x0[(nn)] = 0.0;
    velocity_y0[(nn)] = 0.0;
    velocity_z0[(nn)] = 0.0;

    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;

    for (int cc = 0; cc < ncells_by_node; ++cc) {
    }

    total_vx += velocity_x0[(nn)];
    total_vy += velocity_y0[(nn)];
    total_vz += velocity_z0[(nn)];
  }

  printf("Total Scattered Velocity %.12f %.12f %.12f\n", total_vx, total_vy,
         total_vz);
#endif // if 0
}

// Scatter the subcell energy and mass quantities back to the cell centers
void scatter_energy_and_mass(
    const int ncells, const double total_mass, const double total_ie,
    const double* rezoned_nodes_x, const double* rezoned_nodes_y,
    const double* rezoned_nodes_z, double* cell_volume, double* energy0,
    double* energy1, double* density0, double* cell_mass,
    double* subcell_ie_mass, double* subcell_mass, double* subcell_ie_mass_flux,
    double* subcell_mass_flux, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_offsets,
    int* cells_to_nodes) {

  // Scatter energy and density, and print the conservation of mass
  double rz_total_mass = 0.0;
  double rz_total_ie = 0.0;
#pragma omp parallel for reduction(+ : rz_total_mass, rz_total_ie)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    cell_mass[(cc)] = 0.0;
    energy1[(cc)] = 0.0;

    // Update the volume of the cell to the new rezoned mesh
    vec_t rz_cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, rezoned_nodes_x, rezoned_nodes_y,
                  rezoned_nodes_z, cells_to_nodes, cell_to_nodes_off,
                  &rz_cell_centroid);
    calc_volume(cell_to_faces_off, nfaces_by_cell, cells_to_faces,
                faces_to_nodes, faces_to_nodes_offsets, rezoned_nodes_x,
                rezoned_nodes_y, rezoned_nodes_z, &rz_cell_centroid,
                &cell_volume[(cc)]);

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int subcell_index = cell_to_nodes_off + nn;

      // Scatter the subcell mass data back to the cell
      subcell_mass[(subcell_index)] -= subcell_mass_flux[(subcell_index)];
      subcell_ie_mass[(subcell_index)] -= subcell_ie_mass_flux[(subcell_index)];
      cell_mass[(cc)] += subcell_mass[(subcell_index)];
      energy1[(cc)] += subcell_ie_mass[(subcell_index)];

      subcell_mass_flux[(subcell_index)] = 0.0;
      subcell_ie_mass_flux[(subcell_index)] = 0.0;
    }

    // Scatter the energy and density
    density0[(cc)] = cell_mass[(cc)] / cell_volume[(cc)];
    energy0[(cc)] = energy1[(cc)] / cell_mass[(cc)];

    // Calculate the conservation data
    rz_total_mass += cell_mass[(cc)];
    rz_total_ie += energy1[(cc)];
  }

  printf(
      "Rezoned Total Mass %.12f, Initial Total Mass %.12f, Difference %.12f\n",
      rz_total_mass, total_mass, total_mass - rz_total_mass);
  printf("Rezoned Total Internal Energy %.12f, Initial Total Energy %.12f, "
         "Difference "
         "%.12f\n",
         rz_total_ie, total_ie, rz_total_ie - total_ie);
}
