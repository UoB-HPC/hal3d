#include "hale.h"
#include "../../shared.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_3d(
    Mesh* mesh, HaleData* hale_data, const int ncells, const int nnodes,
    const int nsubcell_nodes, const int nsubcells_per_cell,
    const double visc_coeff1, const double visc_coeff2,
    double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_to_nodes, int* cells_offsets,
    int* nodes_to_cells, int* nodes_offsets, double* nodes_x0, double* nodes_y0,
    double* nodes_z0, double* nodes_x1, double* nodes_y1, double* nodes_z1,
    int* boundary_index, int* boundary_type, double* boundary_normal_x,
    double* boundary_normal_y, double* boundary_normal_z, double* cell_volume,
    double* energy0, double* energy1, double* density0, double* density1,
    double* pressure0, double* pressure1, double* velocity_x0,
    double* velocity_y0, double* velocity_z0, double* velocity_x1,
    double* velocity_y1, double* velocity_z1, double* subcell_force_x,
    double* subcell_force_y, double* subcell_force_z, double* cell_mass,
    double* nodal_mass, double* nodal_volumes, double* nodal_soundspeed,
    double* limiter, double* subcell_volume, double* subcell_ie_mass0,
    double* subcell_mass0, double* subcell_ie_mass1, double* subcell_mass1,
    double* subcell_momentum_x, double* subcell_momentum_y,
    double* subcell_momentum_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* subcell_kinetic_energy, int* subcells_to_nodes,
    double* subcell_data_x, double* subcell_data_y, double* subcell_data_z,
    double* rezoned_nodes_x, double* rezoned_nodes_y, double* rezoned_nodes_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* subcell_face_offsets,
    int* subcells_to_subcells) {

  //
  //
  //
  //
  //

  for (int nn = 0; nn < nnodes; ++nn) {
    int x = nn % (mesh->local_nx + 1);
    int y = (nn / (mesh->local_nx + 1)) % (mesh->local_ny + 1);

    if (x == 0 || x == (mesh->local_nx + 1) - 1) {
      continue;
    }
#if 0
    if (y == 0 || y == (mesh->local_nx + 1) - 1) {
      continue;
    }
#endif // if 0

    nodes_x0[(nn)] += 0.01;
#if 0
    nodes_y0[(nn)] += 0.01;
#endif // if 0
  }

  init_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0, nodes_y0,
                      nodes_z0, cell_centroids_x, cell_centroids_y,
                      cell_centroids_z);

  calc_volumes_centroids(ncells, cells_to_faces_offsets, cell_centroids_x,
                         cell_centroids_y, cell_centroids_z, cells_to_faces,
                         faces_to_nodes, faces_to_nodes_offsets,
                         subcell_face_offsets, nodes_x0, nodes_y0, nodes_z0,
                         cell_volume, subcell_centroids_x, subcell_centroids_y,
                         subcell_centroids_z, subcell_volume);

  for (int z = 0; z < mesh->local_nz; ++z) {
    for (int y = 0; y < mesh->local_ny; ++y) {
      for (int x = 0; x < mesh->local_nx; ++x) {
        int c = z * mesh->local_nx * mesh->local_ny + y * mesh->local_nx + x;
        density0[(c)] = 1.0 + 3.0 * (double)x + (double)y + 2.0 * (double)z;
        energy0[(c)] = 1.0 + 3.0 * (double)x + (double)y + 2.0 * (double)z;
        cell_mass[(c)] = 0.0;
      }
    }
  }

  init_mesh_mass(ncells, cells_offsets, cell_centroids_x, cell_centroids_y,
                 cell_centroids_z, cells_to_nodes, density0, nodes_x0, nodes_y0,
                 nodes_z0, cell_mass, subcell_mass0, cells_to_faces_offsets,
                 cells_to_faces, faces_to_nodes_offsets, faces_to_nodes,
                 subcell_face_offsets);

  write_unstructured_to_visit_3d(nnodes, ncells, 10, nodes_x0, nodes_y0,
                                 nodes_z0, cells_to_nodes, density0, 0, 1);

  //
  //
  //
  //
  //

  // Describe the subcell node layout
  printf("\nPerforming the Lagrangian Phase\n");

#if 0
  // Perform the Lagrangian phase of the ALE algorithm where the mesh will move
  // due to the pressure (ideal gas) and artificial viscous forces
  lagrangian_phase(
      mesh, ncells, nnodes, visc_coeff1, visc_coeff2, cell_centroids_x,
      cell_centroids_y, cell_centroids_z, cells_to_nodes, cells_offsets,
      nodes_to_cells, nodes_offsets, nodes_x0, nodes_y0, nodes_z0, nodes_x1,
      nodes_y1, nodes_z1, boundary_index, boundary_type, boundary_normal_x,
      boundary_normal_y, boundary_normal_z, energy0, energy1, density0,
      density1, pressure0, pressure1, velocity_x0, velocity_y0, velocity_z0,
      velocity_x1, velocity_y1, velocity_z1, subcell_force_x, subcell_force_y,
      subcell_force_z, cell_mass, nodal_mass, nodal_volumes, nodal_soundspeed,
      limiter, nodes_to_faces_offsets, nodes_to_faces, faces_to_nodes,
      faces_to_nodes_offsets, faces_to_cells0, faces_to_cells1,
      cells_to_faces_offsets, cells_to_faces);
#endif // if 0

///
//
//
//
//
//
//
//
//

#if 0
  for (int nn = 0; nn < nnodes; ++nn) {
    int x = nn % (mesh->local_nx + 1);
    int y = nn / (mesh->local_nx + 1) % (mesh->local_ny + 1);

    if (x == 0 || x == (mesh->local_nx + 1) - 1 || y == 0 ||
        y == (mesh->local_nx + 1) - 1) {
      continue;
    }

    rezoned_nodes_x[(nn)] += 0.01;
    rezoned_nodes_y[(nn)] += 0.01;
  }
#endif // if 0

  ///
  //
  //
  //
  //
  //
  //
  //
  //

  printf("\nPerforming Gathering Phase\n");

  // Gather the subcell quantities for mass, internal and kinetic energy
  // density, and momentum
  gather_subcell_quantities(
      ncells, nnodes, nodal_volumes, nodal_mass, cell_centroids_x,
      cell_centroids_y, cell_centroids_z, cells_offsets, nodes_x0, nodes_y0,
      nodes_z0, energy0, density0, velocity_x0, velocity_y0, velocity_z0,
      cell_mass, subcell_volume, subcell_ie_mass0, subcell_momentum_x,
      subcell_momentum_y, subcell_momentum_z, subcell_centroids_x,
      subcell_centroids_y, subcell_centroids_z, cell_volume,
      subcell_face_offsets, faces_to_nodes, faces_to_nodes_offsets,
      faces_to_cells0, faces_to_cells1, cells_to_faces_offsets, cells_to_faces,
      cells_to_nodes);

  for (int ss = 0; ss < ncells * 24; ++ss) {
    subcell_mass0[(ss)] = 0.0;
  }

  const int subcell_index = 13;
  subcell_mass0[(subcell_index)] = 5.0;
  for (int ss = 0; ss < NSUBCELL_NEIGHBOURS; ++ss) {
    const int neighbour_subcell_index =
        subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + ss)];
    if (neighbour_subcell_index != -1) {
      subcell_mass0[(neighbour_subcell_index)] = (double)(ss + 1.0);
    }
  }

  subcells_to_visit(nsubcell_nodes, ncells * nsubcells_per_cell, 10,
                    subcell_data_x, subcell_data_y, subcell_data_z,
                    subcells_to_nodes, subcell_mass0, 0, 1);

#if 0
  // Store the total mass and internal energy
  double total_mass = 0.0;
  double total_ie = 0.0;
#pragma omp parallel for reduction(+ : total_mass, total_ie)
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
    total_ie += energy0[(cc)] * cell_mass[(cc)];
  }

  printf("\nPerforming Remap Phase\n");

  // Performs a remap and some scattering of the subcell values
  remap_phase(ncells, cell_centroids_x, cell_centroids_y, cell_centroids_z,
              cells_to_nodes, cells_offsets, nodes_x0, nodes_y0, nodes_z0,
              cell_volume, velocity_x0, velocity_y0, velocity_z0,
              subcell_volume, subcell_ie_mass0, subcell_ie_mass1, subcell_mass0,
              subcell_mass1, subcell_momentum_x, subcell_momentum_y,
              subcell_momentum_z, subcell_centroids_x, subcell_centroids_y,
              subcell_centroids_z, rezoned_nodes_x, rezoned_nodes_y,
              rezoned_nodes_z, faces_to_nodes, faces_to_nodes_offsets,
              cells_to_faces_offsets, cells_to_faces, subcell_face_offsets,
              subcells_to_subcells);

  printf("\nPerforming the Scattering Phase\n");

  // Perform the scatter step of the ALE remapping algorithm
  scatter_phase(ncells, nnodes, total_mass, total_ie, cell_volume, energy0,
                energy1, density0, velocity_x0, velocity_y0, velocity_z0,
                cell_mass, nodal_mass, subcell_ie_mass0, subcell_mass0,
                subcell_ie_mass1, subcell_mass1, subcell_momentum_x,
                subcell_momentum_y, subcell_momentum_z, nodes_to_faces_offsets,
                nodes_to_faces, faces_to_nodes, faces_to_nodes_offsets,
                faces_to_cells0, faces_to_cells1, cells_to_faces_offsets,
                cells_to_faces, subcell_face_offsets);

  printf("\nEulerian Mesh Rezone\n");

  // Finalise the mesh rezone
  apply_mesh_rezoning(nnodes, rezoned_nodes_x, rezoned_nodes_y, rezoned_nodes_z,
                      nodes_x0, nodes_y0, nodes_z0);

  // Determine the new cell centroids
  init_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0, nodes_y0,
                      nodes_z0, cell_centroids_x, cell_centroids_y,
                      cell_centroids_z);

  write_unstructured_to_visit_3d(nnodes, ncells, 11, nodes_x0, nodes_y0,
                                 nodes_z0, cells_to_nodes, density0, 0, 1);
#endif // if 0
}
