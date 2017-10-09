#include "hale.h"
#include "../../shared.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_3d(Mesh* mesh, HaleData* hale_data,
                                 UnstructuredMesh* umesh, const int timestep) {

  // Describe the subcell node layout
  printf("\nPerforming the Lagrangian Phase\n");

  // Perform the Lagrangian phase of the ALE algorithm where the mesh will move
  // due to the pressure (ideal gas) and artificial viscous forces
  lagrangian_phase(
      mesh, umesh->ncells, umesh->nnodes, hale_data->visc_coeff1,
      hale_data->visc_coeff2, umesh->cell_centroids_x, umesh->cell_centroids_y,
      umesh->cell_centroids_z, umesh->cells_to_nodes, umesh->cells_offsets,
      umesh->nodes_to_cells, umesh->nodes_offsets, umesh->nodes_x0,
      umesh->nodes_y0, umesh->nodes_z0, umesh->nodes_x1, umesh->nodes_y1,
      umesh->nodes_z1, umesh->boundary_index, umesh->boundary_type,
      umesh->boundary_normal_x, umesh->boundary_normal_y,
      umesh->boundary_normal_z, hale_data->energy0, hale_data->energy1,
      hale_data->density0, hale_data->density1, hale_data->pressure0,
      hale_data->pressure1, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0, hale_data->velocity_x1, hale_data->velocity_y1,
      hale_data->velocity_z1, hale_data->subcell_force_x,
      hale_data->subcell_force_y, hale_data->subcell_force_z,
      hale_data->cell_mass, hale_data->nodal_mass, hale_data->nodal_volumes,
      hale_data->nodal_soundspeed, hale_data->limiter,
      umesh->nodes_to_faces_offsets, umesh->nodes_to_faces,
      umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
      umesh->faces_to_cells0, umesh->faces_to_cells1,
      umesh->cells_to_faces_offsets, umesh->cells_to_faces);

  if (hale_data->visit_dump) {
    write_unstructured_to_visit_3d(umesh->nnodes, umesh->ncells, timestep * 2,
                                   umesh->nodes_x0, umesh->nodes_y0,
                                   umesh->nodes_z0, umesh->cells_to_nodes,
                                   hale_data->velocity_x0, 1, 1);
  }

  if (hale_data->perform_remap) {
    printf("\nPerforming Gathering Phase\n");

    // Gathers all of the subcell quantities on the mesh
    gather_subcell_quantities(
        umesh->ncells, umesh->nnodes, hale_data->nnodes_by_subcell,
        hale_data->nodal_volumes, hale_data->nodal_mass,
        umesh->cell_centroids_x, umesh->cell_centroids_y,
        umesh->cell_centroids_z, umesh->cells_offsets, umesh->nodes_to_cells,
        umesh->nodes_offsets, umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
        hale_data->energy0, hale_data->density0, hale_data->velocity_x0,
        hale_data->velocity_y0, hale_data->velocity_z0, hale_data->cell_mass,
        hale_data->subcell_volume, hale_data->subcell_ie_mass,
        hale_data->subcell_momentum_x, hale_data->subcell_momentum_y,
        hale_data->subcell_momentum_z, hale_data->subcell_centroids_x,
        hale_data->subcell_centroids_y, hale_data->subcell_centroids_z,
        hale_data->cell_volume, hale_data->subcells_to_faces_offsets,
        umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
        umesh->faces_to_cells0, umesh->faces_to_cells1,
        umesh->cells_to_faces_offsets, umesh->cells_to_faces,
        umesh->cells_to_nodes, hale_data->subcells_to_faces);

    init_subcell_data_structures(mesh, hale_data, umesh);
    write_unstructured_to_visit_3d(
        hale_data->nsubcell_nodes, umesh->ncells * hale_data->nsubcells_by_cell,
        timestep * 2 + 1, hale_data->subcell_nodes_x,
        hale_data->subcell_nodes_y, hale_data->subcell_nodes_z,
        hale_data->subcells_to_nodes, hale_data->subcell_momentum_x, 0, 1);

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
                cells_to_nodes, cells_offsets, nodes_to_cells, nodes_offsets,
                nodes_x0, nodes_y0, nodes_z0, cell_volume, velocity_x0,
                velocity_y0, velocity_z0, subcell_volume, subcell_ie_mass,
                subcell_ie_mass_flux, subcell_mass, subcell_mass_flux,
                subcell_momentum_x, subcell_momentum_y, subcell_momentum_z,
                subcell_centroids_x, subcell_centroids_y, subcell_centroids_z,
                rezoned_nodes_x, rezoned_nodes_y, rezoned_nodes_z,
                faces_to_nodes, faces_to_nodes_offsets, cells_to_faces_offsets,
                cells_to_faces, subcells_to_faces_offsets, subcells_to_subcells);

    printf("\nPerforming the Scattering Phase\n");

    // Finalise the mesh rezone
    apply_mesh_rezoning(nnodes, rezoned_nodes_x, rezoned_nodes_y,
                        rezoned_nodes_z, nodes_x0, nodes_y0, nodes_z0);

    // Determine the new cell centroids
    init_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0,
                        nodes_y0, nodes_z0, cell_centroids_x, cell_centroids_y,
                        cell_centroids_z);

    // Scatter the primary variables into the new mesh cells
    scatter_phase(ncells, nnodes, total_mass, total_ie, cell_volume, energy0,
                  energy1, density0, velocity_x0, velocity_y0, velocity_z0,
                  cell_mass, nodal_mass, subcell_ie_mass, subcell_mass,
                  subcell_ie_mass_flux, subcell_mass_flux, subcell_momentum_x,
                  subcell_momentum_y, subcell_momentum_z,
                  nodes_to_faces_offsets, nodes_to_faces, faces_to_nodes,
                  faces_to_nodes_offsets, faces_to_cells0, faces_to_cells1,
                  cells_to_faces_offsets, cells_to_faces, subcells_to_faces_offsets);

    subcells_to_visit(nsubcell_nodes, ncells * nsubcells_by_cell,
                      1000 + timestep, subcell_nodes_x, subcell_nodes_y,
                      subcell_nodes_z, subcells_to_nodes, subcell_mass_flux, 0,
                      1);

    printf("\nEulerian Mesh Rezone\n");

    if (hale_data->visit_dump) {
      write_unstructured_to_visit_3d(nnodes, ncells, timestep * 2 + 1, nodes_x0,
                                     nodes_y0, nodes_z0, cells_to_nodes,
                                     density0, 0, 1);
    }
#endif // if 0
  }
}
