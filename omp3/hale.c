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
      hale_data->cell_mass, hale_data->nodal_mass, hale_data->cell_volume,
      hale_data->nodal_volumes, hale_data->nodal_soundspeed, hale_data->limiter,
      umesh->nodes_to_faces_offsets, umesh->nodes_to_faces,
      umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
      umesh->faces_to_cells0, umesh->faces_to_cells1,
      umesh->cells_to_faces_offsets, umesh->cells_to_faces);

#if 0
  write_unstructured_to_visit_3d(umesh->nnodes, umesh->ncells, timestep * 2,
                                 umesh->nodes_x0, umesh->nodes_y0,
                                 umesh->nodes_z0, umesh->cells_to_nodes,
                                 hale_data->velocity_x0, 1, 1);
#endif // if 0

  if (hale_data->perform_remap) {
    printf("\nPerforming Gathering Phase\n");

    double initial_mass = 0.0;
    double initial_ie_mass = 0.0;
    vec_t initial_momentum = {0.0, 0.0, 0.0};

    // gathers all of the subcell quantities on the mesh
    gather_subcell_quantities(umesh, hale_data, &initial_momentum,
                              &initial_mass, &initial_ie_mass);

    init_subcell_data_structures(mesh, hale_data, umesh);
    write_unstructured_to_visit_3d(
        hale_data->nsubcell_nodes, umesh->ncells * hale_data->nsubcells_by_cell,
        timestep * 2 + 1, hale_data->subcell_nodes_x,
        hale_data->subcell_nodes_y, hale_data->subcell_nodes_z,
        hale_data->subcells_to_nodes, hale_data->subcell_momentum_x, 0, 1);

    printf("\nPerforming Remap Phase\n");

    // Performs a remap and some scattering of the subcell values
    advection_phase(umesh, hale_data);

    printf("\nEulerian Mesh Rezone\n");

    eulerian_rezone(umesh, hale_data);

    printf("\nPerforming the Scattering Phase\n");

    for (int cc = 0; cc < umesh->ncells; ++cc) {
      const int cell_to_nodes_off = umesh->cells_offsets[(cc)];
      const int nnodes_by_cell =
          umesh->cells_offsets[(cc + 1)] - cell_to_nodes_off;

      for (int nn = 0; nn < nnodes_by_cell; ++nn) {
        const int subcell_index = cell_to_nodes_off + nn;

        // Calculate the changes due to flux
        hale_data->subcell_mass[(subcell_index)] -=
            hale_data->subcell_mass_flux[(subcell_index)];
        hale_data->subcell_ie_mass[(subcell_index)] -=
            hale_data->subcell_ie_mass_flux[(subcell_index)];
        hale_data->subcell_momentum_x[(subcell_index)] -=
            hale_data->subcell_momentum_flux_x[(subcell_index)];
        hale_data->subcell_momentum_y[(subcell_index)] -=
            hale_data->subcell_momentum_flux_y[(subcell_index)];
        hale_data->subcell_momentum_z[(subcell_index)] -=
            hale_data->subcell_momentum_flux_z[(subcell_index)];

        // Clear the array that we will be reducing into during next timestep
        hale_data->subcell_mass_flux[(subcell_index)] = 0.0;
        hale_data->subcell_ie_mass_flux[(subcell_index)] = 0.0;
        hale_data->subcell_momentum_flux_x[(subcell_index)] = 0.0;
        hale_data->subcell_momentum_flux_y[(subcell_index)] = 0.0;
        hale_data->subcell_momentum_flux_z[(subcell_index)] = 0.0;
      }
    }

    repair_phase(umesh, hale_data);

    // Perform the scatter step of the ALE remapping algorithm
    scatter_phase(umesh, hale_data, &initial_momentum, initial_mass,
                  initial_ie_mass);

    printf("\nPerforming the Repair Phase\n");
  }
}
