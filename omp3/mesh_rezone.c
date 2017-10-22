#include "../../shared.h"
#include "hale.h"

// Correct the subcell data by the determined fluxes
void correct_for_fluxes(const int ncells, const int* cells_offsets,
                        double* subcell_mass, double* subcell_ie_mass,
                        double* subcell_mass_flux, double* subcell_ie_mass_flux,
                        double* subcell_momentum_x, double* subcell_momentum_y,
                        double* subcell_momentum_z,
                        double* subcell_momentum_flux_x,
                        double* subcell_momentum_flux_y,
                        double* subcell_momentum_flux_z);

// Performs an Eulerian rezone of the mesh
void eulerian_rezone(UnstructuredMesh* umesh, HaleData* hale_data) {

  // Correct the subcell data by the determined fluxes
  correct_for_fluxes(
      umesh->ncells, umesh->cells_offsets, hale_data->subcell_mass,
      hale_data->subcell_ie_mass, hale_data->subcell_mass_flux,
      hale_data->subcell_ie_mass_flux, hale_data->subcell_momentum_x,
      hale_data->subcell_momentum_y, hale_data->subcell_momentum_z,
      hale_data->subcell_momentum_flux_x, hale_data->subcell_momentum_flux_y,
      hale_data->subcell_momentum_flux_z);

  // Finalise the mesh rezone
  apply_mesh_rezoning(umesh->nnodes, hale_data->rezoned_nodes_x,
                      hale_data->rezoned_nodes_y, hale_data->rezoned_nodes_z,
                      umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0);

  // Determine the new cell centroids
  init_cell_centroids(umesh->ncells, umesh->cells_offsets,
                      umesh->cells_to_nodes, umesh->nodes_x0, umesh->nodes_y0,
                      umesh->nodes_z0, umesh->cell_centroids_x,
                      umesh->cell_centroids_y, umesh->cell_centroids_z);
}

// Correct the subcell data by the determined fluxes
void correct_for_fluxes(const int ncells, const int* cells_offsets,
                        double* subcell_mass, double* subcell_ie_mass,
                        double* subcell_mass_flux, double* subcell_ie_mass_flux,
                        double* subcell_momentum_x, double* subcell_momentum_y,
                        double* subcell_momentum_z,
                        double* subcell_momentum_flux_x,
                        double* subcell_momentum_flux_y,
                        double* subcell_momentum_flux_z) {

  double dm = 0.0;
  double die = 0.0;
  double dmom_x = 0.0;
  double dmom_y = 0.0;
  double dmom_z = 0.0;

  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int subcell_index = cell_to_nodes_off + nn;

      // Calculate the changes due to flux
      subcell_mass[(subcell_index)] -= subcell_mass_flux[(subcell_index)];
      subcell_ie_mass[(subcell_index)] -= subcell_ie_mass_flux[(subcell_index)];
      subcell_momentum_x[(subcell_index)] -=
          subcell_momentum_flux_x[(subcell_index)];
      subcell_momentum_y[(subcell_index)] -=
          subcell_momentum_flux_y[(subcell_index)];
      subcell_momentum_z[(subcell_index)] -=
          subcell_momentum_flux_z[(subcell_index)];

      dm += subcell_mass_flux[(subcell_index)];
      die += subcell_ie_mass_flux[(subcell_index)];
      dmom_x += subcell_momentum_flux_x[(subcell_index)];
      dmom_y += subcell_momentum_flux_y[(subcell_index)];
      dmom_z += subcell_momentum_flux_z[(subcell_index)];

      if (subcell_mass[(subcell_index)] < 0.0) {
        printf("Subcell Mass has turned negative.\n");
      }
      if (subcell_ie_mass[(subcell_index)] < 0.0) {
        printf("Subcell Energy has turned negative.\n");
      }

      // Clear the array that we will be reducing into during next timestep
      subcell_mass_flux[(subcell_index)] = 0.0;
      subcell_ie_mass_flux[(subcell_index)] = 0.0;
      subcell_momentum_flux_x[(subcell_index)] = 0.0;
      subcell_momentum_flux_y[(subcell_index)] = 0.0;
      subcell_momentum_flux_z[(subcell_index)] = 0.0;
    }
  }
}
