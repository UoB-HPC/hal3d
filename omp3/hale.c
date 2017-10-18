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

  if (hale_data->perform_remap) {
    printf("\nPerforming Gathering Phase\n");

    double total_ie = 0.0;
    double total_mass = 0.0;
    vec_t initial_momentum = {0.0, 0.0, 0.0};

    // gathers all of the subcell quantities on the mesh
    gather_subcell_quantities(
        umesh->ncells, umesh->nnodes, hale_data->nnodes_by_subcell,
        hale_data->nodal_volumes, hale_data->nodal_mass,
        umesh->cell_centroids_x, umesh->cell_centroids_y,
        umesh->cell_centroids_z, umesh->nodes_to_cells, umesh->nodes_x0,
        umesh->nodes_y0, umesh->nodes_z0, hale_data->energy0,
        hale_data->density0, hale_data->velocity_x0, hale_data->velocity_y0,
        hale_data->velocity_z0, hale_data->cell_mass, hale_data->subcell_volume,
        hale_data->subcell_ie_mass, hale_data->subcell_momentum_x,
        hale_data->subcell_momentum_y, hale_data->subcell_momentum_z,
        hale_data->subcell_centroids_x, hale_data->subcell_centroids_y,
        hale_data->subcell_centroids_z, hale_data->cell_volume,
        hale_data->subcells_to_faces_offsets, umesh->faces_to_nodes,
        umesh->faces_to_nodes_offsets, umesh->faces_to_cells0,
        umesh->faces_to_cells1, umesh->cells_to_faces_offsets,
        umesh->cells_to_faces, hale_data->subcells_to_faces,
        umesh->nodes_offsets, umesh->cells_offsets, umesh->cells_to_nodes,
        umesh->nodes_to_nodes_offsets, umesh->nodes_to_nodes,
        &initial_momentum);

    printf("\nPerforming Remap Phase\n");

    // Performs a remap and some scattering of the subcell values
    remap_phase(
        umesh->ncells, umesh->cells_offsets, umesh->nodes_x0, umesh->nodes_y0,
        umesh->nodes_z0, hale_data->rezoned_nodes_x, hale_data->rezoned_nodes_y,
        hale_data->rezoned_nodes_z, hale_data->subcell_momentum_x,
        hale_data->subcell_momentum_y, hale_data->subcell_momentum_z,
        umesh->cells_to_nodes, umesh->faces_to_nodes_offsets,
        umesh->faces_to_nodes, hale_data->subcells_to_faces_offsets,
        hale_data->subcells_to_faces, hale_data->subcells_to_subcells_offsets,
        hale_data->subcells_to_subcells, umesh->faces_to_cells0,
        umesh->faces_to_cells1, hale_data->subcell_momentum_flux_x,
        hale_data->subcell_momentum_flux_y, hale_data->subcell_momentum_flux_z,
        hale_data->subcell_centroids_x, hale_data->subcell_centroids_y,
        hale_data->subcell_centroids_z, hale_data->subcell_volume,
        hale_data->subcell_mass, hale_data->subcell_mass_flux,
        hale_data->subcell_ie_mass, hale_data->subcell_ie_mass_flux);

    printf("\nEulerian Mesh Rezone\n");

    // Finalise the mesh rezone
    apply_mesh_rezoning(umesh->nnodes, hale_data->rezoned_nodes_x,
                        hale_data->rezoned_nodes_y, hale_data->rezoned_nodes_z,
                        umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0);

    // Determine the new cell centroids
    init_cell_centroids(umesh->ncells, umesh->cells_offsets,
                        umesh->cells_to_nodes, umesh->nodes_x0, umesh->nodes_y0,
                        umesh->nodes_z0, umesh->cell_centroids_x,
                        umesh->cell_centroids_y, umesh->cell_centroids_z);

    printf("\nPerforming the Scattering Phase\n");

    // Perform the scatter step of the ALE remapping algorithm
    scatter_phase(
        umesh->ncells, umesh->nnodes, &initial_momentum,
        hale_data->rezoned_nodes_x, hale_data->rezoned_nodes_y,
        hale_data->rezoned_nodes_z, hale_data->cell_volume, hale_data->energy0,
        hale_data->density0, hale_data->velocity_x0, hale_data->velocity_y0,
        hale_data->velocity_z0, hale_data->cell_mass, hale_data->nodal_mass,
        hale_data->subcell_ie_mass, hale_data->subcell_mass,
        hale_data->subcell_ie_mass_flux, hale_data->subcell_mass_flux,
        hale_data->subcell_momentum_x, hale_data->subcell_momentum_y,
        hale_data->subcell_momentum_z, hale_data->subcell_momentum_flux_x,
        hale_data->subcell_momentum_flux_y, hale_data->subcell_momentum_flux_z,
        umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
        umesh->cells_to_faces_offsets, umesh->cells_to_faces,
        umesh->nodes_offsets, umesh->nodes_to_cells, umesh->cells_offsets,
        umesh->cells_to_nodes, &total_mass, &total_ie);

    init_subcell_data_structures(mesh, hale_data, umesh);
    write_unstructured_to_visit_3d(
        hale_data->nsubcell_nodes, umesh->ncells * hale_data->nsubcells_by_cell,
        timestep * 2 + 1, hale_data->subcell_nodes_x,
        hale_data->subcell_nodes_y, hale_data->subcell_nodes_z,
        hale_data->subcells_to_nodes, hale_data->subcell_mass, 0, 1);
  }

#if 0
  const int swept_edge_to_faces[] = {0, 1, 2, 3, 4, 5};
  const int swept_edge_faces_to_nodes_offsets[] = {0, 4, 8, 12, 16, 20, 24};
  const int swept_edge_faces_to_nodes[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 7, 4,
                                           7, 6, 2, 3, 1, 5, 6, 2, 0, 4, 5, 1};

  double nodes_x[] = {
      7.022991994606413e-01, 6.564142503612233e-01, 6.564142824995303e-01,
      7.022992176736477e-01, 7.000000000000000e-01, 6.500000000000001e-01,
      6.500000000000001e-01, 7.000000000000000e-01,
  };
  double nodes_y[] = {
      4.499999999999999e-01, 4.499999999999999e-01, 4.499999999999999e-01,
      4.499999999999999e-01, 4.500000000000000e-01, 4.500000000000000e-01,
      4.500000000000000e-01, 4.500000000000000e-01,
  };
  double nodes_z[] = {
      1.000000016593187e-01, 1.000000009180668e-01, 5.000000045903342e-02,
      5.000000082965937e-02, 1.000000000000000e-01, 1.000000000000000e-01,
      5.000000000000000e-02, 5.000000000000000e-02,
  };

  double swept_edge_vol = 0.0;

  int overlap =
      test_prism_overlap(NNODES_BY_SUBCELL_FACE, swept_edge_faces_to_nodes,
                         nodes_x, nodes_y, nodes_z);

  printf("THE PRISM DID %sOVERLAP\n", overlap ? "" : "NOT ");

  // The face centroid is the same for all nodes on the face
  vec_t face_c = {0.0, 0.0, 0.0};
  calc_centroid(NNODES_BY_SUBCELL_FACE, nodes_x, nodes_y, nodes_z,
                swept_edge_faces_to_nodes, 0, &face_c);
  printf("%.12e %.12e %.12e\n", face_c.x, face_c.y, face_c.z);

  vec_t cell_c = {0.0, 0.0, 0.0};
  calc_centroid(2 * NNODES_BY_SUBCELL_FACE, nodes_x, nodes_y, nodes_z,
                swept_edge_faces_to_nodes, 0, &cell_c);
  printf("%.12e %.12e %.12e\n", cell_c.x, cell_c.y, cell_c.z);

  vec_t swept_edge_c = {0.0, 0.0, 0.0};
  calc_centroid(2 * NNODES_BY_SUBCELL_FACE, nodes_x, nodes_y, nodes_z,
                swept_edge_faces_to_nodes, 0, &swept_edge_c);
  calc_volume(0, 2 + NNODES_BY_SUBCELL_FACE, swept_edge_to_faces,
              swept_edge_faces_to_nodes, swept_edge_faces_to_nodes_offsets,
              nodes_x, nodes_y, nodes_z, &swept_edge_c, &swept_edge_vol);
  write_unstructured_to_visit_3d(2 * NNODES_BY_SUBCELL_FACE, 1, 10000, nodes_x,
                                 nodes_y, nodes_z, swept_edge_faces_to_nodes,
                                 &swept_edge_vol, 0, 1);

  printf("swept_edge_vol %.12e\n", swept_edge_vol);
#endif // if 0
}

#if 0
vec_t face_normal;
calc_surface_normal(NNODES_BY_SUBCELL_FACE, 0, swept_edge_faces_to_nodes,
                    nodes_x, nodes_y, nodes_z, &face_c, &cell_c,
                    &face_normal);
#endif // if 0

#if 0
  double nodes_x[] = {0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0};
  double nodes_y[] = {0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
  double nodes_z[] = {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
#endif // if 0

#if 0
  double nodes_x[] = {0.0, 1.0, 1.0, 0.0};
  double nodes_y[] = {0.0, 0.0, 1.0, 1.0};
  double nodes_z[] = {0.5, 0.0, 0.5, 0.0};
  double nodes_x[] = {
      6.10225185718835e-01, 6.10225185718933e-01, 6.56224127425556e-01,
      6.56224127425173e-01, 6.00000000000000e-01, 6.00000000000000e-01,
      6.50000000000000e-01, 6.50000000000000e-01,
  };
  double nodes_y[] = {
      3.00000001704799e-01, 3.50000000000185e-01, 3.50000000000821e-01,
      2.99999989178878e-01, 3.00000000000000e-01, 3.50000000000000e-01,
      3.50000000000000e-01, 3.00000000000000e-01,
  };
  double nodes_z[] = {
      6.50000009101991e-01, 6.50000009101992e-01, 6.49999999146579e-01,
      6.49999999146578e-01, 6.50000000000000e-01, 6.50000000000000e-01,
      6.50000000000000e-01, 6.50000000000000e-01,
  };
#if 0
  const int swept_edge_faces_to_nodes[] = {0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 7, 3,
                                           7, 6, 2, 3, 1, 2, 6, 5, 0, 1, 5, 4};
  const int swept_edge_faces_to_nodes[] = {0, 1, 3, 2, 4, 5, 7, 6, 1, 5, 7, 3,
                                           0, 2, 6, 4, 0, 4, 5, 1, 2, 6, 7, 3};
#endif // if 0

#endif // if 0
