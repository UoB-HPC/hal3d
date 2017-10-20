#include "hale.h"

// Performs an Eulerian rezone of the mesh
void eulerian_rezone(UnstructuredMesh* umesh, HaleData* hale_data) {

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
