#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <math.h>

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_nodal_volumes(const int nnodes, const int* nodes_to_faces_offsets,
                        const int* nodes_to_faces,
                        const int* faces_to_nodes_offsets,
                        const int* faces_to_nodes, const int* faces_to_cells0,
                        const int* faces_to_cells1,
                        const double* cell_centroids_x,
                        const double* cell_centroids_y,
                        const double* cell_centroids_z, const double* nodes_x,
                        const double* nodes_y, const double* nodes_z,
                        double* nodal_volumes) {

#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;

    nodal_volumes[(nn)] = 0.0;

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
      vec_t face_c = {0.0, 0.0, 0.0};
      int node_in_face_c;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c.x += nodes_x[(node_index)] / nnodes_by_face;
        face_c.y += nodes_y[(node_index)] / nnodes_by_face;
        face_c.z += nodes_z[(node_index)] / nnodes_by_face;

        // Choose the node in the list of nodes attached to the face
        if (nn == node_index) {
          node_in_face_c = nn2;
        }
      }

      // Fetch the nodes attached to our current node on the current face
      int local_nodes[2];
      local_nodes[0] =
          (node_in_face_c - 1 >= 0)
              ? faces_to_nodes[(face_to_nodes_off + node_in_face_c - 1)]
              : faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)];
      local_nodes[1] =
          (node_in_face_c + 1 < nnodes_by_face)
              ? faces_to_nodes[(face_to_nodes_off + node_in_face_c + 1)]
              : faces_to_nodes[(face_to_nodes_off)];

      // Fetch the cells attached to our current face
      int local_cells[2];
      local_cells[0] = faces_to_cells0[(face_index)];
      local_cells[1] = faces_to_cells1[(face_index)];

      // Add contributions from both of the cells attached to the face
      for (int cc = 0; cc < 2; ++cc) {
        if (local_cells[(cc)] == -1) {
          continue;
        }

        const int cell_index = local_cells[(cc)];

        // Add contributions for both edges attached to our current node
        for (int nn2 = 0; nn2 < 2; ++nn2) {
          const double subsubcell_volume = calc_subsubcell_volume(
              cell_index, local_nodes[(nn2)], nn, face_c, nodes_x, nodes_y,
              nodes_z, cell_centroids_x, cell_centroids_y, cell_centroids_z);
          nodal_volumes[(nn)] += subsubcell_volume;
        }
      }
    }
  }
}

// Calculates the cell volume, subcell volume and the subcell centroids

// Initialises the centroids for each cell
void init_cell_centroids(const int ncells, const int* cells_offsets,
                         const int* cells_to_nodes, const double* nodes_x,
                         const double* nodes_y, const double* nodes_z,
                         double* cell_centroids_x, double* cell_centroids_y,
                         double* cell_centroids_z) {

  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cells_off, &cell_c);

    cell_centroids_x[(cc)] = cell_c.x;
    cell_centroids_y[(cc)] = cell_c.y;
    cell_centroids_z[(cc)] = cell_c.z;
  }
  STOP_PROFILING(&compute_profile, __func__);
}
