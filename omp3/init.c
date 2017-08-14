#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <math.h>

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(const int ncells, const int* cells_offsets,
                    const double* cell_centroids_x,
                    const double* cell_centroids_y,
                    const double* cell_centroids_z, const int* cells_to_nodes,
                    const double* density, const double* nodes_x,
                    const double* nodes_y, const double* nodes_z,
                    double* cell_mass, double* subcell_mass,
                    int* cells_to_faces_offsets, int* cells_to_faces,
                    int* faces_to_nodes_offsets, int* faces_to_nodes,
                    int* subcell_face_offsets) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // Calculate the face center... SHOULD WE PRECOMPUTE?
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                    face_to_nodes_off, &face_c);

      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        // Fetch the nodes attached to our current node on the current face
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn2)];
        const int next_node =
            (nn2 + 1 < nnodes_by_face)
                ? faces_to_nodes[(face_to_nodes_off + nn2 + 1)]
                : faces_to_nodes[(face_to_nodes_off)];

        // Get the halfway point on the right edge
        const double half_edge_x =
            0.5 * (nodes_x[(current_node)] + nodes_x[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y[(current_node)] + nodes_y[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z[(current_node)] + nodes_z[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (half_edge_x - face_c.x);
        const double a_y = (half_edge_y - face_c.y);
        const double a_z = (half_edge_z - face_c.z);
        const double b_x = (cell_centroids_x[(cc)] - face_c.x);
        const double b_y = (cell_centroids_y[(cc)] - face_c.y);
        const double b_z = (cell_centroids_z[(cc)] - face_c.z);

        // Calculate the area vector S using cross product
        const double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        const double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        const double S_z = 0.5 * (a_x * b_y - a_y * b_x);

        double subcell_volume =
            2.0 * fabs(((half_edge_x - nodes_x[(current_node)]) * S_x +
                        (half_edge_y - nodes_y[(current_node)]) * S_y +
                        (half_edge_z - nodes_z[(current_node)]) * S_z) /
                       3.0);

        subcell_mass[(subcell_off + nn2)] = density[(cc)] * subcell_volume;
        cell_mass[(cc)] += density[(cc)] * subcell_volume;
      }
    }

    total_mass += cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, __func__);

  printf("Initial total mesh mash: %.15f\n", total_mass);
}

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

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cells_off + nn)];
      cx += nodes_x[(node_index)];
      cy += nodes_y[(node_index)];
      cz += nodes_z[(node_index)];
    }

    cell_centroids_x[(cc)] = cx / (double)nnodes_by_cell;
    cell_centroids_y[(cc)] = cy / (double)nnodes_by_cell;
    cell_centroids_z[(cc)] = cz / (double)nnodes_by_cell;
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const int* faces_to_cells0,
    const int* faces_to_cells1, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    int* cells_to_faces_offsets, int* cells_to_faces, int* subcells_to_subcells,
    int* subcell_face_offsets, int* subcell_to_neighbour_face) {

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      subcell_face_offsets[(cell_to_faces_off + ff + 1)] = nnodes_by_face;
    }
  }

  // TODO: Can only do serially at the moment... Fix this ideally
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      subcell_face_offsets[(cell_to_faces_off + ff + 1)] +=
          subcell_face_offsets[(cell_to_faces_off + ff)];
    }
  }

// This routine uses the previously described subcell faces to determine a ring
// ordering for the subcell neighbours. Essentially, each subcell has a pair of
// neighbours for every external face, and those neighbours are stored in the
// same order as the faces.
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      const int neighbour_cell_index = (faces_to_cells0[(face_index)] == cc)
                                           ? faces_to_cells1[(face_index)]
                                           : faces_to_cells0[(face_index)];

      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // Have to check the orientation of the face in order to pick
      // the correct subcell neighbour at this point
      const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
      const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
      const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];

      vec_t normal = {0.0, 0.0, 0.0};
      const int face_clockwise = calc_surface_normal(
          n0, n1, n2, nodes_x, nodes_y, nodes_z, &cell_centroid, &normal);

      // We have a subcell per node on a face
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        /*
         * Find the subcell that exists in the neighbouring cell on this face.
         */
        const int subcell_index = subcell_off + nn;
        if (neighbour_cell_index != -1) {
          const int neighbour_to_faces_off =
              cells_to_faces_offsets[(neighbour_cell_index)];
          const int nfaces_by_neighbour =
              cells_to_faces_offsets[(neighbour_cell_index + 1)] -
              neighbour_to_faces_off;

          // Look at all of the faces on the neighbour cell
          for (int ff2 = 0; ff2 < nfaces_by_neighbour; ++ff2) {
            const int neighbour_face_index =
                cells_to_faces[(neighbour_to_faces_off + ff2)];
            const int neighbour_face_to_nodes_off =
                faces_to_nodes_offsets[(neighbour_face_index)];
            const int nnodes_by_neighbour_face =
                faces_to_nodes_offsets[(neighbour_face_index + 1)] -
                neighbour_face_to_nodes_off;

            // We have found the adjoining face in the neighbour cell
            if (face_index != neighbour_face_index) {
              continue;
            }

            // Find the node that determines the subcell on this neighbour
            for (int nn2 = 0; nn2 < nnodes_by_neighbour_face; ++nn2) {
              if (faces_to_nodes[(neighbour_face_to_nodes_off + nn2)] ==
                  faces_to_nodes[(face_to_nodes_off + nn)]) {
                const int neighbour_subcell_index =
                    subcell_face_offsets[(neighbour_to_faces_off + ff)] + nn2;
                subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS)] =
                    neighbour_subcell_index;
              }
            }
          }
        } else {
          subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS)] = -1;
        }

        // Choose the right face node from our current 'subcell' face node
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        const int prev_node = ((nn == 0) ? nnodes_by_face - 1 : nn - 1);
        const int next_node = ((nn == nnodes_by_face - 1) ? 0 : nn + 1);
        const int lnode_off = (face_clockwise) ? next_node : prev_node;
        const int rnode_off = (face_clockwise) ? prev_node : next_node;
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + next_node)];

        /*
         * Find the internal face neighbour, with a brute force search...
         */
        for (int ff2 = 0; ff2 < nfaces_by_cell; ++ff2) {
          // Don't check the current face
          if (ff2 == ff) {
            continue;
          }

          const int face2_index = cells_to_faces[(cell_to_faces_off + ff2)];
          const int face2_to_nodes_off = faces_to_nodes_offsets[(face2_index)];
          const int nnodes_by_face2 =
              faces_to_nodes_offsets[(face2_index + 1)] - face2_to_nodes_off;

          // Check all nodes to see if we match
          for (int nn2 = 0; nn2 < nnodes_by_face2; ++nn2) {
            const int node_index2 = faces_to_nodes[(face2_to_nodes_off + nn2)];
            const int prev_node2 = ((nn2 == 0) ? nnodes_by_face2 - 1 : nn2 - 1);
            const int next_node2 = ((nn2 == nnodes_by_face2 - 1) ? 0 : nn2 + 1);
            const int prev_index =
                faces_to_nodes[(face2_to_nodes_off + prev_node2)];
            const int next_index =
                faces_to_nodes[(face2_to_nodes_off + next_node2)];

            // Check the nodes coincide on the face
            if (node_index == node_index2 &&
                (prev_index == rnode_index || next_index == rnode_index)) {

              const int node =
                  (prev_index == rnode_index) ? prev_node2 : next_node2;

              // Use the orientation of this face to determine how we set the
              // neighbouring subcell index
              const int n20 = faces_to_nodes[(face2_to_nodes_off + 0)];
              const int n21 = faces_to_nodes[(face2_to_nodes_off + 1)];
              const int n22 = faces_to_nodes[(face2_to_nodes_off + 2)];
              vec_t normal = {0.0, 0.0, 0.0};
              const int face2_clockwise =
                  calc_surface_normal(n20, n21, n22, nodes_x, nodes_y, nodes_z,
                                      &cell_centroid, &normal);

              subcell_to_neighbour_face[(subcell_index)] = face2_index;
              const int face_neighbour_subcell_index =
                  subcell_face_offsets[(cell_to_faces_off + ff2)] +
                  (face_clockwise != face2_clockwise ? nn2 : node);
              subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + 1)] =
                  face_neighbour_subcell_index;
            }
          }
        }

        /*
         * Store the left and right subcells on the same face
         */

        subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + 2)] =
            subcell_off + rnode_off;
        subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + 3)] =
            subcell_off + lnode_off;
      }
    }
  }
}
