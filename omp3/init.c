#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <math.h>

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(
    const int ncells, const int nnodes, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const double* density, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, double* cell_mass, double* subcell_mass,
    double* nodal_mass, int* cells_to_faces_offsets, int* cells_to_faces,
    int* faces_to_nodes_offsets, int* faces_to_nodes,
    int* faces_to_subcells_offsets, int* nodes_to_faces_offsets,
    int* nodes_to_faces, int* faces_to_cells0, int* faces_to_cells1) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
  double total_subcell_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass, total_subcell_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double cm = 0.0;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off =
          faces_to_subcells_offsets[(cell_to_faces_off + ff)];

      // Calculate the face center... SHOULD WE PRECOMPUTE?
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                    face_to_nodes_off, &face_c);

      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        // Fetch the nodes attached to our current node on
        // the current face
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
        total_subcell_mass += subcell_mass[(subcell_off + nn2)];
        cm += density[(cc)] * subcell_volume;
      }
    }

    cell_mass[(cc)] = cm;
    total_mass += cm;
  }
  STOP_PROFILING(&compute_profile, __func__);

  printf("Initial Total Mesh Mass: %.12f\n", total_mass);
  printf("Initial Total Subcell Mass: %.12f\n", total_subcell_mass);

  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;

    vec_t node_c = {nodes_x[(nn)], nodes_y[(nn)], nodes_z[(nn)]};

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

      // Find node center and location of current node on
      // face
      vec_t face_c = {0.0, 0.0, 0.0};
      int node_in_face_c;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c.x += nodes_x[(node_index)] / nnodes_by_face;
        face_c.y += nodes_y[(node_index)] / nnodes_by_face;
        face_c.z += nodes_z[(node_index)] / nnodes_by_face;

        // Choose the node in the list of nodes attached to
        // the face
        if (nn == node_index) {
          node_in_face_c = nn2;
        }
      }

      // Fetch the nodes attached to our current node on the
      // current face
      int nodes[2];
      nodes[0] = (node_in_face_c - 1 >= 0)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c - 1)]
                     : faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)];
      nodes[1] = (node_in_face_c + 1 < nnodes_by_face)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c + 1)]
                     : faces_to_nodes[(face_to_nodes_off)];

      // Fetch the cells attached to our current face
      int cells[2];
      cells[0] = faces_to_cells0[(face_index)];
      cells[1] = faces_to_cells1[(face_index)];

      // Add contributions from all of the cells attached to
      // the face
      for (int cc = 0; cc < 2; ++cc) {
        if (cells[(cc)] == -1) {
          continue;
        }

        // Add contributions for both edges attached to our
        // current node
        for (int nn2 = 0; nn2 < 2; ++nn2) {
          // Get the halfway point on the right edge
          vec_t half_edge = {0.5 * (nodes_x[(nodes[(nn2)])] + nodes_x[(nn)]),
                             0.5 * (nodes_y[(nodes[(nn2)])] + nodes_y[(nn)]),
                             0.5 * (nodes_z[(nodes[(nn2)])] + nodes_z[(nn)])};

          // Setup basis on plane of tetrahedron
          vec_t a = {(face_c.x - node_c.x), (face_c.y - node_c.y),
                     (face_c.z - node_c.z)};
          vec_t b = {(face_c.x - half_edge.x), (face_c.y - half_edge.y),
                     (face_c.z - half_edge.z)};
          vec_t ab = {(cell_centroids_x[(cells[cc])] - face_c.x),
                      (cell_centroids_y[(cells[cc])] - face_c.y),
                      (cell_centroids_z[(cells[cc])] - face_c.z)};

          // Calculate the area vector S using cross product
          vec_t A = {0.5 * (a.y * b.z - a.z * b.y),
                     -0.5 * (a.x * b.z - a.z * b.x),
                     0.5 * (a.x * b.y - a.y * b.x)};

          const double subcell_volume =
              fabs((ab.x * A.x + ab.y * A.y + ab.z * A.z) / 3.0);

          nodal_mass[(nn)] += density[(cells[(cc)])] * subcell_volume;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");
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

    vec_t cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cells_off, &cell_centroid);

    cell_centroids_x[(cc)] = cell_centroid.x;
    cell_centroids_y[(cc)] = cell_centroid.y;
    cell_centroids_z[(cc)] = cell_centroid.z;
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const int* faces_to_cells0, const int* faces_to_cells1,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, int* cells_to_faces_offsets,
    int* cells_to_faces, int* subcells_to_subcells,
    int* faces_to_subcells_offsets) {

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
      faces_to_subcells_offsets[(cell_to_faces_off + ff + 1)] = nnodes_by_face;
    }
  }

  // TODO: Can only do serially at the moment... Fix this
  // ideally
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      faces_to_subcells_offsets[(cell_to_faces_off + ff + 1)] +=
          faces_to_subcells_offsets[(cell_to_faces_off + ff)];
    }
  }

// This routine has ended up becoming messy and kludgy. It needs tidying up and
// fixing, but for now it needs to work as it is fundamental to the correct
// function of the application. It feels like it should be possible to re-remove
// all of the checks for whether the face is clockwise if we are able to make
// some better ordering or assumptions.
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

      const int subcell_off =
          faces_to_subcells_offsets[(cell_to_faces_off + ff)];

      // We have a subcell per node on a face
      for (int nn = 0; nn < nnodes_by_face; ++nn) {

        // Choose the right face node from our current 'subcell' face node
        const int subcell_index = subcell_off + nn;
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        const int next_node_off = ((nn == nnodes_by_face - 1) ? 0 : nn + 1);
        const int next_node_index =
            faces_to_nodes[(face_to_nodes_off + next_node_off)];

        /*
         * Find the subcell that exists in the neighbouring cell on this face.
         */
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

            // Check if we have found the adjoining face in the neighbour cell
            if (face_index != neighbour_face_index) {
              continue;
            }

            // Find the node that determines the subcell on this neighbour
            for (int nn2 = 0; nn2 < nnodes_by_neighbour_face; ++nn2) {
              const int neighbour_node_index =
                  faces_to_nodes[(neighbour_face_to_nodes_off + nn2)];

              if (neighbour_node_index == node_index) {
                subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS)] =
                    faces_to_subcells_offsets[(neighbour_to_faces_off + ff2)] +
                    nn2;
              }
            }
          }
        } else {
          subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS)] = -1;
        }

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

          // We have a subcell per node on a face
          for (int nn2 = 0; nn2 < nnodes_by_face2; ++nn2) {
            const int node_index2 = faces_to_nodes[(face2_to_nodes_off + nn2)];
            const int next_node_off2 =
                ((nn2 == nnodes_by_face2 - 1) ? 0 : nn2 + 1);
            const int next_node_index2 =
                faces_to_nodes[(face2_to_nodes_off + next_node_off2)];

            // Order is not guaranteed due to the fact that the faces are
            // declared clockwise or counterclockwise arbitrarily
            if ((node_index2 == node_index &&
                 next_node_index2 == next_node_index) ||
                (node_index2 == next_node_index &&
                 next_node_index2 == node_index)) {
              const int subcell_off2 =
                  faces_to_subcells_offsets[(cell_to_faces_off + ff2)];
              subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + 1)] =
                  subcell_off2 + nn2;
            }
          }
        }

        /*
         * Store the left and right subcells on the same face
         */
        const int prev_node_off = (nn == 0 ? nnodes_by_face - 1 : nn - 1);
        subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + 2)] =
            subcell_off + next_node_off;
        subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + 3)] =
            subcell_off + prev_node_off;
      }
    }
  }
}
