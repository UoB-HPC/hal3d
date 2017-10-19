#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <math.h>

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(const int ncells, const int nnodes,
                    const int nnodes_by_subcell, const double* density,
                    const double* nodes_x, const double* nodes_y,
                    const double* nodes_z, double* subcell_mass,
                    double* nodal_mass, int* cells_to_faces_offsets,
                    int* cells_to_faces, int* faces_to_nodes_offsets,
                    int* faces_to_nodes, int* faces_cclockwise_cell,
                    int* cells_offsets, int* cells_to_nodes,
                    int* subcells_to_faces_offsets, int* subcells_to_faces,
                    int* nodes_offsets, int* nodes_to_cells,
                    double* subcell_centroids_x, double* subcell_centroids_y,
                    double* subcell_centroids_z, double* subcell_volume,
                    double* cell_volume, double* nodal_volumes,
                    double* cell_mass) {

  // Calculates the cell volume, subcell volume and the subcell centroids
  calc_volumes_centroids(
      ncells, nnodes, nnodes_by_subcell, cells_offsets, cells_to_nodes,
      cells_to_faces_offsets, cells_to_faces, subcells_to_faces_offsets,
      subcells_to_faces, faces_to_nodes, faces_to_nodes_offsets,
      faces_cclockwise_cell, nodes_x, nodes_y, nodes_z, subcell_centroids_x,
      subcell_centroids_y, subcell_centroids_z, subcell_volume, cell_volume,
      nodal_volumes, nodes_offsets, nodes_to_cells);

  double total_mass_in_cells = 0.0;
  double total_mass_in_subcells = 0.0;

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // looping over corner subcells here
    double total_mass = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int subcell_index = cell_to_nodes_off + nn;
      subcell_mass[(subcell_index)] =
          density[(cc)] * subcell_volume[(subcell_index)];

      total_mass += subcell_mass[(subcell_index)];
      total_mass_in_subcells += subcell_mass[(subcell_index)];
    }

    cell_mass[(cc)] = total_mass;
    total_mass_in_cells += cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, __func__);

  printf("Total Mass in Cells    %.12f\n", total_mass_in_cells);
  printf("Total Mass in Subcells %.12f\n", total_mass_in_subcells);
  printf("Difference             %.12f\n\n",
         total_mass_in_subcells - total_mass_in_cells);

#if 0
#pragma omp parallel for
#endif // if 0
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - node_to_cells_off;

    nodal_mass[(nn)] = 0.0;

    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_offsets[(cell_index + 1)] - cell_to_nodes_off;

      for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
          const int subcell_index = cell_to_nodes_off + nn2;
          nodal_mass[(nn)] += subcell_mass[(subcell_index)];
          break;
        }
      }
    }
  }

  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");
}

// Calculates the cell volume, subcell volume and the subcell centroids
void calc_volumes_centroids(
    const int ncells, const int nnodes, const int nnodes_by_subcell,
    const int* cells_offsets, const int* cells_to_nodes,
    const int* cells_to_faces_offsets, const int* cells_to_faces,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes, const int* faces_to_nodes_offsets,
    const int* faces_cclockwise_cell, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* subcell_volume, double* cell_volume, double* nodal_volumes,
    int* nodes_offsets, int* nodes_to_cells) {

  double total_cell_volume = 0.0;
  double total_subcell_volume = 0.0;
#if 0
#pragma omp parallel for reduction(+ : total_cell_volume, total_subcell_volume)
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // Calculates the weighted volume dist for a provided cell along x-y-z
    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_c);

    total_cell_volume += cell_volume[(cc)];

    // Looping over corner subcells here
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      subcell_centroids_x[(subcell_index)] = 0.0;
      subcell_centroids_y[(subcell_index)] = 0.0;
      subcell_centroids_z[(subcell_index)] = 0.0;

      // Consider all faces attached to node
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // The face centroid is the same for all nodes on the face
        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                      face_to_nodes_off, &face_c);

        const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);

        // Determine the position of the node in the face list of nodes
        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
        const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
        const int rnode_off = (face_clockwise ? prev_node : next_node);
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];

        subcell_centroids_x[(subcell_index)] +=
            0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]) + face_c.x;
        subcell_centroids_y[(subcell_index)] +=
            0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]) + face_c.y;
        subcell_centroids_z[(subcell_index)] +=
            0.5 * (nodes_z[(node_index)] + nodes_z[(rnode_index)]) + face_c.z;
      }

      subcell_centroids_x[(subcell_index)] =
          (subcell_centroids_x[(subcell_index)] + cell_c.x +
           nodes_x[(node_index)]) /
          nnodes_by_subcell;
      subcell_centroids_y[(subcell_index)] =
          (subcell_centroids_y[(subcell_index)] + cell_c.y +
           nodes_y[(node_index)]) /
          nnodes_by_subcell;
      subcell_centroids_z[(subcell_index)] =
          (subcell_centroids_z[(subcell_index)] + cell_c.z +
           nodes_z[(node_index)]) /
          nnodes_by_subcell;

      vec_t subcell_c = {subcell_centroids_x[(subcell_index)],
                         subcell_centroids_y[(subcell_index)],
                         subcell_centroids_z[(subcell_index)]};

      subcell_volume[(subcell_index)] = 0.0;

      // Consider all faces attached to node
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // The face centroid is the same for all nodes on the face
        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                      face_to_nodes_off, &face_c);

        const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);

        // Determine the position of the node in the face list of nodes
        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
        const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
        const int rnode_off = (face_clockwise ? prev_node : next_node);
        const int lnode_off = (face_clockwise ? next_node : prev_node);
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];
        const int lnode_index = faces_to_nodes[(face_to_nodes_off + lnode_off)];

        /* EXTERNAL FACE */

        const int subcell_faces_to_nodes[NNODES_BY_SUBCELL_FACE] = {0, 1, 2, 3};

        double enodes_x[NNODES_BY_SUBCELL_FACE] = {
            nodes_x[(node_index)],
            0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]), face_c.x,
            0.5 * (nodes_x[(node_index)] + nodes_x[(lnode_index)])};
        double enodes_y[NNODES_BY_SUBCELL_FACE] = {
            nodes_y[(node_index)],
            0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]), face_c.y,
            0.5 * (nodes_y[(node_index)] + nodes_y[(lnode_index)])};
        double enodes_z[NNODES_BY_SUBCELL_FACE] = {
            nodes_z[(node_index)],
            0.5 * (nodes_z[(node_index)] + nodes_z[(rnode_index)]), face_c.z,
            0.5 * (nodes_z[(node_index)] + nodes_z[(lnode_index)])};

        contribute_face_volume(NNODES_BY_SUBCELL_FACE, subcell_faces_to_nodes,
                               enodes_x, enodes_y, enodes_z, &subcell_c,
                               &subcell_volume[(subcell_index)]);

        /* INTERNAL FACE */

        const int r_face_off = (ff == nfaces_by_subcell - 1) ? 0 : ff + 1;
        const int l_face_off = (ff == 0) ? nfaces_by_subcell - 1 : ff - 1;
        const int r_face_index =
            subcells_to_faces[(subcell_to_faces_off + r_face_off)];
        const int l_face_index =
            subcells_to_faces[(subcell_to_faces_off + l_face_off)];
        const int r_face_to_nodes_off = faces_to_nodes_offsets[(r_face_index)];
        const int l_face_to_nodes_off = faces_to_nodes_offsets[(l_face_index)];
        const int nnodes_by_rface =
            faces_to_nodes_offsets[(r_face_index + 1)] - r_face_to_nodes_off;
        const int nnodes_by_lface =
            faces_to_nodes_offsets[(l_face_index + 1)] - l_face_to_nodes_off;

        vec_t rface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_rface, nodes_x, nodes_y, nodes_z,
                      faces_to_nodes, r_face_to_nodes_off, &rface_c);

        const int r_face_clockwise =
            (faces_cclockwise_cell[(r_face_index)] != cc);

        // Determine the position of the node in the face list of nodes
        for (nn2 = 0; nn2 < nnodes_by_rface; ++nn2) {
          if (faces_to_nodes[(r_face_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        const int rface_next_node = (nn2 == nnodes_by_rface - 1) ? 0 : nn2 + 1;
        const int rface_prev_node = (nn2 == 0) ? nnodes_by_rface - 1 : nn2 - 1;
        const int rface_rnode_off =
            (r_face_clockwise ? rface_prev_node : rface_next_node);
        const int rface_rnode_index =
            faces_to_nodes[(r_face_to_nodes_off + rface_rnode_off)];

        vec_t lface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_lface, nodes_x, nodes_y, nodes_z,
                      faces_to_nodes, l_face_to_nodes_off, &lface_c);

        double inodes_x[NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_x[(node_index)] + nodes_x[(rface_rnode_index)]),
            rface_c.x, cell_c.x, lface_c.x};
        double inodes_y[NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_y[(node_index)] + nodes_y[(rface_rnode_index)]),
            rface_c.y, cell_c.y, lface_c.y};
        double inodes_z[NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_z[(node_index)] + nodes_z[(rface_rnode_index)]),
            rface_c.z, cell_c.z, lface_c.z};

        contribute_face_volume(NNODES_BY_SUBCELL_FACE, subcell_faces_to_nodes,
                               inodes_x, inodes_y, inodes_z, &subcell_c,
                               &subcell_volume[(subcell_index)]);

        if (isnan(subcell_volume[(subcell_index)])) {
          subcell_volume[(subcell_index)] = 0.0;
          break;
        }
      }

      subcell_volume[(subcell_index)] = fabs(subcell_volume[(subcell_index)]);
      total_subcell_volume += subcell_volume[(subcell_index)];
    }
  }

#if 0
#pragma omp parallel for
#endif // if 0
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - node_to_cells_off;

    nodal_volumes[(nn)] = 0.0;

    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_offsets[(cell_index + 1)] - cell_to_nodes_off;

      for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
          const int subcell_index = cell_to_nodes_off + nn2;
          nodal_volumes[(nn)] += subcell_volume[(subcell_index)];
          break;
        }
      }
    }
  }

  printf("Total Cell Volume    %.12f\n", total_cell_volume);
  printf("Total Subcell Volume %.12f\n", total_subcell_volume);
}

// Initialises the centroids for each cell
void init_cell_centroids(const int ncells, const int* cells_offsets,
                         const int* cells_to_nodes, const double* nodes_x,
                         const double* nodes_y, const double* nodes_z,
                         double* cell_centroids_x, double* cell_centroids_y,
                         double* cell_centroids_z) {

  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#if 0
#pragma omp parallel for
#endif // if 0
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

void init_subcells_to_faces(
    const int ncells, const int nsubcells, const int* cells_offsets,
    const int* nodes_to_faces_offsets, const int* cells_to_nodes,
    const int* faces_to_cells0, const int* faces_to_cells1,
    const int* nodes_to_faces, const int* faces_to_nodes,
    const int* faces_to_nodes_offsets, const int* faces_cclockwise_cell,
    int* subcells_to_faces, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, int* subcells_to_faces_offsets) {

#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;
      const int subcell_index = cell_to_nodes_off + nn;

      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if (face_index != -1 && (faces_to_cells0[(face_index)] == cc ||
                                 faces_to_cells1[(face_index)] == cc)) {
          subcells_to_faces_offsets[(subcell_index + 1)]++;
        }
      }
    }
  }

  // TODO: CAN WE IMPROVE THIS PREFIX SUM?
  for (int ss = 0; ss < nsubcells; ++ss) {
    subcells_to_faces_offsets[(ss + 1)] += subcells_to_faces_offsets[(ss)];
  }

#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_c);

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;
      const int subcell_index = cell_to_nodes_off + nn;

      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // Fetch all of the faces that are attached to a subcell
      int f = 0;
      int faces[nfaces_by_subcell];
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if (faces_to_cells0[(face_index)] != cc &&
            faces_to_cells1[(face_index)] != cc) {
          continue;
        }

        if (face_index != -1) {
          faces[(f++)] = face_index;
        }
      }

      subcells_to_faces[(subcell_to_faces_off)] = faces[0];

      for (int ff = 0; ff < nfaces_by_subcell - 1; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                      face_to_nodes_off, &face_c);

        const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);
        const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
        const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
        const int rnode_off = (face_clockwise ? prev_node : next_node);
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];

        subcells_to_faces[(subcell_to_faces_off + ff + 1)] = -1;

        // Essentially have to perform a search through the faces.
        for (int ff2 = 1; ff2 < nfaces_by_subcell; ++ff2) {
          const int face_index2 = faces[(ff2)];

          if (face_index == face_index2) {
            continue;
          }

          const int face_to_nodes_off2 = faces_to_nodes_offsets[(face_index2)];
          const int nnodes_by_face2 =
              faces_to_nodes_offsets[(face_index2 + 1)] - face_to_nodes_off2;

          // Check if this face shares an edge
          for (int nn2 = 0; nn2 < nnodes_by_face2; ++nn2) {
            if (faces_to_nodes[(face_to_nodes_off2 + nn2)] == rnode_index) {
              subcells_to_faces[(subcell_to_faces_off + ff + 1)] = face_index2;
              break;
            }
          }

          if (subcells_to_faces[(subcell_to_faces_off + ff + 1)] != -1) {
            break;
          }
        }
      }
    }
  }
}

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const int nsubcells, const int* faces_to_cells0,
    const int* faces_to_cells1, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const int* faces_cclockwise_cell,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    int* subcells_to_subcells, int* subcells_to_subcells_offsets,
    int* cells_offsets, int* nodes_to_faces_offsets, int* nodes_to_faces,
    int* cells_to_nodes, int* subcells_to_faces,
    int* subcells_to_faces_offsets) {

#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;
      const int subcell_index = cell_to_nodes_off + nn;

      // Consider all faces attached to node
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if (face_index != -1 && (faces_to_cells0[(face_index)] == cc ||
                                 faces_to_cells1[(face_index)] == cc)) {
          // A pair of subcell neighbours for every face
          subcells_to_subcells_offsets[(subcell_index + 1)] += 2;
        }
      }
    }
  }

  for (int ss = 0; ss < nsubcells; ++ss) {
    subcells_to_subcells_offsets[(ss + 1)] +=
        subcells_to_subcells_offsets[(ss)];
  }

#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_c);

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      const int subcell_to_subcells_off =
          subcells_to_subcells_offsets[(subcell_index)];
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // Consider all faces attached to node
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
        const int neighbour_cell_index = (faces_to_cells0[(face_index)] == cc)
                                             ? faces_to_cells1[(face_index)]
                                             : faces_to_cells0[(face_index)];

        // We can consider two neighbour contributions loosely associated with a
        // face, the external and the internal

        /* INTERNAL NEIGHBOUR */

        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                      face_to_nodes_off, &face_c);

        const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);

        // We have to find our position on the face
        // The nodes will mostly be in cache anyway so should be cheap
        int rnode_index;
        for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
            const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
            const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
            const int rnode_off = (face_clockwise) ? prev_node : next_node;
            rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];
            break;
          }
        }

        // Now we have to work back to find our location in the cell
        for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
          if (cells_to_nodes[(cell_to_nodes_off + nn2)] == rnode_index) {
            subcells_to_subcells[(subcell_to_subcells_off + ff * 2)] =
                cell_to_nodes_off + nn2;
            break;
          }
        }

        /* EXTERNAL NEIGHBOUR */

        if (neighbour_cell_index == -1) {
          subcells_to_subcells[(subcell_to_subcells_off + ff * 2 + 1)] = -1;
          continue;
        }

        const int neighbour_to_nodes_off =
            cells_offsets[(neighbour_cell_index)];
        const int nnodes_by_neighbour =
            cells_offsets[(neighbour_cell_index + 1)] - neighbour_to_nodes_off;

        // NOTE: Cells to nodes is essentially cells to subcells here
        for (int nn2 = 0; nn2 < nnodes_by_neighbour; ++nn2) {
          const int neighbour_subcell_index = neighbour_to_nodes_off + nn2;
          const int neighbour_node_index =
              cells_to_nodes[(neighbour_subcell_index)];
          if (neighbour_node_index == node_index) {
            subcells_to_subcells[(subcell_to_subcells_off + ff * 2 + 1)] =
                neighbour_subcell_index;
            break;
          }
        }
      }
    }
  }
}

// NOTE: This is not intended to be a production device, rather used for
// debugging the code against a well tested description of the subcell mesh.
void init_subcell_data_structures(Mesh* mesh, HaleData* hale_data,
                                  UnstructuredMesh* umesh) {

  const int nx = mesh->local_nx;
  const int ny = mesh->local_ny;
  const int nz = mesh->local_nz;

  // Construct the subcell mesh description
  const int subcell_half_nodes_x_off = (nx + 1) * (ny + 1) * (nz + 1);
  const int subcell_half_nodes_y_off =
      subcell_half_nodes_x_off + (nx) * (ny + 1) * (nz + 1);
  const int subcell_half_nodes_z_off =
      subcell_half_nodes_y_off + (nx + 1) * (ny) * (nz + 1);
  const int subcell_face_c_xy_off =
      subcell_half_nodes_z_off + (nx + 1) * (ny + 1) * (nz);
  const int subcell_face_c_yz_off = subcell_face_c_xy_off + nx * ny * (nz + 1);
  const int subcell_face_c_zx_off = subcell_face_c_yz_off + (nx + 1) * ny * nz;
  const int subcell_cell_c_off = subcell_face_c_zx_off + nx * (ny + 1) * nz;
  const int nsubcell_nodes = subcell_cell_c_off + nx * ny * nz;
  hale_data->nsubcell_nodes = nsubcell_nodes;

  if (!hale_data->subcell_nodes_x) {
    size_t allocated =
        allocate_data(&hale_data->subcell_nodes_x, nsubcell_nodes);
    allocated += allocate_data(&hale_data->subcell_nodes_y, nsubcell_nodes);
    allocated += allocate_data(&hale_data->subcell_nodes_z, nsubcell_nodes);
    allocated +=
        allocate_int_data(&hale_data->subcells_to_nodes,
                          umesh->ncells * hale_data->nsubcells_by_cell *
                              hale_data->nnodes_by_subcell);
    printf("Allocated %.4lf GB for subcell debugging output\n", allocated / GB);
  }

// Determine subcell connectivity in a planar fashion
#define NODE_IND(i, j, k) ((i) * (nx + 1) * (ny + 1) + (j) * (nx + 1) + (k))
#define HALF_NODE_X_IND(i, j, k)                                               \
  (subcell_half_nodes_x_off + ((i) * (nx) * (ny + 1) + (j) * (nx) + (k)))
#define HALF_NODE_Y_IND(i, j, k)                                               \
  (subcell_half_nodes_y_off + ((i) * (nx + 1) * (ny) + (j) * (nx + 1) + (k)))
#define HALF_NODE_Z_IND(i, j, k)                                               \
  (subcell_half_nodes_z_off +                                                  \
   ((i) * (nx + 1) * (ny + 1) + (j) * (nx + 1) + (k)))
#define FACE_C_XY_IND(i, j, k)                                                 \
  (subcell_face_c_xy_off + ((i)*nx * ny + (j)*nx + (k)))
#define FACE_C_YZ_IND(i, j, k)                                                 \
  (subcell_face_c_yz_off + ((i) * (nx + 1) * ny + (j) * (nx + 1) + (k)))
#define FACE_C_ZX_IND(i, j, k)                                                 \
  (subcell_face_c_zx_off + ((i)*nx * (ny + 1) + (j)*nx + (k)))
#define CELL_C_IND(i, j, k) (subcell_cell_c_off + ((i)*nx * ny + (j)*nx + (k)))

// Construct the nodal positions
#if 0
#pragma omp parallel for
#endif // if 0
  for (int ii = 0; ii < nz + 1; ++ii) {
    for (int jj = 0; jj < ny + 1; ++jj) {
      for (int kk = 0; kk < nx + 1; ++kk) {
        // Corner nodes
        hale_data->subcell_nodes_x[NODE_IND(ii, jj, kk)] =
            umesh->nodes_x0[(NODE_IND(ii, jj, kk))];
        hale_data->subcell_nodes_y[NODE_IND(ii, jj, kk)] =
            umesh->nodes_y0[(NODE_IND(ii, jj, kk))];
        hale_data->subcell_nodes_z[NODE_IND(ii, jj, kk)] =
            umesh->nodes_z0[(NODE_IND(ii, jj, kk))];

        if (kk < nx) {
          hale_data->subcell_nodes_x[HALF_NODE_X_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_x0[(NODE_IND(ii, jj, kk + 1))]);
          hale_data->subcell_nodes_y[HALF_NODE_X_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_y0[(NODE_IND(ii, jj, kk + 1))]);
          hale_data->subcell_nodes_z[HALF_NODE_X_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_z0[(NODE_IND(ii, jj, kk + 1))]);
        }

        if (jj < ny) {
          hale_data->subcell_nodes_x[HALF_NODE_Y_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk))]);
          hale_data->subcell_nodes_y[HALF_NODE_Y_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk))]);
          hale_data->subcell_nodes_z[HALF_NODE_Y_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk))]);
        }

        if (ii < nz) {
          hale_data->subcell_nodes_x[HALF_NODE_Z_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk))]);
          hale_data->subcell_nodes_y[HALF_NODE_Z_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk))]);
          hale_data->subcell_nodes_z[HALF_NODE_Z_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk))]);
        }

        if (kk < nx && jj < ny) {
          hale_data->subcell_nodes_x[(FACE_C_XY_IND(ii, jj, kk))] =
              (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk))]) /
              4.0;
          hale_data->subcell_nodes_y[(FACE_C_XY_IND(ii, jj, kk))] =
              (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk))]) /
              4.0;
          hale_data->subcell_nodes_z[(FACE_C_XY_IND(ii, jj, kk))] =
              (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk))]) /
              4.0;
        }

        if (jj < ny && ii < nz) {
          hale_data->subcell_nodes_x[(FACE_C_YZ_IND(ii, jj, kk))] =
              (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj + 1, kk))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
          hale_data->subcell_nodes_y[(FACE_C_YZ_IND(ii, jj, kk))] =
              (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj + 1, kk))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
          hale_data->subcell_nodes_z[(FACE_C_YZ_IND(ii, jj, kk))] =
              (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj + 1, kk))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
        }

        if (kk < nx && ii < nz) {
          hale_data->subcell_nodes_x[(FACE_C_ZX_IND(ii, jj, kk))] =
              (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
          hale_data->subcell_nodes_y[(FACE_C_ZX_IND(ii, jj, kk))] =
              (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
          hale_data->subcell_nodes_z[(FACE_C_ZX_IND(ii, jj, kk))] =
              (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
        }

        if (ii < nz && jj < ny && kk < nx) {
          hale_data->subcell_nodes_x[CELL_C_IND(ii, jj, kk)] =
              (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj + 1, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj + 1, kk))]) /
              8.0;
          hale_data->subcell_nodes_y[CELL_C_IND(ii, jj, kk)] =
              (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj + 1, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj + 1, kk))]) /
              8.0;
          hale_data->subcell_nodes_z[CELL_C_IND(ii, jj, kk)] =
              (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj + 1, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj + 1, kk))]) /
              8.0;
        }
      }
    }
  }

#if 0
#pragma omp parallel for
#endif // if 0
  for (int ii = 0; ii < nz; ++ii) {
    for (int jj = 0; jj < ny; ++jj) {
      for (int kk = 0; kk < nx; ++kk) {
        const int cell_index = (ii * nx * ny + jj * nx + kk);
        const int c_off = cell_index * (hale_data->nnodes_by_subcell *
                                        hale_data->nsubcells_by_cell);

        // Bottom left subcell
        hale_data->subcells_to_nodes[(c_off + 0)] = NODE_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 1)] = HALF_NODE_X_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 2)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 3)] = HALF_NODE_Y_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 4)] = HALF_NODE_Z_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 5)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 6)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 7)] = FACE_C_YZ_IND(ii, jj, kk);

        // Bottom right subcell
        hale_data->subcells_to_nodes[(c_off + 8)] = HALF_NODE_X_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 9)] = NODE_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 10)] =
            HALF_NODE_Y_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 11)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 12)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 13)] =
            HALF_NODE_Z_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 14)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 15)] = CELL_C_IND(ii, jj, kk);

        // Top right subcell
        hale_data->subcells_to_nodes[(c_off + 16)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 17)] =
            HALF_NODE_Y_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 18)] =
            NODE_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 19)] =
            HALF_NODE_X_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 20)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 21)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 22)] =
            HALF_NODE_Z_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 23)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);

        // Top left subcell
        hale_data->subcells_to_nodes[(c_off + 24)] =
            HALF_NODE_Y_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 25)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 26)] =
            HALF_NODE_X_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 27)] = NODE_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 28)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 29)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 30)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 31)] =
            HALF_NODE_Z_IND(ii, jj + 1, kk);

        // Bottom back left subcell
        hale_data->subcells_to_nodes[(c_off + 32)] =
            HALF_NODE_Z_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 33)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 34)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 35)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 36)] = NODE_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 37)] =
            HALF_NODE_X_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 38)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 39)] =
            HALF_NODE_Y_IND(ii + 1, jj, kk);

        // Bottom back right subcell
        hale_data->subcells_to_nodes[(c_off + 40)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 41)] =
            HALF_NODE_Z_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 42)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 43)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 44)] =
            HALF_NODE_X_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 45)] =
            NODE_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 46)] =
            HALF_NODE_Y_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 47)] =
            FACE_C_XY_IND(ii + 1, jj, kk);

        // Top back right subcell
        hale_data->subcells_to_nodes[(c_off + 48)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 49)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 50)] =
            HALF_NODE_Z_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 51)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 52)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 53)] =
            HALF_NODE_Y_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 54)] =
            NODE_IND(ii + 1, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 55)] =
            HALF_NODE_X_IND(ii + 1, jj + 1, kk);

        // Top back left subcell
        hale_data->subcells_to_nodes[(c_off + 56)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 57)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 58)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 59)] =
            HALF_NODE_Z_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 60)] =
            HALF_NODE_Y_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 61)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 62)] =
            HALF_NODE_X_IND(ii + 1, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 63)] =
            NODE_IND(ii + 1, jj + 1, kk);
      }
    }
  }
}
