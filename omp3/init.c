#include "../../shared.h"
#include "../hale_data.h"
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
                    int* faces_to_nodes_offsets, int* faces_to_nodes) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Calculate the face center... SHOULD WE PRECOMPUTE?
      double face_c_x = 0.0;
      double face_c_y = 0.0;
      double face_c_z = 0.0;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c_x += nodes_x[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z[(node_index)] / nnodes_by_face;
      }

      // Now we will sum the contributions at each of the nodes
      // TODO: THERE IS SOME SYMMETRY HERE THAT MEANS WE MIGHT BE ABLE TO
      // OPTIMISE
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
        const double a_x = (half_edge_x - face_c_x);
        const double a_y = (half_edge_y - face_c_y);
        const double a_z = (half_edge_z - face_c_z);
        const double b_x = (cell_centroids_x[(cc)] - face_c_x);
        const double b_y = (cell_centroids_y[(cc)] - face_c_y);
        const double b_z = (cell_centroids_z[(cc)] - face_c_z);

        // Calculate the area vector S using cross product
        const double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        const double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        const double S_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: WE MULTIPLY BY 2 HERE BECAUSE WE ARE ADDING THE VOLUME TO
        // BOTH
        // THE CURRENT AND NEXT NODE, OTHERWISE WE ONLY ACCOUNT FOR HALF OF
        // THE
        // 'HALF' TETRAHEDRONS
        double subcell_volume =
            fabs(2.0 * ((half_edge_x - nodes_x[(current_node)]) * S_x +
                        (half_edge_y - nodes_y[(current_node)]) * S_y +
                        (half_edge_z - nodes_z[(current_node)]) * S_z) /
                 3.0);

        // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER CLOSED
        // FORM SOLUTION?
        for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
          if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
            subcell_mass[(cell_to_nodes_off + nn3)] +=
                density[(cc)] * subcell_volume;
          }
        }

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

// Initialise the subcells to faces connectivity list
void init_subcells_to_faces(
    const int ncells, const int* cells_offsets, const int* cells_to_nodes,
    const int* cells_to_faces_offsets, const int* cells_to_faces,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    int* subcells_to_faces_offsets, int* subcells_to_faces) {

// NOTE: Some of these steps might be mergable, but I feel liek the current
// implementation leads to a better read through of the code
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // We will calculate the flux at every face of the subcells
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + ss)];

      // Find all of the faces that share the node (was slightly easier to
      // understand if this and subsequent step were separated)
      int nfaces_on_node = 0;
      for (int ff = 0; ff < nfaces_by_cell; ++ff) {
        const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Attempt to the node on the face
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (faces_to_nodes[(face_to_nodes_off + nn)] == node_index) {
            // We found a touching face
            nfaces_on_node++;
            break;
          }
        }
      }

      subcells_to_faces_offsets[(cell_to_nodes_off + ss + 1)] = nfaces_on_node;
    }
  }

  // TODO: This is another serial conversion from counts to offsets. Need to
  // find a way of paralellising these.
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_subcells_off = cells_offsets[(cc)];
    const int nsubcells_by_cell =
        cells_offsets[(cc + 1)] - cell_to_subcells_off;

    // We will calculate the flux at every face of the subcells
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      subcells_to_faces_offsets[(cell_to_subcells_off + ss + 1)] +=
          subcells_to_faces_offsets[(cell_to_subcells_off + ss)];
    }
  }

// NOTE: Some of these steps might be mergable, but I feel liek the current
// implementation leads to a better read through of the code
// We also do too much work in this, as we have knowledge about those faces
// that have already been processed, but this should be quite minor overall
// and it's and initialisation function so just keep an eye on the
// initialisation performance.
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    // The list of face indices attached to a node
    int faces_on_node[] = {-1, -1, -1, -1};
    int face_rorientation[] = {0, 0, 0, 0};

    // We will calculate the flux at every face of the subcells
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + ss)];

      // Find all of the faces that share the node (was slightly easier to
      // understand if this and subsequent step were separated)
      int nfaces_on_node = 0;
      for (int ff = 0; ff < nfaces_by_cell; ++ff) {
        const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Attempt to find the node on the face
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (faces_to_nodes[(face_to_nodes_off + nn)] == node_index) {
            // TODO: This is duplicate, and all of this just to determine
            // orientation is annoying ><

            // Get two vectors on the face plane
            vec_t dn0 = {0.0, 0.0, 0.0};
            vec_t dn1 = {0.0, 0.0, 0.0};
            dn0.x = nodes_x[(faces_to_nodes[(face_to_nodes_off + 2)])] -
                    nodes_x[faces_to_nodes[(face_to_nodes_off + 1)]];
            dn0.y = nodes_y[(faces_to_nodes[(face_to_nodes_off + 2)])] -
                    nodes_y[faces_to_nodes[(face_to_nodes_off + 1)]];
            dn0.z = nodes_z[(faces_to_nodes[(face_to_nodes_off + 2)])] -
                    nodes_z[faces_to_nodes[(face_to_nodes_off + 1)]];
            dn1.x = nodes_x[(faces_to_nodes[(face_to_nodes_off + 1)])] -
                    nodes_x[faces_to_nodes[(face_to_nodes_off + 0)]];
            dn1.y = nodes_y[(faces_to_nodes[(face_to_nodes_off + 1)])] -
                    nodes_y[faces_to_nodes[(face_to_nodes_off + 0)]];
            dn1.z = nodes_z[(faces_to_nodes[(face_to_nodes_off + 1)])] -
                    nodes_z[faces_to_nodes[(face_to_nodes_off + 0)]];

            // Calculate a vector from face to cell centroid
            vec_t ab;
            ab.x = (cell_centroid.x -
                    nodes_x[(faces_to_nodes[(face_to_nodes_off)])]);
            ab.y = (cell_centroid.y -
                    nodes_y[(faces_to_nodes[(face_to_nodes_off)])]);
            ab.z = (cell_centroid.z -
                    nodes_z[(faces_to_nodes[(face_to_nodes_off)])]);

            // Cross product to get the normal
            vec_t normal;
            normal.x = (dn0.y * dn1.z - dn0.z * dn1.y);
            normal.y = (dn0.z * dn1.x - dn0.x * dn1.z);
            normal.z = (dn0.x * dn1.y - dn0.y * dn1.x);
            face_rorientation[(nfaces_on_node)] =
                (ab.x * normal.x + ab.y * normal.y + ab.z * normal.z < 0.0);
            faces_on_node[(nfaces_on_node++)] = face_index;
            break;
          }
        }
      }

      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(cell_to_nodes_off + ss)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(cell_to_nodes_off + ss + 1)] -
          subcell_to_faces_off;

      // Look at all of the faces we have discovered so far and see if
      // there is a connection between the faces
      subcells_to_faces[(subcell_to_faces_off)] = faces_on_node[(0)];
      int previous_fn = 0;
      for (int fn = 0; fn < nfaces_by_subcell - 1; ++fn) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + fn)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Re-find the node on the face
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (faces_to_nodes[(face_to_nodes_off + nn)] == node_index) {
            int side_node;
            if (face_rorientation[(previous_fn)]) {
              side_node =
                  faces_to_nodes[(face_to_nodes_off +
                                  ((nn == nnodes_by_face - 1) ? 0 : nn + 1))];
            } else {
              side_node =
                  faces_to_nodes[(face_to_nodes_off +
                                  ((nn == 0) ? nnodes_by_face - 1 : nn - 1))];
            }

            // Find all of the faces that connect to this face in the list
            for (int fn2 = 0; fn2 < nfaces_by_subcell; ++fn2) {
              const int face_index2 = faces_on_node[(fn2)];

              // No self connectivity
              if (face_index2 == face_index) {
                continue;
              }

              const int face_to_nodes_off2 =
                  faces_to_nodes_offsets[(face_index2)];
              const int nnodes_by_face2 =
                  faces_to_nodes_offsets[(face_index2 + 1)] -
                  face_to_nodes_off2;

              // Check whether the face is connected
              for (int nn2 = 0; nn2 < nnodes_by_face2; ++nn2) {
                if (faces_to_nodes[(face_to_nodes_off2 + nn2)] == side_node) {
                  subcells_to_faces[(subcell_to_faces_off + fn + 1)] =
                      face_index2;
                  previous_fn = fn2;
                  break;
                }
              }
            }

            break;
          }
        }
      }
    }
  }
}

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const int* cells_offsets, const int* cells_to_nodes,
    const int* faces_to_cells0, const int* faces_to_cells1,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const int* subcells_to_faces_offsets,
    const int* subcells_to_faces, int* subcells_to_subcells) {

// This routine uses the previously described subcell faces to determine a ring
// ordering for the subcell neighbours. Essentially, each subcell has a pair of
// neighbours for every external face, and those neighbours are stored in the
// same order as the faces.
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_in_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    // For every face on a subcell we have a pair of neighbours that are
    // attached
    for (int ss = 0; ss < nsubcells_in_cell; ++ss) {
      const int subcell_index = (cell_to_nodes_off + ss);
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // For every face there are two faces
      const int subcell_to_subcells_off = subcell_to_faces_off * 2;

      // Look at all of the faces to find the pair of neighbours associated with
      // each of them
      int neighbour_index = 0;
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Determine the neighbouring cell
        const int neighbour_cell_index = (faces_to_cells0[(face_index)] == cc)
                                             ? faces_to_cells1[(face_index)]
                                             : faces_to_cells0[(face_index)];

        // Pick the neighbouring subcell from the adjoining cell
        if (neighbour_cell_index != -1) {
          const int neighbour_to_nodes_off =
              cells_offsets[(neighbour_cell_index)];
          const int nnodes_by_neighbour_cell =
              cells_offsets[(neighbour_cell_index + 1)] -
              neighbour_to_nodes_off;
          for (int nn = 0; nn < nnodes_by_neighbour_cell; ++nn) {
            if (cells_to_nodes[(neighbour_to_nodes_off + nn)] ==
                cells_to_nodes[(subcell_index)]) {
              subcells_to_subcells[(subcell_to_subcells_off +
                                    neighbour_index++)] =
                  neighbour_to_nodes_off + nn;
              break;
            }
          }
        } else {
          subcells_to_subcells[(subcell_to_subcells_off + neighbour_index++)] =
              -1;
        }

        // We again need to determine the orientation in order to calculate the
        // correct right handed node
        vec_t dn0 = {0.0, 0.0, 0.0};
        vec_t dn1 = {0.0, 0.0, 0.0};
        const int fn_off0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int fn_off1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int fn_off2 = faces_to_nodes[(face_to_nodes_off + 2)];
        dn0.x = nodes_x[(fn_off2)] - nodes_x[(fn_off1)];
        dn0.y = nodes_y[(fn_off2)] - nodes_y[(fn_off1)];
        dn0.z = nodes_z[(fn_off2)] - nodes_z[(fn_off1)];
        dn1.x = nodes_x[(fn_off1)] - nodes_x[(fn_off0)];
        dn1.y = nodes_y[(fn_off1)] - nodes_y[(fn_off0)];
        dn1.z = nodes_z[(fn_off1)] - nodes_z[(fn_off0)];

        // Calculate a vector from face to cell centroid
        vec_t ab;
        ab.x = (cell_centroid.x - nodes_x[(fn_off0)]);
        ab.y = (cell_centroid.y - nodes_y[(fn_off0)]);
        ab.z = (cell_centroid.z - nodes_z[(fn_off0)]);

        // Cross product to get the normal
        vec_t normal;
        normal.x = (dn0.y * dn1.z - dn0.z * dn1.y);
        normal.y = (dn0.z * dn1.x - dn0.x * dn1.z);
        normal.z = (dn0.x * dn1.y - dn0.y * dn1.x);
        const int face_rorientation =
            (ab.x * normal.x + ab.y * normal.y + ab.z * normal.z < 0.0);

        // Determine the right oriented next node after the subcell node on the
        // face that represents the subcell node in the cell
        int rnode;
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (faces_to_nodes[(face_to_nodes_off + nn)] ==
              cells_to_nodes[(subcell_index)]) {
            const int loff = ((nn == 0) ? nnodes_by_face - 1 : nn - 1);
            const int roff = ((nn == nnodes_by_face - 1) ? 0 : nn + 1);
            rnode = (face_rorientation)
                        ? faces_to_nodes[(face_to_nodes_off + roff)]
                        : faces_to_nodes[(face_to_nodes_off + loff)];
            break;
          }
        }

        // Search for the node (subcell) in the cell
        for (int nn = 0; nn < nsubcells_in_cell; ++nn) {
          const int subcell_index2 = cell_to_nodes_off + nn;
          if (cells_to_nodes[(subcell_index2)] == rnode) {
            subcells_to_subcells[(subcell_to_subcells_off +
                                  neighbour_index++)] = subcell_index2;
            break;
          }
        }
      }
    }
  }
}
