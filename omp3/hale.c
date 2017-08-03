#include "hale.h"
#include "../../comms.h"
#include "../../params.h"
#include "../../shared.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// TODO: At this stage, there are so many additional fields required
// to handle the sub-cell data for the remapping phase, there will be some use
// in considering whether some of the fields could be shared or whether
// adaptations to the algorithm are even necessary for this particular point

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_2d(
    Mesh* mesh, const int ncells, const int nnodes,
    const int nsubcell_neighbours, const double visc_coeff1,
    const double visc_coeff2, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_to_nodes,
    int* cells_offsets, int* nodes_to_cells, int* nodes_offsets,
    double* nodes_x0, double* nodes_y0, double* nodes_z0, double* nodes_x1,
    double* nodes_y1, double* nodes_z1, int* boundary_index, int* boundary_type,
    double* boundary_normal_x, double* boundary_normal_y,
    double* boundary_normal_z, double* energy0, double* energy1,
    double* density0, double* density1, double* pressure0, double* pressure1,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* velocity_x1, double* velocity_y1, double* velocity_z1,
    double* subcell_force_x, double* subcell_force_y, double* subcell_force_z,
    double* cell_mass, double* nodal_mass, double* nodal_volumes,
    double* nodal_soundspeed, double* limiter, double* subcell_volume,
    double* subcell_internal_energy, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_integrals_x,
    double* subcell_integrals_y, double* subcell_integrals_z,
    double* subcell_kinetic_energy, double* rezoned_nodes_x,
    double* rezoned_nodes_y, double* rezoned_nodes_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces,
    int* subcells_to_faces_offsets, int* subcells_to_faces) {

  double total_mass = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
  }

  printf("total mass %.12f\n", total_mass);

// Calculate the sub-cell internal energies
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

// Here we are constructing a reference subcell prism for the target face, this
// reference element is used for all of the faces of the subcell, in fact it's
// the same for all cells too, so this could be moved into some global space
// it's going to spill anyway. The reference shape is by construction when
// considering a swept edge remap.
#define NSUBCELL_FACES 6
#define NSUBCELL_NODES 8
#define NSUBCELL_NODES_PER_FACE 4
    const int subcell_faces_to_nodes_offsets[NSUBCELL_FACES + 1] = {
        0, 4, 8, 12, 16, 20, 24};
    const int subcell_faces_to_nodes[NSUBCELL_FACES * NSUBCELL_NODES_PER_FACE] =
        {0, 1, 2, 3, 0, 1, 5, 4, 0, 3, 7, 4,
         1, 2, 6, 5, 4, 5, 6, 7, 3, 2, 6, 7};
    const int subcell_to_faces[NSUBCELL_FACES] = {0, 1, 2, 3, 4, 5};
    double subcell_nodes_x[NSUBCELL_NODES] = {0.0};
    double subcell_nodes_y[NSUBCELL_NODES] = {0.0};
    double subcell_nodes_z[NSUBCELL_NODES] = {0.0};

    /*
     * Calculate the swept-edge region
     */

    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int subcell_index = cell_to_nodes_off + ss;
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // We will calculate the swept edge region for the internal and external
      // face here, this relies on the faces being ordered in a ring.
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes[(face_index + 1)] - face_to_nodes_off;

        double face_c_x = 0.0;
        double face_c_y = 0.0;
        double face_c_z = 0.0;
        for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
          face_c_x += nodes_x1[(node_index)] / nnodes_by_face;
          face_c_y += nodes_y1[(node_index)] / nnodes_by_face;
          face_c_z += nodes_z1[(node_index)] / nnodes_by_face;
        }
      }
    }

    /*
     * Calculate the subcell gradients
     */

    for (int nn = 0; nn < nsubcells_by_cell; ++nn) {
      // The coefficients of the 3x3 gradient coefficient matrix
      vec_t coeff[3] = {{0.0, 0.0, 0.0}};

      // Store the neighbouring cell's contribution to the coefficients
      coeff[0].x += (2.0 * subcell_integrals_x[(cell_to_nodes_off + nn)] *
                     subcell_integrals_x[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);
      coeff[0].y += (2.0 * subcell_integrals_x[(cell_to_nodes_off + nn)] *
                     subcell_integrals_y[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);
      coeff[0].z += (2.0 * subcell_integrals_x[(cell_to_nodes_off + nn)] *
                     subcell_integrals_z[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);
      coeff[1].x += (2.0 * subcell_integrals_y[(cell_to_nodes_off + nn)] *
                     subcell_integrals_x[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);
      coeff[1].y += (2.0 * subcell_integrals_y[(cell_to_nodes_off + nn)] *
                     subcell_integrals_y[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);
      coeff[1].z += (2.0 * subcell_integrals_y[(cell_to_nodes_off + nn)] *
                     subcell_integrals_z[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);
      coeff[2].x += (2.0 * subcell_integrals_z[(cell_to_nodes_off + nn)] *
                     subcell_integrals_x[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);
      coeff[2].y += (2.0 * subcell_integrals_z[(cell_to_nodes_off + nn)] *
                     subcell_integrals_y[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);
      coeff[2].z += (2.0 * subcell_integrals_z[(cell_to_nodes_off + nn)] *
                     subcell_integrals_z[(cell_to_nodes_off + nn)]) /
                    (subcell_volume[(cell_to_nodes_off + nn)] *
                     subcell_volume[(cell_to_nodes_off + nn)]);

      // Calculate the subcell gradients for all of the variables
      vec_t rhs = {0.0, 0.0, 0.0};
      vec_t subcell_grad = {0.0, 0.0, 0.0};

#if 0
      // Prepare the RHS, which includes energy differential
      const double de =
          (subcell_internal_energy[(neighbour_index)] - subcell_internal_energy[(cc)]);
      rhs.x += (2.0 * subcell_integrals_x[(cell_to_nodes_off + nn)] * de /
                subcell_volume[(cell_to_nodes_off + nn)]);
      rhs.y += (2.0 * subcell_integrals_y[(cell_to_nodes_off + nn)] * de /
                subcell_volume[(cell_to_nodes_off + nn)]);
      rhs.z += (2.0 * subcell_integrals_z[(cell_to_nodes_off + nn)] * de /
                subcell_volume[(cell_to_nodes_off + nn)]);
#endif // if 0
    }
  }
}

// Initialise the subcells to faces connectivity list
void init_subcells_to_faces(
    const int ncells, const int* cells_offsets, const int* cells_to_nodes,
    const int* nodes_to_faces_offsets, const int* nodes_to_faces,
    const int* cells_to_faces_offsets, const int* cells_to_faces,
    const int* faces_to_cells0, const int* faces_to_cells1,
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

    // This is a map between one element of faces_on_node and another, with
    // the unique pairings, which is essentially a ring of faces
    int faces_to_faces[] = {-1, -1, -1, -1, -1, -1, -1, -1};

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
        int found = 0;
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
            int connect_face_index = 0;
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
#if 0

  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // We will calculate the flux at every face of the subcells
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int subcell_index = cell_to_nodes_off + ss;
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      printf("subcell %d : ", subcell_index);
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        printf("%d ", face_index);
      }
      printf("\n");
    }
  }
#endif // if 0
}
