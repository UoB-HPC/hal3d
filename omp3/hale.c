#include "hale.h"
#include "../../shared.h"
#include "../hale_data.h"
#include "../hale_interface.h"
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
    double* subcell_ie_density, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_integrals_x,
    double* subcell_integrals_y, double* subcell_integrals_z,
    double* subcell_kinetic_energy, double* rezoned_nodes_x,
    double* rezoned_nodes_y, double* rezoned_nodes_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces,
    int* subcells_to_faces_offsets, int* subcells_to_faces,
    int* subcells_to_subcells_offsets, int* subcells_to_subcells) {

  double total_mass = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
  }

  printf("total mass %.12f\n", total_mass);

// Calculate the swept edge remap for each of the subcells.
// TODO: There a again many different ways to crack this nut. One consideration
// is that teh whole algorithm will allow you to precompute and store for
// re-use, but artefacts such as the inverse coefficient matrix for the least
// squares, which stays the same for all density calculations of a subcell, are
// essentially prohibitively large for storage.
//
// The approach I am currently taking here is to calculate the inverse
// coefficient matrix and then perform the gradient calculations and swept edge
// remaps for all of the densities for a particular subcell in a single
// timestep. The impplication is that the whole calculation will be repeated
// many times, but it seems like this will be the case regardless and my
// intuition is that this path leads to the fewest expensive and repetitious
// calculations.
//
// The choices I can see are:
//  (1) remap all of the variables for a subcell
//  (2) remap each variable individually for every subcell
//
//  option (1) will have to recompute the gradient coefficients every time
//  option (2) will have to recompute the gradients for local subcells every
//  time
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
     * Perform the swept edge remaps for the subcells...
     */

    // We discover the subcell gradients using a least squares fit for the
    // gradient between the subcell and its neighbours
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int subcell_index = cells_to_nodes[(cell_to_nodes_off + ss)];
      const int subcell_to_subcells_off =
          subcells_to_subcells_offsets[(subcell_index)];
      const int nsubcells_by_subcell =
          subcells_to_subcells[(subcell_index + 1)] - subcell_to_subcells_off;

      /*
       * Calculate the coefficients for all density gradients
       */

      // The coefficients of the 3x3 gradient coefficient matrix
      vec_t coeff[3] = {{0.0, 0.0, 0.0}};
      for (int ss2 = 0; ss2 < nsubcells_by_subcell; ++ss2) {
        const int neighbour_subcell_index =
            subcells_to_subcells[(subcell_to_subcells_off + ss2)];

        const double ix = subcell_integrals_x[(neighbour_subcell_index)];
        const double iy = subcell_integrals_y[(neighbour_subcell_index)];
        const double iz = subcell_integrals_z[(neighbour_subcell_index)];
        const double vol = subcell_volume[(neighbour_subcell_index)];

        // Store the neighbouring cell's contribution to the coefficients
        coeff[0].x += (2.0 * ix * ix) / (vol * vol);
        coeff[0].y += (2.0 * ix * iy) / (vol * vol);
        coeff[0].z += (2.0 * ix * iz) / (vol * vol);
        coeff[1].x += (2.0 * iy * ix) / (vol * vol);
        coeff[1].y += (2.0 * iy * iy) / (vol * vol);
        coeff[1].z += (2.0 * iy * iz) / (vol * vol);
        coeff[2].x += (2.0 * iz * ix) / (vol * vol);
        coeff[2].y += (2.0 * iz * iy) / (vol * vol);
        coeff[2].z += (2.0 * iz * iz) / (vol * vol);
      }

      // Calculate the inverse of the coefficients for swept edge of all faces
      vec_t inv[3];
      calc_3x3_inverse(&coeff, &inv);

      // Calculate the gradient for the internal energy density
      vec_t rhs = {0.0, 0.0, 0.0};
      for (int ss2 = 0; ss2 < nsubcells_by_subcell; ++ss2) {
        const int neighbour_subcell_index =
            subcells_to_subcells[(subcell_to_subcells_off + ss2)];

        // Prepare differential
        const double de = (subcell_ie_density[(neighbour_subcell_index)] -
                           subcell_ie_density[(subcell_index)]);

        // Calculate the subcell gradients for all of the variables
        rhs.x += (2.0 * subcell_integrals_x[(cell_to_nodes_off + ss)] * de /
                  subcell_volume[(cell_to_nodes_off + ss)]);
        rhs.y += (2.0 * subcell_integrals_y[(cell_to_nodes_off + ss)] * de /
                  subcell_volume[(cell_to_nodes_off + ss)]);
        rhs.z += (2.0 * subcell_integrals_z[(cell_to_nodes_off + ss)] * de /
                  subcell_volume[(cell_to_nodes_off + ss)]);
      }

      vec_t grad_ie_density;
      grad_ie_density.x =
          inv[0].x * rhs.x + inv[0].y * rhs.y + inv[0].z * rhs.z;
      grad_ie_density.y =
          inv[1].x * rhs.x + inv[1].y * rhs.y + inv[1].z * rhs.z;
      grad_ie_density.z =
          inv[2].x * rhs.x + inv[2].y * rhs.y + inv[2].z * rhs.z;

      // NOTE: At this stage we are *currently* making a decision to
      // overcalculate the face sweeps in order to ensure that there are no data
      // races. The idea is that each of the faces is calculated twice, once for
      // eac of the coinciding subcells, which means that we could potentially
      // improve this routine by a factor of 2 if we can devise a scheme that
      // allows all of the work to occur independently while updating both
      // subcells that coincide with a face.
    }
  }
}
