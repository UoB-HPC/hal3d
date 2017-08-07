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
    Mesh* mesh, const int ncells, const int nnodes, const double visc_coeff1,
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
    int* subcells_to_subcells) {

  lagrangian_phase(
      mesh, ncells, nnodes, visc_coeff1, visc_coeff2, cell_centroids_x,
      cell_centroids_y, cell_centroids_z, cells_to_nodes, cells_offsets,
      nodes_to_cells, nodes_offsets, nodes_x0, nodes_y0, nodes_z0, nodes_x1,
      nodes_y1, nodes_z1, boundary_index, boundary_type, boundary_normal_x,
      boundary_normal_y, boundary_normal_z, energy0, energy1, density0,
      density1, pressure0, pressure1, velocity_x0, velocity_y0, velocity_z0,
      velocity_x1, velocity_y1, velocity_z1, subcell_force_x, subcell_force_y,
      subcell_force_z, cell_mass, nodal_mass, nodal_volumes, nodal_soundspeed,
      limiter, nodes_to_faces_offsets, nodes_to_faces, faces_to_nodes,
      faces_to_nodes_offsets, faces_to_cells0, faces_to_cells1,
      cells_to_faces_offsets, cells_to_faces);

#if 0
  gather_subcell_quantities(
      ncells, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      cells_to_nodes, cells_offsets, nodes_x0, nodes_y0, nodes_z0, energy0,
      density0, velocity_x0, velocity_y0, velocity_z0, cell_mass,
      subcell_volume, subcell_ie_density, subcell_mass, subcell_velocity_x,
      subcell_velocity_y, subcell_velocity_z, subcell_integrals_x,
      subcell_integrals_y, subcell_integrals_z, nodes_to_faces_offsets,
      nodes_to_faces, faces_to_nodes, faces_to_nodes_offsets, faces_to_cells0,
      faces_to_cells1, cells_to_faces_offsets, cells_to_faces);

  double total_mass = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
  }

  printf("total mass %.12f\n", total_mass);

  for (int nn = 0; nn < nnodes; ++nn) {
    rezoned_nodes_x[(nn)] += 1.0;
    rezoned_nodes_y[(nn)] += 1.0;
    rezoned_nodes_z[(nn)] += 1.0;
  }

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

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    // Calculate the cell centroids for the rezoned mesh
    vec_t rz_cell_centroid = {0.0, 0.0, 0.0};
    for (int nn = 0; nn < nsubcells_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      rz_cell_centroid.x += rezoned_nodes_x[(node_index)] / nsubcells_by_cell;
      rz_cell_centroid.y += rezoned_nodes_y[(node_index)] / nsubcells_by_cell;
      rz_cell_centroid.z += rezoned_nodes_z[(node_index)] / nsubcells_by_cell;
    }

    /*
     * Here we are constructing a reference subcell prism for the target face,
    * this reference element is used for all of the faces of the subcell, in
    * fact it's the same for all cells too, so this could be moved into some
    * global space it's going to spill anyway. The reference shape is by
    * construction when considering a swept edge remap.
    */

    const int prism_faces_to_nodes_offsets[] = {0, 4, 8, 12, 16, 20, 24};
    const int prism_faces_to_nodes[] = {0, 1, 2, 3, 0, 1, 5, 4, 0, 4, 7, 3,
                                        1, 5, 6, 2, 4, 5, 6, 7, 3, 2, 6, 7};
    const int prism_to_faces[] = {0, 1, 2, 3, 4, 5};
    double prism_nodes_x[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double prism_nodes_y[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double prism_nodes_z[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // We discover the subcell gradients using a least squares fit for the
    // gradient between the subcell and its neighbours
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int subcell_index = (cell_to_nodes_off + ss);
      const int subcell_node_index = cells_to_nodes[(subcell_index)];
      const int subcell_to_subcells_off =
          subcells_to_faces_offsets[(subcell_index)] * 2;
      const int nsubcells_by_subcell =
          subcells_to_subcells[(subcell_index + 1)] - subcell_to_subcells_off;
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // We will calculate the swept edge region for the internal and external
      // face here, this relies on the faces being ordered in a ring.
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_index2 =
            (ff == nfaces_by_subcell - 1)
                ? subcells_to_faces[(subcell_to_faces_off)]
                : subcells_to_faces[(subcell_to_faces_off + ff + 1)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
        const int face2_to_nodes_off = faces_to_nodes_offsets[(face_index2)];
        const int nnodes_by_face2 =
            faces_to_nodes_offsets[(face_index2 + 1)] - face2_to_nodes_off;

        /*
         * Determine all of the nodes for the swept edge region inside the
         * current mesh and in the rezoned mesh
         */

        // Calculate the face center for the current and rezoned meshes
        vec_t face_c = {0.0, 0.0, 0.0};
        vec_t rz_face_c = {0.0, 0.0, 0.0};
        int sn_off;
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
          face_c.x += nodes_x0[(node_index)] / nnodes_by_face;
          face_c.y += nodes_y0[(node_index)] / nnodes_by_face;
          face_c.z += nodes_z0[(node_index)] / nnodes_by_face;
          rz_face_c.x += rezoned_nodes_x[(node_index)] / nnodes_by_face;
          rz_face_c.y += rezoned_nodes_y[(node_index)] / nnodes_by_face;
          rz_face_c.z += rezoned_nodes_z[(node_index)] / nnodes_by_face;

          // Store the offset of our subcell's node on the face for calculating
          // the half edges later
          if (subcell_node_index == node_index) {
            sn_off = nn;
          }
        }

        // Calculate the face center for the current and rezoned meshes of the
        // neighbouring face
        vec_t face2_c = {0.0, 0.0, 0.0};
        vec_t rz_face2_c = {0.0, 0.0, 0.0};
        for (int nn = 0; nn < nnodes_by_face2; ++nn) {
          const int node_index = faces_to_nodes[(face2_to_nodes_off + nn)];
          face2_c.x += nodes_x0[(node_index)] / nnodes_by_face2;
          face2_c.y += nodes_y0[(node_index)] / nnodes_by_face2;
          face2_c.z += nodes_z0[(node_index)] / nnodes_by_face2;
          rz_face2_c.x += rezoned_nodes_x[(node_index)] / nnodes_by_face2;
          rz_face2_c.y += rezoned_nodes_y[(node_index)] / nnodes_by_face2;
          rz_face2_c.z += rezoned_nodes_z[(node_index)] / nnodes_by_face2;
        }

        // The half edges are the points between the node at the subcell and the
        // right and left nodes on our current face
        vec_t half_edge_l = {0.0, 0.0, 0.0};
        vec_t half_edge_r = {0.0, 0.0, 0.0};
        const int l_off = (sn_off == 0) ? nnodes_by_face - 1 : sn_off - 1;
        const int r_off = (sn_off == nnodes_by_face - 1) ? 0 : sn_off + 1;
        const int node_l_index = faces_to_nodes[(face_to_nodes_off + l_off)];
        const int node_r_index = faces_to_nodes[(face_to_nodes_off + r_off)];
        half_edge_l.x =
            0.5 * (nodes_x0[(subcell_node_index)] + nodes_x0[(node_l_index)]);
        half_edge_l.y =
            0.5 * (nodes_y0[(subcell_node_index)] + nodes_y0[(node_l_index)]);
        half_edge_l.z =
            0.5 * (nodes_z0[(subcell_node_index)] + nodes_z0[(node_l_index)]);
        half_edge_r.x =
            0.5 * (nodes_x0[(subcell_node_index)] + nodes_x0[(node_r_index)]);
        half_edge_r.y =
            0.5 * (nodes_y0[(subcell_node_index)] + nodes_y0[(node_r_index)]);
        half_edge_r.z =
            0.5 * (nodes_z0[(subcell_node_index)] + nodes_z0[(node_r_index)]);

        vec_t rz_half_edge_l = {0.0, 0.0, 0.0};
        vec_t rz_half_edge_r = {0.0, 0.0, 0.0};
        rz_half_edge_l.x = 0.5 * (rezoned_nodes_x[(subcell_node_index)] +
                                  rezoned_nodes_x[(node_l_index)]);
        rz_half_edge_l.y = 0.5 * (rezoned_nodes_y[(subcell_node_index)] +
                                  rezoned_nodes_y[(node_l_index)]);
        rz_half_edge_l.z = 0.5 * (rezoned_nodes_z[(subcell_node_index)] +
                                  rezoned_nodes_z[(node_l_index)]);
        rz_half_edge_r.x = 0.5 * (rezoned_nodes_x[(subcell_node_index)] +
                                  rezoned_nodes_x[(node_r_index)]);
        rz_half_edge_r.y = 0.5 * (rezoned_nodes_y[(subcell_node_index)] +
                                  rezoned_nodes_y[(node_r_index)]);
        rz_half_edge_r.z = 0.5 * (rezoned_nodes_z[(subcell_node_index)] +
                                  rezoned_nodes_z[(node_r_index)]);

/*
 * Construct the swept edge prism for the internal and external face
 * that is described by the above nodes, and determine the weighted
 * volume integrals
 */

// Firstly we will determine the external swept region

        prism_nodes_x[(0)] = nodes_x0[(subcell_node_index)];
        prism_nodes_y[(0)] = nodes_y0[(subcell_node_index)];
        prism_nodes_z[(0)] = nodes_z0[(subcell_node_index)];
        prism_nodes_x[(1)] = half_edge_r.x;
        prism_nodes_y[(1)] = half_edge_r.y;
        prism_nodes_z[(1)] = half_edge_r.z;
        prism_nodes_x[(2)] = face_c.x;
        prism_nodes_y[(2)] = face_c.y;
        prism_nodes_z[(2)] = face_c.z;
        prism_nodes_x[(3)] = half_edge_l.x;
        prism_nodes_y[(3)] = half_edge_l.y;
        prism_nodes_z[(3)] = half_edge_l.z;
        prism_nodes_x[(4)] = rezoned_nodes_x[(subcell_node_index)];
        prism_nodes_y[(4)] = rezoned_nodes_y[(subcell_node_index)];
        prism_nodes_z[(4)] = rezoned_nodes_z[(subcell_node_index)];
        prism_nodes_x[(5)] = rz_half_edge_r.x;
        prism_nodes_y[(5)] = rz_half_edge_r.y;
        prism_nodes_z[(5)] = rz_half_edge_r.z;
        prism_nodes_x[(6)] = rz_face_c.x;
        prism_nodes_y[(6)] = rz_face_c.y;
        prism_nodes_z[(6)] = rz_face_c.z;
        prism_nodes_x[(7)] = rz_half_edge_l.x;
        prism_nodes_y[(7)] = rz_half_edge_l.y;
        prism_nodes_z[(7)] = rz_half_edge_l.z;

        vec_t prism_centroid = {0.0, 0.0, 0.0};
        for (int pp = 0; pp < NSUBCELL_NODES; ++pp) {
          prism_centroid.x += prism_nodes_x[(pp)] / NSUBCELL_NODES;
          prism_centroid.y += prism_nodes_y[(pp)] / NSUBCELL_NODES;
          prism_centroid.z += prism_nodes_z[(pp)] / NSUBCELL_NODES;
        }

        vec_t integrals = {0.0, 0.0, 0.0};
        double vol = 0.0;
        calc_weighted_volume_integrals(
            0, NSUBCELL_FACES, prism_to_faces, prism_faces_to_nodes,
            prism_faces_to_nodes_offsets, prism_nodes_x, prism_nodes_y,
            prism_nodes_z, &prism_centroid, &integrals, &vol);

        // Secondly we will determine the internal swept region

        // Get the orientation of the face
        vec_t dn0 = {0.0, 0.0, 0.0};
        vec_t dn1 = {0.0, 0.0, 0.0};
        const int fn0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int fn1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int fn2 = faces_to_nodes[(face_to_nodes_off + 2)];
        dn0.x = nodes_x0[(fn2)] - nodes_x0[(fn1)];
        dn0.y = nodes_y0[(fn2)] - nodes_y0[(fn1)];
        dn0.z = nodes_z0[(fn2)] - nodes_z0[(fn1)];
        dn1.x = nodes_x0[(fn1)] - nodes_x0[(fn0)];
        dn1.y = nodes_y0[(fn1)] - nodes_y0[(fn0)];
        dn1.z = nodes_z0[(fn1)] - nodes_z0[(fn0)];

        // Calculate a vector from face to cell centroid
        vec_t ab;
        ab.x = (cell_centroid.x - nodes_x0[(fn0)]);
        ab.y = (cell_centroid.y - nodes_y0[(fn0)]);
        ab.z = (cell_centroid.z - nodes_z0[(fn0)]);

        // Cross product to get the normal
        vec_t normal;
        normal.x = (dn0.y * dn1.z - dn0.z * dn1.y);
        normal.y = (dn0.z * dn1.x - dn0.x * dn1.z);
        normal.z = (dn0.x * dn1.y - dn0.y * dn1.x);

        const int face_rorientation =
            (ab.x * normal.x + ab.y * normal.y + ab.z * normal.z < 0.0);

        construct_internal_swept_region(
            face_rorientation, &half_edge_l, &half_edge_r, &rz_half_edge_l,
            &rz_half_edge_r, &face_c, &face2_c, &rz_face_c, &rz_face2_c,
            &cell_centroid, &rz_cell_centroid, &prism_centroid, prism_nodes_x,
            prism_nodes_y, prism_nodes_z);
        calc_weighted_volume_integrals(
            0, NSUBCELL_FACES, prism_to_faces, prism_faces_to_nodes,
            prism_faces_to_nodes_offsets, prism_nodes_x, prism_nodes_y,
            prism_nodes_z, &prism_centroid, &integrals, &vol);
        /*
         * Calculate the coefficients for all density gradients
         * The gradient that we calculate may be for the external or the
         * internal cell.
         *
         * We can save improve performance here by caching the gradients in the
         * case that the swept edge region is primarily overlapping our current
         * cell. In the cases where it is not, presumably half of the
         * calculations, we have to perform redundant computation. My assumption
         * here is that the calculation of the gradients is significantly
         * cheaper than the calculation of the swept edge regions. If this turns
         * out to be a faulty assumption, perhaps due to the computational
         * intensity, then another of the three obvious algorithms for this step
         * could be chosen.
         *
         * The best case algorithm in terms of reducing repeated or redundant
         * computation will lead to a data race, and we may see that atomics are
         * able to win out for some architectures instead of this approach here.
         */

        /*
         * TODO: We need to perform the test here to determine if the signed
         * volume of the prism indicates that the swept region overlaps with the
         * current cell or the neighbour.
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

        /*
         * Perform the swept edge remaps for the subcells...
         */
      }
    }
  }
#endif // if 0
}

// Constructs the prism for swept region of a subcell face internal to a cell
void construct_internal_swept_region(
    const int face_rorientation, const vec_t* half_edge_l,
    const vec_t* half_edge_r, const vec_t* rz_half_edge_l,
    const vec_t* rz_half_edge_r, const vec_t* face_c, const vec_t* face2_c,
    const vec_t* rz_face_c, const vec_t* rz_face2_c, const vec_t* cell_centroid,
    const vec_t* rz_cell_centroid, vec_t* prism_centroid, double* prism_nodes_x,
    double* prism_nodes_y, double* prism_nodes_z) {

  // Constructing a prism from all known points in mesh and rezoned mesh
  prism_nodes_x[(0)] = face_rorientation ? half_edge_r->x : half_edge_l->x;
  prism_nodes_y[(0)] = face_rorientation ? half_edge_r->y : half_edge_l->y;
  prism_nodes_z[(0)] = face_rorientation ? half_edge_r->z : half_edge_l->z;
  prism_nodes_x[(1)] = face2_c->x;
  prism_nodes_y[(1)] = face2_c->y;
  prism_nodes_z[(1)] = face2_c->z;
  prism_nodes_x[(2)] = cell_centroid->x;
  prism_nodes_y[(2)] = cell_centroid->y;
  prism_nodes_z[(2)] = cell_centroid->z;
  prism_nodes_x[(3)] = face_c->x;
  prism_nodes_y[(3)] = face_c->y;
  prism_nodes_z[(3)] = face_c->z;
  prism_nodes_x[(4)] =
      face_rorientation ? rz_half_edge_r->x : rz_half_edge_l->x;
  prism_nodes_y[(4)] =
      face_rorientation ? rz_half_edge_r->y : rz_half_edge_l->y;
  prism_nodes_z[(4)] =
      face_rorientation ? rz_half_edge_r->z : rz_half_edge_l->z;
  prism_nodes_x[(5)] = rz_face2_c->x;
  prism_nodes_y[(5)] = rz_face2_c->y;
  prism_nodes_z[(5)] = rz_face2_c->z;
  prism_nodes_x[(6)] = rz_cell_centroid->x;
  prism_nodes_y[(6)] = rz_cell_centroid->y;
  prism_nodes_z[(6)] = rz_cell_centroid->z;
  prism_nodes_x[(7)] = rz_face_c->x;
  prism_nodes_y[(7)] = rz_face_c->y;
  prism_nodes_z[(7)] = rz_face_c->z;

  // Determine the prism's centroid
  prism_centroid->x = 0.0;
  prism_centroid->y = 0.0;
  prism_centroid->z = 0.0;
  for (int pp = 0; pp < NSUBCELL_NODES; ++pp) {
    prism_centroid->x += prism_nodes_x[(pp)] / NSUBCELL_NODES;
    prism_centroid->y += prism_nodes_y[(pp)] / NSUBCELL_NODES;
    prism_centroid->z += prism_nodes_z[(pp)] / NSUBCELL_NODES;
  }
}

#if 0
// A test where a single cell would expand evenly.
  rezoned_nodes_x[(0)] -= 0.1;
  rezoned_nodes_y[(0)] -= 0.1;
  rezoned_nodes_z[(0)] -= 0.1;
  rezoned_nodes_x[(1)] += 0.1;
  rezoned_nodes_y[(1)] -= 0.1;
  rezoned_nodes_z[(1)] -= 0.1;
  rezoned_nodes_x[(2)] += 0.1;
  rezoned_nodes_y[(2)] += 0.1;
  rezoned_nodes_z[(2)] -= 0.1;
  rezoned_nodes_x[(3)] -= 0.1;
  rezoned_nodes_y[(3)] += 0.1;
  rezoned_nodes_z[(3)] -= 0.1;
  rezoned_nodes_x[(4)] -= 0.1;
  rezoned_nodes_y[(4)] -= 0.1;
  rezoned_nodes_z[(4)] += 0.1;
  rezoned_nodes_x[(5)] += 0.1;
  rezoned_nodes_y[(5)] -= 0.1;
  rezoned_nodes_z[(5)] += 0.1;
  rezoned_nodes_x[(6)] += 0.1;
  rezoned_nodes_y[(6)] += 0.1;
  rezoned_nodes_z[(6)] += 0.1;
  rezoned_nodes_x[(7)] -= 0.1;
  rezoned_nodes_y[(7)] += 0.1;
  rezoned_nodes_z[(7)] += 0.1;
#endif // if 0
