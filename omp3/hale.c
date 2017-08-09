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
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, double* subcell_kinetic_energy,
    double* rezoned_nodes_x, double* rezoned_nodes_y, double* rezoned_nodes_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces,
    int* subcells_to_faces_offsets, int* subcells_to_faces,
    int* subcells_to_subcells) {

  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
  }
  printf("total mass %.12f\n", total_mass);

  // We are storing our original mesh to allow an Eulerian remap
  store_rezoned_mesh(nnodes, nodes_x0, nodes_y0, nodes_z0, rezoned_nodes_x,
                     rezoned_nodes_y, rezoned_nodes_z);

  // Perform the Lagrangian phase of the ALE algorithm where the mesh will move
  // due to the pressure (ideal gas) and artificial viscous forces
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

  // Gather the subcell quantities for mass, internal and kinetic energy
  // density, and momentum
  gather_subcell_quantities(
      ncells, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      cells_to_nodes, cells_offsets, nodes_x0, nodes_y0, nodes_z0, energy0,
      density0, velocity_x0, velocity_y0, velocity_z0, cell_mass,
      subcell_volume, subcell_ie_density, subcell_mass, subcell_velocity_x,
      subcell_velocity_y, subcell_velocity_z, subcell_integrals_x,
      subcell_integrals_y, subcell_integrals_z, nodes_to_faces_offsets,
      nodes_to_faces, faces_to_nodes, faces_to_nodes_offsets, faces_to_cells0,
      faces_to_cells1, cells_to_faces_offsets, cells_to_faces);

  calc_subcell_centroids(ncells, cells_offsets, cell_centroids_x,
                         cell_centroids_y, cell_centroids_z, cells_to_nodes,
                         subcells_to_faces_offsets, subcells_to_faces,
                         faces_to_nodes_offsets, faces_to_nodes, nodes_x0,
                         nodes_y0, nodes_z0, subcell_centroids_x,
                         subcell_centroids_y, subcell_centroids_z);

#if 0
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

    vec_t rz_cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nsubcells_by_cell, rezoned_nodes_x, rezoned_nodes_y,
                  rezoned_nodes_z, cells_to_nodes, cell_to_nodes_off,
                  &rz_cell_centroid);

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
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // we will calculate the swept edge region for the internal and external
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
         * determine all of the nodes for the swept edge region inside the
         * current mesh and in the rezoned mesh
         */

        // calculate the face center for the current and rezoned meshes
        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x0, nodes_y0, nodes_z0,
                      faces_to_nodes, face_to_nodes_off, &face_c);
        vec_t face2_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face2, nodes_x0, nodes_y0, nodes_z0,
                      faces_to_nodes, face2_to_nodes_off, &face2_c);

        // calculate the subsequent face center for current and rezoned meshes
        vec_t rz_face_c = {0.0, 0.0, 0.0};
        int sn_off;
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
          rz_face_c.x += rezoned_nodes_x[(node_index)] / nnodes_by_face;
          rz_face_c.y += rezoned_nodes_y[(node_index)] / nnodes_by_face;
          rz_face_c.z += rezoned_nodes_z[(node_index)] / nnodes_by_face;

          // store the offset of our subcell's node on the face for
          // calculating
          // the half edges later
          if (subcell_node_index == node_index) {
            sn_off = nn;
          }
        }
        vec_t rz_face2_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face2, rezoned_nodes_x, rezoned_nodes_y,
                      rezoned_nodes_z, faces_to_nodes, face2_to_nodes_off,
                      &face2_c);

        // the half edges are the points between the node at the subcell and the
        // right and left nodes on our current face
        const int l_off = (sn_off == 0) ? nnodes_by_face - 1 : sn_off - 1;
        const int r_off = (sn_off == nnodes_by_face - 1) ? 0 : sn_off + 1;
        const int node_l_index = faces_to_nodes[(face_to_nodes_off + l_off)];
        const int node_r_index = faces_to_nodes[(face_to_nodes_off + r_off)];

        vec_t half_edge_l = {
            0.5 * (nodes_x0[(subcell_node_index)] + nodes_x0[(node_l_index)]),
            0.5 * (nodes_y0[(subcell_node_index)] + nodes_y0[(node_l_index)]),
            0.5 * (nodes_z0[(subcell_node_index)] + nodes_z0[(node_l_index)])};
        vec_t half_edge_r = {
            0.5 * (nodes_x0[(subcell_node_index)] + nodes_x0[(node_r_index)]),
            0.5 * (nodes_y0[(subcell_node_index)] + nodes_y0[(node_r_index)]),
            0.5 * (nodes_z0[(subcell_node_index)] + nodes_z0[(node_r_index)])};

        vec_t rz_half_edge_l = {0.5 * (rezoned_nodes_x[(subcell_node_index)] +
                                       rezoned_nodes_x[(node_l_index)]),
                                0.5 * (rezoned_nodes_y[(subcell_node_index)] +
                                       rezoned_nodes_y[(node_l_index)]),
                                0.5 * (rezoned_nodes_z[(subcell_node_index)] +
                                       rezoned_nodes_z[(node_l_index)])};
        vec_t rz_half_edge_r = {0.5 * (rezoned_nodes_x[(subcell_node_index)] +
                                       rezoned_nodes_x[(node_r_index)]),
                                0.5 * (rezoned_nodes_y[(subcell_node_index)] +
                                       rezoned_nodes_y[(node_r_index)]),
                                0.5 * (rezoned_nodes_z[(subcell_node_index)] +
                                       rezoned_nodes_z[(node_r_index)])};

        /*
         * Construct the swept edge prism for the internal and external face
         * that is described by the above nodes, and determine the weighted
         * volume swept_edge_integrals
         */

        // Firstly we will determine the external swept region
        vec_t nodes = {nodes_x0[(subcell_node_index)],
                       nodes_y0[(subcell_node_index)],
                       nodes_z0[(subcell_node_index)]};
        vec_t rz_nodes = {rezoned_nodes_x[(subcell_node_index)],
                          rezoned_nodes_y[(subcell_node_index)],
                          rezoned_nodes_z[(subcell_node_index)]};

        vec_t prism_centroid = {0.0, 0.0, 0.0};
        construct_external_swept_region(
            &nodes, &rz_nodes, &half_edge_l, &half_edge_r, &rz_half_edge_l,
            &rz_half_edge_r, &face_c, &rz_face_c, &prism_centroid,
            prism_nodes_x, prism_nodes_y, prism_nodes_z);

        vec_t swept_edge_integrals = {0.0, 0.0, 0.0};
        double swept_edge_vol = 0.0;
        calc_weighted_volume_integrals(
            0, NSUBCELL_FACES, prism_to_faces, prism_faces_to_nodes,
            prism_faces_to_nodes_offsets, prism_nodes_x, prism_nodes_y,
            prism_nodes_z, &prism_centroid, &swept_edge_integrals,
            &swept_edge_vol);

        // Secondly we will determine the internal swept region

        // Choose threee points on the planar face
        const int fn0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int fn1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int fn2 = faces_to_nodes[(face_to_nodes_off + 2)];

        vec_t normal;
        calc_normal(fn0, fn1, fn2, nodes_x0, nodes_y0, nodes_z0, &normal);

        // TODO: Fairly sure that this returns the opposite value to what we
        // are
        // expecting later on...
        const int face_rorientation = check_normal_orientation(
            fn0, nodes_x0, nodes_y0, nodes_z0, &cell_centroid, &normal);

        construct_internal_swept_region(
            face_rorientation, &half_edge_l, &half_edge_r, &rz_half_edge_l,
            &rz_half_edge_r, &face_c, &face2_c, &rz_face_c, &rz_face2_c,
            &cell_centroid, &rz_cell_centroid, &prism_centroid, prism_nodes_x,
            prism_nodes_y, prism_nodes_z);
        calc_weighted_volume_integrals(
            0, NSUBCELL_FACES, prism_to_faces, prism_faces_to_nodes,
            prism_faces_to_nodes_offsets, prism_nodes_x, prism_nodes_y,
            prism_nodes_z, &prism_centroid, &swept_edge_integrals,
            &swept_edge_vol);

        /*
         * Calculate the coefficients for all density gradients
         * The gradient that we calculate may be for the external or the
         * internal cell.
         *
         * We can save improve performance here by caching the gradients in
         * the
         * case that the swept edge region is primarily overlapping our
         * current
         * cell. In the cases where it is not, presumably half of the
         * calculations, we have to perform redundant computation. My
         * assumption
         * here is that the calculation of the gradients is significantly
         * cheaper than the calculation of the swept edge regions. If this
         * turns
         * out to be a faulty assumption, perhaps due to the computational
         * intensity, then another of the three obvious algorithms for this
         * step
         * could be chosen.
         *
         * The best case algorithm in terms of reducing repeated or redundant
         * computation will lead to a data race, and we may see that atomics
         * are
         * able to win out for some architectures instead of this approach
         * here.
         */

        /*
         * TODO: We need to perform the test here to determine if the signed
         * volume of the prism indicates that the swept region overlaps with
         * the
         * current cell or the neighbour.
         */

        const int sweeping_subcell = subcell_index;

        // Calculate the inverse coefficient matrix for the least squares
        // regression of the gradient, which is the same for all quantities.
        vec_t inv[3];
        int nsubcells_by_subcell;
        int subcell_to_subcells_off;
        calculate_inverse_coefficient_matrix(
            sweeping_subcell, subcells_to_faces_offsets, subcells_to_subcells,
            subcell_integrals_x, subcell_integrals_y, subcell_integrals_z,
            subcell_volume, &nsubcells_by_subcell, &subcell_to_subcells_off,
            &inv);

        // For all of the subcell centered quantities, determine the flux.
        vec_t ie_gradient;
        calc_gradient(sweeping_subcell, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_ie_density, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &ie_gradient);

        // Calculate the flux for internal energy density in the subcell
        const double ie_flux =
            subcell_ie_density[(sweeping_subcell)] * swept_edge_vol +
            ie_gradient.x * swept_edge_integrals.x -
            ie_gradient.x * subcell_centroids_x[(sweeping_subcell)] +
            ie_gradient.y * swept_edge_integrals.y -
            ie_gradient.x * subcell_centroids_y[(sweeping_subcell)] +
            ie_gradient.z * swept_edge_integrals.z -
            ie_gradient.x * subcell_centroids_z[(sweeping_subcell)];

        vec_t mass_gradient;
        calc_gradient(sweeping_subcell, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_mass, subcell_integrals_x, subcell_integrals_y,
                      subcell_integrals_z, subcell_volume, &inv,
                      &mass_gradient);

        vec_t ke_gradient;
        calc_gradient(sweeping_subcell, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_kinetic_energy, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &ke_gradient);

        vec_t xmomentum_gradient;
        calc_gradient(sweeping_subcell, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_velocity_x, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &xmomentum_gradient);

        vec_t ymomentum_gradient;
        calc_gradient(sweeping_subcell, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_velocity_y, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &ymomentum_gradient);

        vec_t zmomentum_gradient;
        calc_gradient(sweeping_subcell, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_velocity_z, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &zmomentum_gradient);

        /*
         * Perform the swept edge remaps for the subcells...
         */
      }
    }
  }
#endif // if 0
}
