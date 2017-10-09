#include "../../comms.h"
#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>

// Performs a remap and some scattering of the subcell values
void remap_phase(
    const int ncells, const int* cells_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const double* rezoned_nodes_x,
    const double* rezoned_nodes_y, const double* rezoned_nodes_z,
    const int* cells_to_nodes, const int* cells_to_faces_offsets,
    const int* cells_to_faces, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const int* subcells_to_faces_offsets,
    const int* subcells_to_faces, const double* subcell_centroids_x,
    const double* subcell_centroids_y, const double* subcell_centroids_z,
    double* subcell_volume, double* cell_volume) {

  /* REMAP THE MASS AND ENERGY */

  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_c);

    // Looping over corner subcells here
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      vec_t subcell_c = {subcell_centroids_x[(subcell_index)],
                         subcell_centroids_y[(subcell_index)],
                         subcell_centroids_z[(subcell_index)]};

      //
      //
      //
      //
      //
      //
      //
      subcell_volume[(subcell_index)] = 0.0;
      //
      //
      //
      //
      //
      //
      //
      //

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
        vec_t rz_face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, rezoned_nodes_x, rezoned_nodes_y,
                      rezoned_nodes_z, faces_to_nodes, face_to_nodes_off,
                      &rz_face_c);

        // Determine the orientation of the face
        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        vec_t face_normal;
        const int face_clockwise = calc_surface_normal(
            n0, n1, n2, nodes_x, nodes_y, nodes_z, &cell_c, &face_normal);

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

        double eswept_edge_vol = 0.0;
        const int swept_edge_to_faces[] = {0, 1, 2, 3, 4, 5};
        const int swept_edge_faces_to_nodes[] = {0, 1, 2, 3, 4, 5, 6, 7,
                                                 0, 4, 7, 3, 7, 6, 2, 3,
                                                 1, 2, 6, 5, 0, 1, 5, 4};
        const int swept_edge_faces_to_nodes_offsets[] = {0,  4,  8, 12,
                                                         16, 20, 24};

        double enodes_x[2 * NNODES_BY_SUBCELL_FACE] = {
            nodes_x[(node_index)],
            0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]), face_c.x,
            0.5 * (nodes_x[(node_index)] + nodes_x[(lnode_index)]),
            rezoned_nodes_x[(node_index)],
            0.5 * (rezoned_nodes_x[(node_index)] +
                   rezoned_nodes_x[(rnode_index)]),
            rz_face_c.x, 0.5 * (rezoned_nodes_x[(node_index)] +
                                rezoned_nodes_x[(lnode_index)])};
        double enodes_y[2 * NNODES_BY_SUBCELL_FACE] = {
            nodes_y[(node_index)],
            0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]), face_c.y,
            0.5 * (nodes_y[(node_index)] + nodes_y[(lnode_index)]),
            rezoned_nodes_y[(node_index)],
            0.5 * (rezoned_nodes_y[(node_index)] +
                   rezoned_nodes_y[(rnode_index)]),
            rz_face_c.y, 0.5 * (rezoned_nodes_y[(node_index)] +
                                rezoned_nodes_y[(lnode_index)])};
        double enodes_z[2 * NNODES_BY_SUBCELL_FACE] = {
            nodes_z[(node_index)],
            0.5 * (nodes_z[(node_index)] + nodes_z[(rnode_index)]), face_c.z,
            0.5 * (nodes_z[(node_index)] + nodes_z[(lnode_index)]),
            rezoned_nodes_z[(node_index)],
            0.5 * (rezoned_nodes_z[(node_index)] +
                   rezoned_nodes_z[(rnode_index)]),
            rz_face_c.z, 0.5 * (rezoned_nodes_z[(node_index)] +
                                rezoned_nodes_z[(lnode_index)])};

        // Determine the swept edge prism's centroid
        vec_t eswept_edge_c = {0.0, 0.0, 0.0};
        for (int pn = 0; pn < 2 * NNODES_BY_SUBCELL_FACE; ++pn) {
          eswept_edge_c.x += enodes_x[(pn)] / (NNODES_BY_SUBCELL_FACE * 2);
          eswept_edge_c.y += enodes_y[(pn)] / (NNODES_BY_SUBCELL_FACE * 2);
          eswept_edge_c.z += enodes_z[(pn)] / (NNODES_BY_SUBCELL_FACE * 2);
        }

        // Calculate the volume of the swept edge prism
        calc_volume(0, 2 + NNODES_BY_SUBCELL_FACE, swept_edge_to_faces,
                    swept_edge_faces_to_nodes,
                    swept_edge_faces_to_nodes_offsets, enodes_x, enodes_y,
                    enodes_z, &eswept_edge_c, &eswept_edge_vol);

        //
        //
        //
        //
        //
        subcell_volume[(subcell_index)] += eswept_edge_vol;
//
//
//
//
//

#if 0
        // Ignore the special case of an empty swept edge region
        if (swept_edge_vol < EPS) {
          if (swept_edge_vol < -EPS) {
            printf("Negative swept edge volume %d %.12f\n", cc, swept_edge_vol);
          }
          continue;
        }

        // Determine the centroids of the subcell and rezoned faces
        vec_t subcell_face_c = {
            (subcell[(sf0)].x + subcell[(sf1)].x + subcell[(sf2)].x) /
                NTET_NODES_PER_FACE,
            (subcell[(sf0)].y + subcell[(sf1)].y + subcell[(sf2)].y) /
                NTET_NODES_PER_FACE,
            (subcell[(sf0)].z + subcell[(sf1)].z + subcell[(sf2)].z) /
                NTET_NODES_PER_FACE};
        vec_t rz_subcell_face_c = {
            (rz_subcell[(sf0)].x + rz_subcell[(sf1)].x + rz_subcell[(sf2)].x) /
                NTET_NODES_PER_FACE,
            (rz_subcell[(sf0)].y + rz_subcell[(sf1)].y + rz_subcell[(sf2)].y) /
                NTET_NODES_PER_FACE,
            (rz_subcell[(sf0)].z + rz_subcell[(sf1)].z + rz_subcell[(sf2)].z) /
                NTET_NODES_PER_FACE};

        vec_t ab = {rz_subcell_face_c.x - subcell_face_c.x,
                    rz_subcell_face_c.y - subcell_face_c.y,
                    rz_subcell_face_c.z - subcell_face_c.z};
        vec_t ac = {subcell_face_c.x - subcell_centroids_x[(subcell_index)],
                    subcell_face_c.y - subcell_centroids_y[(subcell_index)],
                    subcell_face_c.z - subcell_centroids_z[(subcell_index)]};

        const int is_outflux = (ab.x * ac.x + ab.y * ac.y + ab.z * ac.z > 0.0);

        // Depending upon which subcell we are sweeping into, choose the
        // subcell index with which to reconstruct the density
        const int subcell_neighbour_index =
            subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + pp)];
        const int sweep_subcell_index =
            (is_outflux ? subcell_index : subcell_neighbour_index);

        // Only perform the sweep on the external face if it isn't a
        // boundary
        if (subcell_neighbour_index == -1) {
          TERMINATE("We should not be attempting to flux from boundary.");
        }

        /* CALCULATE INVERSE COEFFICIENT MATRIX */

        // The 3x3 gradient coefficient matrix, and inverse
        vec_t inv[3] = {{0.0, 0.0, 0.0}};
        vec_t coeff[3] = {{0.0, 0.0, 0.0}};

        for (int ss2 = 0; ss2 < NSUBCELL_NEIGHBOURS; ++ss2) {
          const int neighbour_subcell_index = subcells_to_subcells[(
              sweep_subcell_index * NSUBCELL_NEIGHBOURS + ss2)];

          // Ignore boundary neighbours
          if (neighbour_subcell_index == -1) {
            continue;
          }

          double vol = subcell_volume[(neighbour_subcell_index)];
          vec_t i = {subcell_centroids_x[(neighbour_subcell_index)] * vol -
                         subcell_centroids_x[(sweep_subcell_index)] * vol,
                     subcell_centroids_y[(neighbour_subcell_index)] * vol -
                         subcell_centroids_y[(sweep_subcell_index)] * vol,
                     subcell_centroids_z[(neighbour_subcell_index)] * vol -
                         subcell_centroids_z[(sweep_subcell_index)] * vol};

          // Store the neighbouring cell's contribution to the coefficients
          coeff[0].x += 2.0 * (i.x * i.x) / (vol * vol);
          coeff[0].y += 2.0 * (i.x * i.y) / (vol * vol);
          coeff[0].z += 2.0 * (i.x * i.z) / (vol * vol);
          coeff[1].x += 2.0 * (i.y * i.x) / (vol * vol);
          coeff[1].y += 2.0 * (i.y * i.y) / (vol * vol);
          coeff[1].z += 2.0 * (i.y * i.z) / (vol * vol);
          coeff[2].x += 2.0 * (i.z * i.x) / (vol * vol);
          coeff[2].y += 2.0 * (i.z * i.y) / (vol * vol);
          coeff[2].z += 2.0 * (i.z * i.z) / (vol * vol);
        }

        calc_3x3_inverse(&coeff, &inv);

        /* ADVECT MASS */

        double gmax_m = -DBL_MAX;
        double gmin_m = DBL_MAX;
        double subcell_m = subcell_mass[(sweep_subcell_index)] /
                           subcell_volume[(sweep_subcell_index)];

        vec_t m_rhs = {0.0, 0.0, 0.0};
        for (int ss2 = 0; ss2 < NSUBCELL_NEIGHBOURS; ++ss2) {
          const int neighbour_subcell_index = subcells_to_subcells[(
              sweep_subcell_index * NSUBCELL_NEIGHBOURS + ss2)];

          if (neighbour_subcell_index == -1) {
            continue;
          }

          // Prepare differential
          const double vol = subcell_volume[(neighbour_subcell_index)];
          const double phi = subcell_mass[(neighbour_subcell_index)] / vol;
          const double dphi = phi - subcell_m;

          m_rhs.x += 2.0 * dphi *
                     (subcell_centroids_x[(neighbour_subcell_index)] * vol -
                      subcell_centroids_x[(sweep_subcell_index)] * vol) /
                     vol;
          m_rhs.y += 2.0 * dphi *
                     (subcell_centroids_y[(neighbour_subcell_index)] * vol -
                      subcell_centroids_y[(sweep_subcell_index)] * vol) /
                     vol;
          m_rhs.z += 2.0 * dphi *
                     (subcell_centroids_z[(neighbour_subcell_index)] * vol -
                      subcell_centroids_z[(sweep_subcell_index)] * vol) /
                     vol;

          gmax_m = max(gmax_m, phi);
          gmin_m = min(gmin_m, phi);
        }

        vec_t grad_m = {
            inv[0].x * m_rhs.x + inv[0].y * m_rhs.y + inv[0].z * m_rhs.z,
            inv[1].x * m_rhs.x + inv[1].y * m_rhs.y + inv[1].z * m_rhs.z,
            inv[2].x * m_rhs.x + inv[2].y * m_rhs.y + inv[2].z * m_rhs.z};

        apply_cell_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                           &grad_m, &cell_c, nodes_x0, nodes_y0, nodes_z0,
                           subcell_m, gmax_m, gmin_m);

        // Calculate the flux for internal energy density in the subcell
        const double local_mass_flux =
            swept_edge_vol *
            (subcell_m +
             grad_m.x *
                 (swept_edge_c.x - subcell_centroids_x[(sweep_subcell_index)]) +
             grad_m.y *
                 (swept_edge_c.y - subcell_centroids_y[(sweep_subcell_index)]) +
             grad_m.z *
                 (swept_edge_c.z - subcell_centroids_z[(sweep_subcell_index)]));

        /* ADVECT ENERGY */

        double gmax_ie = -DBL_MAX;
        double gmin_ie = DBL_MAX;
        const double subcell_ie = subcell_ie_mass[(sweep_subcell_index)] /
                                  subcell_volume[(sweep_subcell_index)];

        vec_t ie_rhs = {0.0, 0.0, 0.0};
        for (int ss2 = 0; ss2 < NSUBCELL_NEIGHBOURS; ++ss2) {
          const int neighbour_subcell_index = subcells_to_subcells[(
              sweep_subcell_index * NSUBCELL_NEIGHBOURS + ss2)];

          if (neighbour_subcell_index == -1) {
            continue;
          }

          // Prepare differential
          const double phi = subcell_ie_mass[(neighbour_subcell_index)] /
                             subcell_volume[(neighbour_subcell_index)];
          const double dphi = phi - subcell_ie;

          // Calculate the subcell gradients for all
          // of the variables
          ie_rhs.x += dphi * (subcell_centroids_x[(neighbour_subcell_index)] -
                              subcell_centroids_x[(sweep_subcell_index)]);
          ie_rhs.y += dphi * (subcell_centroids_y[(neighbour_subcell_index)] -
                              subcell_centroids_y[(sweep_subcell_index)]);
          ie_rhs.z += dphi * (subcell_centroids_z[(neighbour_subcell_index)] -
                              subcell_centroids_z[(sweep_subcell_index)]);

          gmax_ie = max(gmax_ie, phi);
          gmin_ie = min(gmin_ie, phi);
        }

        vec_t grad_ie = {
            inv[0].x * ie_rhs.x + inv[0].y * ie_rhs.y + inv[0].z * ie_rhs.z,
            inv[1].x * ie_rhs.x + inv[1].y * ie_rhs.y + inv[1].z * ie_rhs.z,
            inv[2].x * ie_rhs.x + inv[2].y * ie_rhs.y + inv[2].z * ie_rhs.z};

        apply_cell_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                           &grad_ie, &cell_c, nodes_x0, nodes_y0, nodes_z0,
                           subcell_ie, gmax_ie, gmin_ie);

        // Calculate the flux for internal energy density in the subcell
        const double local_energy_flux =
            swept_edge_vol *
            (subcell_ie +
             grad_ie.x *
                 (swept_edge_c.x - subcell_centroids_x[(sweep_subcell_index)]) +
             grad_ie.y *
                 (swept_edge_c.y - subcell_centroids_y[(sweep_subcell_index)]) +
             grad_ie.z *
                 (swept_edge_c.z - subcell_centroids_z[(sweep_subcell_index)]));

        // Either mass and energy is flowing into or out of the subcell
        if (is_outflux) {
          subcell_mass_flux[(subcell_index)] += local_mass_flux;
          subcell_ie_mass_flux[(subcell_index)] += local_energy_flux;
        } else {
          subcell_mass_flux[(subcell_index)] -= local_mass_flux;
          subcell_ie_mass_flux[(subcell_index)] -= local_energy_flux;
        }

#if 0
        /* INTERNAL FACE */

        const int rface_off = (ff == nfaces_by_subcell - 1) ? 0 : ff + 1;
        const int lface_off = (ff == 0) ? nfaces_by_subcell - 1 : ff - 1;
        const int rface_index =
            subcells_to_faces[(subcell_to_faces_off + rface_off)];
        const int lface_index =
            subcells_to_faces[(subcell_to_faces_off + lface_off)];
        const int rface_to_nodes_off = faces_to_nodes_offsets[(rface_index)];
        const int lface_to_nodes_off = faces_to_nodes_offsets[(lface_index)];
        const int nnodes_by_rface =
            faces_to_nodes_offsets[(rface_index + 1)] - rface_to_nodes_off;
        const int nnodes_by_lface =
            faces_to_nodes_offsets[(lface_index + 1)] - lface_to_nodes_off;

        // Determine the orientation of the face
        const int rn0 = faces_to_nodes[(rface_to_nodes_off + 0)];
        const int rn1 = faces_to_nodes[(rface_to_nodes_off + 1)];
        const int rn2 = faces_to_nodes[(rface_to_nodes_off + 2)];
        vec_t rface_normal;
        const int rface_clockwise = calc_surface_normal(
            rn0, rn1, rn2, nodes_x, nodes_y, nodes_z, &cell_c, &rface_normal);

        // Determine the position of the node in the face list of nodes
        for (nn2 = 0; nn2 < nnodes_by_rface; ++nn2) {
          if (faces_to_nodes[(rface_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        const int rface_next_node = (nn2 == nnodes_by_rface - 1) ? 0 : nn2 + 1;
        const int rface_prev_node = (nn2 == 0) ? nnodes_by_rface - 1 : nn2 - 1;
        const int rface_rnode_off =
            (rface_clockwise ? rface_prev_node : rface_next_node);
        const int rface_rnode_index =
            faces_to_nodes[(rface_to_nodes_off + rface_rnode_off)];

        vec_t rface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_rface, nodes_x, nodes_y, nodes_z,
                      faces_to_nodes, rface_to_nodes_off, &rface_c);
        vec_t lface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_lface, nodes_x, nodes_y, nodes_z,
                      faces_to_nodes, lface_to_nodes_off, &lface_c);

        double inodes_x[NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_x[(node_index)] + nodes_x[(rface_rnode_index)]),
            rface_c.x, cell_c.x, lface_c.x};
        double inodes_y[NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_y[(node_index)] + nodes_y[(rface_rnode_index)]),
            rface_c.y, cell_c.y, lface_c.y};
        double inodes_z[NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_z[(node_index)] + nodes_z[(rface_rnode_index)]),
            rface_c.z, cell_c.z, lface_c.z};

        // TODO: Need to perform the remap again here
#endif // if 0
#endif // if 0
      }
    }
  }

#if 0
#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // TODO: I think it will be faster to calculate these on the fly, like so
    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x0, nodes_y0, nodes_z0, cells_to_nodes,
                  cell_to_nodes_off, &cell_c);

    vec_t rz_cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, rezoned_nodes_x, rezoned_nodes_y,
                  rezoned_nodes_z, cells_to_nodes, cell_to_nodes_off,
                  &rz_cell_centroid);

    /* LOOP OVER CELL FACES */
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      vec_t rz_face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x0, nodes_y0, nodes_z0,
                    faces_to_nodes, face_to_nodes_off, &face_c);
      calc_centroid(nnodes_by_face, rezoned_nodes_x, rezoned_nodes_y,
                    rezoned_nodes_z, faces_to_nodes, face_to_nodes_off,
                    &rz_face_c);

      /* LOOP OVER FACE NODES */
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        const int subcell_index = subcell_off + nn;

        subcell_mass_flux[(subcell_index)] = 0.0;
        subcell_ie_mass_flux[(subcell_index)] = 0.0;

        // We always determine the subcell with the value that is prescribed in
        // the subcell index to relate to the current node and its subsequent
        // node in the face list.
        const int next_off = ((nn == nnodes_by_face - 1) ? 0 : nn + 1);
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + next_off)];

        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        vec_t face_normal;
        const int face_clockwise = calc_surface_normal(
            n0, n1, n2, nodes_x0, nodes_y0, nodes_z0, &cell_c, &face_normal);

        // We may need to reorder the two corner nodes for the tet given that
        // the face may be ordered clockwise, rather than counter-clockwise.
        const int ccw_node_index = (face_clockwise) ? rnode_index : node_index;
        const int ccw_rnode_index = (face_clockwise) ? node_index : rnode_index;

        // Describe the subcell tetrahedron connectivity
        const int subcell_faces_to_nodes[] = {0, 1, 2, 0, 3, 1,
                                              1, 3, 2, 0, 2, 3};
        const vec_t subcell[] = {
            {nodes_x0[(ccw_node_index)], nodes_y0[(ccw_node_index)],
             nodes_z0[(ccw_node_index)]},
            {nodes_x0[(ccw_rnode_index)], nodes_y0[(ccw_rnode_index)],
             nodes_z0[(ccw_rnode_index)]},
            {face_c.x, face_c.y, face_c.z},
            {cell_c.x, cell_c.y, cell_c.z}};
        const vec_t rz_subcell[] = {
            {rezoned_nodes_x[(ccw_node_index)],
             rezoned_nodes_y[(ccw_node_index)],
             rezoned_nodes_z[(ccw_node_index)]},
            {rezoned_nodes_x[(ccw_rnode_index)],
             rezoned_nodes_y[(ccw_rnode_index)],
             rezoned_nodes_z[(ccw_rnode_index)]},
            {rz_face_c.x, rz_face_c.y, rz_face_c.z},
            {rz_cell_centroid.x, rz_cell_centroid.y, rz_cell_centroid.z}};

        // TODO: CHECK THE SUBCELL VOLUME IS POSITIVE

        /* LOOP OVER SUBCELL FACES */
        for (int pp = 0; pp < NTET_FACES; ++pp) {
        }
      }
    }

    // Update the volume of the cell to the new rezoned mesh
    calc_volume(cell_to_faces_off, nfaces_by_cell, cells_to_faces,
                faces_to_nodes, faces_to_nodes_offsets, rezoned_nodes_x,
                rezoned_nodes_y, rezoned_nodes_z, &rz_cell_centroid,
                &cell_volume[(cc)]);
  }
#endif // if 0

#if 0
/* REMAP MOMENTUM */

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // Determine aprpopiate cell centroids
    vec_t cell_c = {cell_centroids_x[(cc)], cell_centroids_y[(cc)],
                           cell_centroids_z[(cc)]};
    vec_t rz_cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, rezoned_nodes_x, rezoned_nodes_y,
                  rezoned_nodes_z, cells_to_nodes, cell_to_nodes_off,
                  &rz_cell_centroid);

    /* LOOP OVER CELL FACES */
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      vec_t rz_face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x0, nodes_y0, nodes_z0,
                    faces_to_nodes, face_to_nodes_off, &face_c);
      calc_centroid(nnodes_by_face, rezoned_nodes_x, rezoned_nodes_y,
                    rezoned_nodes_z, faces_to_nodes, face_to_nodes_off,
                    &rz_face_c);

      /* LOOP OVER FACE NODES */
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        double x_momentum_flux = 0.0;
        double y_momentum_flux = 0.0;
        double z_momentum_flux = 0.0;

        // We always determine the subcell with the value that is prescribed in
        // the subcell index to relate to the current node and its subsequent
        // node in the face list.
        const int next_off = ((nn == nnodes_by_face - 1) ? 0 : nn + 1);
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + next_off)];
        const int subcell_index = subcell_off + nn;

        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        vec_t face_normal;
        const int face_clockwise =
            calc_surface_normal(n0, n1, n2, nodes_x0, nodes_y0, nodes_z0,
                                &cell_c, &face_normal);

        // We may need to reorder the two corner nodes for the tet given that
        // the face may be ordered clockwise, rather than counter-clockwise.
        const int ccw_node_index = (face_clockwise) ? rnode_index : node_index;
        const int ccw_rnode_index = (face_clockwise) ? node_index : rnode_index;

        // Describe the subcell tetrahedron connectivity
        const int subcell_faces_to_nodes[] = {0, 1, 2, 0, 3, 1,
                                              1, 3, 2, 0, 2, 3};

        // Loop over both of the subsubcells that exist within the subcell
        for (int ss = 0; ss < NSUBSUBCELLS; ++ss) {
          const int subsubcell_index = subcell_index * NSUBSUBCELLS + ss;

          // Here for the momentum we advect using sub-subcells, which are
          // essentially half subcells, in order to maintain the conservation of
          // momentum at the corners. The result of performing the advection in
          // this manner is that we are able to return the nodal values
          // trivially and accurately.
          const vec_t subcell[] = {
              {nodes_x0[(ccw_node_index)], nodes_y0[(ccw_node_index)],
               nodes_z0[(ccw_node_index)]},
              {0.5 * (nodes_x0[(ccw_node_index)] + nodes_x0[(ccw_rnode_index)]),
               0.5 * (nodes_y0[(ccw_node_index)] + nodes_y0[(ccw_rnode_index)]),
               0.5 *
                   (nodes_z0[(ccw_node_index)] + nodes_z0[(ccw_rnode_index)])},
              {face_c.x, face_c.y, face_c.z},
              {cell_c.x, cell_c.y, cell_c.z}};
          const vec_t rz_subcell[] = {
              {rezoned_nodes_x[(ccw_node_index)],
               rezoned_nodes_y[(ccw_node_index)],
               rezoned_nodes_z[(ccw_node_index)]},
              {0.5 * (rezoned_nodes_x[(ccw_node_index)] +
                      rezoned_nodes_x[(ccw_rnode_index)]),
               0.5 * (rezoned_nodes_y[(ccw_node_index)] +
                      rezoned_nodes_y[(ccw_rnode_index)]),
               0.5 * (rezoned_nodes_z[(ccw_node_index)] +
                      rezoned_nodes_z[(ccw_rnode_index)])},
              {rz_face_c.x, rz_face_c.y, rz_face_c.z},
              {rz_cell_centroid.x, rz_cell_centroid.y, rz_cell_centroid.z}};

          /* LOOP OVER SUBCELL FACES */
          for (int pp = 0; pp < NTET_FACES; ++pp) {

            // Describe the swept edge prism
            const int swept_edge_faces_to_nodes_offsets[] = {0,  4,  7,
                                                             10, 14, 18};
            const int swept_edge_faces_to_nodes[] = {0, 1, 2, 3, 0, 3, 4, 2, 1,
                                                     5, 3, 2, 5, 4, 0, 4, 5, 1};
            const int swept_edge_to_faces[] = {0, 1, 2, 3, 4};

            const int sf0 =
                subcell_faces_to_nodes[(pp * NTET_NODES_PER_FACE + 0)];
            const int sf1 =
                subcell_faces_to_nodes[(pp * NTET_NODES_PER_FACE + 1)];
            const int sf2 =
                subcell_faces_to_nodes[(pp * NTET_NODES_PER_FACE + 2)];

            double swept_edge_nodes_x[] = {
                subcell[(sf0)].x, rz_subcell[(sf0)].x, rz_subcell[(sf1)].x,
                subcell[(sf1)].x, subcell[(sf2)].x,    rz_subcell[(sf2)].x};
            double swept_edge_nodes_y[] = {
                subcell[(sf0)].y, rz_subcell[(sf0)].y, rz_subcell[(sf1)].y,
                subcell[(sf1)].y, subcell[(sf2)].y,    rz_subcell[(sf2)].y};
            double swept_edge_nodes_z[] = {
                subcell[(sf0)].z, rz_subcell[(sf0)].z, rz_subcell[(sf1)].z,
                subcell[(sf1)].z, subcell[(sf2)].z,    rz_subcell[(sf2)].z};

            // Determine the swept edge prism's centroid
            vec_t swept_edge_c = {0.0, 0.0, 0.0};
            for (int pn = 0; pn < NPRISM_NODES; ++pn) {
              swept_edge_c.x += swept_edge_nodes_x[(pn)] / NPRISM_NODES;
              swept_edge_c.y += swept_edge_nodes_y[(pn)] / NPRISM_NODES;
              swept_edge_c.z += swept_edge_nodes_z[(pn)] / NPRISM_NODES;
            }

            // Calculate the volume of the swept edge prism
            double swept_edge_vol = 0.0;
            calc_volume(0, NPRISM_FACES, swept_edge_to_faces,
                        swept_edge_faces_to_nodes,
                        swept_edge_faces_to_nodes_offsets, swept_edge_nodes_x,
                        swept_edge_nodes_y, swept_edge_nodes_z,
                        &swept_edge_c, &swept_edge_vol);

            // Ignore the special case of an empty swept edge region
            if (fabs(swept_edge_vol - EPS) > 0.0) {
              continue;
            }

            // Determine the centroids of the subcell and rezoned faces
            vec_t subcell_face_c = {
                (subcell[(sf0)].x + subcell[(sf1)].x + subcell[(sf2)].x) /
                    NTET_NODES_PER_FACE,
                (subcell[(sf0)].y + subcell[(sf1)].y + subcell[(sf2)].y) /
                    NTET_NODES_PER_FACE,
                (subcell[(sf0)].z + subcell[(sf1)].z + subcell[(sf2)].z) /
                    NTET_NODES_PER_FACE};
            vec_t rz_subcell_face_c = {
                (rz_subcell[(sf0)].x + rz_subcell[(sf1)].x +
                 rz_subcell[(sf2)].x) /
                    NTET_NODES_PER_FACE,
                (rz_subcell[(sf0)].y + rz_subcell[(sf1)].y +
                 rz_subcell[(sf2)].y) /
                    NTET_NODES_PER_FACE,
                (rz_subcell[(sf0)].z + rz_subcell[(sf1)].z +
                 rz_subcell[(sf2)].z) /
                    NTET_NODES_PER_FACE};

            vec_t ab = {rz_subcell_face_c.x - subcell_face_c.x,
                        rz_subcell_face_c.y - subcell_face_c.y,
                        rz_subcell_face_c.z - subcell_face_c.z};
            vec_t ac = {subcell_face_c.x - subcell_centroids_x[(subcell_index)],
                        subcell_face_c.y - subcell_centroids_y[(subcell_index)],
                        subcell_face_c.z -
                            subcell_centroids_z[(subcell_index)]};

            const int is_outflux =
                (ab.x * ac.x + ab.y * ac.y + ab.z * ac.z < 0.0);

            // Depending upon which subcell we are sweeping into, choose the
            // subcell index with which to reconstruct the density
            const int subcell_neighbour_index = subcells_to_subcells[(
                subcell_index * NSUBCELL_NEIGHBOURS + pp)];
            const int sweep_subcell_index =
                (is_outflux ? subcell_index : subcell_neighbour_index);

            // Only perform the sweep on the external face if it isn't a
            // boundary
            if (subcell_neighbour_index == -1) {
              TERMINATE("We should not be attempting to flux from boundary.");
            }

            /* CALCULATE INVERSE COEFFICIENT MATRIX */

            // The 3x3 gradient coefficient matrix, and inverse
            vec_t inv[3] = {{0.0, 0.0, 0.0}};
            vec_t coeff[3] = {{0.0, 0.0, 0.0}};

            for (int ss2 = 0; ss2 < NSUBCELL_NEIGHBOURS; ++ss2) {
              const int neighbour_subcell_index = subcells_to_subcells[(
                  sweep_subcell_index * NSUBCELL_NEIGHBOURS + ss2)];

              // Ignore boundary neighbours
              if (neighbour_subcell_index == -1) {
                continue;
              }

              vec_t i = {subcell_centroids_x[(neighbour_subcell_index)] -
                             subcell_centroids_x[(sweep_subcell_index)],
                         subcell_centroids_y[(neighbour_subcell_index)] -
                             subcell_centroids_y[(sweep_subcell_index)],
                         subcell_centroids_z[(neighbour_subcell_index)] -
                             subcell_centroids_z[(sweep_subcell_index)]};

              // Store the neighbouring cell's contribution to the coefficients
              coeff[0].x += (i.x * i.x);
              coeff[0].y += (i.x * i.y);
              coeff[0].z += (i.x * i.z);
              coeff[1].x += (i.y * i.x);
              coeff[1].y += (i.y * i.y);
              coeff[1].z += (i.y * i.z);
              coeff[2].x += (i.z * i.x);
              coeff[2].y += (i.z * i.y);
              coeff[2].z += (i.z * i.z);
            }

            calc_3x3_inverse(&coeff, &inv);

            double gmax_vx = -DBL_MAX;
            double gmin_vx = DBL_MAX;
            double gmax_vy = -DBL_MAX;
            double gmin_vy = DBL_MAX;
            double gmax_vz = -DBL_MAX;
            double gmin_vz = DBL_MAX;

            double subcell_vx = velocity_x0[(node_index)] *
                                subcell_mass[(sweep_subcell_index)] /
                                subcell_volume[(sweep_subcell_index)];
            double subcell_vy = velocity_y0[(node_index)] *
                                subcell_mass[(sweep_subcell_index)] /
                                subcell_volume[(sweep_subcell_index)];
            double subcell_vz = velocity_z0[(node_index)] *
                                subcell_mass[(sweep_subcell_index)] /
                                subcell_volume[(sweep_subcell_index)];

            vec_t vx_rhs = {0.0, 0.0, 0.0};
            vec_t vy_rhs = {0.0, 0.0, 0.0};
            vec_t vz_rhs = {0.0, 0.0, 0.0};
            for (int ss2 = 0; ss2 < NSUBCELL_NEIGHBOURS; ++ss2) {
              const int neighbour_subcell_index = subcells_to_subcells[(
                  sweep_subcell_index * NSUBCELL_NEIGHBOURS + ss2)];

              if (neighbour_subcell_index == -1) {
                continue;
              }

              // Prepare differential
              // TODO: IS it correct to use the node_index here?
              const double dphi_vx =
                  (velocity_x0[(node_index)] *
                       subcell_mass[(neighbour_subcell_index)] /
                       subcell_volume[(neighbour_subcell_index)] -
                   subcell_vx);
              const double dphi_vy =
                  (velocity_y0[(node_index)] *
                       subcell_mass[(neighbour_subcell_index)] /
                       subcell_volume[(neighbour_subcell_index)] -
                   subcell_vy);
              const double dphi_vz =
                  (velocity_z0[(node_index)] *
                       subcell_mass[(neighbour_subcell_index)] /
                       subcell_volume[(neighbour_subcell_index)] -
                   subcell_vz);

              vx_rhs.x +=
                  dphi_vx * (subcell_centroids_x[(neighbour_subcell_index)] -
                             subcell_centroids_x[(sweep_subcell_index)]);
              vx_rhs.y +=
                  dphi_vx * (subcell_centroids_y[(neighbour_subcell_index)] -
                             subcell_centroids_y[(sweep_subcell_index)]);
              vx_rhs.z +=
                  dphi_vx * (subcell_centroids_z[(neighbour_subcell_index)] -
                             subcell_centroids_z[(sweep_subcell_index)]);

              vy_rhs.x +=
                  dphi_vy * (subcell_centroids_x[(neighbour_subcell_index)] -
                             subcell_centroids_x[(sweep_subcell_index)]);
              vy_rhs.y +=
                  dphi_vy * (subcell_centroids_y[(neighbour_subcell_index)] -
                             subcell_centroids_y[(sweep_subcell_index)]);
              vy_rhs.z +=
                  dphi_vy * (subcell_centroids_z[(neighbour_subcell_index)] -
                             subcell_centroids_z[(sweep_subcell_index)]);

              vz_rhs.x +=
                  dphi_vz * (subcell_centroids_x[(neighbour_subcell_index)] -
                             subcell_centroids_x[(sweep_subcell_index)]);
              vz_rhs.y +=
                  dphi_vz * (subcell_centroids_y[(neighbour_subcell_index)] -
                             subcell_centroids_y[(sweep_subcell_index)]);
              vz_rhs.z +=
                  dphi_vz * (subcell_centroids_z[(neighbour_subcell_index)] -
                             subcell_centroids_z[(sweep_subcell_index)]);

              gmax_vx = max(gmax_vx, dphi_vx);
              gmin_vx = min(gmin_vx, dphi_vx);
              gmax_vy = max(gmax_vy, dphi_vy);
              gmin_vy = min(gmin_vy, dphi_vy);
              gmax_vz = max(gmax_vz, dphi_vz);
              gmin_vz = min(gmin_vz, dphi_vz);
            }

            vec_t grad_vx = {
                inv[0].x * vx_rhs.x + inv[0].y * vx_rhs.y + inv[0].z * vx_rhs.z,
                inv[1].x * vx_rhs.x + inv[1].y * vx_rhs.y + inv[1].z * vx_rhs.z,
                inv[2].x * vx_rhs.x + inv[2].y * vx_rhs.y +
                    inv[2].z * vx_rhs.z};
            vec_t grad_vy = {
                inv[0].x * vy_rhs.x + inv[0].y * vy_rhs.y + inv[0].z * vy_rhs.z,
                inv[1].x * vy_rhs.x + inv[1].y * vy_rhs.y + inv[1].z * vy_rhs.z,
                inv[2].x * vy_rhs.x + inv[2].y * vy_rhs.y +
                    inv[2].z * vy_rhs.z};
            vec_t grad_vz = {
                inv[0].x * vz_rhs.x + inv[0].y * vz_rhs.y + inv[0].z * vz_rhs.z,
                inv[1].x * vz_rhs.x + inv[1].y * vz_rhs.y + inv[1].z * vz_rhs.z,
                inv[2].x * vz_rhs.x + inv[2].y * vz_rhs.y +
                    inv[2].z * vz_rhs.z};

            // TODO: SHOULD THIS BE BASED ON THE SUBCELLS???
            vec_t node = {nodes_x0[(node_index)], nodes_y0[(node_index)],
                          nodes_z0[(node_index)]};
            const int node_to_cells_off = nodes_offsets[(node_index)];
#if 0
            apply_node_limiter(ncells_by_node, node_to_cells_off,
                               cells_to_nodes, &grad_vx, &node,
                               cell_centroids_x, cell_centroids_y,
                               cell_centroids_z, subcell_vx, gmax_vx, gmin_vx);
            apply_node_limiter(ncells_by_node, node_to_cells_off,
                               cells_to_nodes, &grad_vy, &node,
                               cell_centroids_x, cell_centroids_y,
                               cell_centroids_z, subcell_vy, gmax_vy, gmin_vy);
            apply_node_limiter(ncells_by_node, node_to_cells_off,
                               cells_to_nodes, &grad_vz, &node,
                               cell_centroids_x, cell_centroids_y,
                               cell_centroids_z, subcell_vz, gmax_vz, gmin_vz);
#endif // if 0

            // Calculate the flux for internal energy density in the subcell
            const double local_x_momentum_flux =
                swept_edge_vol *
                (velocity_x0[(node_index)] *
                     subcell_mass[(sweep_subcell_index)] /
                     subcell_volume[(sweep_subcell_index)] +
                 grad_vx.x * (swept_edge_c.x -
                              subcell_centroids_x[(sweep_subcell_index)]) +
                 grad_vx.y * (swept_edge_c.y -
                              subcell_centroids_y[(sweep_subcell_index)]) +
                 grad_vx.z * (swept_edge_c.z -
                              subcell_centroids_z[(sweep_subcell_index)]));
            const double local_y_momentum_flux =
                swept_edge_vol *
                (velocity_y0[(node_index)] *
                     subcell_mass[(sweep_subcell_index)] /
                     subcell_volume[(sweep_subcell_index)] +
                 grad_vy.x * (swept_edge_c.x -
                              subcell_centroids_x[(sweep_subcell_index)]) +
                 grad_vy.y * (swept_edge_c.y -
                              subcell_centroids_y[(sweep_subcell_index)]) +
                 grad_vy.z * (swept_edge_c.z -
                              subcell_centroids_z[(sweep_subcell_index)]));
            const double local_z_momentum_flux =
                swept_edge_vol *
                (velocity_z0[(node_index)] *
                     subcell_mass[(sweep_subcell_index)] /
                     subcell_volume[(sweep_subcell_index)] +
                 grad_vz.x * (swept_edge_c.x -
                              subcell_centroids_x[(sweep_subcell_index)]) +
                 grad_vz.y * (swept_edge_c.y -
                              subcell_centroids_y[(sweep_subcell_index)]) +
                 grad_vz.z * (swept_edge_c.z -
                              subcell_centroids_z[(sweep_subcell_index)]));

            // Either the momentum is flowing in or out
            if (is_outflux) {
              x_momentum_flux += local_x_momentum_flux;
              y_momentum_flux += local_y_momentum_flux;
              z_momentum_flux += local_z_momentum_flux;
            } else {
              x_momentum_flux -= local_x_momentum_flux;
              y_momentum_flux -= local_y_momentum_flux;
              z_momentum_flux -= local_z_momentum_flux;
            }
          }

          // TODO: Check this is valid, we needed to use a slightly different
          // approach with the mass and energy so I'm not sure if this should be
          // split as well. The capacity will be very  large if we do have to...
          subcell_momentum_flux_x[(subsubcell_index)] -= x_momentum_flux;
          subcell_momentum_flux_y[(subsubcell_index)] -= y_momentum_flux;
          subcell_momentum_flux_z[(subsubcell_index)] -= z_momentum_flux;
        }
      }
    }

    // Calculates the weighted volume dist for a provided cell along x-y-z
    calc_volume(cell_to_faces_off, nfaces_by_cell, cells_to_faces,
                faces_to_nodes, faces_to_nodes_offsets, rezoned_nodes_x, rezoned_nodes_y,
                rezoned_nodes_z, &cell_c, &cell_volume[(cc)]);
  }

  double total_x_flux = 0.0;
  double total_y_flux = 0.0;
  double total_z_flux = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    total_x_flux += subcell_momentum_flux_x[(cc)];
    total_y_flux += subcell_momentum_flux_y[(cc)];
    total_z_flux += subcell_momentum_flux_z[(cc)];
  }
  printf("Total flux momentum %.12f %.12f %.12f\n", total_x_flux, total_y_flux,
         total_z_flux);
#endif // if 0
}

// Checks if the normal vector is pointing inward or outward
// n0 is just a point on the plane
int check_normal_orientation(const int n0, const double* nodes_x,
                             const double* nodes_y, const double* nodes_z,
                             const vec_t* centroid, vec_t* normal) {

  // Calculate a vector from face to cell centroid
  vec_t ab;
  ab.x = (centroid->x - nodes_x[(n0)]);
  ab.y = (centroid->y - nodes_y[(n0)]);
  ab.z = (centroid->z - nodes_z[(n0)]);

  return (ab.x * normal->x + ab.y * normal->y + ab.z * normal->z > 0.0);
}

// Calculates the surface normal of a vector pointing outwards
int calc_surface_normal(const int n0, const int n1, const int n2,
                        const double* nodes_x, const double* nodes_y,
                        const double* nodes_z, const vec_t* cell_c,
                        vec_t* normal) {

  // Calculate the unit normal vector
  calc_unit_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, normal);

  // Determine the orientation of the normal
  const int face_clockwise =
      check_normal_orientation(n0, nodes_x, nodes_y, nodes_z, cell_c, normal);

  // Flip the vector if necessary
  normal->x *= (face_clockwise ? -1.0 : 1.0);
  normal->y *= (face_clockwise ? -1.0 : 1.0);
  normal->z *= (face_clockwise ? -1.0 : 1.0);

  return face_clockwise;
}

// Calculate the normal vector from the provided nodes
void calc_unit_normal(const int n0, const int n1, const int n2,
                      const double* nodes_x, const double* nodes_y,
                      const double* nodes_z, vec_t* normal) {

  // Calculate the normal
  calc_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, normal);

  // Normalise the normal
  double len = sqrt(normal->x * normal->x + normal->y * normal->y +
                    normal->z * normal->z);

  // Force propagation of zero length normal
  if (len < EPS) {
    len = 0.0;
  }

  normal->x /= len;
  normal->y /= len;
  normal->z /= len;
}

// Calculate the normal for a plane
void calc_normal(const int n0, const int n1, const int n2,
                 const double* nodes_x, const double* nodes_y,
                 const double* nodes_z, vec_t* normal) {
  // Get two vectors on the face plane
  vec_t dn0 = {0.0, 0.0, 0.0};
  vec_t dn1 = {0.0, 0.0, 0.0};

  // Outwards facing normal for clockwise ordering
  dn0.x = nodes_x[(n0)] - nodes_x[(n1)];
  dn0.y = nodes_y[(n0)] - nodes_y[(n1)];
  dn0.z = nodes_z[(n0)] - nodes_z[(n1)];
  dn1.x = nodes_x[(n2)] - nodes_x[(n1)];
  dn1.y = nodes_y[(n2)] - nodes_y[(n1)];
  dn1.z = nodes_z[(n2)] - nodes_z[(n1)];

  // Cross product to get the normal
  normal->x = (dn0.y * dn1.z - dn1.y * dn0.z);
  normal->y = (dn0.z * dn1.x - dn1.z * dn0.x);
  normal->z = (dn0.x * dn1.y - dn1.x * dn0.y);
}

// Contributes a face to the volume of some cell
void contribute_face_volume(const int nnodes_by_face, const int* faces_to_nodes,
                            const double* nodes_x, const double* nodes_y,
                            const double* nodes_z, const vec_t* cell_c,
                            double* vol) {

  // Determine the outward facing unit normal vector
  vec_t normal = {0.0, 0.0, 0.0};
  const int face_clockwise = calc_surface_normal(
      faces_to_nodes[(0)], faces_to_nodes[(1)], faces_to_nodes[(2)], nodes_x,
      nodes_y, nodes_z, cell_c, &normal);

  vec_t face_c = {0.0, 0.0, 0.0};
  calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes, 0,
                &face_c);

  double tn_x[3];
  double tn_y[3];
  double tn_z[3];

  // We have a triangle per edge, which are per node on the face
  for (int tt = 0; tt < nnodes_by_face; ++tt) {
    const int next_node = (tt == nnodes_by_face - 1) ? 0 : tt + 1;
    const int prev_node = (tt == 0) ? nnodes_by_face - 1 : tt - 1;
    const int n0 = faces_to_nodes[(tt)];
    const int n1n = faces_to_nodes[(next_node)];
    const int n1p = faces_to_nodes[(prev_node)];
    const int faces_to_nodes_tri[3] = {0, 1, 2};

    // Construct the face triangle associated with the node
    tn_x[0] = nodes_x[(n0)];
    tn_y[0] = nodes_y[(n0)];
    tn_z[0] = nodes_z[(n0)];
    tn_x[1] = (face_clockwise ? nodes_x[(n1p)] : nodes_x[(n1n)]);
    tn_y[1] = (face_clockwise ? nodes_y[(n1p)] : nodes_y[(n1n)]);
    tn_z[1] = (face_clockwise ? nodes_z[(n1p)] : nodes_z[(n1n)]);
    tn_x[2] = face_c.x;
    tn_y[2] = face_c.y;
    tn_z[2] = face_c.z;

    vec_t tnormal = {0.0, 0.0, 0.0};
    calc_surface_normal(0, 1, 2, tn_x, tn_y, tn_z, cell_c, &tnormal);

    // The projection of the normal vector onto a point on the face
    double omega = -(tnormal.x * tn_x[(2)] + tnormal.y * tn_y[(2)] +
                     tnormal.z * tn_z[(2)]);

    // Select the orientation based on the face area
    int basis;
    if (fabs(tnormal.x) > fabs(tnormal.y)) {
      basis = (fabs(tnormal.x) > fabs(tnormal.z)) ? YZX : XYZ;
    } else {
      basis = (fabs(tnormal.z) > fabs(tnormal.y)) ? XYZ : ZXY;
    }

    // The basis ensures that gamma is always maximised
    if (basis == XYZ) {
      calc_face_integrals(3, 0, omega, faces_to_nodes_tri, tn_x, tn_y, tnormal,
                          vol);
    } else if (basis == YZX) {
      dswap(tnormal.x, tnormal.y);
      dswap(tnormal.y, tnormal.z);
      calc_face_integrals(3, 0, omega, faces_to_nodes_tri, tn_y, tn_z, tnormal,
                          vol);
    } else if (basis == ZXY) {
      dswap(tnormal.x, tnormal.y);
      dswap(tnormal.x, tnormal.z);
      calc_face_integrals(3, 0, omega, faces_to_nodes_tri, tn_z, tn_x, tnormal,
                          vol);
    }
  }
}

// Calculates the weighted volume dist for a provided cell along x-y-z
void calc_volume(const int cell_to_faces_off, const int nfaces_by_cell,
                 const int* cells_to_faces, const int* faces_to_nodes,
                 const int* faces_to_nodes_offsets, const double* nodes_x,
                 const double* nodes_y, const double* nodes_z,
                 const vec_t* cell_c, double* vol) {

  // Prepare to accumulate the volume
  *vol = 0.0;

  for (int ff = 0; ff < nfaces_by_cell; ++ff) {
    const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    contribute_face_volume(nnodes_by_face, &faces_to_nodes[(face_to_nodes_off)],
                           nodes_x, nodes_y, nodes_z, cell_c, vol);
  }

  if (isnan(*vol)) {
    *vol = 0.0;
    return;
  }

  *vol = fabs(*vol);
}

// Resolves the volume dist in alpha-beta-gamma basis
void calc_face_integrals(const int nnodes_by_face, const int face_to_nodes_off,
                         const double omega, const int* faces_to_nodes,
                         const double* nodes_alpha, const double* nodes_beta,
                         vec_t normal, double* vol) {

  double pione = 0.0;
  double pialpha = 0.0;
  double pibeta = 0.0;

  // Calculate the coefficients for the projected face integral
  for (int nn = 0; nn < nnodes_by_face; ++nn) {
    const int n0 = faces_to_nodes[(face_to_nodes_off + nn)];
    const int n1 = (nn == nnodes_by_face - 1)
                       ? faces_to_nodes[(face_to_nodes_off)]
                       : faces_to_nodes[(face_to_nodes_off + nn + 1)];

    // Calculate all of the coefficients
    const double a0 = nodes_alpha[(n0)];
    const double a1 = nodes_alpha[(n1)];
    const double b0 = nodes_beta[(n0)];
    const double b1 = nodes_beta[(n1)];
    const double dalpha = a1 - a0;
    const double dbeta = b1 - b0;
    const double Calpha = a1 * (a1 + a0) + a0 * a0;
    const double Cbeta = b1 * (b1 + b0) + b0 * b0;

    // Accumulate the projection dist
    pione += dbeta * (a1 + a0) / 2.0;
    pialpha += dbeta * (Calpha) / 6.0;
    pibeta -= dalpha * (Cbeta) / 6.0;
  }

  // Finalise the weighted face dist
  const double Falpha = pialpha / normal.z;
  const double Fbeta = pibeta / normal.z;
  const double Fgamma =
      -(normal.x * pialpha + normal.y * pibeta + omega * pione) /
      (normal.z * normal.z);

  *vol += (normal.x * Falpha + normal.y * Fbeta + normal.z * Fgamma) / 3.0;
}

// Stores the rezoned mesh specification as the original mesh. Until we
// determine a reasonable rezoning algorithm, this makes us Eulerian
void store_rezoned_mesh(const int nnodes, const double* nodes_x,
                        const double* nodes_y, const double* nodes_z,
                        double* rezoned_nodes_x, double* rezoned_nodes_y,
                        double* rezoned_nodes_z) {

// Store the rezoned nodes
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    rezoned_nodes_x[(nn)] = nodes_x[(nn)];
    rezoned_nodes_y[(nn)] = nodes_y[(nn)];
    rezoned_nodes_z[(nn)] = nodes_z[(nn)];
  }
}

// Calculate the centroid
void calc_centroid(const int nnodes, const double* nodes_x,
                   const double* nodes_y, const double* nodes_z,
                   const int* indirection, const int offset, vec_t* centroid) {

  centroid->x = 0.0;
  centroid->y = 0.0;
  centroid->z = 0.0;
  for (int nn2 = 0; nn2 < nnodes; ++nn2) {
    const int node_index = indirection[(offset + nn2)];
    centroid->x += nodes_x[(node_index)] / nnodes;
    centroid->y += nodes_y[(node_index)] / nnodes;
    centroid->z += nodes_z[(node_index)] / nnodes;
  }
}

// Calculates the inverse of a 3x3 matrix, out-of-place
void calc_3x3_inverse(vec_t (*a)[3], vec_t (*inv)[3]) {
  // Calculate the determinant of the 3x3
  const double det =
      (*a)[0].x * ((*a)[1].y * (*a)[2].z - (*a)[1].z * (*a)[2].y) -
      (*a)[0].y * ((*a)[1].x * (*a)[2].z - (*a)[1].z * (*a)[2].x) +
      (*a)[0].z * ((*a)[1].x * (*a)[2].y - (*a)[1].y * (*a)[2].x);

  // Check if the matrix is singular
  if (det == 0.0) {
    TERMINATE("singular coefficient matrix");
  } else {
    // Perform the simple and fast 3x3 matrix inverstion
    (*inv)[0].x = ((*a)[1].y * (*a)[2].z - (*a)[1].z * (*a)[2].y) / det;
    (*inv)[0].y = ((*a)[0].z * (*a)[2].y - (*a)[0].y * (*a)[2].z) / det;
    (*inv)[0].z = ((*a)[0].y * (*a)[1].z - (*a)[0].z * (*a)[1].y) / det;

    (*inv)[1].x = ((*a)[1].z * (*a)[2].x - (*a)[1].x * (*a)[2].z) / det;
    (*inv)[1].y = ((*a)[0].x * (*a)[2].z - (*a)[0].z * (*a)[2].x) / det;
    (*inv)[1].z = ((*a)[0].z * (*a)[1].x - (*a)[0].x * (*a)[1].z) / det;

    (*inv)[2].x = ((*a)[1].x * (*a)[2].y - (*a)[1].y * (*a)[2].x) / det;
    (*inv)[2].y = ((*a)[0].y * (*a)[2].x - (*a)[0].x * (*a)[2].y) / det;
    (*inv)[2].z = ((*a)[0].x * (*a)[1].y - (*a)[0].y * (*a)[1].x) / det;
  }
}

// Calculate the gradient for the
void calc_gradient(const int subcell_index, const int nsubcells_by_subcell,
                   const int subcell_to_subcells_off,
                   const int* subcells_to_subcells, const double* phi,
                   const double* subcell_centroids_x,
                   const double* subcell_centroids_y,
                   const double* subcell_centroids_z, const vec_t (*inv)[3],
                   vec_t* gradient) {

  // Calculate the gradient for the internal energy density
  vec_t rhs = {0.0, 0.0, 0.0};
  for (int ss2 = 0; ss2 < nsubcells_by_subcell; ++ss2) {
    const int neighbour_subcell_index =
        subcells_to_subcells[(subcell_to_subcells_off + ss2)];

    if (neighbour_subcell_index == -1) {
      continue;
    }

    // Prepare differential
    const double dphi = (phi[(neighbour_subcell_index)] - phi[(subcell_index)]);

    // Calculate the subcell gradients for all of the variables
    rhs.x += dphi * (subcell_centroids_x[(neighbour_subcell_index)] -
                     subcell_centroids_x[(subcell_index)]);
    rhs.y += dphi * (subcell_centroids_y[(neighbour_subcell_index)] -
                     subcell_centroids_y[(subcell_index)]);
    rhs.z += dphi * (subcell_centroids_z[(neighbour_subcell_index)] -
                     subcell_centroids_z[(subcell_index)]);
  }

  gradient->x = (*inv)[0].x * rhs.x + (*inv)[0].y * rhs.y + (*inv)[0].z * rhs.z;
  gradient->y = (*inv)[1].x * rhs.x + (*inv)[1].y * rhs.y + (*inv)[1].z * rhs.z;
  gradient->z = (*inv)[2].x * rhs.x + (*inv)[2].y * rhs.y + (*inv)[2].z * rhs.z;
}

// Calculates the limiter for the provided gradient
double apply_cell_limiter(const int nnodes_by_cell, const int cell_to_nodes_off,
                          const int* cells_to_nodes, vec_t* grad,
                          const vec_t* cell_c, const double* nodes_x0,
                          const double* nodes_y0, const double* nodes_z0,
                          const double phi, const double gmax,
                          const double gmin) {

  // Calculate the limiter for the gradient
  double limiter = 1.0;
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
    double g_unlimited = phi + grad->x * (nodes_x0[(node_index)] - cell_c->x) +
                         grad->y * (nodes_y0[(node_index)] - cell_c->y) +
                         grad->z * (nodes_z0[(node_index)] - cell_c->z);

    double node_limiter = 1.0;
    if (g_unlimited - phi > 0.0) {
      node_limiter = min(1.0, ((gmax - phi) / (g_unlimited - phi)));
    } else if (g_unlimited - phi < 0.0) {
      node_limiter = min(1.0, ((gmin - phi) / (g_unlimited - phi)));
    }
    limiter = min(limiter, node_limiter);
  }

  grad->x *= limiter;
  grad->y *= limiter;
  grad->z *= limiter;

  return limiter;
}

// Calculates the limiter for the provided gradient
double apply_node_limiter(const int ncells_by_node, const int node_to_cells_off,
                          const int* nodes_to_cells, vec_t* grad,
                          const vec_t* node, const double* cell_centroids_x,
                          const double* cell_centroids_y,
                          const double* cell_centroids_z, const double phi,
                          const double gmax, const double gmin) {

  // Calculate the limiter for the gradient
  double limiter = DBL_MAX;
  for (int cc = 0; cc < ncells_by_node; ++cc) {
    const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
    double g_unlimited = phi +
                         grad->x * (cell_centroids_x[(cell_index)] - node->x) +
                         grad->y * (cell_centroids_y[(cell_index)] - node->y) +
                         grad->z * (cell_centroids_z[(cell_index)] - node->z);

    double cell_limiter = 1.0;
    if (g_unlimited - phi > 0.0) {
      if (fabs(g_unlimited - phi) > EPS) {
        cell_limiter = min(1.0, ((gmax - phi) / (g_unlimited - phi)));
      }
    } else if (g_unlimited - phi < 0.0) {
      if (fabs(g_unlimited - phi) > EPS) {
        cell_limiter = min(1.0, ((gmin - phi) / (g_unlimited - phi)));
      }
    }
    limiter = min(limiter, cell_limiter);
  }

  grad->x *= limiter;
  grad->y *= limiter;
  grad->z *= limiter;

  return limiter;
}

// Applies the mesh rezoning strategy. This is a pure Eulerian strategy.
void apply_mesh_rezoning(const int nnodes, const double* rezoned_nodes_x,
                         const double* rezoned_nodes_y,
                         const double* rezoned_nodes_z, double* nodes_x,
                         double* nodes_y, double* nodes_z) {

// Apply the rezoned mesh into the main mesh
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x[(nn)] = rezoned_nodes_x[(nn)];
    nodes_y[(nn)] = rezoned_nodes_y[(nn)];
    nodes_z[(nn)] = rezoned_nodes_z[(nn)];
  }
}
