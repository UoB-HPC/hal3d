#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <stdio.h>

// Gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(
    const int ncells, const int nnodes, const double* nodal_volumes,
    const double* nodal_mass, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_offsets,
    double* nodes_x0, const double* nodes_y0, const double* nodes_z0,
    double* energy0, double* density0, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* cell_mass, double* subcell_volume,
    double* subcell_ie_mass0, double* subcell_momentum_x,
    double* subcell_momentum_y, double* subcell_momentum_z,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, double* cell_volume, int* subcell_face_offsets,
    int* faces_to_nodes, int* faces_to_nodes_offsets, int* faces_to_cells0,
    int* faces_to_cells1, int* cells_to_faces_offsets, int* cells_to_faces,
    int* cells_to_nodes) {

  /*
  *      GATHERING STAGE OF THE REMAP
  */

  calc_volumes_centroids(ncells, cells_to_faces_offsets, cell_centroids_x,
                         cell_centroids_y, cell_centroids_z, cells_to_faces,
                         faces_to_nodes, faces_to_nodes_offsets,
                         subcell_face_offsets, nodes_x0, nodes_y0, nodes_z0,
                         cell_volume, subcell_centroids_x, subcell_centroids_y,
                         subcell_centroids_z, subcell_volume);

  gather_subcell_momentum(
      ncells, nnodes, nodal_volumes, nodal_mass, cell_centroids_x,
      cell_centroids_y, cell_centroids_z, cells_offsets, nodes_x0, nodes_y0,
      nodes_z0, velocity_x0, velocity_y0, velocity_z0, subcell_volume,
      subcell_momentum_x, subcell_momentum_y, subcell_momentum_z,
      subcell_face_offsets, faces_to_nodes, faces_to_nodes_offsets,
      cells_to_faces_offsets, cells_to_faces, cells_to_nodes);

  // Gathers all of the subcell quantities on the mesh
  gather_subcell_energy(
      ncells, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      cells_offsets, nodes_x0, nodes_y0, nodes_z0, energy0, density0, cell_mass,
      subcell_volume, subcell_ie_mass0, subcell_centroids_x,
      subcell_centroids_y, subcell_centroids_z, subcell_face_offsets,
      faces_to_nodes, faces_to_nodes_offsets, faces_to_cells0, faces_to_cells1,
      cells_to_faces_offsets, cells_to_faces, cells_to_nodes);
}

// Gathers all of the subcell quantities on the mesh
void gather_subcell_energy(
    const int ncells, double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_offsets, const double* nodes_x0,
    const double* nodes_y0, const double* nodes_z0, double* energy0,
    double* density0, double* cell_mass, double* subcell_volume,
    double* subcell_ie_mass, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    int* subcell_face_offsets, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* faces_to_cells0, int* faces_to_cells1, int* cells_to_faces_offsets,
    int* cells_to_faces, int* cells_to_nodes) {

// Calculate the sub-cell internal energies
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    // Calculating the volume dist necessary for the least squares
    // regression
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    const double cell_ie = density0[(cc)] * energy0[(cc)];
    vec_t cell_centroid = {cell_centroids_x[(cc)], cell_centroids_y[(cc)],
                           cell_centroids_z[(cc)]};

    vec_t rhs = {0.0, 0.0, 0.0};
    vec_t coeff[3] = {{0.0, 0.0, 0.0}};

    // Determine the weighted volume dist for neighbouring cells
    double gmax = -DBL_MAX;
    double gmin = DBL_MAX;
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int neighbour_index = (faces_to_cells0[(face_index)] == cc)
                                      ? faces_to_cells1[(face_index)]
                                      : faces_to_cells0[(face_index)];

      // Check if boundary face
      if (neighbour_index == -1) {
        continue;
      }

      vec_t dist = {cell_centroids_x[(neighbour_index)] - cell_centroid.x,
                    cell_centroids_y[(neighbour_index)] - cell_centroid.y,
                    cell_centroids_z[(neighbour_index)] - cell_centroid.z};

      // Store the neighbouring cell's contribution to the coefficients
      coeff[0].x += (dist.x * dist.x);
      coeff[0].y += (dist.x * dist.y);
      coeff[0].z += (dist.x * dist.z);
      coeff[1].x += (dist.y * dist.x);
      coeff[1].y += (dist.y * dist.y);
      coeff[1].z += (dist.y * dist.z);
      coeff[2].x += (dist.z * dist.x);
      coeff[2].y += (dist.z * dist.y);
      coeff[2].z += (dist.z * dist.z);

      const double neighbour_ie =
          density0[(neighbour_index)] * energy0[(neighbour_index)];

      gmax = max(gmax, neighbour_ie);
      gmin = min(gmin, neighbour_ie);

      // Prepare the RHS, which includes energy differential
      const double de = (neighbour_ie - cell_ie);
      rhs.x += (dist.x * de);
      rhs.y += (dist.y * de);
      rhs.z += (dist.z * de);
    }

    // Determine the inverse of the coefficient matrix
    vec_t inv[3];
    calc_3x3_inverse(&coeff, &inv);

    // Solve for the energy gradient
    vec_t grad_energy = {inv[0].x * rhs.x + inv[0].y * rhs.y + inv[0].z * rhs.z,
                         inv[1].x * rhs.x + inv[1].y * rhs.y + inv[1].z * rhs.z,
                         inv[2].x * rhs.x + inv[2].y * rhs.y +
                             inv[2].z * rhs.z};

    apply_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                  &grad_energy, &cell_centroid, nodes_x0, nodes_y0, nodes_z0,
                  cell_ie, gmax, gmin);

    // Determine the weighted volume dist for neighbouring cells
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x0, nodes_y0, nodes_z0,
                    faces_to_nodes, face_to_nodes_off, &face_c);

      // Each face/node pair has two sub-cells
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // The left and right nodes on the face for this anchor node
        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        const int subcell_index = subcell_off + nn;

        vec_t normal;
        const int face_clockwise = calc_surface_normal(
            n0, n1, n2, nodes_x0, nodes_y0, nodes_z0, &cell_centroid, &normal);

        int rnode_index;
        if (face_clockwise) {
          rnode_index = faces_to_nodes[(
              face_to_nodes_off + ((nn == 0) ? nnodes_by_face - 1 : nn - 1))];
        } else {
          rnode_index = faces_to_nodes[(
              face_to_nodes_off + ((nn == nnodes_by_face - 1) ? 0 : nn + 1))];
        }

        // Calculate the center of mass
        vec_t dist = {subcell_centroids_x[(subcell_index)] - cell_centroid.x,
                      subcell_centroids_y[(subcell_index)] - cell_centroid.y,
                      subcell_centroids_z[(subcell_index)] - cell_centroid.z};

        // Determine subcell energy from linear function at cell
        subcell_ie_mass[(subcell_index)] =
            subcell_volume[(subcell_index)] *
            (cell_ie + (grad_energy.x * (dist.x) + grad_energy.y * (dist.y) +
                        grad_energy.z * (dist.z)));

        if (subcell_ie_mass[(subcell_index)] < 0.0) {
          printf("neg ie mass %d %.12f\n", subcell_index,
                 subcell_ie_mass[(subcell_index)]);
        }
      }
    }
  }

  // Print out the conservation of energy following the gathering
  double total_ie = 0.0;
  double total_ie_in_subcells = 0.0;
#pragma omp parallel for reduction(+ : total_ie, total_ie_in_subcells)
  for (int cc = 0; cc < ncells; ++cc) {
    double ie = cell_mass[(cc)] * energy0[(cc)];
    total_ie += ie;
    double total_ie_in_cell = 0.0;
    for (int ss = 0; ss < 24; ++ss) {
      total_ie_in_subcells += subcell_ie_mass[(cc * 24 + ss)];
      total_ie_in_cell += subcell_ie_mass[(cc * 24 + ss)];
    }
  }

  printf("Total Energy in Cells    %.12f\n", total_ie);
  printf("Total Energy in Subcells %.12f\n", total_ie_in_subcells);
  printf("Difference %.12f\n", total_ie - total_ie_in_subcells);
}

void gather_subcell_momentum(
    const int ncells, const int nnodes, const double* nodal_volumes,
    const double* nodal_mass, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_offsets,
    const double* nodes_x0, const double* nodes_y0, const double* nodes_z0,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* subcell_volume, double* subcell_momentum_x,
    double* subcell_momentum_y, double* subcell_momentum_z,
    int* subcell_face_offsets, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes) {

  // The following method is a homegrown solution. It doesn't feel totally
  // precise, but it is a quite reasonable approach based on the popular
  // methods and seems to end up with lots of computational work (much of which
  // is redundant).
  double total_subcell_vx = 0.0;
  double total_subcell_vy = 0.0;
  double total_subcell_vz = 0.0;
#pragma omp parallel for reduction(+ : total_subcell_vx, total_subcell_vy,     \
                                   total_subcell_vz)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid = {cell_centroids_x[(cc)], cell_centroids_y[(cc)],
                           cell_centroids_z[(cc)]};

    /* LOOP OVER CELL FACES */
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x0, nodes_y0, nodes_z0,
                    faces_to_nodes, face_to_nodes_off, &face_c);

      /* LOOP OVER FACE NODES */
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // The subcell index is doubled into sub-subcells, hence the index
        const int subsubcell_index = (subcell_off + nn) * NSUBSUBCELLS;

        // Determine the outward facing unit normal vector
        vec_t normal = {0.0, 0.0, 0.0};
        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        const int face_clockwise = calc_surface_normal(
            n0, n1, n2, nodes_x0, nodes_y0, nodes_z0, &cell_centroid, &normal);
        const int rnode = (nn == nnodes_by_face - 1) ? 0 : nn + 1;
        const int lnode = (nn == 0) ? nnodes_by_face - 1 : nn - 1;
        const int rnode_index = faces_to_nodes[(
            face_to_nodes_off + (face_clockwise ? lnode : rnode))];
        const int lnode_index = faces_to_nodes[(
            face_to_nodes_off + (!face_clockwise ? lnode : rnode))];

        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        vec_t gmax = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
        vec_t gmin = {DBL_MAX, DBL_MAX, DBL_MAX};
        vec_t rhsx = {0.0, 0.0, 0.0};
        vec_t rhsy = {0.0, 0.0, 0.0};
        vec_t rhsz = {0.0, 0.0, 0.0};
        vec_t coeff[3] = {{0.0, 0.0, 0.0}};

        const double nodal_density =
            nodal_mass[(node_index)] / nodal_volumes[(node_index)];
        vec_t node_v = {nodal_density * velocity_x0[(node_index)],
                        nodal_density * velocity_y0[(node_index)],
                        nodal_density * velocity_z0[(node_index)]};

        // Calculate the gradient for the node
        for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
          const int neighbour_index = cells_to_nodes[(cell_to_nodes_off + nn2)];

          if (neighbour_index == node_index) {
            continue;
          }

          // Calculate the center of mass distance
          vec_t dist = {nodes_x0[(neighbour_index)] - nodes_x0[(node_index)],
                        nodes_y0[(neighbour_index)] - nodes_y0[(node_index)],
                        nodes_z0[(neighbour_index)] - nodes_z0[(node_index)]};

          // Store the neighbouring cell's contribution to the coefficients
          coeff[0].x += (dist.x * dist.x);
          coeff[0].y += (dist.x * dist.y);
          coeff[0].z += (dist.x * dist.z);
          coeff[1].x += (dist.y * dist.x);
          coeff[1].y += (dist.y * dist.y);
          coeff[1].z += (dist.y * dist.z);
          coeff[2].x += (dist.z * dist.x);
          coeff[2].y += (dist.z * dist.y);
          coeff[2].z += (dist.z * dist.z);

          gmax.x = max(gmax.x, nodal_density * velocity_x0[(neighbour_index)]);
          gmin.x = min(gmin.x, nodal_density * velocity_x0[(neighbour_index)]);
          gmax.y = max(gmax.y, nodal_density * velocity_y0[(neighbour_index)]);
          gmin.y = min(gmin.y, nodal_density * velocity_y0[(neighbour_index)]);
          gmax.z = max(gmax.z, nodal_density * velocity_z0[(neighbour_index)]);
          gmin.z = min(gmin.z, nodal_density * velocity_z0[(neighbour_index)]);

          // Prepare the RHSs for the different momentums
          const double neighbour_nodal_density =
              nodal_mass[(neighbour_index)] / nodal_volumes[(neighbour_index)];
          vec_t dv = {
              (neighbour_nodal_density * velocity_x0[(neighbour_index)] -
               node_v.x),
              (neighbour_nodal_density * velocity_y0[(neighbour_index)] -
               node_v.y),
              (neighbour_nodal_density * velocity_z0[(neighbour_index)] -
               node_v.z)};

          rhsx.x += (dist.x * dv.x);
          rhsx.y += (dist.y * dv.x);
          rhsx.z += (dist.z * dv.x);
          rhsy.x += (dist.x * dv.y);
          rhsy.y += (dist.y * dv.y);
          rhsy.z += (dist.z * dv.y);
          rhsz.x += (dist.x * dv.z);
          rhsz.y += (dist.y * dv.z);
          rhsz.z += (dist.z * dv.z);
        }

        // Determine the inverse of the coefficient matrix
        vec_t inv[3];
        calc_3x3_inverse(&coeff, &inv);

        // Solve for the x velocity gradient
        vec_t grad_vx;
        grad_vx.x = inv[0].x * rhsx.x + inv[0].y * rhsx.y + inv[0].z * rhsx.z;
        grad_vx.y = inv[1].x * rhsx.x + inv[1].y * rhsx.y + inv[1].z * rhsx.z;
        grad_vx.z = inv[2].x * rhsx.x + inv[2].y * rhsx.y + inv[2].z * rhsx.z;

#if 0
        apply_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                      &grad_vx, &cell_centroid, nodes_x0, nodes_y0, nodes_z0,
                      node_v.x, gmax.x, gmin.x);
#endif // if 0

        // Solve for the y velocity gradient
        vec_t grad_vy;
        grad_vy.x = inv[0].x * rhsy.x + inv[0].y * rhsy.y + inv[0].z * rhsy.z;
        grad_vy.y = inv[1].x * rhsy.x + inv[1].y * rhsy.y + inv[1].z * rhsy.z;
        grad_vy.z = inv[2].x * rhsy.x + inv[2].y * rhsy.y + inv[2].z * rhsy.z;

#if 0
        apply_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                      &grad_vy, &cell_centroid, nodes_x0, nodes_y0, nodes_z0,
                      node_v.y, gmax.y, gmin.y);
#endif // if 0

        // Solve for the z velocity gradient
        vec_t grad_vz;
        grad_vz.x = inv[0].x * rhsz.x + inv[0].y * rhsz.y + inv[0].z * rhsz.z;
        grad_vz.y = inv[1].x * rhsz.x + inv[1].y * rhsz.y + inv[1].z * rhsz.z;
        grad_vz.z = inv[2].x * rhsz.x + inv[2].y * rhsz.y + inv[2].z * rhsz.z;

#if 0
        apply_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                      &grad_vz, &cell_centroid, nodes_x0, nodes_y0, nodes_z0,
                      node_v.z, gmax.z, gmin.z);
#endif // if 0

        // Now that we have the gradient for the node, we can determine the
        // values at the two adjoining sub-subcells on the face.

        // Calculate the center of mass
        const int subcell_index = (subcell_off + nn);
        double vol = 0.5 * subcell_volume[(subcell_index)];

        // Determine the sub-sub-cell centroid
        vec_t subsubcell_c = {
            (nodes_x0[(node_index)] +
             0.5 * (nodes_x0[(node_index)] + nodes_x0[(rnode_index)]) +
             cell_centroid.x + face_c.x) /
                NTET_NODES,
            (nodes_y0[(node_index)] +
             0.5 * (nodes_y0[(node_index)] + nodes_y0[(rnode_index)]) +
             cell_centroid.y + face_c.y) /
                NTET_NODES,
            (nodes_z0[(node_index)] +
             0.5 * (nodes_z0[(node_index)] + nodes_z0[(rnode_index)]) +
             cell_centroid.z + face_c.z) /
                NTET_NODES};

        subcell_momentum_x[(subsubcell_index)] =
            vol *
            (node_v.x + grad_vx.x * (subsubcell_c.x - nodes_x0[(node_index)]) +
             grad_vx.y * (subsubcell_c.y - nodes_y0[(node_index)]) +
             grad_vx.z * (subsubcell_c.z - nodes_z0[(node_index)]));
        subcell_momentum_y[(subsubcell_index)] =
            vol *
            (node_v.y + grad_vy.x * (subsubcell_c.x - nodes_x0[(node_index)]) +
             grad_vy.y * (subsubcell_c.y - nodes_y0[(node_index)]) +
             grad_vy.z * (subsubcell_c.z - nodes_z0[(node_index)]));
        subcell_momentum_z[(subsubcell_index)] =
            vol *
            (node_v.z + grad_vz.x * (subsubcell_c.x - nodes_x0[(node_index)]) +
             grad_vz.y * (subsubcell_c.y - nodes_y0[(node_index)]) +
             grad_vz.z * (subsubcell_c.z - nodes_z0[(node_index)]));

        const int lsubsubcell_index =
            (subcell_off + (face_clockwise ? lnode : rnode)) * 2 + 1;
        const int lsubcell_index = (subcell_off + nn);

        // Calculate the center of mass
        vol = subcell_volume[(lsubcell_index)] / NSUBSUBCELLS;

        // Determine the sub-sub-cell centroid
        vec_t lsubsubcell_c = {
            (nodes_x0[(node_index)] +
             0.5 * (nodes_x0[(lnode_index)] + nodes_x0[(node_index)]) +
             cell_centroid.x + face_c.x) /
                NTET_NODES,
            (nodes_y0[(node_index)] +
             0.5 * (nodes_y0[(lnode_index)] + nodes_y0[(node_index)]) +
             cell_centroid.y + face_c.y) /
                NTET_NODES,
            (nodes_z0[(node_index)] +
             0.5 * (nodes_z0[(lnode_index)] + nodes_z0[(node_index)]) +
             cell_centroid.z + face_c.z) /
                NTET_NODES};

        subcell_momentum_x[(lsubsubcell_index)] =
            vol *
            (node_v.x + grad_vx.x * (lsubsubcell_c.x - nodes_x0[(node_index)]) +
             grad_vx.y * (lsubsubcell_c.y - nodes_y0[(node_index)]) +
             grad_vx.z * (lsubsubcell_c.z - nodes_z0[(node_index)]));
        subcell_momentum_y[(lsubsubcell_index)] =
            vol *
            (node_v.y + grad_vy.x * (lsubsubcell_c.x - nodes_x0[(node_index)]) +
             grad_vy.y * (lsubsubcell_c.y - nodes_y0[(node_index)]) +
             grad_vy.z * (lsubsubcell_c.z - nodes_z0[(node_index)]));
        subcell_momentum_z[(lsubsubcell_index)] =
            vol *
            (node_v.z + grad_vz.x * (lsubsubcell_c.x - nodes_x0[(node_index)]) +
             grad_vz.y * (lsubsubcell_c.y - nodes_y0[(node_index)]) +
             grad_vz.z * (lsubsubcell_c.z - nodes_z0[(node_index)]));

        total_subcell_vx += subcell_momentum_x[(subsubcell_index)];
        total_subcell_vy += subcell_momentum_y[(subsubcell_index)];
        total_subcell_vz += subcell_momentum_z[(subsubcell_index)];
        total_subcell_vx += subcell_momentum_x[(lsubsubcell_index)];
        total_subcell_vy += subcell_momentum_y[(lsubsubcell_index)];
        total_subcell_vz += subcell_momentum_z[(lsubsubcell_index)];
      }
    }
  }

  double total_vx = 0.0;
  double total_vy = 0.0;
  double total_vz = 0.0;
#pragma omp parallel for reduction(+ : total_vx, total_vy, total_vz)
  for (int nn = 0; nn < nnodes; ++nn) {
    total_vx += nodal_mass[nn] * velocity_x0[nn];
    total_vy += nodal_mass[nn] * velocity_y0[nn];
    total_vz += nodal_mass[nn] * velocity_z0[nn];
  }

  printf("\nTotal Momentum in Cells    (%.12f,%.12f,%.12f)\n", total_vx,
         total_vy, total_vz);
  printf("Total Momentum in Subcells (%.12f,%.12f,%.12f)\n", total_subcell_vx,
         total_subcell_vy, total_subcell_vz);
  printf("Difference                 (%.12f,%.12f,%.12f)\n",
         total_vx - total_subcell_vx, total_vy - total_subcell_vy,
         total_vz - total_subcell_vz);
}
