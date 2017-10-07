#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

// Gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(
    const int ncells, const int nnodes, const double* nodal_volumes,
    const double* nodal_mass, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_offsets,
    int* nodes_to_cells, int* nodes_offsets, double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* energy,
    double* density, double* velocity_x, double* velocity_y, double* velocity_z,
    double* cell_mass, double* subcell_volume, double* subcell_ie_mass,
    double* subcell_momentum_flux_x, double* subcell_momentum_flux_y,
    double* subcell_momentum_flux_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* cell_volume, int* subcells_to_faces_offsets, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes,
    int* nodes_to_faces_offsets, int* subcells_to_subcells_offsets,
    int* subcells_to_subcells, int* subcells_to_faces) {

  /*
  *      GATHERING STAGE OF THE REMAP
  */

  // Calculates the cell volume, subcell volume and the subcell centroids
  calc_volumes_centroids(ncells, cells_offsets, cells_to_nodes,
                         subcells_to_faces_offsets, subcells_to_faces,
                         faces_to_nodes, faces_to_nodes_offsets, nodes_x,
                         nodes_y, nodes_z, subcell_volume, cell_volume);

  // Gathers all of the subcell quantities on the mesh
  gather_subcell_energy(
      ncells, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      cells_offsets, nodes_x, nodes_y, nodes_z, energy, density, cell_mass,
      subcell_volume, subcell_ie_mass, subcell_centroids_x, subcell_centroids_y,
      subcell_centroids_z, subcells_to_faces_offsets, faces_to_nodes,
      faces_to_nodes_offsets, faces_to_cells0, faces_to_cells1,
      cells_to_faces_offsets, cells_to_faces, cells_to_nodes);

#if 0
  gather_subcell_momentum(
      ncells, nnodes, nodal_volumes, nodal_mass, cell_centroids_x,
      cell_centroids_y, cell_centroids_z, cells_offsets, nodes_to_cells,
      nodes_offsets, nodes_x, nodes_y, nodes_z, velocity_x, velocity_y,
      velocity_z, subcell_volume, subcell_momentum_flux_x,
      subcell_momentum_flux_y, subcell_momentum_flux_z, subcells_to_faces_offsets,
      faces_to_nodes, faces_to_nodes_offsets, cells_to_faces_offsets,
      cells_to_faces, cells_to_nodes);
#endif // if 0
}

// Calculates the cell volume, subcell volume and the subcell centroids
void calc_volumes_centroids(
    const int ncells, const int* cells_offsets, const int* cells_to_nodes,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes, const int* faces_to_nodes_offsets,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    double* subcell_volume, double* cell_volume) {

  double total_subcell_volume = 0.0;
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_centroid);

    // Looping over corner subcells here
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

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

        // Determine the orientation of the face
        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        vec_t face_normal;
        const int face_clockwise =
            calc_surface_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z,
                                &cell_centroid, &face_normal);

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

        const int faces_to_nodes[NNODES_BY_SUBCELL_FACE] = {0, 1, 2, 3};

        {
          double nodes_x[NNODES_BY_SUBCELL_FACE] = {
              nodes_x[(node_index)],
              0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]), face_c.x,
              0.5 * (nodes_x[(node_index)] + nodes_x[(lnode_index)])};
          double nodes_y[NNODES_BY_SUBCELL_FACE] = {
              nodes_y[(node_index)],
              0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]), face_c.y,
              0.5 * (nodes_y[(node_index)] + nodes_y[(lnode_index)])};
          double nodes_z[NNODES_BY_SUBCELL_FACE] = {
              nodes_x[(node_index)],
              0.5 * (nodes_z[(node_index)] + nodes_z[(rnode_index)]), face_c.z,
              0.5 * (nodes_z[(node_index)] + nodes_z[(lnode_index)])};

          contribute_face_volume(NNODES_BY_SUBCELL_FACE, faces_to_nodes,
                                 nodes_x, nodes_y, nodes_z, &cell_centroid,
                                 &subcell_volume[(subcell_index)]);
        }

        if (isnan(subcell_volume[(subcell_index)])) {
          subcell_volume[(subcell_index)] = 0.0;
        }

        /* INTERNAL FACE */

        const int rface_off = (ff == nfaces_by_subcell - 1) ? 0 : ff + 1;
        const int lface_off = (ff == 0) ? nfaces_by_subcell - 1 : ff - 1;
        const int rface_index =
            subcells_to_faces[(subcell_to_faces_off + rface_off)];
        const int rface_to_nodes_off = faces_to_nodes_offsets[(rface_index)];
        const int nnodes_by_rface =
            faces_to_nodes_offsets[(rface_index + 1)] - rface_to_nodes_off;

        const int lface_index =
            subcells_to_faces[(subcell_to_faces_off + lface_off)];
        const int lface_to_nodes_off = faces_to_nodes_offsets[(lface_index)];
        const int nnodes_by_lface =
            faces_to_nodes_offsets[(lface_index + 1)] - lface_to_nodes_off;

        // Determine the orientation of the face
        const int rn0 = faces_to_nodes[(rface_to_nodes_off + 0)];
        const int rn1 = faces_to_nodes[(rface_to_nodes_off + 1)];
        const int rn2 = faces_to_nodes[(rface_to_nodes_off + 2)];
        vec_t rface_normal;
        const int rface_clockwise =
            calc_surface_normal(rn0, rn1, rn2, nodes_x, nodes_y, nodes_z,
                                &cell_centroid, &rface_normal);

#if 0
        const int next_rnode = (nn2 == nnodes_by_rface - 1) ? 0 : nn2 + 1;
        const int prev_rnode = (nn2 == 0) ? nnodes_by_rface - 1 : nn2 - 1;
        const int rface_rnode_index = faces_to_nodes[(
            rface_to_nodes_off + (rface_clockwise ? prev_rnode : next_rnode))];

        // Determine the position of the node in the face list of nodes
        for (nn2 = 0; nn2 < nnodes_by_rface; ++nn2) {
          if (faces_to_nodes[(rface_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        vec_t rface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_rface, nodes_x, nodes_y, nodes_z,
                      faces_to_nodes, rface_to_nodes_off, &rface_c);
        vec_t lface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_lface, nodes_x, nodes_y, nodes_z,
                      faces_to_nodes, lface_to_nodes_off, &lface_c);

        {
          double nodes_x[NNODES_BY_SUBCELL_FACE] = {
              0.5 * (nodes_x[(node_index)] + nodes_x[(rface_rnode_index)]),
              rface_c.x, cell_centroid.x, lface_c.x};
          double nodes_y[NNODES_BY_SUBCELL_FACE] = {
              0.5 * (nodes_y[(node_index)] + nodes_y[(rface_rnode_index)]),
              rface_c.y, cell_centroid.y, lface_c.y};
          double nodes_z[NNODES_BY_SUBCELL_FACE] = {
              0.5 * (nodes_z[(node_index)] + nodes_z[(rface_rnode_index)]),
              rface_c.z, cell_centroid.z, lface_c.z};

          contribute_face_volume(NNODES_BY_SUBCELL_FACE, faces_to_nodes,
                                 nodes_x, nodes_y, nodes_z, &cell_centroid,
                                 &subcell_volume[(subcell_index)]);
        }
#endif // if 0
      }

      total_subcell_volume += subcell_volume[(subcell_index)];
    }
  }

  printf("Total Subcell Volume at Gather %.12f\n", total_subcell_volume);
}

// Gathers all of the subcell quantities on the mesh
void gather_subcell_energy(
    const int ncells, double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* energy,
    double* density, double* cell_mass, double* subcell_volume,
    double* subcell_ie_mass, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    int* subcells_to_faces_offsets, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes) {

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

    const double cell_ie = density[(cc)] * energy[(cc)];
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
          density[(neighbour_index)] * energy[(neighbour_index)];

      gmax = max(gmax, neighbour_ie - cell_ie);
      gmin = min(gmin, neighbour_ie - cell_ie);

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
    vec_t grad_ie = {inv[0].x * rhs.x + inv[0].y * rhs.y + inv[0].z * rhs.z,
                     inv[1].x * rhs.x + inv[1].y * rhs.y + inv[1].z * rhs.z,
                     inv[2].x * rhs.x + inv[2].y * rhs.y + inv[2].z * rhs.z};

    apply_cell_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                       &grad_ie, &cell_centroid, nodes_x, nodes_y, nodes_z,
                       cell_ie, gmax, gmin);

    // Determine the weighted volume dist for neighbouring cells
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off =
          subcells_to_faces_offsets[(cell_to_faces_off + ff)];

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                    face_to_nodes_off, &face_c);

      // Subcells are ordered with the nodes on a face
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int subcell_index = subcell_off + nn;

        // Calculate the center of mass distance
        vec_t dist = {subcell_centroids_x[(subcell_index)] - cell_centroid.x,
                      subcell_centroids_y[(subcell_index)] - cell_centroid.y,
                      subcell_centroids_z[(subcell_index)] - cell_centroid.z};

        // Determine subcell energy from linear function at cell
        subcell_ie_mass[(subcell_index)] =
            subcell_volume[(subcell_index)] *
            (cell_ie + grad_ie.x * dist.x + grad_ie.y * dist.y +
             grad_ie.z * dist.z);

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
    double ie = cell_mass[(cc)] * energy[(cc)];
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
    int* nodes_to_cells, int* nodes_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* velocity_x,
    double* velocity_y, double* velocity_z, double* subcell_volume,
    double* subcell_momentum_flux_x, double* subcell_momentum_flux_y,
    double* subcell_momentum_flux_z, int* subcells_to_faces_offsets,
    int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes) {

#if 0
  // The following method is a homegrown solution. It doesn't feel totally
  // precise, but it is a quite reasonable approach based on the popular
  // methods but seems to end up with lots of computational work (much of which
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
      const int subcell_off =
          subcells_to_faces_offsets[(cell_to_faces_off + ff)];

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                    face_to_nodes_off, &face_c);

      /* LOOP OVER FACE NODES */
      for (int nn = 0; nn < nnodes_by_face; ++nn) {

        // Fetch the right and left node indices
        const int rnode_off = (nn == nnodes_by_face - 1) ? 0 : nn + 1;
        const int lnode_off = (nn == 0) ? nnodes_by_face - 1 : nn - 1;
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];
        const int lnode_index = faces_to_nodes[(face_to_nodes_off + lnode_off)];
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];

        vec_t rhsx = {0.0, 0.0, 0.0};
        vec_t rhsy = {0.0, 0.0, 0.0};
        vec_t rhsz = {0.0, 0.0, 0.0};
        vec_t gmin = {DBL_MAX, DBL_MAX, DBL_MAX};
        vec_t gmax = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
        vec_t coeff[3] = {{0.0, 0.0, 0.0}};

        const double nodal_density =
            nodal_mass[(node_index)] / nodal_volumes[(node_index)];
        vec_t node_v = {nodal_density * velocity_x[(node_index)],
                        nodal_density * velocity_y[(node_index)],
                        nodal_density * velocity_z[(node_index)]};

        // Calculate the gradient for the node
        for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
          const int neighbour_index = cells_to_nodes[(cell_to_nodes_off + nn2)];

          if (neighbour_index == node_index) {
            continue;
          }

          // Calculate the center of mass distance
          vec_t dist = {nodes_x[(neighbour_index)] - nodes_x[(node_index)],
                        nodes_y[(neighbour_index)] - nodes_y[(node_index)],
                        nodes_z[(neighbour_index)] - nodes_z[(node_index)]};

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

          gmax.x = max(gmax.x, nodal_density * velocity_x[(neighbour_index)]);
          gmin.x = min(gmin.x, nodal_density * velocity_x[(neighbour_index)]);
          gmax.y = max(gmax.y, nodal_density * velocity_y[(neighbour_index)]);
          gmin.y = min(gmin.y, nodal_density * velocity_y[(neighbour_index)]);
          gmax.z = max(gmax.z, nodal_density * velocity_z[(neighbour_index)]);
          gmin.z = min(gmin.z, nodal_density * velocity_z[(neighbour_index)]);

          // Prepare the RHSs for the different momentums
          const double neighbour_nodal_density =
              nodal_mass[(neighbour_index)] / nodal_volumes[(neighbour_index)];
          vec_t dv = {(neighbour_nodal_density * velocity_x[(neighbour_index)] -
                       node_v.x),
                      (neighbour_nodal_density * velocity_y[(neighbour_index)] -
                       node_v.y),
                      (neighbour_nodal_density * velocity_z[(neighbour_index)] -
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

        const int node_to_cells_off = nodes_offsets[(node_index)];
        const int ncells_by_node =
            nodes_offsets[(node_index + 1)] - node_to_cells_off;
        apply_node_limiter(ncells_by_node, node_to_cells_off, nodes_to_cells,
                           &grad_vx, &cell_centroid, nodes_x, nodes_y, nodes_z,
                           node_v.x, gmax.x, gmin.x);

        // Solve for the y velocity gradient
        vec_t grad_vy;
        grad_vy.x = inv[0].x * rhsy.x + inv[0].y * rhsy.y + inv[0].z * rhsy.z;
        grad_vy.y = inv[1].x * rhsy.x + inv[1].y * rhsy.y + inv[1].z * rhsy.z;
        grad_vy.z = inv[2].x * rhsy.x + inv[2].y * rhsy.y + inv[2].z * rhsy.z;

        apply_node_limiter(ncells_by_node, cell_to_nodes_off, nodes_to_cells,
                           &grad_vy, &cell_centroid, nodes_x, nodes_y, nodes_z,
                           node_v.y, gmax.y, gmin.y);

        // Solve for the z velocity gradient
        vec_t grad_vz;
        grad_vz.x = inv[0].x * rhsz.x + inv[0].y * rhsz.y + inv[0].z * rhsz.z;
        grad_vz.y = inv[1].x * rhsz.x + inv[1].y * rhsz.y + inv[1].z * rhsz.z;
        grad_vz.z = inv[2].x * rhsz.x + inv[2].y * rhsz.y + inv[2].z * rhsz.z;

        apply_node_limiter(ncells_by_node, cell_to_nodes_off, nodes_to_cells,
                           &grad_vz, &cell_centroid, nodes_x, nodes_y, nodes_z,
                           node_v.z, gmax.z, gmin.z);

        // Now we will construct and determine flux for right subsubcell
        const int subcell_index = (subcell_off + nn);
        double vol = 0.5 * subcell_volume[(subcell_index)];

        // Determine the sub-sub-cell centroid
        vec_t subsubcell_c = {
            (nodes_x[(node_index)] +
             0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]) +
             cell_centroid.x + face_c.x) /
                NTET_NODES,
            (nodes_y[(node_index)] +
             0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]) +
             cell_centroid.y + face_c.y) /
                NTET_NODES,
            (nodes_z[(node_index)] +
             0.5 * (nodes_z[(node_index)] + nodes_z[(rnode_index)]) +
             cell_centroid.z + face_c.z) /
                NTET_NODES};

        const int subsubcell_index = (subcell_off + nn) * NSUBSUBCELLS;
        subcell_momentum_flux_x[(subsubcell_index)] =
            vol *
            (node_v.x + grad_vx.x * (subsubcell_c.x - nodes_x[(node_index)]) +
             grad_vx.y * (subsubcell_c.y - nodes_y[(node_index)]) +
             grad_vx.z * (subsubcell_c.z - nodes_z[(node_index)]));
        subcell_momentum_flux_y[(subsubcell_index)] =
            vol *
            (node_v.y + grad_vy.x * (subsubcell_c.x - nodes_x[(node_index)]) +
             grad_vy.y * (subsubcell_c.y - nodes_y[(node_index)]) +
             grad_vy.z * (subsubcell_c.z - nodes_z[(node_index)]));
        subcell_momentum_flux_z[(subsubcell_index)] =
            vol *
            (node_v.z + grad_vz.x * (subsubcell_c.x - nodes_x[(node_index)]) +
             grad_vz.y * (subsubcell_c.y - nodes_y[(node_index)]) +
             grad_vz.z * (subsubcell_c.z - nodes_z[(node_index)]));

        const int lsubcell_index = (subcell_off + lnode_off);
        const int lsubsubcell_index = (subcell_off + nn * NSUBSUBCELLS + 1);

        // Calculate the center of mass
        vol = subcell_volume[(lsubcell_index)] / NSUBSUBCELLS;

        // Determine the sub-sub-cell centroid
        vec_t lsubsubcell_c = {
            (nodes_x[(node_index)] +
             0.5 * (nodes_x[(lnode_index)] + nodes_x[(node_index)]) +
             cell_centroid.x + face_c.x) /
                NTET_NODES,
            (nodes_y[(node_index)] +
             0.5 * (nodes_y[(lnode_index)] + nodes_y[(node_index)]) +
             cell_centroid.y + face_c.y) /
                NTET_NODES,
            (nodes_z[(node_index)] +
             0.5 * (nodes_z[(lnode_index)] + nodes_z[(node_index)]) +
             cell_centroid.z + face_c.z) /
                NTET_NODES};

        subcell_momentum_flux_x[(lsubsubcell_index)] =
            vol *
            (node_v.x + grad_vx.x * (lsubsubcell_c.x - nodes_x[(node_index)]) +
             grad_vx.y * (lsubsubcell_c.y - nodes_y[(node_index)]) +
             grad_vx.z * (lsubsubcell_c.z - nodes_z[(node_index)]));
        subcell_momentum_flux_y[(lsubsubcell_index)] =
            vol *
            (node_v.y + grad_vy.x * (lsubsubcell_c.x - nodes_x[(node_index)]) +
             grad_vy.y * (lsubsubcell_c.y - nodes_y[(node_index)]) +
             grad_vy.z * (lsubsubcell_c.z - nodes_z[(node_index)]));
        subcell_momentum_flux_z[(lsubsubcell_index)] =
            vol *
            (node_v.z + grad_vz.x * (lsubsubcell_c.x - nodes_x[(node_index)]) +
             grad_vz.y * (lsubsubcell_c.y - nodes_y[(node_index)]) +
             grad_vz.z * (lsubsubcell_c.z - nodes_z[(node_index)]));

        total_subcell_vx += subcell_momentum_flux_x[(subsubcell_index)] +
                            subcell_momentum_flux_x[(lsubsubcell_index)];
        total_subcell_vy += subcell_momentum_flux_y[(subsubcell_index)] +
                            subcell_momentum_flux_y[(lsubsubcell_index)];
        total_subcell_vz += subcell_momentum_flux_z[(subsubcell_index)] +
                            subcell_momentum_flux_z[(lsubsubcell_index)];
      }
    }
  }

  double total_vx = 0.0;
  double total_vy = 0.0;
  double total_vz = 0.0;
#pragma omp parallel for reduction(+ : total_vx, total_vy, total_vz)
  for (int nn = 0; nn < nnodes; ++nn) {
    total_vx += nodal_mass[nn] * velocity_x[nn];
    total_vy += nodal_mass[nn] * velocity_y[nn];
    total_vz += nodal_mass[nn] * velocity_z[nn];
  }

  printf("\nTotal Momentum in Cells    (%.12f,%.12f,%.12f)\n", total_vx,
         total_vy, total_vz);
  printf("Total Momentum in Subcells (%.12f,%.12f,%.12f)\n", total_subcell_vx,
         total_subcell_vy, total_subcell_vz);
  printf("Difference                 (%.12f,%.12f,%.12f)\n",
         total_vx - total_subcell_vx, total_vy - total_subcell_vy,
         total_vz - total_subcell_vz);
#endif // if 0
}
