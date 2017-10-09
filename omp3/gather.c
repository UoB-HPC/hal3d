#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

// Gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(
    const int ncells, const int nnodes, const int nnodes_by_subcell,
    double* nodal_volumes, const double* nodal_mass, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_offsets,
    int* nodes_to_cells, int* nodes_offsets, double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* energy,
    double* density, double* velocity_x, double* velocity_y, double* velocity_z,
    double* cell_mass, double* subcell_volume, double* subcell_ie_mass,
    double* subcell_momentum_x, double* subcell_momentum_y,
    double* subcell_momentum_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* cell_volume, int* subcells_to_faces_offsets, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes,
    int* subcells_to_faces) {

  /*
  *      GATHERING STAGE OF THE REMAP
  */

  // Calculates the cell volume, subcell volume and the subcell centroids

  // Calculates the cell volume, subcell volume and the subcell centroids
  calc_volumes_centroids(
      ncells, nnodes, nnodes_by_subcell, cells_offsets, cells_to_nodes,
      cells_to_faces_offsets, cells_to_faces, subcells_to_faces_offsets,
      subcells_to_faces, faces_to_nodes, faces_to_nodes_offsets, nodes_x,
      nodes_y, nodes_z, subcell_centroids_x, subcell_centroids_y,
      subcell_centroids_z, subcell_volume, cell_volume, nodal_volumes,
      nodes_offsets, nodes_to_cells);

  // Gathers all of the subcell quantities on the mesh
  gather_subcell_energy(
      ncells, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      cells_offsets, nodes_x, nodes_y, nodes_z, cell_volume, energy, density,
      cell_mass, subcell_volume, subcell_ie_mass, subcell_centroids_x,
      subcell_centroids_y, subcell_centroids_z, faces_to_cells0,
      faces_to_cells1, cells_to_faces_offsets, cells_to_faces, cells_to_nodes);

  // Gathers the momentum into the subcells
  gather_subcell_momentum(
      ncells, nnodes, nodal_volumes, nodal_mass, cell_centroids_x,
      cell_centroids_y, cell_centroids_z, cells_offsets, nodes_to_cells,
      nodes_offsets, nodes_x, nodes_y, nodes_z, velocity_x, velocity_y,
      velocity_z, subcell_volume, subcell_momentum_x, subcell_momentum_y,
      subcell_momentum_z, subcell_centroids_x, subcell_centroids_y,
      subcell_centroids_z, faces_to_nodes, faces_to_nodes_offsets,
      cells_to_faces_offsets, cells_to_faces, cells_to_nodes);
}

// Gathers all of the subcell quantities on the mesh
void gather_subcell_energy(
    const int ncells, double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const double* cell_volume,
    double* energy, double* density, double* cell_mass, double* subcell_volume,
    double* subcell_ie_mass, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
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

    const double cell_ie = density[(cc)] * energy[(cc)];
    vec_t cell_c = {cell_centroids_x[(cc)], cell_centroids_y[(cc)],
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

      vec_t dist = {cell_centroids_x[(neighbour_index)] - cell_c.x,
                    cell_centroids_y[(neighbour_index)] - cell_c.y,
                    cell_centroids_z[(neighbour_index)] - cell_c.z};

      // Store the neighbouring cell's contribution to the coefficients
      double neighbour_vol = cell_volume[(neighbour_index)];
      coeff[0].x += 2.0 * (dist.x * dist.x) / (neighbour_vol * neighbour_vol);
      coeff[0].y += 2.0 * (dist.x * dist.y) / (neighbour_vol * neighbour_vol);
      coeff[0].z += 2.0 * (dist.x * dist.z) / (neighbour_vol * neighbour_vol);
      coeff[1].x += 2.0 * (dist.y * dist.x) / (neighbour_vol * neighbour_vol);
      coeff[1].y += 2.0 * (dist.y * dist.y) / (neighbour_vol * neighbour_vol);
      coeff[1].z += 2.0 * (dist.y * dist.z) / (neighbour_vol * neighbour_vol);
      coeff[2].x += 2.0 * (dist.z * dist.x) / (neighbour_vol * neighbour_vol);
      coeff[2].y += 2.0 * (dist.z * dist.y) / (neighbour_vol * neighbour_vol);
      coeff[2].z += 2.0 * (dist.z * dist.z) / (neighbour_vol * neighbour_vol);

      const double neighbour_ie =
          density[(neighbour_index)] * energy[(neighbour_index)];

      gmax = max(gmax, neighbour_ie);
      gmin = min(gmin, neighbour_ie);

      // Prepare the RHS, which includes energy differential
      const double die = (neighbour_ie - cell_ie);
      rhs.x += 2.0 * (dist.x * die) / neighbour_vol;
      rhs.y += 2.0 * (dist.y * die) / neighbour_vol;
      rhs.z += 2.0 * (dist.z * die) / neighbour_vol;
    }

    // Determine the inverse of the coefficient matrix
    vec_t inv[3];
    calc_3x3_inverse(&coeff, &inv);

    // Solve for the energy gradient
    vec_t grad_ie = {inv[0].x * rhs.x + inv[0].y * rhs.y + inv[0].z * rhs.z,
                     inv[1].x * rhs.x + inv[1].y * rhs.y + inv[1].z * rhs.z,
                     inv[2].x * rhs.x + inv[2].y * rhs.y + inv[2].z * rhs.z};

    apply_cell_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                       &grad_ie, &cell_c, nodes_x, nodes_y, nodes_z, cell_ie,
                       gmax, gmin);

    // Subcells are ordered with the nodes on a face
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int subcell_index = cell_to_nodes_off + nn;

      // Calculate the center of mass distance
      vec_t dist = {subcell_centroids_x[(subcell_index)] - cell_c.x,
                    subcell_centroids_y[(subcell_index)] - cell_c.y,
                    subcell_centroids_z[(subcell_index)] - cell_c.z};

      // Determine subcell energy from linear function at cell
      subcell_ie_mass[(subcell_index)] =
          subcell_volume[(subcell_index)] *
          (cell_ie + grad_ie.x * dist.x + grad_ie.y * dist.y +
           grad_ie.z * dist.z);

      if (subcell_ie_mass[(subcell_index)] < 0.0) {
        printf("Negative internal energy mass %d %.12f\n", subcell_index,
               subcell_ie_mass[(subcell_index)]);
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
    for (int ss = 0; ss < 8; ++ss) {
      total_ie_in_subcells += subcell_ie_mass[(cc * 8 + ss)];
      total_ie_in_cell += subcell_ie_mass[(cc * 8 + ss)];
    }
  }

  printf("Total Energy in Cells    %.12f\n", total_ie);
  printf("Total Energy in Subcells %.12f\n", total_ie_in_subcells);
  printf("Difference %.12f\n", total_ie - total_ie_in_subcells);
}

// Gathers the momentum into the subcells
void gather_subcell_momentum(
    const int ncells, const int nnodes, const double* nodal_volumes,
    const double* nodal_mass, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_offsets,
    int* nodes_to_cells, int* nodes_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* velocity_x,
    double* velocity_y, double* velocity_z, double* subcell_volume,
    double* subcell_momentum_x, double* subcell_momentum_y,
    double* subcell_momentum_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes) {

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
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // Loop over subcells
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      const int node_to_cells_off = nodes_offsets[(node_index)];
      const int ncells_by_node =
          nodes_offsets[(node_index + 1)] - node_to_cells_off;

      vec_t rhsx = {0.0, 0.0, 0.0};
      vec_t rhsy = {0.0, 0.0, 0.0};
      vec_t rhsz = {0.0, 0.0, 0.0};
      vec_t gmin = {DBL_MAX, DBL_MAX, DBL_MAX};
      vec_t gmax = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
      vec_t coeff[3] = {{0.0, 0.0, 0.0}};

      const double nodal_density =
          nodal_mass[(node_index)] / nodal_volumes[(node_index)];
      vec_t node_mom = {nodal_density * velocity_x[(node_index)],
                        nodal_density * velocity_y[(node_index)],
                        nodal_density * velocity_z[(node_index)]};

      // TODO: This gradient could be replaced by a calculation that operates
      // from node to node
      //
      // Calculate the gradient around the node
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
        double neighbour_vol = nodal_volumes[(node_index)];
        coeff[0].x += 2.0 * (dist.x * dist.x) / (neighbour_vol * neighbour_vol);
        coeff[0].y += 2.0 * (dist.x * dist.y) / (neighbour_vol * neighbour_vol);
        coeff[0].z += 2.0 * (dist.x * dist.z) / (neighbour_vol * neighbour_vol);
        coeff[1].x += 2.0 * (dist.y * dist.x) / (neighbour_vol * neighbour_vol);
        coeff[1].y += 2.0 * (dist.y * dist.y) / (neighbour_vol * neighbour_vol);
        coeff[1].z += 2.0 * (dist.y * dist.z) / (neighbour_vol * neighbour_vol);
        coeff[2].x += 2.0 * (dist.z * dist.x) / (neighbour_vol * neighbour_vol);
        coeff[2].y += 2.0 * (dist.z * dist.y) / (neighbour_vol * neighbour_vol);
        coeff[2].z += 2.0 * (dist.z * dist.z) / (neighbour_vol * neighbour_vol);

        const double neighbour_nodal_density =
            nodal_mass[(neighbour_index)] / nodal_volumes[(neighbour_index)];

        vec_t neighbour_node_mom = {
            neighbour_nodal_density * velocity_x[(neighbour_index)],
            neighbour_nodal_density * velocity_y[(neighbour_index)],
            neighbour_nodal_density * velocity_z[(neighbour_index)]};

        gmax.x = max(gmax.x, neighbour_node_mom.x);
        gmin.x = min(gmin.x, neighbour_node_mom.x);
        gmax.y = max(gmax.y, neighbour_node_mom.y);
        gmin.y = min(gmin.y, neighbour_node_mom.y);
        gmax.z = max(gmax.z, neighbour_node_mom.z);
        gmin.z = min(gmin.z, neighbour_node_mom.z);

        vec_t dv = {(neighbour_node_mom.x - node_mom.x),
                    (neighbour_node_mom.y - node_mom.y),
                    (neighbour_node_mom.z - node_mom.z)};

        rhsx.x += 2.0 * (dist.x * dv.x) / neighbour_vol;
        rhsx.y += 2.0 * (dist.y * dv.x) / neighbour_vol;
        rhsx.z += 2.0 * (dist.z * dv.x) / neighbour_vol;
        rhsy.x += 2.0 * (dist.x * dv.y) / neighbour_vol;
        rhsy.y += 2.0 * (dist.y * dv.y) / neighbour_vol;
        rhsy.z += 2.0 * (dist.z * dv.y) / neighbour_vol;
        rhsz.x += 2.0 * (dist.x * dv.z) / neighbour_vol;
        rhsz.y += 2.0 * (dist.y * dv.z) / neighbour_vol;
        rhsz.z += 2.0 * (dist.z * dv.z) / neighbour_vol;
      }

      // Determine the inverse of the coefficient matrix
      vec_t inv[3];
      calc_3x3_inverse(&coeff, &inv);

      vec_t node = {nodes_x[(node_index)], nodes_y[(node_index)],
                    nodes_z[(node_index)]};

      // Solve for the x velocity gradient
      vec_t grad_vx = {
          inv[0].x * rhsx.x + inv[0].y * rhsx.y + inv[0].z * rhsx.z,
          inv[1].x * rhsx.x + inv[1].y * rhsx.y + inv[1].z * rhsx.z,
          inv[2].x * rhsx.x + inv[2].y * rhsx.y + inv[2].z * rhsx.z};

      apply_node_limiter(ncells_by_node, node_to_cells_off, nodes_to_cells,
                         &grad_vx, &node, nodes_x, nodes_y, nodes_z, node_mom.x,
                         gmax.x, gmin.x);

      // Solve for the y velocity gradient
      vec_t grad_vy = {
          inv[0].x * rhsy.x + inv[0].y * rhsy.y + inv[0].z * rhsy.z,
          inv[1].x * rhsy.x + inv[1].y * rhsy.y + inv[1].z * rhsy.z,
          inv[2].x * rhsy.x + inv[2].y * rhsy.y + inv[2].z * rhsy.z};

      apply_node_limiter(ncells_by_node, node_to_cells_off, nodes_to_cells,
                         &grad_vy, &node, nodes_x, nodes_y, nodes_z, node_mom.y,
                         gmax.y, gmin.y);

      // Solve for the z velocity gradient
      vec_t grad_vz = {
          inv[0].x * rhsz.x + inv[0].y * rhsz.y + inv[0].z * rhsz.z,
          inv[1].x * rhsz.x + inv[1].y * rhsz.y + inv[1].z * rhsz.z,
          inv[2].x * rhsz.x + inv[2].y * rhsz.y + inv[2].z * rhsz.z};

      apply_node_limiter(ncells_by_node, node_to_cells_off, nodes_to_cells,
                         &grad_vz, &node, nodes_x, nodes_y, nodes_z, node_mom.z,
                         gmax.z, gmin.z);

      double vol = subcell_volume[(subcell_index)];
      double dx = subcell_centroids_x[(subcell_index)] - nodes_x[(node_index)];
      double dy = subcell_centroids_y[(subcell_index)] - nodes_y[(node_index)];
      double dz = subcell_centroids_z[(subcell_index)] - nodes_z[(node_index)];

      subcell_momentum_x[(subcell_index)] =
          vol * (node_mom.x + grad_vx.x * dx + grad_vx.y * dy + grad_vx.z * dz);
      subcell_momentum_y[(subcell_index)] =
          vol * (node_mom.y + grad_vy.x * dx + grad_vy.y * dy + grad_vy.z * dz);
      subcell_momentum_z[(subcell_index)] =
          vol * (node_mom.z + grad_vz.x * dx + grad_vz.y * dy + grad_vz.z * dz);

      total_subcell_vx += subcell_momentum_x[(subcell_index)];
      total_subcell_vy += subcell_momentum_y[(subcell_index)];
      total_subcell_vz += subcell_momentum_z[(subcell_index)];
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
}
