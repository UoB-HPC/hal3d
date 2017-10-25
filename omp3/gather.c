#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

// Gathers all of the subcell quantities on the mesh
void gather_subcell_mass_and_energy(
    const int ncells, const int nnodes, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z,
    int* cells_to_nodes_offsets, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const double* cell_volume, double* energy,
    double* density, double* velocity_x, double* velocity_y, double* velocity_z,
    double* ke_mass, double* cell_mass, double* subcell_mass,
    double* subcell_volume, double* subcell_ie_mass, double* subcell_ke_mass,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes,
    int* nodes_to_cells_offsets, int* nodes_to_cells, double* initial_mass,
    double* initial_ie_mass, double* initial_ke_mass);

// Gathers the momentum into the subcells
void gather_subcell_momentum(
    const int nnodes, const double* nodal_volumes, const double* nodal_mass,
    int* nodes_to_cells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, double* velocity_x, double* velocity_y,
    double* velocity_z, double* subcell_volume, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z,
    double* subcell_momentum_x, double* subcell_momentum_y,
    double* subcell_momentum_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    int* nodes_to_cells_offsets, int* cells_to_nodes_offsets,
    int* cells_to_nodes, int* nodes_to_nodes_offsets, int* nodes_to_nodes,
    vec_t* initial_momentum);

// gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(UnstructuredMesh* umesh, HaleData* hale_data,
                               vec_t* initial_momentum, double* initial_mass,
                               double* initial_ie_mass,
                               double* initial_ke_mass) {

  /*
  *      GATHERING STAGE OF THE REMAP
  */

  // Calculates the cell volume, subcell volume and the subcell centroids
  calc_volumes_centroids(
      umesh->ncells, umesh->nnodes, hale_data->nnodes_by_subcell,
      umesh->cells_offsets, umesh->cells_to_nodes,
      hale_data->subcells_to_faces_offsets, hale_data->subcells_to_faces,
      umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
      umesh->faces_cclockwise_cell, umesh->nodes_x0, umesh->nodes_y0,
      umesh->nodes_z0, hale_data->subcell_centroids_x,
      hale_data->subcell_centroids_y, hale_data->subcell_centroids_z,
      hale_data->subcell_volume, hale_data->cell_volume,
      hale_data->nodal_volumes, umesh->nodes_offsets, umesh->nodes_to_cells);

  // Gathers all of the subcell quantities on the mesh
  gather_subcell_mass_and_energy(
      umesh->ncells, umesh->nnodes, umesh->cell_centroids_x,
      umesh->cell_centroids_y, umesh->cell_centroids_z, umesh->cells_offsets,
      umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0, hale_data->cell_volume,
      hale_data->energy0, hale_data->density0, hale_data->velocity_x0,
      hale_data->velocity_y0, hale_data->velocity_z0, hale_data->ke_mass,
      hale_data->cell_mass, hale_data->subcell_mass, hale_data->subcell_volume,
      hale_data->subcell_ie_mass, hale_data->subcell_ke_mass,
      hale_data->subcell_centroids_x, hale_data->subcell_centroids_y,
      hale_data->subcell_centroids_z, umesh->faces_to_cells0,
      umesh->faces_to_cells1, umesh->cells_to_faces_offsets,
      umesh->cells_to_faces, umesh->cells_to_nodes, umesh->nodes_offsets,
      umesh->nodes_to_cells, initial_mass, initial_ie_mass, initial_ke_mass);

  // Gathers the momentum  the subcells
  gather_subcell_momentum(
      umesh->nnodes, hale_data->nodal_volumes, hale_data->nodal_mass,
      umesh->nodes_to_cells, umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
      hale_data->velocity_x0, hale_data->velocity_y0, hale_data->velocity_z0,
      hale_data->subcell_volume, umesh->cell_centroids_x,
      umesh->cell_centroids_y, umesh->cell_centroids_z,
      hale_data->subcell_momentum_x, hale_data->subcell_momentum_y,
      hale_data->subcell_momentum_z, hale_data->subcell_centroids_x,
      hale_data->subcell_centroids_y, hale_data->subcell_centroids_z,
      umesh->nodes_offsets, umesh->cells_offsets, umesh->cells_to_nodes,
      umesh->nodes_to_nodes_offsets, umesh->nodes_to_nodes, initial_momentum);
}

// Gathers all of the subcell quantities on the mesh
void gather_subcell_mass_and_energy(
    const int ncells, const int nnodes, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z,
    int* cells_to_nodes_offsets, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const double* cell_volume, double* energy,
    double* density, double* velocity_x, double* velocity_y, double* velocity_z,
    double* ke_mass, double* cell_mass, double* subcell_mass,
    double* subcell_volume, double* subcell_ie_mass, double* subcell_ke_mass,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes,
    int* nodes_to_cells_offsets, int* nodes_to_cells, double* initial_mass,
    double* initial_ie_mass, double* initial_ke_mass) {

  double total_mass = 0.0;
  double total_ie_mass = 0.0;
  double total_ke_mass = 0.0;
  double total_ie_in_subcells = 0.0;
  double total_ke_in_subcells = 0.0;

// We first have to determine the cell centered kinetic energy
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
    const int nnodes_by_cell =
        cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;

    ke_mass[(cc)] = 0.0;

    // Subcells are ordered with the nodes on a face
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      ke_mass[(cc)] += subcell_mass[(subcell_index)] *
                       (velocity_x[(node_index)] * velocity_x[(node_index)] +
                        velocity_y[(node_index)] * velocity_y[(node_index)] +
                        velocity_z[(node_index)] * velocity_z[(node_index)]);
    }
  }

// Calculate the sub-cell internal and kinetic energies
#pragma omp parallel for reduction(+ : total_mass, total_ie_mass,              \
                                   total_ke_mass, total_ie_in_subcells)
  for (int cc = 0; cc < ncells; ++cc) {
    // Calculating the volume dist necessary for the least squares
    // regression
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
    const int nnodes_by_cell =
        cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;

    const double cell_ie = density[(cc)] * energy[(cc)];
    const double cell_ke = ke_mass[(cc)] / cell_volume[(cc)];
    vec_t cell_c = {cell_centroids_x[(cc)], cell_centroids_y[(cc)],
                    cell_centroids_z[(cc)]};

    vec_t ie_rhs = {0.0, 0.0, 0.0};
    vec_t ke_rhs = {0.0, 0.0, 0.0};
    vec_t coeff[3] = {{0.0, 0.0, 0.0}};

    total_mass += cell_mass[(cc)];
    total_ie_mass += cell_mass[(cc)] * energy[(cc)];

    // Determine the weighted volume dist for neighbouring cells
    double gmax_ie = -DBL_MAX;
    double gmin_ie = DBL_MAX;
    double gmax_ke = -DBL_MAX;
    double gmin_ke = DBL_MAX;
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
      const double neighbour_ke = ke_mass[(neighbour_index)] / neighbour_vol;

      gmax_ie = max(gmax_ie, neighbour_ie);
      gmin_ie = min(gmin_ie, neighbour_ie);
      gmax_ke = max(gmax_ke, neighbour_ke);
      gmin_ke = min(gmin_ke, neighbour_ke);

      // Prepare the RHS, which includes energy differential
      const double die = (neighbour_ie - cell_ie);
      const double dke = (neighbour_ke - cell_ke);
      ie_rhs.x += 2.0 * (dist.x * die) / neighbour_vol;
      ie_rhs.y += 2.0 * (dist.y * die) / neighbour_vol;
      ie_rhs.z += 2.0 * (dist.z * die) / neighbour_vol;
      ke_rhs.x += 2.0 * (dist.x * dke) / neighbour_vol;
      ke_rhs.y += 2.0 * (dist.y * dke) / neighbour_vol;
      ke_rhs.z += 2.0 * (dist.z * dke) / neighbour_vol;
    }

    // Determine the inverse of the coefficient matrix
    vec_t inv[3];
    calc_3x3_inverse(&coeff, &inv);

    // Solve for the internal energy gradient
    vec_t grad_ie = {
        inv[0].x * ie_rhs.x + inv[0].y * ie_rhs.y + inv[0].z * ie_rhs.z,
        inv[1].x * ie_rhs.x + inv[1].y * ie_rhs.y + inv[1].z * ie_rhs.z,
        inv[2].x * ie_rhs.x + inv[2].y * ie_rhs.y + inv[2].z * ie_rhs.z};

    apply_cell_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                       &grad_ie, &cell_c, nodes_x, nodes_y, nodes_z, cell_ie,
                       gmax_ie, gmin_ie);

    // Solve for the kinetic energy gradient
    vec_t grad_ke = {
        inv[0].x * ke_rhs.x + inv[0].y * ke_rhs.y + inv[0].z * ke_rhs.z,
        inv[1].x * ke_rhs.x + inv[1].y * ke_rhs.y + inv[1].z * ke_rhs.z,
        inv[2].x * ke_rhs.x + inv[2].y * ke_rhs.y + inv[2].z * ke_rhs.z};

    apply_cell_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                       &grad_ke, &cell_c, nodes_x, nodes_y, nodes_z, cell_ke,
                       gmax_ke, gmin_ke);

    // Subcells are ordered with the nodes on a face
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int subcell_index = cell_to_nodes_off + nn;

      // Calculate the center of mass distance
      vec_t dist = {subcell_centroids_x[(subcell_index)] - cell_c.x,
                    subcell_centroids_y[(subcell_index)] - cell_c.y,
                    subcell_centroids_z[(subcell_index)] - cell_c.z};

      // Determine subcell internal energy from linear function at cell
      subcell_ie_mass[(subcell_index)] =
          subcell_volume[(subcell_index)] *
          (cell_ie + grad_ie.x * dist.x + grad_ie.y * dist.y +
           grad_ie.z * dist.z);

      // Determine subcell kinetic energy from linear function at cell
      subcell_ke_mass[(subcell_index)] =
          subcell_volume[(subcell_index)] *
          (cell_ke + grad_ke.x * dist.x + grad_ke.y * dist.y +
           grad_ke.z * dist.z);

      total_ie_in_subcells += subcell_ie_mass[(subcell_index)];
      total_ke_in_subcells += subcell_ke_mass[(subcell_index)];

      if (subcell_ie_mass[(subcell_index)] < 0.0) {
        printf("Negative internal energy mass %d %.12f\n", subcell_index,
               subcell_ie_mass[(subcell_index)]);
      }
    }
  }

  *initial_mass = total_mass;
  *initial_ie_mass = total_ie_mass;
  *initial_ke_mass = total_ie_mass;

  printf("Total Energy in Cells    %.12f\n", total_ie_mass + total_ke_mass);
  printf("Total Energy in Subcells %.12f\n",
         total_ie_in_subcells + total_ke_in_subcells);
  printf("Difference               %.12f\n\n",
         (total_ie_mass + total_ke_mass) -
             (total_ie_in_subcells + total_ke_in_subcells));
}

// Gathers the momentum into the subcells
void gather_subcell_momentum(
    const int nnodes, const double* nodal_volumes, const double* nodal_mass,
    int* nodes_to_cells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, double* velocity_x, double* velocity_y,
    double* velocity_z, double* subcell_volume, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z,
    double* subcell_momentum_x, double* subcell_momentum_y,
    double* subcell_momentum_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    int* nodes_to_cells_offsets, int* cells_to_nodes_offsets,
    int* cells_to_nodes, int* nodes_to_nodes_offsets, int* nodes_to_nodes,
    vec_t* initial_momentum) {

  double initial_momentum_x = 0.0;
  double initial_momentum_y = 0.0;
  double initial_momentum_z = 0.0;
  double total_subcell_vx = 0.0;
  double total_subcell_vy = 0.0;
  double total_subcell_vz = 0.0;

#pragma omp parallel for reduction(+ : initial_momentum_x, initial_momentum_y, \
                                   initial_momentum_z, total_subcell_vx,       \
                                   total_subcell_vy, total_subcell_vz)
  for (int nn = 0; nn < nnodes; ++nn) {

    // Calculate the gradient for the nodal momentum
    vec_t rhsx = {0.0, 0.0, 0.0};
    vec_t rhsy = {0.0, 0.0, 0.0};
    vec_t rhsz = {0.0, 0.0, 0.0};
    vec_t coeff[3] = {{0.0, 0.0, 0.0}};
    vec_t gmin = {DBL_MAX, DBL_MAX, DBL_MAX};
    vec_t gmax = {-DBL_MAX, -DBL_MAX, -DBL_MAX};

    const double nodal_density = nodal_mass[(nn)] / nodal_volumes[(nn)];
    vec_t node_mom_density = {nodal_density * velocity_x[(nn)],
                              nodal_density * velocity_y[(nn)],
                              nodal_density * velocity_z[(nn)]};

    initial_momentum_x += nodal_mass[(nn)] * velocity_x[(nn)];
    initial_momentum_y += nodal_mass[(nn)] * velocity_y[(nn)];
    initial_momentum_z += nodal_mass[(nn)] * velocity_z[(nn)];

    const int node_to_nodes_off = nodes_to_nodes_offsets[(nn)];
    const int nnodes_by_node =
        nodes_to_nodes_offsets[(nn + 1)] - node_to_nodes_off;

    for (int nn2 = 0; nn2 < nnodes_by_node; ++nn2) {
      const int neighbour_index = nodes_to_nodes[(node_to_nodes_off + nn2)];

      if (neighbour_index == -1) {
        continue;
      }

      // Calculate the center of mass distance
      vec_t i = {nodes_x[(neighbour_index)] - nodes_x[(nn)],
                 nodes_y[(neighbour_index)] - nodes_y[(nn)],
                 nodes_z[(neighbour_index)] - nodes_z[(nn)]};

      // Store the neighbouring cell's contribution to the coefficients
      double neighbour_vol = nodal_volumes[(neighbour_index)];
      coeff[0].x += 2.0 * (i.x * i.x) / (neighbour_vol * neighbour_vol);
      coeff[0].y += 2.0 * (i.x * i.y) / (neighbour_vol * neighbour_vol);
      coeff[0].z += 2.0 * (i.x * i.z) / (neighbour_vol * neighbour_vol);
      coeff[1].x += 2.0 * (i.y * i.x) / (neighbour_vol * neighbour_vol);
      coeff[1].y += 2.0 * (i.y * i.y) / (neighbour_vol * neighbour_vol);
      coeff[1].z += 2.0 * (i.y * i.z) / (neighbour_vol * neighbour_vol);
      coeff[2].x += 2.0 * (i.z * i.x) / (neighbour_vol * neighbour_vol);
      coeff[2].y += 2.0 * (i.z * i.y) / (neighbour_vol * neighbour_vol);
      coeff[2].z += 2.0 * (i.z * i.z) / (neighbour_vol * neighbour_vol);

      const double neighbour_nodal_density =
          nodal_mass[(neighbour_index)] / nodal_volumes[(neighbour_index)];

      vec_t neighbour_mom_density = {
          neighbour_nodal_density * velocity_x[(neighbour_index)],
          neighbour_nodal_density * velocity_y[(neighbour_index)],
          neighbour_nodal_density * velocity_z[(neighbour_index)]};

      gmax.x = max(gmax.x, neighbour_mom_density.x);
      gmin.x = min(gmin.x, neighbour_mom_density.x);
      gmax.y = max(gmax.y, neighbour_mom_density.y);
      gmin.y = min(gmin.y, neighbour_mom_density.y);
      gmax.z = max(gmax.z, neighbour_mom_density.z);
      gmin.z = min(gmin.z, neighbour_mom_density.z);

      vec_t dv = {(neighbour_mom_density.x - node_mom_density.x),
                  (neighbour_mom_density.y - node_mom_density.y),
                  (neighbour_mom_density.z - node_mom_density.z)};

      rhsx.x += 2.0 * i.x * dv.x / neighbour_vol;
      rhsx.y += 2.0 * i.y * dv.x / neighbour_vol;
      rhsx.z += 2.0 * i.z * dv.x / neighbour_vol;
      rhsy.x += 2.0 * i.x * dv.y / neighbour_vol;
      rhsy.y += 2.0 * i.y * dv.y / neighbour_vol;
      rhsy.z += 2.0 * i.z * dv.y / neighbour_vol;
      rhsz.x += 2.0 * i.x * dv.z / neighbour_vol;
      rhsz.y += 2.0 * i.y * dv.z / neighbour_vol;
      rhsz.z += 2.0 * i.z * dv.z / neighbour_vol;
    }

    // Determine the inverse of the coefficient matrix
    vec_t inv[3];
    calc_3x3_inverse(&coeff, &inv);

    // Solve for the x velocity gradient
    vec_t grad_vx = {inv[0].x * rhsx.x + inv[0].y * rhsx.y + inv[0].z * rhsx.z,
                     inv[1].x * rhsx.x + inv[1].y * rhsx.y + inv[1].z * rhsx.z,
                     inv[2].x * rhsx.x + inv[2].y * rhsx.y + inv[2].z * rhsx.z};

    // Solve for the y velocity gradient
    vec_t grad_vy = {inv[0].x * rhsy.x + inv[0].y * rhsy.y + inv[0].z * rhsy.z,
                     inv[1].x * rhsy.x + inv[1].y * rhsy.y + inv[1].z * rhsy.z,
                     inv[2].x * rhsy.x + inv[2].y * rhsy.y + inv[2].z * rhsy.z};

    // Solve for the z velocity gradient
    vec_t grad_vz = {inv[0].x * rhsz.x + inv[0].y * rhsz.y + inv[0].z * rhsz.z,
                     inv[1].x * rhsz.x + inv[1].y * rhsz.y + inv[1].z * rhsz.z,
                     inv[2].x * rhsz.x + inv[2].y * rhsz.y + inv[2].z * rhsz.z};

    // Limit the gradients
    const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
    const int ncells_by_node =
        nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

#if 0
    vec_t node = {nodes_x[(nn)], nodes_y[(nn)], nodes_z[(nn)]};
    double vx_limiter = 1.0;
    double vy_limiter = 1.0;
    double vz_limiter = 1.0;

    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];

      vx_limiter =
          min(vx_limiter,
              calc_node_limiter(node_mom_density.x, gmax.x, gmin.x, &grad_vx,
                                cell_centroids_x[(cell_index)],
                                cell_centroids_y[(cell_index)],
                                cell_centroids_z[(cell_index)], &node));
      vy_limiter =
          min(vy_limiter,
              calc_node_limiter(node_mom_density.y, gmax.y, gmin.y, &grad_vy,
                                cell_centroids_x[(cell_index)],
                                cell_centroids_y[(cell_index)],
                                cell_centroids_z[(cell_index)], &node));
      vz_limiter =
          min(vz_limiter,
              calc_node_limiter(node_mom_density.z, gmax.z, gmin.z, &grad_vz,
                                cell_centroids_x[(cell_index)],
                                cell_centroids_y[(cell_index)],
                                cell_centroids_z[(cell_index)], &node));
    }

    grad_vx.x *= vx_limiter;
    grad_vx.y *= vx_limiter;
    grad_vx.z *= vx_limiter;
    grad_vy.x *= vy_limiter;
    grad_vy.y *= vy_limiter;
    grad_vy.z *= vy_limiter;
    grad_vz.x *= vz_limiter;
    grad_vz.y *= vz_limiter;
    grad_vz.z *= vz_limiter;
#endif // if 0

    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_to_nodes_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_to_nodes_offsets[(cell_index + 1)] - cell_to_nodes_off;

      // Determine the position of the node in the cell
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
          break;
        }
      }

      const int subcell_index = cell_to_nodes_off + nn2;

      double vol = subcell_volume[(subcell_index)];
      double dx = subcell_centroids_x[(subcell_index)] - nodes_x[(nn)];
      double dy = subcell_centroids_y[(subcell_index)] - nodes_y[(nn)];
      double dz = subcell_centroids_z[(subcell_index)] - nodes_z[(nn)];

      subcell_momentum_x[(subcell_index)] =
          vol * (node_mom_density.x + grad_vx.x * dx + grad_vx.y * dy +
                 grad_vx.z * dz);
      subcell_momentum_y[(subcell_index)] =
          vol * (node_mom_density.y + grad_vy.x * dx + grad_vy.y * dy +
                 grad_vy.z * dz);
      subcell_momentum_z[(subcell_index)] =
          vol * (node_mom_density.z + grad_vz.x * dx + grad_vz.y * dy +
                 grad_vz.z * dz);

      total_subcell_vx += subcell_momentum_x[(subcell_index)];
      total_subcell_vy += subcell_momentum_y[(subcell_index)];
      total_subcell_vz += subcell_momentum_z[(subcell_index)];
    }
  }

  initial_momentum->x = initial_momentum_x;
  initial_momentum->y = initial_momentum_y;
  initial_momentum->z = initial_momentum_z;

  printf("\nTotal Momentum in Cells    (%.12f,%.12f,%.12f)\n",
         initial_momentum->x, initial_momentum->y, initial_momentum->z);
  printf("Total Momentum in Subcells (%.12f,%.12f,%.12f)\n", total_subcell_vx,
         total_subcell_vy, total_subcell_vz);
  printf("Difference                 (%.12f,%.12f,%.12f)\n\n",
         initial_momentum->x - total_subcell_vx,
         initial_momentum->y - total_subcell_vy,
         initial_momentum->z - total_subcell_vz);
}
