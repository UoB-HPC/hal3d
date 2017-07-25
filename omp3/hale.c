#include "hale.h"
#include "../../comms.h"
#include "../../params.h"
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
    const int nsub_cell_neighbours, const double visc_coeff1,
    const double visc_coeff2, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_to_nodes,
    int* cells_offsets, int* nodes_to_cells, int* cells_to_cells,
    int* nodes_offsets, double* nodes_x0, double* nodes_y0, double* nodes_z0,
    double* nodes_x1, double* nodes_y1, double* nodes_z1, int* boundary_index,
    int* boundary_type, const double* original_nodes_x,
    const double* original_nodes_y, const double* original_nodes_z,
    double* boundary_normal_x, double* boundary_normal_y,
    double* boundary_normal_z, double* energy0, double* energy1,
    double* density0, double* density1, double* pressure0, double* pressure1,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* velocity_x1, double* velocity_y1, double* velocity_z1,
    double* sub_cell_force_x, double* sub_cell_force_y,
    double* sub_cell_force_z, double* node_force_x, double* node_force_y,
    double* node_force_z, double* cell_mass, double* nodal_mass,
    double* nodal_volumes, double* nodal_soundspeed, double* limiter,
    double* sub_cell_volume, double* sub_cell_energy, double* sub_cell_mass,
    double* sub_cell_velocity_x, double* sub_cell_velocity_y,
    double* sub_cell_velocity_z, double* sub_cell_kinetic_energy,
    double* sub_cell_centroids_x, double* sub_cell_centroids_y,
    double* sub_cell_centroids_z, double* sub_cell_grad_x,
    double* sub_cell_grad_y, double* sub_cell_grad_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces) {

  double total_mass = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
  }
  printf("total mass %.12f\n", total_mass);

// Calculate the sub-cell velocities

#define N 8 // numbers of nodes
#define F 6 // number of faces
  double mn[N] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  int n_to_f[8 * 3] = {0, 1, 2, 0, 2, 3, 2, 3, 5, 1, 2, 5,
                       0, 1, 4, 0, 3, 4, 3, 4, 5, 1, 4, 5};
  int f_to_n[4 * 6] = {0, 1, 5, 4, 0, 3, 7, 4, 0, 1, 2, 3,
                       1, 2, 6, 5, 4, 5, 6, 7, 3, 2, 6, 7};
  int Nf[F] = {4, 4, 4, 4, 4, 4};
  int Fn[N] = {3, 3, 3, 3, 3, 3, 3, 3};
  double u[N] = {2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 4.5, 1.0};

  double mid = 0.0;
  for (int nn = 0; nn < N; ++nn) {
    mid += u[nn] / N;
  }

#if 0
  double mus = 0.0;
  for (int nn = 0; nn < N; ++nn) {
    double us = 0.0;
    for (int ff = 0; ff < Fn[nn]; ++ff) {
      int face = n_to_f[nn * 3 + ff];
      int node;
      double fc = 0.0;
      for (int nn2 = 0; nn2 < Nf[face]; ++nn2) {
        fc += u[f_to_n[face * 4 + nn2]] / Nf[face]; // Get the face center value
        if (f_to_n[face * 4 + nn2] == nn) {
          node = nn2;
        }
      }

      us += (u[nn] + mid + fc +
             0.5 * (u[nn] +
                    u[f_to_n[face * 4 +
                             ((node == Nf[face] - 1) ? (0) : (node + 1))]]));
      us += (u[nn] + mid + fc +
             0.5 * (u[nn] + u[f_to_n[face * 4 + ((node == 0) ? (Nf[face] - 1)
                                                             : (node - 1))]]));
    }
    printf("us %.12f\n", us);
    mus += mn[nn] * 0.25 * us / (2.0 * Fn[nn]);
  }

  double exp = 0.0;
  for (int nn = 0; nn < N; ++nn) {
    exp += mn[nn] * u[nn];
  }

  printf("mus : %.12f, exp: %.12f\n", mus, exp);
#endif // if 0

  double a = 0.0;
  for (int nn = 0; nn < N; ++nn) {
    a += 2.5 * mn[nn] * u[nn];
  }

  double b = 0.0;
  double c = 0.0;
  for (int nn = 0; nn < N; ++nn) {
    double r = 0.0;
    for (int ff = 0; ff < Fn[nn]; ++ff) {
      double fc = 0.0;
      int face = n_to_f[nn * 3 + ff];
      int node;
      for (int nn2 = 0; nn2 < Nf[face]; ++nn2) {
        fc += u[f_to_n[face * 4 + nn2]] / Nf[face]; // Get the face center value
        if (f_to_n[face * 4 + nn2] == nn) {
          node = nn2;
        }
      }
      r += u[f_to_n[face * 4 + ((node == Nf[face] - 1) ? (0) : (node + 1))]];
      r += u[f_to_n[face * 4 + ((node == 0) ? (Nf[face] - 1) : (node - 1))]];
      b += (mn[nn] * 2.0 * fc) / (2.0 * Fn[nn]);
    }
    c += (mn[nn] * r) / (4.0 * Fn[nn]);
  }

  double d = 0.0;
  for (int nn = 0; nn < N; ++nn) {
    d += mn[nn];
  }

  printf("%.6f %.6f %.6f %.6f, %.6f = %.6f\n", a, b, c, d, (a - b - c) / d,
         mid);

  // Construct accurate sub-cell velocities
  // TODO: THIS WONT WORK FOR PRISMS AT THE MOMENT, NEED TO LOOK INTO FIXING
  // THIS ISSUE
  for (int nn = 0; nn < nnodes; ++nn) {
  }

#if 0
  /*
   * REMAP STEP PROTOTYPE
   */

  // Calculate the sub-cell energies
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nedges = cells_offsets[(cc + 1)] - cells_off;
    const int nsub_cells = nedges;
    const int nnodes_by_cell = nedges;

    // Fetch the cell centroids position
    const double cell_c_x = cell_centroids_x[(cc)];
    const double cell_c_y = cell_centroids_y[(cc)];

    // TODO: Does this least squares implementation still work if we break
    // some of the conditions that fall naturally from having even sides
    // on all of our shapes?

    /* Least squares regression taken from nodes in order to determine
     * the gradients that exist across the internal energy */

    // Calculate the coefficents to matrix M
    double MTM[3] = {0.0}; // Describes the three unique quantities in (M^T.M)
    double MT_del_phi[2] = {0.0};

    // Calculate the coefficients for all edges
    for (int ee = 0; ee < nedges; ++ee) {
      const int neighbour_index = cells_to_cells[(cells_off + ee)];

      // TODO: NOT SURE IF THIS IS THE CORRECT THING TO DO WITH BOUNDARY
      // CONDITION
      if (neighbour_index == IS_BOUNDARY) {
        continue;
      }

      // Calculate the vector pointing between the cell centroids
      double es_x = (cell_centroids_x[(neighbour_index)] - cell_c_x);
      double es_y = (cell_centroids_y[(neighbour_index)] - cell_c_y);
      const double centroid_distance = sqrt(es_x * es_x + es_y * es_y);
      es_x /= centroid_distance;
      es_y /= centroid_distance;

      // The edge relating to our current neighbour
      const int node_c_index = cells_to_nodes[(cells_off + ee)];
      const int node_l_index =
          (ee - 1 >= 0) ? cells_to_nodes[(cells_off + ee - 1)]
                        : cells_to_nodes[(cells_off + nnodes_by_cell - 1)];

      // Calculate the area vector for the face that we are looking at
      double A_x = (nodes_y0[(node_l_index)] - nodes_y0[(node_c_index)]);
      double A_y = -(nodes_x0[(node_l_index)] - nodes_x0[(node_c_index)]);

      // Fix the direction that the area vector is pointing in
      if ((A_x * es_x + A_y * es_y) < 0.0) {
        A_x = -A_x;
        A_y = -A_y;
      }

      // Calculate the gradient matrix
      const double phi0 = energy0[(cc)];
      const double phi_ff = energy0[(neighbour_index)];
      MTM[0] += es_x * es_x;
      MTM[1] += es_x * es_y;
      MTM[2] += es_y * es_y;
      MT_del_phi[0] += es_x * (phi_ff - phi0);
      MT_del_phi[1] += es_y * (phi_ff - phi0);
    }

    // Solve the equation for the temperature gradients
    const double MTM_det = (1.0 / (MTM[0] * MTM[2] - MTM[1] * MTM[1]));
    const double grad_e_x =
        MTM_det * (MT_del_phi[0] * MTM[2] - MT_del_phi[1] * MTM[1]);
    const double grad_e_y =
        MTM_det * (MT_del_phi[1] * MTM[0] - MT_del_phi[0] * MTM[1]);

    // Calculate the energy density in the cell
    double energy_density = energy0[(cc)] * density0[(cc)];

    /* Can now determine the sub cell internal energy */

    // Loop over all sub-cells to calculate integrals
    for (int ss = 0; ss < nsub_cells; ++ss) {
      // Determine the three point stencil of nodes around anchor node
      const int node_l_index =
          (ss == 0) ? cells_to_nodes[(cells_off + nnodes_by_cell - 1)]
                    : cells_to_nodes[(cells_off) + (ss - 1)];
      const int node_c_index = cells_to_nodes[(cells_off) + (ss)];
      const int node_r_index = (ss == nnodes_by_cell - 1)
                                   ? cells_to_nodes[(cells_off)]
                                   : cells_to_nodes[(cells_off) + (ss + 1)];

      // TODO: refactor the routines here so that they operate in loops...

      // Get the anchor node position
      const double node_c_x = nodes_x0[(node_c_index)];
      const double node_c_y = nodes_y0[(node_c_index)];

      // Get the midpoints between l and r nodes and current node
      const double node_l_x = 0.5 * (nodes_x0[(node_l_index)] + node_c_x);
      const double node_l_y = 0.5 * (nodes_y0[(node_l_index)] + node_c_y);
      const double node_r_x = 0.5 * (node_c_x + nodes_x0[(node_r_index)]);
      const double node_r_y = 0.5 * (node_c_y + nodes_y0[(node_r_index)]);

      // Calculate the volume integral weighted by x and y
      const double sub_cell_x_volume =
          (1.0 / 6.0) *
          ((node_c_x * node_c_x + node_c_x * node_r_x + node_r_x * node_r_x) *
               (node_r_y - node_c_y) +
           (node_r_x * node_r_x + node_r_x * cell_c_x + cell_c_x * cell_c_x) *
               (cell_c_y - node_r_y) +
           (cell_c_x * cell_c_x + cell_c_x * node_l_x + node_l_x * node_l_x) *
               (node_l_y - cell_c_y) +
           (node_l_x * node_l_x + node_l_x * node_c_x + node_c_x * node_c_x) *
               (node_c_y - node_l_y));
      const double sub_cell_y_volume =
          (1.0 / 6.0) *
          ((node_c_y * node_c_y + node_c_y * node_r_y + node_r_y * node_r_y) *
               (node_r_x - node_c_x) +
           (node_r_y * node_r_y + node_r_y * cell_c_y + cell_c_y * cell_c_y) *
               (cell_c_x - node_r_x) +
           (cell_c_y * cell_c_y + cell_c_y * node_l_y + node_l_y * node_l_y) *
               (node_l_x - cell_c_x) +
           (node_l_y * node_l_y + node_l_y * node_c_y + node_c_y * node_c_y) *
               (node_c_x - node_l_x));

      // Calculate the sub cell energy mass
      double sub_cell_e_mass =
          energy_density * sub_cell_volume[(cells_off + ss)] +
          grad_e_x * (sub_cell_x_volume -
                      sub_cell_volume[(cells_off + ss)] * cell_c_x) +
          grad_e_y * (sub_cell_y_volume -
                      sub_cell_volume[(cells_off + ss)] * cell_c_y);

      sub_cell_energy[(cells_off + ss)] = sub_cell_e_mass;
    }
  }

  // Calculate the sub-cell velocities and kinetic energy
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nsub_cells = cells_offsets[(cc + 1)] - cells_off;
    const int nnodes_by_cell = nsub_cells;

    double velocities_x[nnodes_by_cell];
    double velocities_y[nnodes_by_cell];
    double masses[nsub_cells];

    // Privatise all of the data we will need
    for (int ss = 0; ss < nsub_cells; ++ss) {
      const int node_index = cells_to_nodes[(cells_off) + (ss)];
      velocities_x[(ss)] = velocity_x0[(node_index)];
      velocities_y[(ss)] = velocity_y0[(node_index)];
      masses[(ss)] = sub_cell_mass[(cells_off + ss)];
    }

    // Calculate the constant coefficients for determining the sub-cell
    // velocities and kinetic energies
    double vel_coeff_x = 0.0;
    double vel_coeff_y = 0.0;
    double ke_coeff = 0.0;
    for (int ss = 0; ss < nsub_cells; ++ss) {
      const int p_index = (ss == 0) ? (nsub_cells - 1) : (ss - 1);
      const int n_index = (ss == nsub_cells - 1) ? (0) : (ss + 1);
      vel_coeff_x +=
          velocities_x[(ss)] *
          (4.0 * masses[(ss)] - masses[(p_index)] - masses[(n_index)]) /
          (8.0 * cell_mass[(cc)]);
      vel_coeff_y +=
          velocities_y[(ss)] *
          (4.0 * masses[(ss)] - masses[(p_index)] - masses[(n_index)]) /
          (8.0 * cell_mass[(cc)]);
      ke_coeff += 0.5 * (velocities_x[(ss)] * velocities_x[(ss)] +
                         velocities_y[(ss)] * velocities_y[(ss)]) *
                  (4.0 * masses[(ss)] - masses[(p_index)] - masses[(n_index)]) /
                  (8.0 * cell_mass[(cc)]);
    }

    for (int ss = 0; ss < nsub_cells; ++ss) {
      const int p_index = (ss == 0) ? (nsub_cells - 1) : (ss - 1);
      const int n_index = (ss == nsub_cells - 1) ? (0) : (ss + 1);

      // Store the sub-cell velocities
      sub_cell_velocity_x[(cells_off + ss)] =
          vel_coeff_x +
          0.25 * (2.0 * velocities_x[(ss)] + 0.5 * velocities_x[(n_index)] +
                  0.5 * velocities_x[(p_index)]);
      sub_cell_velocity_y[(cells_off + ss)] =
          vel_coeff_y +
          0.25 * (2.0 * velocities_y[(ss)] + 0.5 * velocities_y[(n_index)] +
                  0.5 * velocities_y[(p_index)]);

      // Calculate the surrounding values of the kinetic energy
      const double ke_c = 0.5 * (velocities_x[(ss)] * velocities_x[(ss)] +
                                 velocities_y[(ss)] * velocities_y[(ss)]);
      const double ke_p =
          0.5 * (velocities_x[(p_index)] * velocities_x[(p_index)] +
                 velocities_y[(p_index)] * velocities_y[(p_index)]);
      const double ke_n =
          0.5 * (velocities_x[(n_index)] * velocities_x[(n_index)] +
                 velocities_y[(n_index)] * velocities_y[(n_index)]);

      sub_cell_kinetic_energy[(cells_off + ss)] =
          ke_coeff + 0.25 * (2.0 * ke_c + 0.5 * ke_n + 0.5 * ke_p);
    }
  }

  initialise_sub_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0,
                                nodes_y0, cell_centroids_x, cell_centroids_y,
                                sub_cell_centroids_x, sub_cell_centroids_y);

  // TODO: Need to handle the boundary conditions here... in the other routine
  // we simply ignore the boundaries in the gradient calculation, still not sure
  // if this a reasonable approach but it's definitely the easiest

  // Here we calculate the gradient for the quantity in each sub-cell
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nsub_cells = cells_offsets[(cc + 1)] - cells_off;

    // Calculate the gradient of the change in density between the sub-cells
    for (int ss = 0; ss < nsub_cells; ++ss) {
      const int sub_cell_c_index = (cells_off + ss);
      const int node_c_index = cells_to_nodes[(cells_off) + (ss)];

      const int nodes_off = nodes_offsets[(node_c_index)];
      const int ncells_by_node = nodes_offsets[(node_c_index + 1)] - nodes_off;

      // Scan through the cells attached to the sub-cell's external node, to
      // fetch the two cells that neighbour the sub-cell
      int out_cell_index[2] = {-1, -1};
      for (int cc2 = 0; cc2 < ncells_by_node; ++cc2) {
        const int cell_index = nodes_to_cells[(nodes_off + cc2)];
        if (cell_index == cc) {
          // We have found our current cell, so get the neighbours
          out_cell_index[0] =
              (cc2 == 0) ? nodes_to_cells[(nodes_off + ncells_by_node - 1)]
                         : nodes_to_cells[(nodes_off) + (cc2 - 1)];
          out_cell_index[1] = (cc2 == ncells_by_node - 1)
                                  ? nodes_to_cells[(nodes_off)]
                                  : nodes_to_cells[(nodes_off) + (cc2 + 1)];
          break;
        }
      }

      if (out_cell_index[0] == -1 || out_cell_index[1] == -1) {
        TERMINATE(
            "We were not able to find the cell neighbouring this sub-cell.");
      }

      // The indices for sub-cells in neighbourhood of current sub-cell
      int sub_cell_neighbour_indices[nsub_cell_edges];

      for (int oo = 0; oo < 2; ++oo) {
        const int out_cells_off = cells_offsets[(out_cell_index[oo])];
        const int out_nnodes =
            cells_offsets[(out_cell_index[oo] + 1)] - cells_off;

        for (int nn = 0; nn < out_nnodes; ++nn) {
          const int out_node_c_index = cells_to_nodes[(out_cells_off + nn)];
          if (node_c_index == node_c_index) {
            // We have found the neighbouring sub-cell
            sub_cell_neighbour_indices[oo] = (out_cells_off + nn);
            break;
          }
        }
      }

      sub_cell_neighbour_indices[2] =
          (ss == nsub_cells - 1) ? (cells_off + 0) : (cells_off + ss + 1);
      sub_cell_neighbour_indices[3] =
          (ss == 0) ? (cells_off + nsub_cells - 1) : (cells_off + ss - 1);

      // Loop over all of the neighbours
      double a = 0.0;
      double b = 0.0;
      double c = 0.0;
      double d = 0.0;
      double e = 0.0;

      // Fetch the density for the sub-cell
      double sub_cell_c_density = sub_cell_mass[(sub_cell_c_index)] /
                                  sub_cell_volume[(sub_cell_c_index)];

      for (int nn = 0; nn < nsub_cell_edges; ++nn) {
        const int sub_cell_n_index = sub_cell_neighbour_indices[(nn)];

        // Fetch the density for the neighbour
        double sub_cell_n_density = sub_cell_mass[(sub_cell_c_index)] /
                                    sub_cell_volume[(sub_cell_c_index)];

        // Calculate the differential quantities
        const double dx = sub_cell_centroids_x[(sub_cell_n_index)] -
                          sub_cell_centroids_x[(sub_cell_c_index)];
        const double dy = sub_cell_centroids_y[(sub_cell_n_index)] -
                          sub_cell_centroids_y[(sub_cell_c_index)];
        const double drho = sub_cell_n_density - sub_cell_c_density;

        double omega2 = 1.0 / (dx * dx + dy * dy);

        // Calculate the coefficients for the minimisation
        a += omega2 * dx * dx;
        b += omega2 * dx * dy;
        c += omega2 * dy * dy;
        d += omega2 * drho * dx;
        e += omega2 * drho * dy;
      }

      // Solve the minimisation problem to get the gradients for the sub-cell
      sub_cell_grad_x[(sub_cell_c_index)] = (c * d - b * e) / (a * c - b * b);
      sub_cell_grad_y[(sub_cell_c_index)] = (a * e - b * d) / (a * c - b * b);
    }
  }

  // Here we are going to update the mass using the linear function that we have
  // now been able to successfully describe for all of the sub-cells
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nsub_cells = cells_offsets[(cc + 1)] - cells_off;

    // Store the cell centroids as these are the same for all of the
    // sub-cells that are within this cell
    const double cell_c_x = cell_centroids_x[(cell_index)];
    const double cell_c_y = cell_centroids_y[(cell_index)];
    const double rezoned_cell_c_x = cell_centroids_x[(cell_index)];
    const double rezoned_cell_c_y = cell_centroids_y[(cell_index)];

    // Consider all of the sub-cells that are in the new grid, assuming that we
    // have maintained identical connectivity
    for (int ss = 0; ss < nsub_cells; ++ss) {
      const int sub_cell_c_index = (cells_off + ss);

      // Determine the three point stencil of nodes around anchor node
      const int node_l_index =
          (ss == 0) ? cells_to_nodes[(cells_off + nnodes_by_cell - 1)]
                    : cells_to_nodes[(cells_off) + (ss - 1)];
      const int node_c_index = cells_to_nodes[(cells_off) + (ss)];
      const int node_r_index = (ss == nnodes_by_cell - 1)
                                   ? cells_to_nodes[(cells_off)]
                                   : cells_to_nodes[(cells_off) + (ss + 1)];

      // Calculate the nodes surrounding the sub-cell
      double sub_cell_nodes_x[nsub_cell_edges];
      double sub_cell_nodes_y[nsub_cell_edges];
      double new_sub_cell_nodes_x[nsub_cell_edges];
      double new_sub_cell_nodes_y[nsub_cell_edges];

      // The sub-cell is surrounded by the anchor node, midpoints on the outer
      // edges and the center of the cell
      sub_cell_nodes_x[0] = nodes_x0[(node_c_index)];
      sub_cell_nodes_y[0] = nodes_y0[(node_c_index)];
      sub_cell_nodes_x[1] = 0.5 * (node_c_x + nodes_x0[(node_r_index)]);
      sub_cell_nodes_y[1] = 0.5 * (node_c_y + nodes_y0[(node_r_index)]);
      sub_cell_nodes_x[2] = cell_c_x;
      sub_cell_nodes_y[2] = cell_c_y;
      sub_cell_nodes_x[3] = 0.5 * (nodes_x0[(node_l_index)] + node_c_x);
      sub_cell_nodes_y[3] = 0.5 * (nodes_y0[(node_l_index)] + node_c_y);

      // Get the same data for the new sub-cell after rezoning
      new_sub_cell_nodes_x[0] = rezoned_nodes_x0[(node_c_index)];
      new_sub_cell_nodes_y[0] = rezoned_nodes_y0[(node_c_index)];
      new_sub_cell_nodes_x[1] =
          0.5 * (node_c_x + rezoned_nodes_x0[(node_r_index)]);
      new_sub_cell_nodes_y[1] =
          0.5 * (node_c_y + rezoned_nodes_y0[(node_r_index)]);
      new_sub_cell_nodes_x[2] = cell_c_x;
      new_sub_cell_nodes_y[2] = cell_c_y;
      new_sub_cell_nodes_x[3] =
          0.5 * (rezoned_nodes_x0[(node_l_index)] + node_c_x);
      new_sub_cell_nodes_y[3] =
          0.5 * (rezoned_nodes_y0[(node_l_index)] + node_c_y);

      for (int ee = 0; ee < nsub_cell_edges; ++ee) {
        const int node0_off = (cells_off + ss);
        const int node1_off =
            (ss == nsub_cells - 1) ? (cells_off) : (cells_off + ss);

        const int node0_index = cells_to_nodes[(node0_off)];
        const int node1_index = cells_to_nodes[(node1_off)];

        // This ordering of nodes encompasses the swept region for the edge
        double nodes_x[nsub_cell_indices];
        double nodes_y[nsub_cell_indices];
        nodes_x[0] = nodes_x0[(node0_index)];
        nodes_x[1] = nodes_x0[(node1_index)];
        nodes_x[2] = rezoned_nodes_x[(node1_index)];
        nodes_x[3] = rezoned_nodes_x[(node0_index)];
        nodes_y[0] = nodes_y0[(node0_index)];
        nodes_y[1] = nodes_y0[(node1_index)];
        nodes_y[2] = rezoned_nodes_y[(node1_index)];
        nodes_y[3] = rezoned_nodes_y[(node0_index)];

        // We want to calculate the volume integral
        double volume = 0.0;
        for (int nn = 0; nn < nsub_cell_indices; ++nn) {
          volume += 0.5 * (nodes_x[0] * nodes_x[1]) * (nodes_y[1] - nodes_y[0]);
        }

        // The sign of the volume determines the donor sub-cell
      }
    }
  }
#endif // if 0

  /*
   *    PREDICTOR
   */

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodal_mass[(nn)] = 0.0;
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }
  STOP_PROFILING(&compute_profile, "zero_node_data");

  // Equation of state, ideal gas law
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    pressure0[(cc)] = (GAM - 1.0) * energy0[(cc)] * density0[(cc)];
  }
  STOP_PROFILING(&compute_profile, "equation_of_state");

  // TODO: SOOO MUCH POTENTIAL FOR OPTIMISATION HERE...!
  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;
    const double node_c_x = nodes_x0[(nn)];
    const double node_c_y = nodes_y0[(nn)];
    const double node_c_z = nodes_z0[(nn)];

    // Consider all faces attached to node
    for (int ff = 0; ff < nfaces_by_node; ++ff) {
      const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
      if (face_index == -1) {
        continue;
      }

      // Determine the offset into the list of nodes
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Find node center and location of current node on face
      int node_in_face_c;
      double face_c_x = 0.0;
      double face_c_y = 0.0;
      double face_c_z = 0.0;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c_x += nodes_x0[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y0[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z0[(node_index)] / nnodes_by_face;

        // Choose the node in the list of nodes attached to the face
        if (nn == node_index) {
          node_in_face_c = nn2;
        }
      }

      // Fetch the nodes attached to our current node on the current face
      int nodes[2];
      nodes[0] = (node_in_face_c - 1 >= 0)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c - 1)]
                     : faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)];
      nodes[1] = (node_in_face_c + 1 < nnodes_by_face)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c + 1)]
                     : faces_to_nodes[(face_to_nodes_off)];

      // Fetch the cells attached to our current face
      int cells[2];
      cells[0] = faces_to_cells0[(face_index)];
      cells[1] = faces_to_cells1[(face_index)];

      // Add contributions from all of the cells attached to the face
      for (int cc = 0; cc < 2; ++cc) {
        if (cells[(cc)] == -1) {
          continue;
        }

        // Add contributions for both edges attached to our current node
        for (int nn2 = 0; nn2 < 2; ++nn2) {
          // Get the halfway point on the right edge
          const double half_edge_x =
              0.5 * (nodes_x0[(nodes[(nn2)])] + nodes_x0[(nn)]);
          const double half_edge_y =
              0.5 * (nodes_y0[(nodes[(nn2)])] + nodes_y0[(nn)]);
          const double half_edge_z =
              0.5 * (nodes_z0[(nodes[(nn2)])] + nodes_z0[(nn)]);

          // Setup basis on plane of tetrahedron
          const double a_x = (face_c_x - node_c_x);
          const double a_y = (face_c_y - node_c_y);
          const double a_z = (face_c_z - node_c_z);
          const double b_x = (face_c_x - half_edge_x);
          const double b_y = (face_c_y - half_edge_y);
          const double b_z = (face_c_z - half_edge_z);
          const double ab_x = (cell_centroids_x[(cells[cc])] - face_c_x);
          const double ab_y = (cell_centroids_y[(cells[cc])] - face_c_y);
          const double ab_z = (cell_centroids_z[(cells[cc])] - face_c_z);

          // Calculate the area vector S using cross product
          double A_x = 0.5 * (a_y * b_z - a_z * b_y);
          double A_y = -0.5 * (a_x * b_z - a_z * b_x);
          double A_z = 0.5 * (a_x * b_y - a_y * b_x);

          // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES SO
          // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
          // CALCULATION
          const double sub_cell_volume =
              fabs((ab_x * A_x + ab_y * A_y + ab_z * A_z) / 3.0);

          nodal_mass[(nn)] += density0[(cells[(cc)])] * sub_cell_volume;
          nodal_soundspeed[(nn)] +=
              sqrt(GAM * (GAM - 1.0) * energy0[(cells[(cc)])]) *
              sub_cell_volume;
          nodal_volumes[(nn)] += sub_cell_volume;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      node_force_x[(cell_to_nodes_off + nn)] = 0.0;
      node_force_y[(cell_to_nodes_off + nn)] = 0.0;
      node_force_z[(cell_to_nodes_off + nn)] = 0.0;
    }
  }

  // Calculate the pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

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
        face_c_x += nodes_x0[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y0[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z0[(node_index)] / nnodes_by_face;
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
            0.5 * (nodes_x0[(current_node)] + nodes_x0[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y0[(current_node)] + nodes_y0[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z0[(current_node)] + nodes_z0[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (face_c_x - nodes_x0[(current_node)]);
        const double a_y = (face_c_y - nodes_y0[(current_node)]);
        const double a_z = (face_c_z - nodes_z0[(current_node)]);
        const double b_x = (face_c_x - half_edge_x);
        const double b_y = (face_c_y - half_edge_y);
        const double b_z = (face_c_z - half_edge_z);
        const double ab_x = (cell_centroids_x[(cc)] - face_c_x);
        const double ab_y = (cell_centroids_y[(cc)] - face_c_y);
        const double ab_z = (cell_centroids_z[(cc)] - face_c_z);

        // Calculate the area vector S using cross product
        double A_x = 0.5 * (a_y * b_z - a_z * b_y);
        double A_y = -0.5 * (a_x * b_z - a_z * b_x);
        double A_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER CLOSED
        // FORM SOLUTION?
        int node_off;
        int next_node_off;
        for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
          if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
            node_off = nn3;
          } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == next_node) {
            next_node_off = nn3;
          }
        }

        // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES SO
        // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
        // CALCULATION
        const int flip = (ab_x * A_x + ab_y * A_y + ab_z * A_z > 0.0);
        node_force_x[(cell_to_nodes_off + node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_x : A_x);
        node_force_y[(cell_to_nodes_off + node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_y : A_y);
        node_force_z[(cell_to_nodes_off + node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_z : A_z);
        node_force_x[(cell_to_nodes_off + next_node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_x : A_x);
        node_force_y[(cell_to_nodes_off + next_node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_y : A_y);
        node_force_z[(cell_to_nodes_off + next_node_off)] +=
            pressure0[(cc)] * ((flip) ? -A_z : A_z);
      }
    }
  }
  STOP_PROFILING(&compute_profile, "node_force_from_pressure");

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }
  STOP_PROFILING(&compute_profile, "scale_soundspeed");

  calculate_artificial_viscosity(
      ncells, nnodes, visc_coeff1, visc_coeff2, cells_offsets, cells_to_nodes,
      nodes_offsets, nodes_to_cells, nodes_x0, nodes_y0, nodes_z0,
      cell_centroids_x, cell_centroids_y, cell_centroids_z, velocity_x0,
      velocity_y0, velocity_z0, nodal_soundspeed, nodal_mass, nodal_volumes,
      limiter, node_force_x, node_force_y, node_force_z, nodes_to_faces_offsets,
      nodes_to_faces, faces_to_nodes_offsets, faces_to_nodes, faces_to_cells0,
      faces_to_cells1, cells_to_faces_offsets, cells_to_faces);

  // Calculate the time centered evolved velocities, by first calculating the
  // predicted values at the new timestep and then averaging with current
  // velocity
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - node_to_cells_off;

    // Accumulate the force at this node
    double node_force_x0 = 0.0;
    double node_force_y0 = 0.0;
    double node_force_z0 = 0.0;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_offsets[(cell_index + 1)] - cell_to_nodes_off;

      // ARRGHHHH
      int node_off;
      for (node_off = 0; node_off < nnodes_by_cell; ++node_off) {
        if (cells_to_nodes[(cell_to_nodes_off + node_off)] == nn) {
          break;
        }
      }

      node_force_x0 += node_force_x[(cell_to_nodes_off + node_off)];
      node_force_y0 += node_force_y[(cell_to_nodes_off + node_off)];
      node_force_z0 += node_force_z[(cell_to_nodes_off + node_off)];
    }

    // Determine the predicted velocity
    velocity_x1[(nn)] =
        velocity_x0[(nn)] + mesh->dt * node_force_x0 / nodal_mass[(nn)];
    velocity_y1[(nn)] =
        velocity_y0[(nn)] + mesh->dt * node_force_y0 / nodal_mass[(nn)];
    velocity_z1[(nn)] =
        velocity_z0[(nn)] + mesh->dt * node_force_z0 / nodal_mass[(nn)];

    // Calculate the time centered velocity
    velocity_x1[(nn)] = 0.5 * (velocity_x0[(nn)] + velocity_x1[(nn)]);
    velocity_y1[(nn)] = 0.5 * (velocity_y0[(nn)] + velocity_y1[(nn)]);
    velocity_z1[(nn)] = 0.5 * (velocity_z0[(nn)] + velocity_z1[(nn)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_velocity");

  // TODO: NEED TO WORK OUT HOW TO HANDLE BOUNDARY CONDITIONS REASONABLY
  handle_unstructured_reflect_3d(nnodes, boundary_index, boundary_type,
                                 boundary_normal_x, boundary_normal_y,
                                 boundary_normal_z, velocity_x1, velocity_y1,
                                 velocity_z1);

  // Move the nodes by the predicted velocity
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x1[(nn)] = nodes_x0[(nn)] + mesh->dt * velocity_x1[(nn)];
    nodes_y1[(nn)] = nodes_y0[(nn)] + mesh->dt * velocity_y1[(nn)];
    nodes_z1[(nn)] = nodes_z0[(nn)] + mesh->dt * velocity_z1[(nn)];
  }
  STOP_PROFILING(&compute_profile, "move_nodes");

  initialise_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x1,
                            nodes_y1, nodes_z1, cell_centroids_x,
                            cell_centroids_y, cell_centroids_z);

  set_timestep(ncells, cells_to_nodes, cells_offsets, nodes_x1, nodes_y1,
               nodes_z1, energy0, &mesh->dt, cells_to_faces_offsets,
               cells_to_faces, faces_to_nodes_offsets, faces_to_nodes);

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    double cell_force = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      cell_force +=
          (velocity_x1[(node_index)] * node_force_x[(cell_to_nodes_off + nn)] +
           velocity_y1[(node_index)] * node_force_y[(cell_to_nodes_off + nn)] +
           velocity_z1[(node_index)] * node_force_z[(cell_to_nodes_off + nn)]);
    }
    energy1[(cc)] = energy0[(cc)] - mesh->dt * cell_force / cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  // Using the new volume, calculate the predicted density
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double cell_volume = 0.0;

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
        face_c_x += nodes_x1[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y1[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z1[(node_index)] / nnodes_by_face;
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
            0.5 * (nodes_x1[(current_node)] + nodes_x1[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y1[(current_node)] + nodes_y1[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z1[(current_node)] + nodes_z1[(next_node)]);

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

        // TODO: WE MULTIPLY BY 2 HERE BECAUSE WE ARE ADDING THE VOLUME TO BOTH
        // THE CURRENT AND NEXT NODE, OTHERWISE WE ONLY ACCOUNT FOR HALF OF THE
        // 'HALF' TETRAHEDRONS
        cell_volume +=
            fabs(2.0 * ((half_edge_x - nodes_x1[(current_node)]) * S_x +
                        (half_edge_y - nodes_y1[(current_node)]) * S_y +
                        (half_edge_z - nodes_z1[(current_node)]) * S_z) /
                 3.0);
      }
    }

    density1[(cc)] = cell_mass[(cc)] / cell_volume;
  }
  STOP_PROFILING(&compute_profile, "calc_new_density");

  // Calculate the time centered pressure from mid point between rezoned and
  // predicted pressures
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    // Calculate the predicted pressure from the equation of state
    pressure1[(cc)] = (GAM - 1.0) * energy1[(cc)] * density1[(cc)];

    // Determine the time centered pressure
    pressure1[(cc)] = 0.5 * (pressure0[(cc)] + pressure1[(cc)]);
  }
  STOP_PROFILING(&compute_profile, "equation_of_state_time_center");

  // Prepare time centered variables for the corrector step
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x1[(nn)] = 0.5 * (nodes_x1[(nn)] + nodes_x0[(nn)]);
    nodes_y1[(nn)] = 0.5 * (nodes_y1[(nn)] + nodes_y0[(nn)]);
    nodes_z1[(nn)] = 0.5 * (nodes_z1[(nn)] + nodes_z0[(nn)]);
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }
  STOP_PROFILING(&compute_profile, "move_nodes2");

/*
 *    CORRECTOR
 */

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      node_force_x[(cell_to_nodes_off + nn)] = 0.0;
      node_force_y[(cell_to_nodes_off + nn)] = 0.0;
      node_force_z[(cell_to_nodes_off + nn)] = 0.0;
    }
  }

  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;
    const double node_c_x = nodes_x1[(nn)];
    const double node_c_y = nodes_y1[(nn)];
    const double node_c_z = nodes_z1[(nn)];

    // Consider all faces attached to node
    for (int ff = 0; ff < nfaces_by_node; ++ff) {
      const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
      if (face_index == -1) {
        continue;
      }

      // Determine the offset into the list of nodes
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Find node center and location of current node on face
      int node_in_face_c;
      double face_c_x = 0.0;
      double face_c_y = 0.0;
      double face_c_z = 0.0;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c_x += nodes_x1[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y1[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z1[(node_index)] / nnodes_by_face;

        // Choose the node in the list of nodes attached to the face
        if (nn == node_index) {
          node_in_face_c = nn2;
        }
      }

      // Fetch the nodes attached to our current node on the current face
      int nodes[2];
      nodes[0] = (node_in_face_c - 1 >= 0)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c - 1)]
                     : faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)];
      nodes[1] = (node_in_face_c + 1 < nnodes_by_face)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c + 1)]
                     : faces_to_nodes[(face_to_nodes_off)];

      // Fetch the cells attached to our current face
      int cells[2];
      cells[0] = faces_to_cells0[(face_index)];
      cells[1] = faces_to_cells1[(face_index)];

      // Add contributions from all of the cells attached to the face
      for (int cc = 0; cc < 2; ++cc) {
        if (cells[(cc)] == -1) {
          continue;
        }

        // Add contributions for both edges attached to our current node
        for (int nn2 = 0; nn2 < 2; ++nn2) {
          // Get the halfway point on the right edge
          const double half_edge_x =
              0.5 * (nodes_x1[(nodes[(nn2)])] + nodes_x1[(nn)]);
          const double half_edge_y =
              0.5 * (nodes_y1[(nodes[(nn2)])] + nodes_y1[(nn)]);
          const double half_edge_z =
              0.5 * (nodes_z1[(nodes[(nn2)])] + nodes_z1[(nn)]);

          // Setup basis on plane of tetrahedron
          const double a_x = (face_c_x - node_c_x);
          const double a_y = (face_c_y - node_c_y);
          const double a_z = (face_c_z - node_c_z);
          const double b_x = (face_c_x - half_edge_x);
          const double b_y = (face_c_y - half_edge_y);
          const double b_z = (face_c_z - half_edge_z);
          const double ab_x = (cell_centroids_x[(cells[cc])] - face_c_x);
          const double ab_y = (cell_centroids_y[(cells[cc])] - face_c_y);
          const double ab_z = (cell_centroids_z[(cells[cc])] - face_c_z);

          // Calculate the area vector S using cross product
          double A_x = 0.5 * (a_y * b_z - a_z * b_y);
          double A_y = -0.5 * (a_x * b_z - a_z * b_x);
          double A_z = 0.5 * (a_x * b_y - a_y * b_x);

          const double sub_cell_volume =
              fabs((ab_x * A_x + ab_y * A_y + ab_z * A_z) / 3.0);

          nodal_soundspeed[(nn)] +=
              sqrt(GAM * (GAM - 1.0) * energy1[(cells[(cc)])]) *
              sub_cell_volume;
          nodal_volumes[(nn)] += sub_cell_volume;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_soundspeed");

  // Calculate the pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

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
        face_c_x += nodes_x1[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y1[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z1[(node_index)] / nnodes_by_face;
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
            0.5 * (nodes_x1[(current_node)] + nodes_x1[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y1[(current_node)] + nodes_y1[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z1[(current_node)] + nodes_z1[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (face_c_x - nodes_x1[(current_node)]);
        const double a_y = (face_c_y - nodes_y1[(current_node)]);
        const double a_z = (face_c_z - nodes_z1[(current_node)]);
        const double b_x = (face_c_x - half_edge_x);
        const double b_y = (face_c_y - half_edge_y);
        const double b_z = (face_c_z - half_edge_z);
        const double ab_x = (cell_centroids_x[(cc)] - face_c_x);
        const double ab_y = (cell_centroids_y[(cc)] - face_c_y);
        const double ab_z = (cell_centroids_z[(cc)] - face_c_z);

        // Calculate the area vector S using cross product
        double A_x = 0.5 * (a_y * b_z - a_z * b_y);
        double A_y = -0.5 * (a_x * b_z - a_z * b_x);
        double A_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER CLOSED
        // FORM SOLUTION?
        int node_off;
        int next_node_off;
        for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
          if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
            node_off = nn3;
          } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == next_node) {
            next_node_off = nn3;
          }
        }

        // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES SO
        // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
        // CALCULATION
        const int flip = (ab_x * A_x + ab_y * A_y + ab_z * A_z > 0.0);
        node_force_x[(cell_to_nodes_off + node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_x : A_x);
        node_force_y[(cell_to_nodes_off + node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_y : A_y);
        node_force_z[(cell_to_nodes_off + node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_z : A_z);
        node_force_x[(cell_to_nodes_off + next_node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_x : A_x);
        node_force_y[(cell_to_nodes_off + next_node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_y : A_y);
        node_force_z[(cell_to_nodes_off + next_node_off)] +=
            pressure1[(cc)] * ((flip) ? -A_z : A_z);
      }
    }
  }
  STOP_PROFILING(&compute_profile, "node_force_from_pressure");

  calculate_artificial_viscosity(
      ncells, nnodes, visc_coeff1, visc_coeff2, cells_offsets, cells_to_nodes,
      nodes_offsets, nodes_to_cells, nodes_x1, nodes_y1, nodes_z1,
      cell_centroids_x, cell_centroids_y, cell_centroids_z, velocity_x1,
      velocity_y1, velocity_z1, nodal_soundspeed, nodal_mass, nodal_volumes,
      limiter, node_force_x, node_force_y, node_force_z, nodes_to_faces_offsets,
      nodes_to_faces, faces_to_nodes_offsets, faces_to_nodes, faces_to_cells0,
      faces_to_cells1, cells_to_faces_offsets, cells_to_faces);

  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_offsets[(nn)];
    const int ncells_by_node = nodes_offsets[(nn + 1)] - node_to_cells_off;

    // Consider all faces attached to node
    double node_force_x0 = 0.0;
    double node_force_y0 = 0.0;
    double node_force_z0 = 0.0;
    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_offsets[(cell_index + 1)] - cell_to_nodes_off;

      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
          break;
        }
      }

      node_force_x0 += node_force_x[(cell_to_nodes_off + nn2)];
      node_force_y0 += node_force_y[(cell_to_nodes_off + nn2)];
      node_force_z0 += node_force_z[(cell_to_nodes_off + nn2)];
    }

    // Calculate the new velocities
    velocity_x1[(nn)] += mesh->dt * node_force_x0 / nodal_mass[(nn)];
    velocity_y1[(nn)] += mesh->dt * node_force_y0 / nodal_mass[(nn)];
    velocity_z1[(nn)] += mesh->dt * node_force_z0 / nodal_mass[(nn)];

    // Calculate the corrected time centered velocities
    velocity_x0[(nn)] = 0.5 * (velocity_x1[(nn)] + velocity_x0[(nn)]);
    velocity_y0[(nn)] = 0.5 * (velocity_y1[(nn)] + velocity_y0[(nn)]);
    velocity_z0[(nn)] = 0.5 * (velocity_z1[(nn)] + velocity_z0[(nn)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_velocity");

  handle_unstructured_reflect_3d(nnodes, boundary_index, boundary_type,
                                 boundary_normal_x, boundary_normal_y,
                                 boundary_normal_z, velocity_x0, velocity_y0,
                                 velocity_z0);

  // Calculate the corrected node movements
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x0[(nn)] += mesh->dt * velocity_x0[(nn)];
    nodes_y0[(nn)] += mesh->dt * velocity_y0[(nn)];
    nodes_z0[(nn)] += mesh->dt * velocity_z0[(nn)];
  }
  STOP_PROFILING(&compute_profile, "move_nodes");

  set_timestep(ncells, cells_to_nodes, cells_offsets, nodes_x0, nodes_y0,
               nodes_z0, energy1, &mesh->dt, cells_to_faces_offsets,
               cells_to_faces, faces_to_nodes_offsets, faces_to_nodes);

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    double cell_force = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      cell_force +=
          (velocity_x0[(node_index)] * node_force_x[(cell_to_nodes_off + nn)] +
           velocity_y0[(node_index)] * node_force_y[(cell_to_nodes_off + nn)] +
           velocity_z0[(node_index)] * node_force_z[(cell_to_nodes_off + nn)]);
    }

    energy0[(cc)] -= mesh->dt * cell_force / cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  initialise_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0,
                            nodes_y0, nodes_z0, cell_centroids_x,
                            cell_centroids_y, cell_centroids_z);

  // Using the new corrected volume, calculate the density
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double cell_volume = 0.0;

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
        face_c_x += nodes_x0[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y0[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z0[(node_index)] / nnodes_by_face;
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
            0.5 * (nodes_x0[(current_node)] + nodes_x0[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y0[(current_node)] + nodes_y0[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z0[(current_node)] + nodes_z0[(next_node)]);

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

        // TODO: WE MULTIPLY BY 2 HERE BECAUSE WE ARE ADDING THE VOLUME TO BOTH
        // THE CURRENT AND NEXT NODE, OTHERWISE WE ONLY ACCOUNT FOR HALF OF THE
        // 'HALF' TETRAHEDRONS
        cell_volume +=
            fabs(2.0 * ((half_edge_x - nodes_x0[(current_node)]) * S_x +
                        (half_edge_y - nodes_y0[(current_node)]) * S_y +
                        (half_edge_z - nodes_z0[(current_node)]) * S_z) /
                 3.0);
      }
    }

    // Update the density using the new volume
    density0[(cc)] = cell_mass[(cc)] / cell_volume;
  }
  STOP_PROFILING(&compute_profile, "calc_new_density");
}

// Controls the timestep for the simulation
void set_timestep(const int ncells, const int* cells_to_nodes,
                  const int* cells_offsets, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes) {

  // TODO: THIS IS A GOOD EXAMPLE OF WHERE WE SHOULD MARRY FACES TO EDGES
  // RATHER THAN DIRECTLY TO NODES.... WE ARE CURRENTLY PERFORMING TWICE
  // AS MANY CALCULATIONS AS WE NEED TO

  // Calculate the timestep based on the computational mesh and CFL condition
  double local_dt = DBL_MAX;
  START_PROFILING(&compute_profile);
#pragma omp parallel for reduction(min : local_dt)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double shortest_edge = DBL_MAX;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // Fetch the nodes attached to our current node on the current face
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn)];

        const int next_node = (nn + 1 < nnodes_by_face)
                                  ? faces_to_nodes[(face_to_nodes_off + nn + 1)]
                                  : faces_to_nodes[(face_to_nodes_off)];
        const double x_component =
            nodes_x[(current_node)] - nodes_x[(next_node)];
        const double y_component =
            nodes_y[(current_node)] - nodes_y[(next_node)];
        const double z_component =
            nodes_z[(current_node)] - nodes_z[(next_node)];

        // Find the shortest edge of this cell
        shortest_edge = min(shortest_edge, sqrt(x_component * x_component +
                                                y_component * y_component +
                                                z_component * z_component));
      }
    }

    const double soundspeed = sqrt(GAM * (GAM - 1.0) * energy[(cc)]);
    local_dt = min(local_dt, shortest_edge / soundspeed);
  }
  STOP_PROFILING(&compute_profile, __func__);

  *dt = CFL * local_dt;
  printf("Timestep %.8fs\n", *dt);
}

// Initialises the cell mass, sub-cell mass and sub-cell volume
void initialise_mesh_mass(
    const int ncells, const int* cells_offsets, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const int* cells_to_nodes, const double* density, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* cell_mass,
    double* sub_cell_volume, double* sub_cell_mass, int* cells_to_faces_offsets,
    int* cells_to_faces, int* faces_to_nodes_offsets, int* faces_to_nodes) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (int cc = 0; cc < ncells; ++cc) {
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

        // TODO: WE MULTIPLY BY 2 HERE BECAUSE WE ARE ADDING THE VOLUME TO BOTH
        // THE CURRENT AND NEXT NODE, OTHERWISE WE ONLY ACCOUNT FOR HALF OF THE
        // 'HALF' TETRAHEDRONS
        double sub_cell_volume =
            fabs(2.0 * ((half_edge_x - nodes_x[(current_node)]) * S_x +
                        (half_edge_y - nodes_y[(current_node)]) * S_y +
                        (half_edge_z - nodes_z[(current_node)]) * S_z) /
                 3.0);

        cell_mass[(cc)] += density[(cc)] * sub_cell_volume;
      }
    }

    total_mass += cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, __func__);

  printf("Initial total mesh mash: %.15f\n", total_mass);
}

// Initialises the centroids for each cell
void initialise_cell_centroids(const int ncells, const int* cells_offsets,
                               const int* cells_to_nodes, const double* nodes_x,
                               const double* nodes_y, const double* nodes_z,
                               double* cell_centroids_x,
                               double* cell_centroids_y,
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

// Initialises the centroids for each cell
void initialise_sub_cell_centroids(
    const int ncells, const int* cells_offsets, const int* cells_to_nodes,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, double* sub_cell_centroids_x,
    double* sub_cell_centroids_y, double* sub_cell_centroids_z) {
  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nsub_cells = cells_offsets[(cc + 1)] - cells_off;

    const double cell_c_x = cell_centroids_x[(cc)];
    const double cell_c_y = cell_centroids_y[(cc)];
    const double cell_c_z = cell_centroids_z[(cc)];

    for (int ss = 0; ss < nsub_cells; ++ss) {
      // TODO: GET THE NODES AROUND A SUB-CELL

      sub_cell_centroids_x[(cc)] = 0.0;
      sub_cell_centroids_y[(cc)] = 0.0;
      sub_cell_centroids_z[(cc)] = 0.0;
    }
  }
  STOP_PROFILING(&compute_profile, __func__);
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

// Calculates the artificial viscous forces for momentum acceleration
void calculate_artificial_viscosity(
    const int ncells, const int nnodes, const double visc_coeff1,
    const double visc_coeff2, const int* cells_offsets,
    const int* cells_to_nodes, const int* nodes_offsets,
    const int* nodes_to_cells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const double* velocity_x, const double* velocity_y,
    const double* velocity_z, const double* nodal_soundspeed,
    const double* nodal_mass, const double* nodal_volumes,
    const double* limiter, double* node_force_x, double* node_force_y,
    double* node_force_z, int* nodes_to_faces_offsets, int* nodes_to_faces,
    int* faces_to_nodes_offsets, int* faces_to_nodes, int* faces_to_cells0,
    int* faces_to_cells1, int* cells_to_faces_offsets, int* cells_to_faces) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

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
        const double ab_x = (nodes_x[(current_node)] - half_edge_x);
        const double ab_y = (nodes_y[(current_node)] - half_edge_y);
        const double ab_z = (nodes_z[(current_node)] - half_edge_z);

        // Calculate the area vector S using cross product
        double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        double S_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES SO
        // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
        // CALCULATION
        if ((ab_x * S_x + ab_y * S_y + ab_z * S_z) > 0.0) {
          S_x *= -1.0;
          S_y *= -1.0;
          S_z *= -1.0;
        }

        // Calculate the velocity gradients
        const double dvel_x =
            velocity_x[(next_node)] - velocity_x[(current_node)];
        const double dvel_y =
            velocity_y[(next_node)] - velocity_y[(current_node)];
        const double dvel_z =
            velocity_z[(next_node)] - velocity_z[(current_node)];
        const double dvel_mag =
            sqrt(dvel_x * dvel_x + dvel_y * dvel_y + dvel_z * dvel_z);

        // Calculate the unit vectors of the velocity gradients
        const double dvel_unit_x = (dvel_mag != 0.0) ? dvel_x / dvel_mag : 0.0;
        const double dvel_unit_y = (dvel_mag != 0.0) ? dvel_y / dvel_mag : 0.0;
        const double dvel_unit_z = (dvel_mag != 0.0) ? dvel_z / dvel_mag : 0.0;

        // Get the edge-centered density
        double nodal_density0 =
            nodal_mass[(current_node)] / nodal_volumes[(current_node)];
        double nodal_density1 =
            nodal_mass[(next_node)] / nodal_volumes[(next_node)];
        const double density_edge = (2.0 * nodal_density0 * nodal_density1) /
                                    (nodal_density0 + nodal_density1);

        // Calculate the artificial viscous force term for the edge
        double expansion_term = (dvel_x * S_x + dvel_y * S_y + dvel_z * S_z);

        // If the cell is compressing, calculate the edge forces and add their
        // contributions to the node forces
        if (expansion_term <= 0.0) {
          // Calculate the minimum soundspeed
          const double cs = min(nodal_soundspeed[(current_node)],
                                nodal_soundspeed[(next_node)]);
          const double t = 0.25 * (GAM + 1.0);
          const double edge_visc_force_x =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_x) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_x * dvel_x +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_x;
          const double edge_visc_force_y =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_y) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_y * dvel_y +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_y;
          const double edge_visc_force_z =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_z) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_z * dvel_z +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_z;

          // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER CLOSED
          // FORM SOLUTION?
          int node_off;
          int next_node_off;
          for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
            if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
              node_off = nn3;
            } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == next_node) {
              next_node_off = nn3;
            }
          }

          // Add the contributions of the edge based artifical viscous terms
          // to the main force terms
          node_force_x[(cell_to_nodes_off + node_off)] -= edge_visc_force_x;
          node_force_y[(cell_to_nodes_off + node_off)] -= edge_visc_force_y;
          node_force_z[(cell_to_nodes_off + node_off)] -= edge_visc_force_z;
          node_force_x[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_x;
          node_force_y[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_y;
          node_force_z[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_z;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, __func__);
}
