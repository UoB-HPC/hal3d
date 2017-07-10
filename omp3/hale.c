#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include "hale.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include "../../comms.h"
#include "../../params.h"
#include "../../shared.h"

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_2d(
    Mesh* mesh, const int ncells, const int nnodes, const double visc_coeff1, 
    const double visc_coeff2, double* cell_centroids_x, double* cell_centroids_y, 
    int* cells_to_nodes, int* cells_to_nodes_off, int* nodes_to_cells, 
    int* nodes_to_cells_off, double* nodes_x0, double* nodes_y0, double* nodes_x1, 
    double* nodes_y1, int* boundary_index, int* boundary_type, double* boundary_normal_x, 
    double* boundary_normal_y, double* energy0, double* energy1, double* density0, 
    double* density1, double* pressure0, double* pressure1, double* velocity_x0, 
    double* velocity_y0, double* velocity_x1, double* velocity_y1, 
    double* cell_force_x, double* cell_force_y, double* node_force_x, 
    double* node_force_y, double* node_force_x2, double* node_force_y2, 
    double* cell_mass, double* nodal_mass, double* nodal_volumes, 
    double* nodal_soundspeed, double* limiter)
{
  /*
   *    PREDICTOR
   */

  // Calculate the pressure using the ideal gas equation of state
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    pressure0[(cc)] = (GAM-1.0)*energy0[(cc)]*density0[(cc)];
  }
  STOP_PROFILING(&compute_profile, "equation_of_state");

  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
    const double inv_Np = 1.0/(double)nnodes_around_cell;

    double cx = 0.0;
    double cy = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = cells_to_nodes[(nodes_off)+(nn)];
      cx += nodes_x0[(node_index)]*inv_Np;
      cy += nodes_y0[(node_index)]*inv_Np;
    }
    cell_centroids_x[(cc)] = cx;
    cell_centroids_y[(cc)] = cy;
  }
  STOP_PROFILING(&compute_profile, "calc_centroids");

  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for(int nn = 0; nn < nnodes; ++nn) {
    nodal_mass[(nn)] = 0.0;
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }
  STOP_PROFILING(&compute_profile, "zero_nodal_arrays");

  // Calculate the cell mass
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
#pragma omp parallel for reduction(+: total_mass)
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
    const double cell_centroid_x = cell_centroids_x[(cc)];
    const double cell_centroid_y = cell_centroids_y[(cc)];

    double cell_volume = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {

      // Determine the three point stencil of nodes around current node
      const int node_l_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];

      const double node_c_x = nodes_x0[(node_c_index)];
      const double node_c_y = nodes_y0[(node_c_index)];

      // Get the midpoints between l and r nodes and current node
      const double node_l_x = 0.5*(nodes_x0[node_l_index]+node_c_x);
      const double node_l_y = 0.5*(nodes_y0[node_l_index]+node_c_y);
      const double node_r_x = 0.5*(node_c_x+nodes_x0[node_r_index]);
      const double node_r_y = 0.5*(node_c_y+nodes_y0[node_r_index]);

      // Use shoelace formula to get the volume between node and cell c
      const double sub_cell_volume =
        0.5*((node_l_x*node_c_y + node_c_x*node_r_y +
              node_r_x*cell_centroid_y + cell_centroid_x*node_l_y) -
            (node_c_x*node_l_y + node_r_x*node_c_y +
             cell_centroid_x*node_r_y + node_l_x*cell_centroid_y));

      // TODO: this should be updated to fix the issues with hourglassing...
      if(sub_cell_volume <= 0.0) {
        TERMINATE("Encountered cell with unphysical volume %d.", cc);
      }

      // Reduce the total cell volume for later calculation
      cell_volume += sub_cell_volume;
    }

    // Calculate the mass and store volume for the whole cell
    cell_mass[(cc)] = density0[(cc)]*cell_volume;
    total_mass += cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_cell_mass");

  printf("total mass %.12f\n", total_mass);

  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    const int node_offset = nodes_to_cells_off[(nn)];
    const int ncells_by_node = nodes_to_cells_off[(nn+1)]-node_offset;
    for(int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_offset+cc)];
      const int cell_offset = cells_to_nodes_off[(cell_index)];
      const int nnodes_by_cell = cells_to_nodes_off[(cell_index+1)]-cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for(nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if(cells_to_nodes[(cell_offset+nn2)] == nn) {
          break;
        }
      }

      const double cell_centroid_x = cell_centroids_x[(cell_index)];
      const double cell_centroid_y = cell_centroids_y[(cell_index)];

      const int node_l_index = (nn2-1 >= 0) ?
        cells_to_nodes[(cell_offset+nn2-1)] : cells_to_nodes[(cell_offset+nnodes_by_cell-1)];
      const int node_r_index = (nn2+1 < nnodes_by_cell) ?
        cells_to_nodes[(cell_offset+nn2+1)] : cells_to_nodes[(cell_offset)];

      const double node_c_x = nodes_x0[(nn)];
      const double node_c_y = nodes_y0[(nn)];

      // Get the midpoints between l and r nodes and current node
      const double node_l_x = 0.5*(nodes_x0[node_l_index]+node_c_x);
      const double node_l_y = 0.5*(nodes_y0[node_l_index]+node_c_y);
      const double node_r_x = 0.5*(node_c_x+nodes_x0[node_r_index]);
      const double node_r_y = 0.5*(node_c_y+nodes_y0[node_r_index]);

      // Use shoelace formula to get the volume between node and cell c
      const double sub_cell_volume =
        0.5*((node_l_x*node_c_y + node_c_x*node_r_y +
              node_r_x*cell_centroid_y + cell_centroid_x*node_l_y) -
            (node_c_x*node_l_y + node_r_x*node_c_y +
             cell_centroid_x*node_r_y + node_l_x*cell_centroid_y));

      // TODO: this should be updated to fix the issues with hourglassing...
      if(sub_cell_volume <= 0.0) {
        TERMINATE("Encountered cell with unphysical volume %d.", cc);
      }

      nodal_mass[(nn)] += density0[(cell_index)]*sub_cell_volume;

      // Calculate the volume and soundspeed at the node
      nodal_soundspeed[(nn)] += 
        sqrt(GAM*(GAM-1.0)*energy0[(cell_index)])*sub_cell_volume;
      nodal_volumes[(nn)] += sub_cell_volume;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    node_force_x[(nn)] = 0.0;
    node_force_y[(nn)] = 0.0;
    node_force_x2[(nn)] = 0.0;
    node_force_y2[(nn)] = 0.0;
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }
  STOP_PROFILING(&compute_profile, "calc_soundspeed");

  // Calculate the force contributions for pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    const int node_offset = nodes_to_cells_off[(nn)];
    const int ncells_by_node = nodes_to_cells_off[(nn+1)]-node_offset;
    for(int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_offset+cc)];
      const int cell_offset = cells_to_nodes_off[(cell_index)];
      const int nnodes_by_cell = cells_to_nodes_off[(cell_index+1)]-cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for(nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if(cells_to_nodes[(cell_offset+nn2)] == nn) {
          break;
        }
      }

      const int node_l_index = (nn2-1 >= 0) ?
        cells_to_nodes[(cell_offset+nn2-1)] : cells_to_nodes[(cell_offset+nnodes_by_cell-1)];
      const int node_r_index = (nn2+1 < nnodes_by_cell) ?
        cells_to_nodes[(cell_offset+nn2+1)] : cells_to_nodes[(cell_offset)];

      // Calculate the area vectors away from cell through node, using
      // the half edge vectors adjacent to the node combined
      const double S_x =
        0.5*((nodes_y0[(nn)]-nodes_y0[(node_l_index)]) +
            (nodes_y0[(node_r_index)]-nodes_y0[(nn)]));
      const double S_y =
        -0.5*((nodes_x0[(nn)]-nodes_x0[(node_l_index)]) +
            (nodes_x0[(node_r_index)]-nodes_x0[(nn)]));

      node_force_x[(nn)] += pressure0[(cell_index)]*S_x;
      node_force_y[(nn)] += pressure0[(cell_index)]*S_y;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_force");

  // Calculate the force contributions for pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 

      // Determine the three point stencil of nodes around current node
      const int node_l_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)]
        : cells_to_nodes[(nodes_off)+(nn+1)];

      // Calculate the area vectors away from cell through node, using
      // the half edge vectors adjacent to the node combined
      const double S_x =
        0.5*((nodes_y0[(node_c_index)]-nodes_y0[(node_l_index)]) +
            (nodes_y0[(node_r_index)]-nodes_y0[(node_c_index)]));
      const double S_y =
        -0.5*((nodes_x0[(node_c_index)]-nodes_x0[(node_l_index)]) +
            (nodes_x0[(node_r_index)]-nodes_x0[(node_c_index)]));

      cell_force_x[(nodes_off)+(nn)] = pressure0[(cc)]*S_x;
      cell_force_y[(nodes_off)+(nn)] = pressure0[(cc)]*S_y;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_cell_forces");

  calculate_artificial_viscosity(
      nnodes, visc_coeff1, visc_coeff2, cells_to_nodes_off, cells_to_nodes, 
      nodes_to_cells_off, nodes_to_cells, nodes_x0, nodes_y0, cell_centroids_x, 
      cell_centroids_y, velocity_x0, velocity_y0, nodal_soundspeed, nodal_mass,
      nodal_volumes, limiter, node_force_x, node_force_y, node_force_x2, node_force_y2);

  // Calculate the time centered evolved velocities, by first calculating the
  // predicted values at the new timestep and then averaging with current velocity
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for(int nn = 0; nn < nnodes; ++nn) {
    // Determine the predicted velocity
    velocity_x1[(nn)] = velocity_x0[(nn)] + mesh->dt*node_force_x[(nn)]/nodal_mass[(nn)];
    velocity_y1[(nn)] = velocity_y0[(nn)] + mesh->dt*node_force_y[(nn)]/nodal_mass[(nn)];

    // Calculate the time centered velocity
    velocity_x1[(nn)] = 0.5*(velocity_x0[(nn)] + velocity_x1[(nn)]);
    velocity_y1[(nn)] = 0.5*(velocity_y0[(nn)] + velocity_y1[(nn)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_velocity");

  handle_unstructured_reflect(
      nnodes, boundary_index, boundary_type, boundary_normal_x, 
      boundary_normal_y, velocity_x1, velocity_y1);

  // Move the nodes by the predicted velocity
  START_PROFILING(&compute_profile);
#pragma omp parallel for simd
  for(int nn = 0; nn < nnodes; ++nn) {
    nodes_x1[(nn)] = nodes_x0[(nn)] + mesh->dt*velocity_x1[(nn)];
    nodes_y1[(nn)] = nodes_y0[(nn)] + mesh->dt*velocity_y1[(nn)];
  }
  STOP_PROFILING(&compute_profile, "move_nodes");

  set_timestep(
      ncells, cells_to_nodes, cells_to_nodes_off, 
      nodes_x1, nodes_y1, energy0, &mesh->dt);

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    // Sum the time centered velocity by the sub-cell forces
    double force = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = cells_to_nodes[(nodes_off)+(nn)];
      force += 
        (velocity_x1[(node_index)]*cell_force_x[(nodes_off)+(nn)] +
         velocity_y1[(node_index)]*cell_force_y[(nodes_off)+(nn)]);
    }

    energy1[(cc)] = energy0[(cc)] - mesh->dt*force/cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  // Using the new volume, calculate the predicted density
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    double cell_volume = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {

      // Determine the three point stencil of nodes around current node
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];

      // Reduce the total cell volume for later calculation
      cell_volume += 0.5*(nodes_x1[node_c_index]+nodes_x1[node_r_index])*
        (nodes_y1[node_r_index]-nodes_y1[node_c_index]);
    }

    density1[(cc)] = cell_mass[(cc)]/cell_volume;
  }
  STOP_PROFILING(&compute_profile, "calc_new_density");

  // Calculate the time centered pressure from mid point between original and 
  // predicted pressures
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    // Calculate the predicted pressure from the equation of state
    pressure1[(cc)] = (GAM-1.0)*energy1[(cc)]*density1[(cc)];

    // Determine the time centered pressure
    pressure1[(cc)] = 0.5*(pressure0[(cc)] + pressure1[(cc)]);
  }
  STOP_PROFILING(&compute_profile, "equation_of_state_time_center");

  // Prepare time centered variables for the corrector step
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    nodes_x1[(nn)] = 0.5*(nodes_x1[(nn)] + nodes_x0[(nn)]);
    nodes_y1[(nn)] = 0.5*(nodes_y1[(nn)] + nodes_y0[(nn)]);
    node_force_x[(nn)] = 0.0;
    node_force_y[(nn)] = 0.0;
    node_force_x2[(nn)] = 0.0;
    node_force_y2[(nn)] = 0.0;
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }
  STOP_PROFILING(&compute_profile, "move_nodes2");

  /*
   *    CORRECTOR
   */

  // Calculate the new cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
    const double inv_Np = 1.0/(double)nnodes_around_cell;

    double cx = 0.0;
    double cy = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = cells_to_nodes[(nodes_off)+(nn)];
      cx += nodes_x1[(node_index)]*inv_Np;
      cy += nodes_y1[(node_index)]*inv_Np;
    }
    cell_centroids_x[(cc)] = cx;
    cell_centroids_y[(cc)] = cy;
  }
  STOP_PROFILING(&compute_profile, "calc_centroids");

  // Calculate the new nodal soundspeed and volumes
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    const int node_offset = nodes_to_cells_off[(nn)];
    const int ncells_by_node = nodes_to_cells_off[(nn+1)]-node_offset;
    double nc = 0.0;
    double nv = 0.0;
    for(int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_offset+cc)];
      const int cell_offset = cells_to_nodes_off[(cell_index)];
      const int nnodes_by_cell = cells_to_nodes_off[(cell_index+1)]-cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for(nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if(cells_to_nodes[(cell_offset+nn2)] == nn) {
          break;
        }
      }

      const double cell_centroid_x = cell_centroids_x[(cell_index)];
      const double cell_centroid_y = cell_centroids_y[(cell_index)];

      const int node_l_index = (nn2-1 >= 0) ?
        cells_to_nodes[(cell_offset+nn2-1)] : cells_to_nodes[(cell_offset+nnodes_by_cell-1)];
      const int node_r_index = (nn2+1 < nnodes_by_cell) ?
        cells_to_nodes[(cell_offset+nn2+1)] : cells_to_nodes[(cell_offset)];

      const double node_c_x = nodes_x1[(nn)];
      const double node_c_y = nodes_y1[(nn)];

      // Get the midpoints between l and r nodes and current node
      const double node_l_x = 0.5*(nodes_x1[node_l_index]+node_c_x);
      const double node_l_y = 0.5*(nodes_y1[node_l_index]+node_c_y);
      const double node_r_x = 0.5*(node_c_x+nodes_x1[node_r_index]);
      const double node_r_y = 0.5*(node_c_y+nodes_y1[node_r_index]);

      // Use shoelace formula to get the volume between node and cell c
      const double sub_cell_volume =
        0.5*((node_l_x*node_c_y + node_c_x*node_r_y +
              node_r_x*cell_centroid_y + cell_centroid_x*node_l_y) -
            (node_c_x*node_l_y + node_r_x*node_c_y +
             cell_centroid_x*node_r_y + node_l_x*cell_centroid_y));

      // Add contributions to the nodal mass from adjacent sub-cells
      nc += sqrt(GAM*(GAM-1.0)*energy1[(cell_index)])*sub_cell_volume;

      // Calculate the volume at the node
      nv += sub_cell_volume;
    }

    nodal_soundspeed[(nn)] = nc;
    nodal_volumes[(nn)] = nv;
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_volume");

  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_soundspeed");

  // Calculate the force contributions for pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    const int node_offset = nodes_to_cells_off[(nn)];
    const int ncells_by_node = nodes_to_cells_off[(nn+1)]-node_offset;
    for(int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_offset+cc)];
      const int cell_offset = cells_to_nodes_off[(cell_index)];
      const int nnodes_by_cell = cells_to_nodes_off[(cell_index+1)]-cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for(nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if(cells_to_nodes[(cell_offset+nn2)] == nn) {
          break;
        }
      }

      const int node_l_index = (nn2-1 >= 0) ?
        cells_to_nodes[(cell_offset+nn2-1)] : cells_to_nodes[(cell_offset+nnodes_by_cell-1)];
      const int node_r_index = (nn2+1 < nnodes_by_cell) ?
        cells_to_nodes[(cell_offset+nn2+1)] : cells_to_nodes[(cell_offset)];

      // Calculate the area vectors away from cell through node, using
      // the half edge vectors adjacent to the node combined
      const double S_x =
        0.5*((nodes_y1[(nn)]-nodes_y1[(node_l_index)]) +
            (nodes_y1[(node_r_index)]-nodes_y1[(nn)]));
      const double S_y =
        -0.5*((nodes_x1[(nn)]-nodes_x1[(node_l_index)]) +
            (nodes_x1[(node_r_index)]-nodes_x1[(nn)]));

      node_force_x[(nn)] += pressure1[(cell_index)]*S_x;
      node_force_y[(nn)] += pressure1[(cell_index)]*S_y;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_force");

  // Calculate the force contributions for pressure gradients
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 

      // Determine the three point stencil of nodes around current node
      const int node_l_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)]
        : cells_to_nodes[(nodes_off)+(nn+1)];

      // Calculate the area vectors away from cell through node, using
      // the half edge vectors adjacent to the node combined
      const double S_x =
        0.5*((nodes_y1[(node_c_index)]-nodes_y1[(node_l_index)]) +
            (nodes_y1[(node_r_index)]-nodes_y1[(node_c_index)]));
      const double S_y =
        -0.5*((nodes_x1[(node_c_index)]-nodes_x1[(node_l_index)]) +
            (nodes_x1[(node_r_index)]-nodes_x1[(node_c_index)]));

      cell_force_x[(nodes_off)+(nn)] = pressure1[(cc)]*S_x;
      cell_force_y[(nodes_off)+(nn)] = pressure1[(cc)]*S_y;
    }
  }
  STOP_PROFILING(&compute_profile, "calc_cell_forces");

  calculate_artificial_viscosity(
      nnodes, visc_coeff1, visc_coeff2, cells_to_nodes_off, cells_to_nodes, 
      nodes_to_cells_off, nodes_to_cells, nodes_x1, nodes_y1, cell_centroids_x, 
      cell_centroids_y, velocity_x1, velocity_y1, nodal_soundspeed, nodal_mass,
      nodal_volumes, limiter, node_force_x, node_force_y, node_force_x2, node_force_y2);

  update_velocity(
      nnodes, mesh->dt, node_force_x, node_force_y, nodal_mass, velocity_x0, 
      velocity_y0, velocity_x1, velocity_y1);

  handle_unstructured_reflect(
      nnodes, boundary_index, boundary_type, boundary_normal_x, 
      boundary_normal_y, velocity_x0, velocity_y0);

  // Calculate the corrected node movements
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    nodes_x0[(nn)] += mesh->dt*velocity_x0[(nn)];
    nodes_y0[(nn)] += mesh->dt*velocity_y0[(nn)];
  }
  STOP_PROFILING(&compute_profile, "move_nodes");

  set_timestep(
      ncells, cells_to_nodes, cells_to_nodes_off,
      nodes_x0, nodes_y0, energy1, &mesh->dt);

  // Calculate the final energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    // Sum the time centered velocity by the sub-cell forces
    double force = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = cells_to_nodes[(nodes_off)+(nn)];
      force += 
        (velocity_x0[(node_index)]*cell_force_x[(nodes_off)+(nn)] +
         velocity_y0[(node_index)]*cell_force_y[(nodes_off)+(nn)]);
    }

    energy0[(cc)] -= mesh->dt*force/cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, "calc_new_energy");

  // Using the new corrected volume, calculate the density
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    double cell_volume = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      // Calculate the new volume of the cell
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];
      cell_volume += 
        0.5*(nodes_x0[node_c_index]+nodes_x0[node_r_index])*
        (nodes_y0[node_r_index]-nodes_y0[node_c_index]);
    }

    // Update the density using the new volume
    density0[(cc)] = cell_mass[(cc)]/cell_volume;
  }
  STOP_PROFILING(&compute_profile, "calc_new_density");
}

// Calculates the artificial viscous forces for momentum acceleration
void calculate_artificial_viscosity(
    const int nnodes, const double visc_coeff1, const double visc_coeff2, 
    const int* cells_to_nodes_off, const int* cells_to_nodes, 
    const int* nodes_to_cells_off, const int* nodes_to_cells,
    const double* nodes_x, const double* nodes_y, 
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* velocity_x, const double* velocity_y,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter,
    double* node_force_x, double* node_force_y,
    double* node_force_x2, double* node_force_y2)
{
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    const int node_offset = nodes_to_cells_off[(nn)];
    const int ncells_by_node = nodes_to_cells_off[(nn+1)]-node_offset;
    for(int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_offset+cc)];
      const int cell_offset = cells_to_nodes_off[(cell_index)];
      const int nnodes_by_cell = cells_to_nodes_off[(cell_index+1)]-cell_offset;

      // Annoying search to find the relevant node in cell list
      int nn2;
      for(nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if(cells_to_nodes[(cell_offset+nn2)] == nn) {
          break;
        }
      }

      const int node_r_index = (nn2+1 < nnodes_by_cell) ?
        cells_to_nodes[(cell_offset+nn2+1)] : cells_to_nodes[(cell_offset)];

      // Get cell center point and edge center point
      const double cell_x = cell_centroids_x[(cell_index)];
      const double cell_y = cell_centroids_y[(cell_index)];
      const double edge_mid_x = 
        0.5*(nodes_x[(nn)] + nodes_x[(node_r_index)]);
      const double edge_mid_y = 
        0.5*(nodes_y[(nn)] + nodes_y[(node_r_index)]);

      // Rotate the vector between cell c and edge midpoint to get normal
      const double S_x = (edge_mid_y-cell_y);
      const double S_y = -(edge_mid_x-cell_x);

      // Velocity gradients
      const double grad_velocity_x = 
        velocity_x[(node_r_index)]-velocity_x[(nn)];
      const double grad_velocity_y = 
        velocity_y[(node_r_index)]-velocity_y[(nn)];
      const double grad_velocity_mag =
        sqrt(grad_velocity_x*grad_velocity_x+grad_velocity_y*grad_velocity_y);
      const double grad_velocity_unit_x = 
        (grad_velocity_x != 0.0) ? grad_velocity_x/grad_velocity_mag : 0.0;
      const double grad_velocity_unit_y = 
        (grad_velocity_y != 0.0) ? grad_velocity_y/grad_velocity_mag : 0.0;

      // Calculate the minimum soundspeed
      const double cs = min(
          nodal_soundspeed[(nn)], nodal_soundspeed[(node_r_index)]);

      // Calculate the edge centered density with a harmonic mean
      double nodal_density_l = nodal_mass[(nn)]/nodal_volumes[(nn)];
      double nodal_density_r = nodal_mass[(node_r_index)]/nodal_volumes[(node_r_index)];
      const double density_edge = 
        (2.0*nodal_density_l*nodal_density_r)/(nodal_density_l+nodal_density_r);

      // Calculate the artificial viscous force term for the edge
      const double t = 0.25*(GAM + 1.0);
      double expansion_term = (grad_velocity_x*S_x + grad_velocity_y*S_y);

      // If the cell is compressing, calculate the edge forces and add their
      // contributions to the node forces
      if(expansion_term <= 0.0) {
        const double edge_visc_force_x = 
          density_edge*(visc_coeff2*t*fabs(grad_velocity_x) + 
              sqrt(visc_coeff2*visc_coeff2*t*t*grad_velocity_x*grad_velocity_x +
                visc_coeff1*visc_coeff1*cs*cs))*
          (1.0 - limiter[(nn)])*(grad_velocity_x*S_x)*grad_velocity_unit_x;
        const double edge_visc_force_y = 
          density_edge*(visc_coeff2*t*fabs(grad_velocity_y) +
              sqrt(visc_coeff2*visc_coeff2*t*t*grad_velocity_y*grad_velocity_y +
                visc_coeff1*visc_coeff1*cs*cs))*
          (1.0 - limiter[(nn)])*(grad_velocity_y*S_y)*grad_velocity_unit_y;

        // Add the contributions of the edge based artifical viscous terms
        // to the main force terms
        node_force_x[(nn)] -= edge_visc_force_x;
        node_force_y[(nn)] -= edge_visc_force_y;

        //
        //
        //
        //
        //
        // TODO : There is a race condition here...
        node_force_x[(node_r_index)] += edge_visc_force_x;
        node_force_y[(node_r_index)] += edge_visc_force_y;
      }
    }
  }

  STOP_PROFILING(&compute_profile, "artificial_viscosity");
}

// Controls the timestep for the simulation
void set_timestep(
    const int ncells, const int* cells_to_nodes, const int* cells_to_nodes_off,
    const double* nodes_x, const double* nodes_y, const double* energy, double* dt)
{
  // Calculate the timestep based on the computational mesh and CFL condition
  double local_dt = DBL_MAX;
  START_PROFILING(&compute_profile);
#pragma omp parallel for reduction(min: local_dt)
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    double shortest_edge = DBL_MAX;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      // Calculate the new volume of the cell
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];
      const double x_component = nodes_x[(node_c_index)]-nodes_x[(node_r_index)];
      const double y_component = nodes_y[(node_c_index)]-nodes_y[(node_r_index)];
      shortest_edge = min(shortest_edge, 
          sqrt(x_component*x_component+y_component*y_component));
    }

    const double soundspeed = sqrt(GAM*(GAM-1.0)*energy[(cc)]);
    local_dt = min(local_dt, shortest_edge/soundspeed);
  }
  STOP_PROFILING(&compute_profile, __func__);

  *dt = CFL*local_dt;
  printf("Timestep %.8fs\n", *dt);
}

// Uodates the velocity due to the pressure gradients
void update_velocity(
    const int nnodes, const double dt, const double* node_force_x, 
    const double* node_force_y, const double* nodal_mass, double* velocity_x0, 
    double* velocity_y0, double* velocity_x1, double* velocity_y1) 
{
  // Calculate the corrected time centered velocities
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    // Calculate the new velocities
    velocity_x1[(nn)] += dt*node_force_x[(nn)]/nodal_mass[(nn)];
    velocity_y1[(nn)] += dt*node_force_y[(nn)]/nodal_mass[(nn)];

    // Calculate the corrected time centered velocities
    velocity_x0[(nn)] = 0.5*(velocity_x1[(nn)] + velocity_x0[(nn)]);
    velocity_y0[(nn)] = 0.5*(velocity_y1[(nn)] + velocity_y0[(nn)]);
  }
  STOP_PROFILING(&compute_profile, "calc_new_velocity_time_center");
}

