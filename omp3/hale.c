#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "hale.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include "../../comms.h"
#include "../../params.h"

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_2d(
    Mesh* mesh, const int ncells, const int nnodes, const double dt, 
    int* cell_centroids_x, int* cell_centroids_y, int* cells_to_nodes, 
    int* nodes_to_cells, int* nodes_to_cells_off, int* cells_to_nodes_off, 
    double* nodes_x, double* nodes_y, double* node_volume, double* energy, 
    double* density, double* velocity_x, double* velocity_y, double* cell_force_x, 
    double* cell_force_y, double* node_force_x, double* node_force_y, 
    double* pressure, double* cell_mass, double* nodal_mass)
{
  // Random constants
  const double c1 = 1.0;
  const double c2 = 1.0;

  // TODO: Calculate the limiter?
  double limiter = 0.0;

  // Calculate the cell centroids
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
    const double inv_Np = 1.0/(double)nnodes_around_cell;

    cell_centroids_x[(cc)] = 0.0;
    cell_centroids_y[(cc)] = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      cell_centroids_x[(cc)] += nodes_x[(nodes_off)+(nn)]*inv_Np;
      cell_centroids_y[(cc)] += nodes_y[(nodes_off)+(nn)]*inv_Np;
    }
  }
  
  // Calculate the nodal and cell mass
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
    const double cell_centroid_x = cell_centroids_x[(cc)];
    const double cell_centroid_y = cell_centroids_x[(cc)];

    double cell_volume = 0.0;
    cell_mass[(cc)] = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      // Determine the three point stencil of nodes around current node
      const int node_left_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_center_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_right_index = (nn == nnodes_around_cell) 
        ? cells_to_nodes[0] : cells_to_nodes[(nodes_off)+(nn+1)];

      // Get the nodal coords of the three point stencil
      const double node_left_x = nodes_x[node_left_index];
      const double node_left_y = nodes_y[node_left_index];
      const double node_center_x = nodes_x[node_center_index];
      const double node_center_y = nodes_y[node_center_index];
      const double node_right_x = nodes_x[node_right_index];
      const double node_right_y = nodes_y[node_right_index];

      // Use shoelace formula to get the volume between node and cell center
      const double sub_cell_volume =
        (node_left_x*node_center_y + node_center_x*node_right_y + 
         node_right_x*cell_centroid_y + cell_centroid_x*node_left_y) -
        (node_center_x*node_left_y + node_right_x*node_center_y + 
         cell_centroid_x*node_right_y + node_left_x*cell_centroid_y);
      nodal_mass[(node_center_index)] += density[(cc)]*sub_cell_volume;

      cell_volume += 
        0.5*(nodes_x[node_center_index]+nodes_x[node_right_index])*
        (nodes_y[node_right_index]+nodes_y[node_center_index]);
    }

    cell_mass[(cc)] = density[(cc)]*cell_volume;
  }

  for(int cc = 0; cc < ncells; ++cc) {
    cell_force_x[(cc)] = 0.0;
    cell_force_y[(cc)] = 0.0;
  }
  for(int nn = 0; nn < nnodes; ++nn) {
    node_force_x[(nn)] = 0.0;
    node_force_y[(nn)] = 0.0;
  }

  // Calculate the force contributions for pressure gradients
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      // Determine the three point stencil of nodes around current node
      const int node_left_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_center_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_right_index = (nn == nnodes_around_cell) 
        ? cells_to_nodes[0] : cells_to_nodes[(nodes_off)+(nn+1)];

      double S_x = 0.0;
      double S_y = 0.0;

      // Get the nodal coords of the three point stencil
      const double node_left_x = nodes_x[node_left_index];
      const double node_left_y = nodes_y[node_left_index];
      const double node_center_x = nodes_x[node_center_index];
      const double node_center_y = nodes_y[node_center_index];
      const double node_right_x = nodes_x[node_right_index];
      const double node_right_y = nodes_y[node_right_index];

      // Calculate the half edge area vectors
      S_x = 0.25*(node_center_y-node_left_y) + 0.25*(node_right_y-node_center_y);
      S_y = -(0.25*(node_center_x-node_left_x) + 0.25*(node_right_x-node_center_x));

      // Add the contributions of the edge based artifical viscous terms
      // to the main force terms
      node_force_x[(node_center_index)] += pressure[(cc)]*S_x;
      node_force_y[(node_center_index)] += pressure[(cc)]*S_y;
      cell_force_x[(nodes_off)+(nn)] += pressure[(cc)]*S_x;
      cell_force_y[(nodes_off)+(nn)] += pressure[(cc)]*S_y;
    }
  }

  // Calculating artificial viscous terms for all edges of all cells
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      int node_index[NNODES_PER_EDGE]; 
      node_index[0] = cells_to_nodes[(nodes_off)+(nn)]; 
      node_index[1] = (nn == nnodes_around_cell) 
        ? cells_to_nodes[(nodes_off)] 
        : cells_to_nodes[(nodes_off)+(nn+1)];

      // Calculate area weighted averages of the density and soundspeed here.
      double density_node[NNODES_PER_EDGE];
      double cs_node[NNODES_PER_EDGE];
      for(int oo = 0; oo < NNODES_PER_EDGE; ++oo) {
        double total_volume = 0.0;
        const int cells_by_node_off = nodes_to_cells_off[(node_index[oo])];
        const int ncells_around_node = 
          cells_to_nodes_off[(node_index[oo]+1)]-cells_by_node_off;
        for(int zz = 0; zz < ncells_around_node; ++zz) {
          const double V = node_volume[(cells_by_node_off)+(zz)];
          const int cell_index = nodes_to_cells[(cells_by_node_off)+(zz)];
          cs_node[oo] += sqrt(GAM*(GAM-1.0)*energy[(cell_index)])*V;
          density_node[oo] += density[(cell_index)]*V;
          total_volume += V;
        }
        density_node[oo] /= total_volume;
        cs_node[oo] /= total_volume;
      }

      // Area vector for cell center to edge midpoint
      const double cell_x = cell_centroids_x[(cc)];
      const double cell_y = cell_centroids_y[(cc)];
      const double edge_mid_x = 
        0.5*(nodes_x[node_index[0]] + nodes_x[node_index[1]]);
      const double edge_mid_y = 
        0.5*(nodes_y[node_index[0]] + nodes_y[node_index[1]]);

      // Rotate the vector between cell center and edge midpoint to get normal
      const double S_x = (edge_mid_y-cell_y);
      const double S_y = -(edge_mid_x-cell_x);

      // Velocity gradients
      const double grad_velocity_x = 
        velocity_x[node_index[1]]-velocity_x[node_index[0]];
      const double grad_velocity_y = 
        velocity_y[node_index[1]]-velocity_y[node_index[0]];
      const double grad_velocity_mag =
        sqrt(grad_velocity_x*grad_velocity_x+grad_velocity_y*grad_velocity_y);
      const double grad_velocity_unit_x = grad_velocity_x/grad_velocity_mag;
      const double grad_velocity_unit_y = grad_velocity_y/grad_velocity_mag;

      // Calculate the minimum soundspeed
      const double cs = min(cs_node[0], cs_node[1]);

      // Calculate the edge centered density with a harmonic mean
      const double density_edge = 
        (2.0*density_node[0]*density_node[1])/(density_node[0]+density_node[1]);

      // Calculate the artificial viscous force term for the edge
      const double t = 0.25*(GAM + 1.0);
      double expansion_term = (grad_velocity_x*S_x + grad_velocity_y*S_y);

      // If the cell is compressing, calculate the edge forces and add their
      // contributions to the node forces
      if(expansion_term <= 0.0) {
        const double edge_visc_force_x = 
          density_edge*(c2*t*fabs(grad_velocity_x) + 
              sqrt(c2*c2*t*t*grad_velocity_x*grad_velocity_x + c1*c1*cs*cs))*
          (1.0 - limiter)*expansion_term*grad_velocity_unit_x;
        const double edge_visc_force_y = 
          density_edge*(c2*t*fabs(grad_velocity_y) + 
              sqrt(c2*c2*t*t*grad_velocity_y*grad_velocity_y + c1*c1*cs*cs))*
          (1.0 - limiter)*expansion_term*grad_velocity_unit_y;

        // Add the contributions of the edge based artifical viscous terms
        // to the main force terms
        node_force_x[(node_index[0])] += edge_visc_force_x;
        node_force_x[(node_index[1])] -= edge_visc_force_x;
        node_force_y[(node_index[0])] += edge_visc_force_y;
        node_force_y[(node_index[1])] -= edge_visc_force_y;
        cell_force_x[(nodes_off)+(nn)] += edge_visc_force_x;
        cell_force_y[(nodes_off)+(nn)] += edge_visc_force_y;
      }
    }
  }

  // Calculate the half timestep evolved velocities, by first calculating the
  // predicted values at the new timestep and then averaging with current velocity
  for(int nn = 0; nn < nnodes; ++nn) {
    velocity_x[(nn)] = 
      velocity_x[(nn)] + 0.5*(dt/nodal_mass[(nn)])*node_force_x[(nn)];
    velocity_y[(nn)] = 
      velocity_y[(nn)] + 0.5*(dt/nodal_mass[(nn)])*node_force_y[(nn)];
  }

  // Calculate the predicted energy
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    // Sum the half timestep velocity by the sub-cell forces
    double force = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = cells_to_nodes[(nodes_off)+(nn)];
      force += 
        (velocity_x[(node_index)]*cell_force_x[(nodes_off)+(nn)] +
         velocity_y[(node_index)]*cell_force_y[(nodes_off)+(nn)]);
    }

    energy[(cc)] -= (dt/cell_mass[(cc)])*force;
  }
}

#if 0
// Calculate the pressure from GAMma law equation of state
void equation_of_state(
    const int nx, const int ny, double* P, const double* density, const double* e)
{
  START_PROFILING(&compute_profile);

#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      // Only invoke simple GAMma law at the moment
      P[(ii*nx+jj)] = (GAM - 1.0)*density[(ii*nx+jj)]*e[(ii*nx+jj)];
    }
  }

  STOP_PROFILING(&compute_profile, __func__);
}

// Calculates the timestep from the current state
void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* density, 
    const double* e, Mesh* mesh, double* reduce_array, const int first_step,
    const double* celldx, const double* celldy)
{
  double local_min_dt = mesh->max_dt;

  START_PROFILING(&compute_profile);
  // Check the minimum timestep from the sound speed in the nx and ny directions
#pragma omp parallel for reduction(min: local_min_dt)
  for(int ii = PAD; ii < ny-PAD; ++ii) {
#pragma omp simd 
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      // Constrain based on the sound speed within the system
      const double c_s = sqrt(GAM*(GAM - 1.0)*e[(ii*nx+jj)]);
      const double thread_min_dt_x = 
        celldx[jj]/sqrt(c_s*c_s + 2.0*Qxx[(ii*nx+jj)]/density[(ii*nx+jj)]);
      const double thread_min_dt_y = 
        celldy[ii]/sqrt(c_s*c_s + 2.0*Qyy[(ii*nx+jj)]/density[(ii*nx+jj)]);
      const double thread_min_dt = min(thread_min_dt_x, thread_min_dt_y);
      local_min_dt = min(local_min_dt, thread_min_dt);
    }
  }
  STOP_PROFILING(&compute_profile, __func__);

  double global_min_dt = reduce_all_min(local_min_dt);

  // Ensure that the timestep does not jump too far from one step to the next
  const double final_min_dt = min(global_min_dt, C_M*mesh->dt_h);
  mesh->dt = 0.5*(C_T*final_min_dt + mesh->dt_h);
  mesh->dt_h = (first_step) ? mesh->dt : C_T*final_min_dt;
}

