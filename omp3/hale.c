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
    Mesh* mesh, int* cell_centroids_x, int* cell_centroids_y, int* cells_nodes, 
    int* nodes_cells, int* nodes_cells_off, int* cells_nodes_off, 
    double* nodes_x, double* nodes_y, double* node_volume, double* energy, 
    double* density, double* velocity_x, double* velocity_y, double* edge_force_x,
    double* edge_force_y)
{
  // Random constants
  const double c1 = 1.0;
  const double c2 = 1.0;
  const int ncells = 100;

  // TODO: Calculate the limiter?
  double limiter = 0.0;

  // Calculating artificial viscosity for all edges of all cells
  for(int cc = 0; cc < ncells; ++cc) {
    for(int ee = 0; ee < cells_nodes[(cc)]; ++ee) {
      const int node_index[NNODES_PER_EDGE] = { 
        cells_nodes[(ee)*NNODES_PER_EDGE+0], cells_nodes[(ee)*NNODES_PER_EDGE+1] };

      // Calculate area weighted averages of the density and soundspeed here.
      double density_node[NNODES_PER_EDGE];
      double cs_node[NNODES_PER_EDGE];
      for(int oo = 0; oo < NNODES_PER_EDGE; ++oo) {
        double total_volume = 0.0;
        const int cells_by_node_off = nodes_cells_off[(node_index[oo])];
        const int ncells_surrounding_node = 
          cells_nodes_off[(node_index[oo])+1]-cells_by_node_off;
        for(int nn = 0; nn < ncells_surrounding_node; ++nn) {
          const double V = node_volume[(cells_by_node_off)+(nn)];
          const int cell_index = nodes_cells[(cells_by_node_off)+(nn)];
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

      // Calculate the minimum soundspeed
      const double cs = min(cs_node[0], cs_node[1]);

      // Calculate the edge centered density with a harmonic mean
      const double density_edge = 
        (2.0*density_node[0]*density_node[1])/(density_node[0]+density_node[1]);

      // Calculate the artificial viscous force term for the edge
      const double t = 0.25*(GAM + 1.0);
      double expansion_term = (grad_velocity_x*S_x + grad_velocity_y*S_y);

      const int edges_by_cell_off = cells_nodes_off[(cc)];
      if(expansion_term <= 0.0) {
        edge_force_x[(edges_by_cell_off)] = 
          density_edge*(c2*t*fabs(grad_velocity_x) + 
              sqrt(c2*c2*t*t*grad_velocity_x*grad_velocity_x + c1*c1*cs*cs))*
          (1.0 - limiter)*expansion_term;
        edge_force_y[(edges_by_cell_off)] = 
          density_edge*(c2*t*fabs(grad_velocity_y) + 
              sqrt(c2*c2*t*t*grad_velocity_y*grad_velocity_y + c1*c1*cs*cs))*
          (1.0 - limiter)*expansion_term;
      }
      else {
        edge_force_x[(edges_by_cell_off)+(ee)] = 0.0;
        edge_force_y[(edges_by_cell_off)+(ee)] = 0.0;
      }
    }
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

