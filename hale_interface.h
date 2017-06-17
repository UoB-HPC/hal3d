#ifndef __HALEINTERFACEHDR
#define __HALEINTERFACEHDR

#pragma once

#include "hale_data.h" // An important part of the interface
#include "../shared.h"
#include "../mesh.h"
#include "../shared_data.h"

// Controllable parameters for the application
#define GAM 1.4
#define C_Q 3.0
#define C_M (1.5/C_T)

#ifdef __cplusplus
extern "C" {
#endif

  // Solve a single timestep on the given mesh
  void solve_unstructured_hydro_2d(
      Mesh* mesh, const int ncells, const int nnodes, const double dt, 
      double* cell_centroids_x, double* cell_centroids_y, int* cells_to_nodes, 
      int* nodes_to_cells, int* nodes_to_cells_off, int* cells_to_nodes_off, 
      double* nodes_x0, double* nodes_y0, double* nodes_x1, double* nodes_y1,
      int* halo_cell, double* energy0, double* energy1, double* density0, double* density1, 
      double* pressure0, double* pressure1, double* velocity_x0, double* velocity_y0, 
      double* velocity_x1, double* velocity_y1, double* cell_force_x, 
      double* cell_force_y, double* node_force_x, double* node_force_y, 
      double* cell_volumes, double* cell_mass, double* nodal_mass, double* nodal_volumes,
      double* nodal_soundspeed, double* limiter);

  // Calculates the timestep from the current state
  void set_timestep(
      const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
      const double* e, Mesh* mesh, double* reduce_array, const int first_step,
      const double* celldx, const double* celldy);

  // Prints some conservation values
  void print_conservation(
      const int nx, const int ny, double* rho, double* e, double* reduce_array, Mesh* mesh);

#ifdef __cplusplus
}
#endif

#endif

