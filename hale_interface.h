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
  void solve_hydro_2d(
      Mesh* mesh, int tt, double* P, double* rho, double* rho_old, 
      double* e, double* u, double* v, double* rho_u, double* rho_v, 
      double* Qxx, double* Qyy, double* F_x, double* F_y, double* uF_x, 
      double* uF_y, double* vF_x, double* vF_y, double* reduce_array);

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

