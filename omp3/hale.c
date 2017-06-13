#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "hale.h"
#include "../hale_interface.h"
#include "../../comms.h"
#include "../../params.h"

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_2d(
    Mesh* mesh, int tt, double* P, double* rho, double* rho_old, 
    double* u, double* v, double* rho_u, double* rho_v, 
    double* Qxx, double* Qyy, double* F_x, double* F_y, double* uF_x, 
    double* uF_y, double* vF_x, double* vF_y, double* reduce_array)
{

}

#if 0
// Calculate the pressure from GAMma law equation of state
void equation_of_state(
    const int nx, const int ny, double* P, const double* rho, const double* e)
{
  START_PROFILING(&compute_profile);

#pragma omp parallel for
  for(int ii = 0; ii < ny; ++ii) {
#pragma omp simd
    for(int jj = 0; jj < nx; ++jj) {
      // Only invoke simple GAMma law at the moment
      P[(ii*nx+jj)] = (GAM - 1.0)*rho[(ii*nx+jj)]*e[(ii*nx+jj)];
    }
  }

  STOP_PROFILING(&compute_profile, __func__);
}

// Calculates the timestep from the current state
void set_timestep(
    const int nx, const int ny, double* Qxx, double* Qyy, const double* rho, 
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
        celldx[jj]/sqrt(c_s*c_s + 2.0*Qxx[(ii*nx+jj)]/rho[(ii*nx+jj)]);
      const double thread_min_dt_y = 
        celldy[ii]/sqrt(c_s*c_s + 2.0*Qyy[(ii*nx+jj)]/rho[(ii*nx+jj)]);
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

