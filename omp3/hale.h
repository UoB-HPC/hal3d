#include "../../mesh.h"

// Calculates the artificial viscous forces for momentum acceleration
void calculate_artificial_viscosity(
    const int ncells, const double visc_coeff1, const double visc_coeff2, 
    const int* cells_to_nodes_off, const int* cells_to_nodes, 
    const double* nodes_x, const double* nodes_y, 
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* velocity_x, const double* velocity_y,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter,
    double* node_force_x, double* node_force_y);

// Uodates the velocity due to the pressure gradients
void update_velocity(
    const int nnodes, const double dt, const double* node_force_x, 
    const double* node_force_y, const double* nodal_mass, double* velocity_x0, 
    double* velocity_y0, double* velocity_x1, double* velocity_y1);

