#include "../../mesh.h"

// Calculates the artificial viscous forces for momentum acceleration
void calculate_artificial_viscosity(
    const int ncells, const double c1, const double c2, const int* halo_cell, 
    const int* cells_to_nodes_off, const int* cells_to_nodes, 
    const double* nodes_x0, const double* nodes_y0, 
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* velocity_x0, const double* velocity_y0,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter,
    double* node_force_x, double* node_force_y,
    double* cell_force_x, double* cell_force_y);

