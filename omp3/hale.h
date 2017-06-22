#include "../../mesh.h"

// Calculates the artificial viscous forces for momentum acceleration
void calculate_artificial_viscosity(
    const int ncells, const double c1, const double c2, const int* halo_cell, 
    const int* cells_to_nodes_off, const int* cells_to_nodes, 
    const double* nodes_x, const double* nodes_y, 
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* velocity_x, const double* velocity_y,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter,
    double* node_visc_x, double* node_visc_y);

