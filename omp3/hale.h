#include "../../mesh.h"

// Calculates the artificial viscous forces for momentum acceleration
void calc_artificial_viscosity(
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
    int* faces_to_cells1, int* cells_to_faces_offsets, int* cells_to_faces);

// Uodates the velocity due to the pressure gradients
void update_velocity(const int nnodes, const double dt,
                     const double* node_force_x, const double* node_force_y,
                     const double* node_force_z, const double* nodal_mass,
                     double* velocity_x0, double* velocity_y0,
                     double* velocity_z0, double* velocity_x1,
                     double* velocity_y1, double* velocity_z1);
