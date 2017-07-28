#include "../../mesh.h"
#include "../hale_data.h"

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

// Calculate the normal vector from the provided nodes
void calc_unit_normal(const int n0, const int n1, const int n2,
                      const double* alpha, const double* beta,
                      const double* gamma, vec_t cell_centroid, vec_t* normal);

// Calculates the face integral for the provided face, projected onto
// the two-dimensional basis
void calc_face_integral(const double nnodes_by_face,
                        const int face_to_nodes_off, const int* faces_to_nodes,
                        const double* alpha, const double* beta,
                        const double* gamma, pi_t* pi);

// Calculates the weighted volume integrals for a provided cell along x-y-z
void calc_weighted_volume_integrals(
    const int cell_to_faces_off, const int nfaces_by_cell,
    const int* cells_to_faces, const int* faces_to_nodes,
    const int* faces_to_nodes_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const vec_t cell_centroid,
    vec_t* T, double* vol);

// Calculates the inverse of a 3x3 matrix, out-of-place
void calc_3x3_inverse(vec_t (*a)[3], vec_t (*inv)[3]);

// Resolves the volume integrals in alpha-beta-gamma basis
void resolve_integrals(const int nnodes_by_face, const int face_to_nodes_off,
                       const int orientation, const int n0,
                       const int* faces_to_nodes, const double* nodes_x,
                       const double* nodes_y, const double* nodes_z,
                       vec_t normal, vec_t* T, double* vol);
