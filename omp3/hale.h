#include "../../mesh.h"
#include "../hale_data.h"

// Performs the Lagrangian step of the hydro solve
void lagrangian_phase(
    Mesh* mesh, const int ncells, const int nnodes, const double visc_coeff1,
    const double visc_coeff2, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_to_nodes,
    int* cells_offsets, int* nodes_to_cells, int* nodes_offsets,
    double* nodes_x0, double* nodes_y0, double* nodes_z0, double* nodes_x1,
    double* nodes_y1, double* nodes_z1, int* boundary_index, int* boundary_type,
    double* boundary_normal_x, double* boundary_normal_y,
    double* boundary_normal_z, double* energy0, double* energy1,
    double* density0, double* density1, double* pressure0, double* pressure1,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* velocity_x1, double* velocity_y1, double* velocity_z1,
    double* subcell_force_x, double* subcell_force_y, double* subcell_force_z,
    double* cell_mass, double* nodal_mass, double* nodal_volumes,
    double* nodal_soundspeed, double* limiter, int* nodes_to_faces_offsets,
    int* nodes_to_faces, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* faces_to_cells0, int* faces_to_cells1, int* cells_to_faces_offsets,
    int* cells_to_faces);

// Gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(
    const int ncells, double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_to_nodes, int* cells_offsets,
    double* nodes_x0, double* nodes_y0, double* nodes_z0, double* energy0,
    double* density0, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* cell_mass, double* subcell_volume,
    double* subcell_ie_density, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_integrals_x,
    double* subcell_integrals_y, double* subcell_integrals_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces);

// Calculates the artificial viscous forces for momentum acceleration
void calc_artificial_viscosity(
    const int ncells, const double visc_coeff1, const double visc_coeff2,
    const int* cells_offsets, const int* cells_to_nodes, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* velocity_x,
    const double* velocity_y, const double* velocity_z,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter, double* subcell_force_x,
    double* subcell_force_y, double* subcell_force_z,
    int* faces_to_nodes_offsets, int* faces_to_nodes,
    int* cells_to_faces_offsets, int* cells_to_faces);

// Uodates the velocity due to the pressure gradients
void update_velocity(const int nnodes, const double dt,
                     const double* node_force_x, const double* node_force_y,
                     const double* node_force_z, const double* nodal_mass,
                     double* velocity_x0, double* velocity_y0,
                     double* velocity_z0, double* velocity_x1,
                     double* velocity_y1, double* velocity_z1);

// Checks if the normal vector is pointing inward or outward
// n0 is just a point on the plane
int check_normal_orientation(const int n0, const double* nodes_x,
                             const double* nodes_y, const double* nodes_z,
                             const vec_t* cell_centroid, vec_t* normal);

// Calculates the surface normal of a vector pointing outwards
int calc_surface_normal(const int n0, const int n1, const int n2,
                        const double* nodes_x, const double* nodes_y,
                        const double* nodes_z, const vec_t* cell_centroid,
                        vec_t* normal);

// Calculate the normal vector from the provided nodes
void calc_unit_normal(const int n0, const int n1, const int n2,
                      const double* nodes_x, const double* nodes_y,
                      const double* nodes_z, vec_t* normal);

// Normalise a vector
void normalise(vec_t* a);

// Calculate the normal for a plane
void calc_normal(const int n0, const int n1, const int n2,
                 const double* nodes_x, const double* nodes_y,
                 const double* nodes_z, vec_t* normal);

// Calculates the face integral for the provided face, projected onto
// the two-dimensional basis
void calc_projections(const int nnodes_by_face, const int face_to_nodes_off,
                      const int* faces_to_nodes, const int face_orientation,
                      const double* alpha, const double* beta, pi_t* pi);

// Resolves the volume integrals in alpha-beta-gamma basis
void calc_face_integrals(const int nnodes_by_face, const int face_to_nodes_off,
                         const int basis, const int face_orientation,
                         const double omega, const int* faces_to_nodes,
                         const double* nodes_alpha, const double* nodes_beta,
                         vec_t normal, vec_t* T, double* vol);

// Calculate the centroid
void calc_centroid(const int nnodes, const double* nodes_x,
                   const double* nodes_y, const double* nodes_z,
                   const int* indirection, const int offset, vec_t* centroid);

// Controls the timestep for the simulation
void set_timestep(const int ncells, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes);

// Calculates the inverse of a 3x3 matrix, out-of-place
void calc_3x3_inverse(vec_t (*a)[3], vec_t (*inv)[3]);

// Constructs the prism for swept region of a subcell face external to a cell
void construct_external_swept_region(
    const vec_t* nodes, const vec_t* rz_nodes, const vec_t* half_edge_l,
    const vec_t* half_edge_r, const vec_t* rz_half_edge_l,
    const vec_t* rz_half_edge_r, const vec_t* face_c, const vec_t* rz_face_c,
    vec_t* prism_centroid, double* prism_nodes_x, double* prism_nodes_y,
    double* prism_nodes_z);

// Constructs the prism for swept region of a subcell face internal to a cell
void construct_internal_swept_region(
    const int face_rorientation, const vec_t* half_edge_l,
    const vec_t* half_edge_r, const vec_t* rz_half_edge_l,
    const vec_t* rz_half_edge_r, const vec_t* face_c, const vec_t* face2_c,
    const vec_t* rz_face_c, const vec_t* rz_face2_c, const vec_t* cell_centroid,
    const vec_t* rz_cell_centroid, vec_t* prism_centroid, double* prism_nodes_x,
    double* prism_nodes_y, double* prism_nodes_z);

// Calculates the weighted volume integrals for a provided cell along x-y-z
void calc_weighted_volume_integrals(
    const int cell_to_faces_off, const int nfaces_by_cell,
    const int* cells_to_faces, const int* faces_to_nodes,
    const int* faces_to_nodes_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const vec_t* cell_centroid,
    vec_t* T, double* vol);

// Calculate the inverse coefficient matrix for a subcell, in order to determine
// the gradients of the subcell quantities using least squares.
void calc_inverse_coefficient_matrix(
    const int subcell_index, const int* subcells_to_faces_offsets,
    const int* subcells_to_subcells, const double* subcell_integrals_x,
    const double* subcell_integrals_y, const double* subcell_integrals_z,
    const double* subcell_centroids_x, const double* subcell_centroids_y,
    const double* subcell_centroids_z, const double* subcell_volume,
    int* nsubcells_by_subcell, int* subcell_to_subcells_off, vec_t (*inv)[3]);

// Calculate the gradient for the
void calc_gradient(const int subcell_index, const int nsubcells_by_subcell,
                   const int subcell_to_subcells_off,
                   const int* subcells_to_subcells, const double* phi,
                   const double* subcell_integrals_x,
                   const double* subcell_integrals_y,
                   const double* subcell_integrals_z,
                   const double* subcell_volume, const vec_t (*inv)[3],
                   vec_t* gradient);

// Calculates the subcells of all centroids
void calc_subcell_centroids(
    const int ncells, const int* cells_offsets, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const int* cells_to_nodes, const int* subcells_to_faces_offsets,
    const int* subcells_to_faces, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const double* nodes_x0, const double* nodes_y0,
    const double* nodes_z0, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z);
