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

// Controls the timestep for the simulation
void set_timestep(const int ncells, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes);

// Calculates the artificial viscous forces for momentum acceleration
void calc_artificial_viscosity(
    const int ncells, const double visc_coeff1, const double visc_coeff2,
    const int* cells_offsets, const int* cells_to_nodes, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* velocity_x,
    const double* velocity_y, const double* velocity_z,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter, double* corner_force_x,
    double* corner_force_y, double* corner_force_z, int* faces_to_nodes_offsets,
    int* faces_to_nodes, int* cells_to_faces_offsets, int* cells_to_faces);

// Gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(
    const int ncells, const int nnodes, const double* nodal_volumes,
    const double* nodal_mass, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_offsets,
    double* nodes_x0, const double* nodes_y0, const double* nodes_z0,
    double* energy0, double* density0, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* cell_mass, double* subcell_volume,
    double* subcell_ie_density0, double* subcell_mass0,
    double* subcell_ie_density1, double* subcell_mass1,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* cell_volume, int* subcell_face_offsets, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes);

void gather_subcell_momentum(
    const int ncells, const int nnodes, const double* nodal_volumes,
    const double* nodal_mass, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_offsets,
    const double* nodes_x0, const double* nodes_y0, const double* nodes_z0,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* subcell_volume, double* subcell_velocity_x,
    double* subcell_velocity_y, double* subcell_velocity_z,
    int* subcell_face_offsets, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes);

// Gathers all of the subcell quantities on the mesh
void gather_subcell_energy(
    const int ncells, double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_offsets, const double* nodes_x0,
    const double* nodes_y0, const double* nodes_z0, double* energy0,
    double* density0, double* cell_mass, double* subcell_volume,
    double* subcell_ie_density, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    int* subcell_face_offsets, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* faces_to_cells0, int* faces_to_cells1, int* cells_to_faces_offsets,
    int* cells_to_faces, int* cells_to_nodes);

// Performs a remap and some scattering of the subcell values
void remap_phase(const int ncells, const int nnodes, double* cell_centroids_x,
                 double* cell_centroids_y, double* cell_centroids_z,
                 int* cells_to_nodes, int* cells_offsets, double* nodes_x0,
                 double* nodes_y0, double* nodes_z0, double* cell_volume,
                 double* energy0, double* energy1, double* density0,
                 double* velocity_x0, double* velocity_y0, double* velocity_z0,
                 double* cell_mass, double* nodal_mass, double* subcell_volume,
                 double* subcell_ie_density0, double* subcell_mass0,
                 double* subcell_ie_density1, double* subcell_mass1,
                 double* subcell_momentum_x, double* subcell_momentum_y,
                 double* subcell_momentum_z, double* subcell_centroids_x,
                 double* subcell_centroids_y, double* subcell_centroids_z,
                 double* rezoned_nodes_x, double* rezoned_nodes_y,
                 double* rezoned_nodes_z, int* nodes_to_faces_offsets,
                 int* nodes_to_faces, int* faces_to_nodes,
                 int* faces_to_nodes_offsets, int* faces_to_cells0,
                 int* faces_to_cells1, int* cells_to_faces_offsets,
                 int* cells_to_faces, int* subcell_face_offsets,
                 int* subcells_to_subcells);

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

// Resolves the volume center_of_mass in alpha-beta-gamma basis
void calc_face_integrals(const int nnodes_by_face, const int face_to_nodes_off,
                         const int basis, const int face_clockwise,
                         const double omega, const int* faces_to_nodes,
                         const double* nodes_alpha, const double* nodes_beta,
                         vec_t normal, double* vol);

// Calculates the weighted volume center_of_mass for a provided cell along
// x-y-z
void calc_volume(const int cell_to_faces_off, const int nfaces_by_cell,
                 const int* cells_to_faces, const int* faces_to_nodes,
                 const int* faces_to_nodes_offsets, const double* nodes_x,
                 const double* nodes_y, const double* nodes_z,
                 const vec_t* cell_centroid, double* vol);

// Stores the rezoned mesh specification as the original mesh. Until we
// determine a reasonable rezoning algorithm, this makes us Eulerian
void store_rezoned_mesh(const int nnodes, const double* nodes_x,
                        const double* nodes_y, const double* nodes_z,
                        double* rezoned_nodes_x, double* rezoned_nodes_y,
                        double* rezoned_nodes_z);

// Calculate the centroid
void calc_centroid(const int nnodes, const double* nodes_x,
                   const double* nodes_y, const double* nodes_z,
                   const int* indirection, const int offset, vec_t* centroid);

// Calculate the inverse coefficient matrix for a subcell, in order to
// determine the gradients of the subcell quantities using least squares.
void calc_inverse_coefficient_matrix(
    const int subcell_index, const int* subcells_to_subcells,
    const double* subcell_centroids_x, const double* subcell_centroids_y,
    const double* subcell_centroids_z, const double* subcell_volume,
    const int nsubcells_by_subcell, const int subcell_to_subcells_off,
    vec_t (*inv)[3]);

// Calculates the inverse of a 3x3 matrix, out-of-place
void calc_3x3_inverse(vec_t (*a)[3], vec_t (*inv)[3]);

// Calculate the gradient for the
void calc_gradient(const int subcell_index, const int nsubcells_by_subcell,
                   const int subcell_to_subcells_off,
                   const int* subcells_to_subcells, const double* phi,
                   const double* subcell_centroids_x,
                   const double* subcell_centroids_y,
                   const double* subcell_centroids_z, const vec_t (*inv)[3],
                   vec_t* gradient);

// Calculates the limiter for the provided gradient
double apply_limiter(const int nnodes_by_cell, const int cell_to_nodes_off,
                     const int* cell_to_nodes, vec_t* grad,
                     const vec_t* cell_centroid, const double* nodes_x0,
                     const double* nodes_y0, const double* nodes_z0,
                     const double dphi, const double gmax, const double gmin);

// Calculates the cell volume, subcell volume and the subcell centroids
void calc_volumes_centroids(
    const int ncells, const int* cells_to_faces_offsets,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const int* cells_to_faces,
    const int* faces_to_nodes, const int* faces_to_nodes_offsets,
    const int* subcell_face_offsets, const double* nodes_x0,
    const double* nodes_y0, const double* nodes_z0, double* cell_volume,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, double* subcell_volume);

void apply_mesh_rezoning(const int nnodes, const double* rezoned_nodes_x,
                         const double* rezoned_nodes_y,
                         const double* rezoned_nodes_z, double* nodes_x0,
                         double* nodes_y0, double* nodes_z0);
