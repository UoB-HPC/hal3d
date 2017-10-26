#include "../../mesh.h"
#include "../hale_data.h"

// Performs the Lagrangian step of the hydro solve
void lagrangian_phase(Mesh* mesh, UnstructuredMesh* umesh, HaleData* hale_data);

// Controls the timestep for the simulation
void set_timestep(const int ncells, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes);

// gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(UnstructuredMesh* umesh, HaleData* hale_data,
                               vec_t* initial_momentum, double* initial_mass,
                               double* initial_ie_mass,
                               double* initial_ke_mass);

// Performs a remap and some scattering of the subcell values
void advection_phase(UnstructuredMesh* umesh, HaleData* hale_data);

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

// Resolves the volume dist in alpha-beta-gamma basis
void calc_face_integrals(const int nnodes_by_face, const int face_to_nodes_off,
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
double apply_cell_limiter(const int nnodes_by_cell, const int cell_to_nodes_off,
                          const int* cell_to_nodes, vec_t* grad,
                          const vec_t* cell_centroid, const double* nodes_x0,
                          const double* nodes_y0, const double* nodes_z0,
                          const double dphi, const double gmax,
                          const double gmin);

// Calculates the cell volume, subcell volume and the subcell centroids
void calc_volumes_centroids(
    const int ncells, const int nnodes, const int nnodes_by_subcell,
    const int* cells_offsets, const int* cells_to_nodes,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes, const int* faces_to_nodes_offsets,
    const int* faces_cclockwise_cell, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* subcell_volume, double* cell_volume, double* nodal_volumes,
    int* nodes_offsets, int* nodes_to_cells);

void apply_mesh_rezoning(const int nnodes, const double* rezoned_nodes_x,
                         const double* rezoned_nodes_y,
                         const double* rezoned_nodes_z, double* nodes_x0,
                         double* nodes_y0, double* nodes_z0);

// Contributes a face to the volume of some cell
void contribute_face_volume(const int nnodes_by_face, const int* faces_to_nodes,
                            const double* nodes_x, const double* nodes_y,
                            const double* nodes_z, const vec_t* cell_centroid,
                            double* vol);

// Calculates the local limiter for a node
double calc_cell_limiter(const double rho, const double gmax, const double gmin,
                         vec_t* grad, const double node_x, const double node_y,
                         const double node_z, const vec_t* cell_c);

// Perform the scatter step of the ALE remapping algorithm
void scatter_phase(UnstructuredMesh* umesh, HaleData* hale_data,
                   vec_t* initial_momentum, double initial_mass,
                   double initial_ie_mass, double initial_ke_mass);

// The construction of the swept edge prisms can result in tangled or coplanar
// faces between the original and rezoned mesh. This must be recognised and
// handled correctly in order to stop the calculations breaking down.
int test_prism_overlap(const int nnodes_by_face, const int* faces_to_nodes,
                       const double* nodes_x, const double* nodes_y,
                       const double* nodes_z);

// Performs an Eulerian rezone of the mesh
void eulerian_rezone(UnstructuredMesh* umesh, HaleData* hale_data);

// Performs a conservative repair of the mesh
void repair_phase(UnstructuredMesh* umesh, HaleData* hale_data);

// Advects mass and energy through the subcell faces using swept edge approx
void perform_advection(
    const int ncells, const int* cells_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const double* rezoned_nodes_x,
    const double* rezoned_nodes_y, const double* rezoned_nodes_z,
    const int* cells_to_nodes, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const int* faces_cclockwise_cell,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* subcells_to_subcells_offsets, const int* subcells_to_subcells,
    const double* subcell_centroids_x, const double* subcell_centroids_y,
    const double* subcell_centroids_z, const int* faces_to_cells0,
    const int* faces_to_cells1, double* subcell_volume,
    double* subcell_momentum_flux_x, double* subcell_momentum_flux_y,
    double* subcell_momentum_flux_z, const double* subcell_momentum_x,
    const double* subcell_momentum_y, const double* subcell_momentum_z,
    const double* subcell_mass, double* subcell_mass_flux,
    const double* subcell_ie_mass, double* subcell_ie_mass_flux,
    const double* subcell_ke_mass, double* subcell_ke_mass_flux);

// Contributes the local mass, energy and momentum flux for a given subcell face
void flux_mass_energy_momentum(
    const int cc, const int neighbour_cc, const int ff, const int subcell_index,
    vec_t* subcell_c, vec_t* cell_c, const double* se_nodes_x,
    const double* se_nodes_y, const double* se_nodes_z,
    const double* subcell_mass, double* subcell_mass_flux,
    const double* subcell_ie_mass, double* subcell_ie_mass_flux,
    const double* subcell_ke_mass, double* subcell_ke_mass_flux,
    const double* subcell_volume, const double* subcell_momentum_x,
    const double* subcell_momentum_y, const double* subcell_momentum_z,
    double* subcell_momentum_flux_x, double* subcell_momentum_flux_y,
    double* subcell_momentum_flux_z, const int* swept_edge_faces_to_nodes,
    const double* subcell_centroids_x, const double* subcell_centroids_y,
    const double* subcell_centroids_z, const int* swept_edge_to_faces,
    const int* swept_edge_faces_to_nodes_offsets,
    const int* subcells_to_subcells_offsets, const int* subcells_to_subcells,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const int* cells_offsets, const int* cells_to_nodes,
    const int* faces_cclockwise_cell, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const int internal);

// Controls the timestep for the simulation
void set_timestep(const int ncells, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes);

// Limits all of the gradients during flux determination
void limit_mass_gradients(
    vec_t nodes, vec_t* sweep_subcell_c, const double sweep_subcell_density,
    const double sweep_subcell_ie_density,
    const double sweep_subcell_ke_density, const double subcell_vx,
    const double subcell_vy, const double subcell_vz, const double gmax_m,
    const double gmin_m, const double gmax_ie, const double gmin_ie,
    const double gmax_ke, const double gmin_ke, const double gmax_vx,
    const double gmin_vx, const double gmax_vy, const double gmin_vy,
    const double gmax_vz, const double gmin_vz, vec_t* grad_m, vec_t* grad_ie,
    vec_t* grad_ke, vec_t* grad_vx, vec_t* grad_vy, vec_t* grad_vz,
    double* m_limiter, double* ie_limiter, double* ke_limiter,
    double* vx_limiter, double* vy_limiter, double* vz_limiter);
