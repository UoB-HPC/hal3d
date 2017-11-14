#include "../hale_data.h"

// Performs the predictor step of the Lagrangian phase
void predictor(Mesh* mesh, UnstructuredMesh* umesh, HaleData* hale_data);

// Performs the corrector step of the Lagrangian phase
void corrector(Mesh* mesh, UnstructuredMesh* umesh, HaleData* hale_data);

// A simple ideal gas equation of state
void equation_of_state(const int ncells, const double* energy,
                       const double* density, double* pressure);

// Calculates the nodal volume and sound speed
void calc_nodal_vol_and_c(const int nnodes, const int* nodes_to_faces_offsets,
                          const int* nodes_to_faces,
                          const int* faces_to_nodes_offsets,
                          const int* faces_to_nodes, const int* faces_to_cells0,
                          const int* faces_to_cells1, const double* nodes_x,
                          const double* nodes_y, const double* nodes_z,
                          const double* cell_centroids_x,
                          const double* cell_centroids_y,
                          const double* cell_centroids_z, const double* energy,
                          double* nodal_volumes, double* nodal_soundspeed);

// Sets all of the subcell forces to 0
void zero_subcell_forces(const int ncells, const int* cells_offsets,
                         double* subcell_force_x, double* subcell_force_y,
                         double* subcell_force_z);

void calc_subcell_force_from_pressure(
    const int ncells, const int* cells_to_faces_offsets,
    const int* cells_offsets, const int* cells_to_faces,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const int* cells_to_nodes, const int* face_cclockwise_cell,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    const double* pressure, double* subcell_force_x, double* subcell_force_y,
    double* subcell_force_z);

// Scale the soundspeed by the inverse of the nodal volume
void scale_soundspeed(const int nnodes, const double* nodal_volumes,
                      double* nodal_soundspeed);

// Calculate the time centered evolved velocities, by calculating the predicted
// values at the new timestep and averaging with current velocity
void calc_new_velocity(const int nnodes, const double dt,
                       const int* nodes_offsets, const int* nodes_to_cells,
                       const int* cells_offsets, const int* cells_to_nodes,
                       const double* subcell_force_x,
                       const double* subcell_force_y,
                       const double* subcell_force_z, const double* nodal_mass,
                       const double* velocity_x0, const double* velocity_y0,
                       const double* velocity_z0, double* velocity_x1,
                       double* velocity_y1, double* velocity_z1);

// Moves the nodes to the next time level
void move_nodes(const int nnodes, const double dt, const double* nodes_x0,
                const double* nodes_y0, const double* nodes_z0,
                const double* velocity_x1, const double* velocity_y1,
                const double* velocity_z1, double* nodes_x1, double* nodes_y1,
                double* nodes_z1);

// calculates a new density from the pressure gradients
void calc_predicted_density(const int ncells, const int* cells_to_faces_offsets,
                            const int* cells_to_faces,
                            const int* faces_to_nodes_offsets,
                            const int* faces_to_nodes, const double* nodes_x1,
                            const double* nodes_y1, const double* nodes_z1,
                            const double* cell_centroids_x,
                            const double* cell_centroids_y,
                            const double* cell_centroids_z,
                            const double* cell_mass, double* density1);

// Time centers the pressure
void time_center_pressure(const int ncells, const double* energy1,
                          const double* density1, const double* pressure0,
                          double* pressure1);

// Time centers the nodal positions
void time_center_nodes(const int nnodes, const double* nodes_x0,
                       const double* nodes_y0, const double* nodes_z0,
                       double* nodes_x1, double* nodes_y1, double* nodes_z1);

// Updates and time center velocity in the corrector step
void update_and_time_center_velocity(
    const int nnodes, const double dt, const int* nodes_offsets,
    const int* nodes_to_cells, const int* cells_offsets,
    const int* cells_to_nodes, const double* nodal_mass,
    const double* subcell_force_x, const double* subcell_force_y,
    const double* subcell_force_z, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* velocity_x1, double* velocity_y1,
    double* velocity_z1);

// Advances the nodes using the corrected velocity
void advance_nodes_corrected(const int nnodes, const double dt,
                             const double* velocity_x0,
                             const double* velocity_y0,
                             const double* velocity_z0, double* nodes_x0,
                             double* nodes_y0, double* nodes_z0);

// Calculate the new energy base on subcell forces
void calc_predicted_energy(const int ncells, const double dt,
                           const int* cells_offsets, const int* cells_to_nodes,
                           const double* velocity_x1, const double* velocity_y1,
                           const double* velocity_z1,
                           const double* subcell_force_x,
                           const double* subcell_force_y,
                           const double* subcell_force_z, const double* energy0,
                           const double* cell_mass, double* energy1);

// Calculates the energy from the correct subcell pressures and velocity
void calc_corrected_energy(const int ncells, const double dt,
                           const int* cells_offsets, const int* cells_to_nodes,
                           const double* velocity_x0, const double* velocity_y0,
                           const double* velocity_z0,
                           const double* subcell_force_x,
                           const double* subcell_force_y,
                           const double* subcell_force_z,
                           const double* cell_mass, double* energy0);

// Calculates the density from the corrected volume
void calc_corrected_density(
    const int ncells, const int* cells_to_faces_offsets,
    const int* cells_to_faces, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const double* cell_mass, double* cell_volume, double* density);

// Calculates the artificial viscous forces for momentum acceleration
void calc_artificial_viscosity(
    const int ncells, const double visc_coeff1, const double visc_coeff2,
    const int* cells_offsets, const int* cells_to_nodes,
    const int* face_cclockwise_cell, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* velocity_x,
    const double* velocity_y, const double* velocity_z,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter, double* subcell_force_x,
    double* subcell_force_y, double* subcell_force_z,
    int* faces_to_nodes_offsets, int* faces_to_nodes,
    int* cells_to_faces_offsets, int* cells_to_faces);

// Calculates the volume in a cell by tetrahedral decomposition
double calc_cell_volume(const int cc, const int nfaces_by_cell,
                        const int cell_to_faces_off, const int* cells_to_faces,
                        const int* faces_to_nodes_offsets,
                        const int* faces_to_nodes, const double* nodes_x,
                        const double* nodes_y, const double* nodes_z,
                        const double* cell_centroids_x,
                        const double* cell_centroids_y,
                        const double* cell_centroids_z);

// Calculates the volume of a subsubcell
double calc_subsubcell_volume(const int cc, const int next_node,
                              const int current_node, vec_t face_c,
                              const double* nodes_x, const double* nodes_y,
                              const double* nodes_z,
                              const double* cell_centroids_x,
                              const double* cell_centroids_y,
                              const double* cell_centroids_z);
