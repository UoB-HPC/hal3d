#ifndef __HALEINTERFACEHDR
#define __HALEINTERFACEHDR

#pragma once

#include "../mesh.h"
#include "../shared.h"
#include "../shared_data.h"
#include "hale_data.h" // An important part of the interface

#ifdef __cplusplus
extern "C" {
#endif

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_3d(
    Mesh* mesh, HaleData* hale_data, const int ncells, const int nnodes,
    const int timestep, const int nsubcell_nodes, const int nsubcells_per_cell,
    const double visc_coeff1, const double visc_coeff2,
    double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_to_nodes, int* cells_offsets,
    int* nodes_to_cells, int* nodes_offsets, double* nodes_x0, double* nodes_y0,
    double* nodes_z0, double* nodes_x1, double* nodes_y1, double* nodes_z1,
    int* boundary_index, int* boundary_type, double* boundary_normal_x,
    double* boundary_normal_y, double* boundary_normal_z, double* cell_volume,
    double* energy0, double* energy1, double* density0, double* density1,
    double* pressure0, double* pressure1, double* velocity_x0,
    double* velocity_y0, double* velocity_z0, double* velocity_x1,
    double* velocity_y1, double* velocity_z1, double* subcell_force_x,
    double* subcell_force_y, double* subcell_force_z, double* cell_mass,
    double* nodal_mass, double* nodal_volumes, double* nodal_soundspeed,
    double* limiter, double* subcell_volume, double* subcell_mass,
    double* subcell_mass_flux, double* subcell_ie_mass,
    double* subcell_ie_mass_flux, double* subcell_momentum_flux_x,
    double* subcell_momentum_flux_y, double* subcell_momentum_flux_z,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, double* subcell_kinetic_energy,
    int* subcells_to_nodes, double* subcell_nodes_x, double* subcell_nodes_y,
    double* subcell_nodes_z, double* rezoned_nodes_x, double* rezoned_nodes_y,
    double* rezoned_nodes_z, int* nodes_to_faces_offsets, int* nodes_to_faces,
    int* faces_to_nodes, int* faces_to_nodes_offsets, int* faces_to_cells0,
    int* faces_to_cells1, int* cells_to_faces_offsets, int* cells_to_faces,
    int* subcells_to_subcells, int* subcells_to_subcells_offsets,
    int* subcells_to_faces, int* subcells_to_faces_offsets);

// Controls the timestep for the simulation
void set_timestep(const int ncells, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes);

#ifdef __cplusplus
}
#endif

#endif
