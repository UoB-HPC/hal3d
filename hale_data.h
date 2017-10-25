#ifndef __HALEHDR
#define __HALEHDR

#pragma once

#include "../mesh.h"
#include "../umesh.h"
#include <stdlib.h>

// Controllable parameters for the application
#define GAM 1.4
#define C_Q 3.0
#define C_M (1.5 / C_T)
#define EPS 1.0e-15

#define CFL 0.25
#define VALIDATE_TOLERANCE 1.0e-5
#define ARCH_ROOT_PARAMS "../arch.params"
#define HALE_PARAMS "hale.params"
#define HALE_TESTS "hale.tests"
#define NNODES_BY_SUBCELL_FACE 4

enum { XYZ, YZX, ZXY };

typedef struct {
  double x;
  double y;
  double z;
} vec_t;

typedef struct {
  double* energy0;
  double* energy1;
  double* kinetic_energy;
  double* density0;
  double* density1;
  double* pressure0;
  double* pressure1;
  double* velocity_x0;
  double* velocity_y0;
  double* velocity_z0;
  double* velocity_x1;
  double* velocity_y1;
  double* velocity_z1;
  double* node_force_x;
  double* node_force_y;
  double* node_force_z;
  double* node_visc_x;
  double* node_visc_y;
  double* node_visc_z;
  double* cell_volume;
  double* subcell_volume;
  double* cell_mass;
  double* nodal_mass;
  double* nodal_volumes;
  double* nodal_soundspeed;
  double* limiter;

  double* mass_flux;
  double* ie_mass_flux;
  double* ke_mass_flux;
  double* momentum_x_flux;
  double* momentum_y_flux;
  double* momentum_z_flux;
  double* subcell_ie_mass;
  double* subcell_ie_mass_flux;
  double* subcell_ke_mass;
  double* subcell_ke_mass_flux;
  double* subcell_mass;
  double* subcell_mass_flux;
  double* subcell_momentum_x;
  double* subcell_momentum_y;
  double* subcell_momentum_z;
  double* subcell_momentum_flux_x;
  double* subcell_momentum_flux_y;
  double* subcell_momentum_flux_z;
  double* subcell_centroids_x;
  double* subcell_centroids_y;
  double* subcell_centroids_z;
  double* subcell_force_x;
  double* subcell_force_y;
  double* subcell_force_z;
  double* rezoned_nodes_x;
  double* rezoned_nodes_y;
  double* rezoned_nodes_z;

  int nsubcells;
  int nsubcell_nodes;
  int nsubcells_by_cell;
  int nnodes_by_subcell;

  double visc_coeff1;
  double visc_coeff2;

  int perform_remap;
  int visit_dump;

  int* subcells_to_nodes;
  int* subcells_to_subcells_offsets;
  int* subcells_to_subcells;
  int* subcells_to_faces;
  int* subcells_to_faces_offsets;

  // Only intended for testing purposes
  double* subcell_nodes_x;
  double* subcell_nodes_y;
  double* subcell_nodes_z;

} HaleData;

// Initialises the shared_data variables for two dimensional applications
size_t init_hale_data(HaleData* hale_data, UnstructuredMesh* umesh);

// NOTE: This is not intended to be a production device, rather used for
// debugging the code against a well tested description of the subcell mesh.
void init_subcell_data_structures(Mesh* mesh, HaleData* hale_data,
                                  UnstructuredMesh* umesh);

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(const int ncells, const int nnodes,
                    const int nnodes_by_subcell, const double* density,
                    const double* nodes_x, const double* nodes_y,
                    const double* nodes_z, double* subcell_mass,
                    double* nodal_mass, int* faces_to_nodes_offsets,
                    int* faces_to_nodes, int* faces_cclockwise_cell,
                    int* cells_offsets, int* cells_to_nodes,
                    int* subcells_to_faces_offsets, int* subcells_to_faces,
                    int* nodes_offsets, int* nodes_to_cells,
                    double* subcell_centroids_x, double* subcell_centroids_y,
                    double* subcell_centroids_z, double* subcell_volume,
                    double* cell_volume, double* nodal_volumes,
                    double* cell_mass);

// Initialises the centroids for each cell
void init_cell_centroids(const int ncells, const int* cells_offsets,
                         const int* cells_to_nodes, const double* nodes_x0,
                         const double* nodes_y0, const double* nodes_z0,
                         double* cell_centroids_x, double* cell_centroids_y,
                         double* cell_centroids_z);

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const int nsubcells, const int* faces_to_cells0,
    const int* faces_to_cells1, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const int* faces_cclockwise_cell,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    int* subcells_to_subcells, int* subcells_to_subcells_offsets,
    int* cells_offsets, int* nodes_to_faces_offsets, int* nodes_to_faces,
    int* cells_to_nodes, int* subcells_to_faces,
    int* subcells_to_faces_offsets);

void init_subcells_to_faces(
    const int ncells, const int nsubcells, const int* cells_offsets,
    const int* nodes_to_faces_offsets, const int* cells_to_nodes,
    const int* faces_to_cells0, const int* faces_to_cells1,
    const int* nodes_to_faces, const int* faces_to_nodes,
    const int* faces_to_nodes_offsets, const int* faces_cclockwise_cell,
    int* subcells_to_faces, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, int* subcells_to_faces_offsets);

// Stores the rezoned grid specification, in case we aren't going to use a
// rezoning strategy and want to perform an Eulerian remap
void store_rezoned_mesh(const int nnodes, const double* nodes_x,
                        const double* nodes_y, const double* nodes_z,
                        double* rezoned_nodes_x, double* rezoned_nodes_y,
                        double* rezoned_nodes_z);

// Deallocates all of the hale specific data
void deallocate_hale_data(HaleData* hale_data);

// Writes out unstructured triangles to visit
void write_unstructured_to_visit_3d(const int nnodes, int ncells,
                                    const int step, double* nodes_x0,
                                    double* nodes_y0, double* nodes_z0,
                                    const int* cells_to_nodes,
                                    const double* arr, const int nodal,
                                    const int quads);

#endif
