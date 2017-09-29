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
#define EPS 1.0e-14

#define CFL 0.3
#define VALIDATE_TOLERANCE 1.0e-5
#define ARCH_ROOT_PARAMS "../arch.params"
#define HALE_PARAMS "hale.params"
#define HALE_TESTS "hale.tests"
#define NNEIGHBOUR_NODES 2

#define NSUBCELL_NEIGHBOURS 4
#define NSUBSUBCELLS 2

// The number of faces of swept edge prism
#define NPRISM_FACES 5
#define NPRISM_NODES 6

// Describes the miniature tetrahedral subcells
#define NTET_FACES 4
#define NTET_NODES 4
#define NTET_NODES_PER_FACE 3

enum { XYZ, YZX, ZXY };

typedef struct {
  double x;
  double y;
  double z;
} vec_t;

typedef struct {
  double* energy0;
  double* energy1;
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

  int* subcells_to_subcells;
  int* subcell_face_offsets;

  double* subcell_ie_density0;
  double* subcell_ie_mass_flux;
  double* subcell_mass0;
  double* subcell_mass_flux;
  double* subcell_momentum_flux_x;
  double* subcell_momentum_flux_y;
  double* subcell_momentum_flux_z;
  double* subcell_centroids_x;
  double* subcell_centroids_y;
  double* subcell_centroids_z;
  double* corner_force_x;
  double* corner_force_y;
  double* corner_force_z;
  double* subcell_kinetic_energy;
  double* rezoned_nodes_x;
  double* rezoned_nodes_y;
  double* rezoned_nodes_z;

  // NOTE: The data here is only really intended to be used for testing purposes
  double* subcell_data_x;
  double* subcell_data_y;
  double* subcell_data_z;
  int* subcells_to_nodes;

  int nsubcell_edges;
  int nsubcells;
  int nsubcell_nodes;
  int nsubcells_per_cell;

  double visc_coeff1;
  double visc_coeff2;

  int perform_remap;
  int visit_dump;

} HaleData;

// Initialises the shared_data variables for two dimensional applications
size_t init_hale_data(HaleData* hale_data, UnstructuredMesh* umesh);

// This method sets up the subcell nodes and connectivity for a structured mesh
// viewed as an unstructured mesh. This is not intended to be used for
// production purposes, but instead should be used for debugging the code.
size_t init_subcell_data_structures(Mesh* mesh, HaleData* hale_data);

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(const int ncells, const double* cell_centroids_x,
                    const double* cell_centroids_y,
                    const double* cell_centroids_z, const double* density,
                    const double* nodes_x, const double* nodes_y,
                    const double* nodes_z, double* cell_mass,
                    double* subcell_mass, int* cells_to_faces_offsets,
                    int* cells_to_faces, int* faces_to_nodes_offsets,
                    int* faces_to_nodes, int* faces_to_subcells_offsets);

// Initialises the centroids for each cell
void init_cell_centroids(const int ncells, const int* cells_offsets,
                         const int* cells_to_nodes, const double* nodes_x0,
                         const double* nodes_y0, const double* nodes_z0,
                         double* cell_centroids_x, double* cell_centroids_y,
                         double* cell_centroids_z);

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const int* faces_to_cells0, const int* faces_to_cells1,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, int* cells_to_faces_offsets,
    int* cells_to_faces, int* subcells_to_subcells, int* subcell_face_offsets);

// Stores the rezoned grid specification, in case we aren't going to use a
// rezoning strategy and want to perform an Eulerian remap
void store_rezoned_mesh(const int nnodes, const double* nodes_x,
                        const double* nodes_y, const double* nodes_z,
                        double* rezoned_nodes_x, double* rezoned_nodes_y,
                        double* rezoned_nodes_z);

// Deallocates all of the hale specific data
void deallocate_hale_data(HaleData* hale_data);

// Writes out unstructured tetrahedral subcells to visit
void subcells_to_visit(const int nnodes, int ncells, const int step,
                       double* nodes_x, double* nodes_y, double* nodes_z,
                       const int* cells_to_nodes, const double* arr,
                       const int nodal, const int quads);

// Writes out unstructured triangles to visit
void write_unstructured_to_visit_3d(const int nnodes, int ncells,
                                    const int step, double* nodes_x0,
                                    double* nodes_y0, double* nodes_z0,
                                    const int* cells_to_nodes,
                                    const double* arr, const int nodal,
                                    const int quads);

#endif
