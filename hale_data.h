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

#define CFL 0.3
#define VALIDATE_TOLERANCE 1.0e-5
#define ARCH_ROOT_PARAMS "../arch.params"
#define HALE_PARAMS "hale.params"
#define HALE_TESTS "hale.tests"
#define NNEIGHBOUR_NODES 2

#define NSUBCELL_NEIGHBOURS 4

// The number of faces of swept edge prism
#define NPRISM_FACES 5

// Describes the miniature tetrahedral subcells
#define NTET_FACES 4
#define NTET_NODES 4
#define NTET_NODES_PER_FACE 3

enum { XYZ, YZX, ZXY };

typedef struct {
  double one;
  double alpha;
  double alpha2;
  double beta;
  double beta2;
  double alpha_beta;
} pi_t;

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
  double* cell_volumes;
  double* cell_mass;
  double* nodal_mass;
  double* nodal_volumes;
  double* nodal_soundspeed;
  double* limiter;

  int* subcells_to_subcells;
  int* subcell_face_offsets;
  int* subcell_to_neighbour_face;

  // TODO: These are large arrays, which definitely aren't used all at the same
  // time and so there are some potential capacity reclaiming optimisations
  double* subcell_ie_density;
  double* subcell_mass;
  double* subcell_volume;
  double* subcell_velocity_x;
  double* subcell_velocity_y;
  double* subcell_velocity_z;
  double* subcell_centroids_x;
  double* subcell_centroids_y;
  double* subcell_centroids_z;
  double* corner_force_x;
  double* corner_force_y;
  double* corner_force_z;
  double* subcell_kinetic_energy;
  double* subcell_integrals_x;
  double* subcell_integrals_y;
  double* subcell_integrals_z;
  double* rezoned_nodes_x;
  double* rezoned_nodes_y;
  double* rezoned_nodes_z;

  int nsubcell_edges;
  int nsubcells;

  double visc_coeff1;
  double visc_coeff2;
} HaleData;

// Initialises the shared_data variables for two dimensional applications
size_t init_hale_data(HaleData* hale_data, UnstructuredMesh* umesh);

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(const int ncells, const int* cells_offsets,
                    const double* cell_centroids_x,
                    const double* cell_centroids_y,
                    const double* cell_centroids_z, const int* cells_to_nodes,
                    const double* density, const double* nodes_x,
                    const double* nodes_y, const double* nodes_z,
                    double* cell_mass, double* subcell_mass,
                    int* cells_to_faces_offsets, int* cells_to_faces,
                    int* faces_to_nodes_offsets, int* faces_to_nodes,
                    int* subcell_face_offsets);

// Initialises the centroids for each cell
void init_cell_centroids(const int ncells, const int* cells_offsets,
                         const int* cells_to_nodes, const double* nodes_x0,
                         const double* nodes_y0, const double* nodes_z0,
                         double* cell_centroids_x, double* cell_centroids_y,
                         double* cell_centroids_z);

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const int* faces_to_cells0,
    const int* faces_to_cells1, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    int* cells_to_faces_offsets, int* cells_to_faces, int* subcells_to_subcells,
    int* subcell_face_offsets, int* subcell_to_neighbour_face);

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
