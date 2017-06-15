#ifndef __HALEHDR
#define __HALEHDR

#pragma once

#include <stdlib.h> 
#include "../mesh.h"

#define C_T 0.3
#define VALIDATE_TOLERANCE     1.0e-5
#define ARCH_ROOT_PARAMS "../arch.params"
#define HALE_PARAMS "hale.params"
#define HALE_TESTS  "hale.tests"
#define NNODES_PER_EDGE 2
#define NCELLS_PER_EDGE 2

typedef struct {
  // Hale-specific state
  double* rho_u;    // Momentum in the x direction
  double* rho_v;    // Momentum in the y direction

  double* F_x;      // Mass flux in the x direction
  double* F_y;      // Mass flux in the y direction

  double* uF_x;     // Momentum in the x direction flux in the x direction 
  double* uF_y;     // Momentum in the x direction flux in the y direction

  double* vF_x;     // Momentum in the y direction flux in the x direction
  double* vF_y;     // Momentum in the y direction flux in the y direction

  double* wF_x;     // Momentum in the z direction flux in the x direction
  double* wF_y;     // Momentum in the z direction flux in the y direction

  double* energy0;
  double* energy1;
  double* density0; 
  double* density1;
  double* pressure0;
  double* pressure1; 
  double* velocity_x0;
  double* velocity_y0;
  double* velocity_x1; 
  double* velocity_y1;
  double* cell_force_x;
  double* cell_force_y; 
  double* node_force_x;
  double* node_force_y;
  double* cell_volumes; 
  double* cell_mass;
  double* nodal_mass;

} HaleData;


typedef struct {

  // Handles unstructured mesh
  double* nodes_x;
  double* nodes_y;
  double* sub_cell_volume;
  int* nodes_cells;
  int* cells_nodes;
  int* nodes_cells_off;
  int* cells_nodes_off;

  int ncells;
  int nnodes;

  int* cell_centroids_x;
  int* cell_centroids_y;
  int* cells_to_nodes; 
  int* nodes_to_cells; 
  int* nodes_to_cells_off; 
  int* cells_to_nodes_off; 

  double* nodes_x0; 
  double* nodes_y0; 
  double* nodes_x1; 
  double* nodes_y1;

} UnstructuredMesh;

// Initialises the state variables for two dimensional applications
size_t initialise_hale_data_2d(
    const int local_nx, const int local_ny, HaleData* hale_data);
void deallocate_hale_data_2d(
    HaleData* hale_data);
size_t initialise_unstructured_mesh(
    Mesh* mesh, UnstructuredMesh* unstructured_mesh);

// Validates the results of the simulation
void validate(
    const int nx, const int ny, const char* params_filename, 
    const int rank, double* density, double* energy);

#endif

