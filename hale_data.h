#ifndef __HALEHDR
#define __HALEHDR

#pragma once

#define C_T 0.3
#define VALIDATE_TOLERANCE     1.0e-5
#define ARCH_ROOT_PARAMS "../arch.params"
#define HALE_PARAMS "hale.params"
#define HALE_TESTS  "hale.tests"

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
} HaleData;

typedef struct {

  // Handles unstructured mesh
  double* vertices_x;
  double* vertices_y;
  double* cell_centroids_x;
  double* cell_centroids_y;
  double* volume;
  int* cells_vertices;
  int* edge_vertex0;
  int* edge_vertex1;
  int* cells_edges;
  int* edges_cells;

  int nedges;

} UnstructuredMesh;

// Initialises the state variables for two dimensional applications
void initialise_hale_data_2d(
    const int local_nx, const int local_ny, HaleData* hale_data);
void deallocate_hale_data_2d(
    HaleData* hale_data);

// Validates the results of the simulation
void validate(
    const int nx, const int ny, const char* params_filename, 
    const int rank, double* density, double* energy);

