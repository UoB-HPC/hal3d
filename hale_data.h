#ifndef __HALEHDR
#define __HALEHDR

#pragma once

#include <stdlib.h> 
#include "../mesh.h"

#define CFL 0.3
#define VALIDATE_TOLERANCE     1.0e-5
#define ARCH_ROOT_PARAMS "../arch.params"
#define HALE_PARAMS "hale.params"
#define HALE_TESTS  "hale.tests"

typedef struct {
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
  double* node_force_x2;
  double* node_force_y2;
  double* node_visc_x;
  double* node_visc_y;
  double* cell_volumes; 
  double* cell_mass;
  double* nodal_mass;
  double* nodal_volumes;
  double* nodal_soundspeed;
  double* limiter;

  double visc_coeff1;
  double visc_coeff2;
} HaleData;

// Initialises the shared_data variables for two dimensional applications
size_t initialise_hale_data_2d(
    HaleData* hale_data, UnstructuredMesh* umesh);

void deallocate_hale_data_2d(
    HaleData* hale_data);

size_t initialise_unstructured_mesh(
    Mesh* mesh, UnstructuredMesh* unstructured_mesh);

// Reflect the node centered velocities on the boundary
void handle_unstructured_reflect(
    const int nnodes, const int* boundary_index, const int* boundary_type,
    const double* boundary_normal_x, const double* boundary_normal_y, 
    double* velocity_x, double* velocity_y);

// Fill boundary cells with interior values
void handle_unstructured_cell_boundary(
    const int ncells, const int* halo_cell, double* arr);

// Fill halo nodes with interior values
void handle_unstructured_node_boundary(
    const int nnodes, const int* halo_index, const int* halo_neighbour, double* arr);

// Reads an unstructured mesh from an input file
size_t read_unstructured_mesh(
    UnstructuredMesh* umesh, double** variables);

// Writes out unstructured triangles to visit
void write_unstructured_to_visit(
    const int nnodes, int ncells, const int step, double* nodes_x0, 
    double* nodes_y0, const int* cells_to_nodes, const double* arr, 
    const int nodal, const int quads);

#endif

