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
void solve_unstructured_hydro_3d(Mesh* mesh, HaleData* hale_data,
                                 UnstructuredMesh* umesh, const int timestep);

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
