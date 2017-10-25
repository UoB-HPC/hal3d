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

#ifdef __cplusplus
}
#endif

#endif
