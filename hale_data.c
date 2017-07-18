#include "hale_data.h"
#include "../mesh.h"
#include "../params.h"
#include "../shared.h"
#include <assert.h>
#include <math.h>
#include <silo.h>
#include <stdlib.h>

// Initialises the shared_data variables for two dimensional applications
size_t initialise_hale_data_2d(HaleData* hale_data, UnstructuredMesh* umesh) {
  size_t allocated = allocate_data(&hale_data->pressure0, umesh->ncells);
  allocated += allocate_data(&hale_data->velocity_x0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y0, umesh->nnodes);
  allocated += allocate_data(&hale_data->energy1, umesh->ncells);
  allocated += allocate_data(&hale_data->density1, umesh->ncells);
  allocated += allocate_data(&hale_data->pressure1, umesh->ncells);
  allocated += allocate_data(&hale_data->velocity_x1, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y1, umesh->nnodes);
  allocated += allocate_data(&hale_data->node_force_x, umesh->nnodes);
  allocated += allocate_data(&hale_data->node_force_y, umesh->nnodes);
  allocated += allocate_data(&hale_data->node_force_x2, umesh->nnodes);
  allocated += allocate_data(&hale_data->node_force_y2, umesh->nnodes);
  allocated += allocate_data(&hale_data->cell_mass, umesh->ncells);
  allocated += allocate_data(&hale_data->nodal_mass, umesh->nnodes);
  allocated += allocate_data(&hale_data->nodal_volumes, umesh->nnodes);
  allocated += allocate_data(&hale_data->nodal_soundspeed, umesh->nnodes);
  allocated += allocate_data(&hale_data->limiter, umesh->nnodes);
  allocated += allocate_data(&hale_data->rezoned_nodes_x, umesh->nnodes);
  allocated += allocate_data(&hale_data->rezoned_nodes_y, umesh->nnodes);
  allocated += allocate_data(&hale_data->sub_cell_volume,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_force_x,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_force_y,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_energy,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_kinetic_energy,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_mass,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_velocity_x,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_velocity_y,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_grad_x,
                             umesh->ncells * umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->sub_cell_grad_y,
                             umesh->ncells * umesh->nnodes_by_cell);

  // In hale, the fundamental principle is that the mass at the cell and
  // sub-cell are conserved, so we can initialise them from the mesh
  // and then only the remapping step will ever adjust them
  initialise_cell_centroids(umesh->ncells, umesh->cells_offsets,
                            umesh->cells_to_nodes, umesh->nodes_x0,
                            umesh->nodes_y0, umesh->cell_centroids_x,
                            umesh->cell_centroids_y);

  initialise_mesh_mass(umesh->ncells, umesh->cells_offsets,
                       umesh->cell_centroids_x, umesh->cell_centroids_y,
                       umesh->cells_to_nodes, hale_data->density0,
                       umesh->nodes_x0, umesh->nodes_y0, hale_data->cell_mass,
                       umesh->sub_cell_volume, hale_data->sub_cell_mass);

  store_rezoned_mesh(umesh->nnodes, umesh->nodes_x0, umesh->nodes_y0,
                     hale_data->rezoned_nodes_x, hale_data->rezoned_nodes_y);

  return allocated;
}

// Deallocates all of the hale specific data
void deallocate_hale_data(HaleData* hale_data) {
  // TODO: Populate this correctly !
}

// Writes out unstructured triangles to visit
void write_unstructured_to_visit(const int nnodes, int ncells, const int step,
                                 double* nodes_x0, double* nodes_y0,
                                 const int* cells_to_nodes, const double* arr,
                                 const int nodal, const int quads) {
  // Only triangles
  double* coords[] = {(double*)nodes_x0, (double*)nodes_y0};
  int shapesize[] = {(quads ? 4 : 3)};
  int shapecounts[] = {ncells};
  int shapetype[] = {(quads ? DB_ZONETYPE_QUAD : DB_ZONETYPE_TRIANGLE)};
  int ndims = 2;
  int nshapes = 1;

  char filename[MAX_STR_LEN];
  sprintf(filename, "output%04d.silo", step);

  DBfile* dbfile =
      DBCreate(filename, DB_CLOBBER, DB_LOCAL, "simulation time step", DB_HDF5);

  DBPutZonelist2(dbfile, "zonelist", ncells, ndims, cells_to_nodes,
                 ncells * shapesize[0], 0, 0, 0, shapetype, shapesize,
                 shapecounts, nshapes, NULL);
  DBPutUcdmesh(dbfile, "mesh", ndims, NULL, coords, nnodes, ncells, "zonelist",
               NULL, DB_DOUBLE, NULL);
  DBPutUcdvar1(dbfile, "arr", "mesh", arr, (nodal ? nnodes : ncells), NULL, 0,
               DB_DOUBLE, (nodal ? DB_NODECENT : DB_ZONECENT), NULL);
  DBClose(dbfile);
}
