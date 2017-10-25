#include "hale_data.h"
#include "../mesh.h"
#include "../params.h"
#include "../shared.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <silo.h>
#include <stdlib.h>

// Initialises the shared_data variables for two dimensional applications
size_t init_hale_data(HaleData* hale_data, UnstructuredMesh* umesh) {
  const int nfaces_by_node = 3;
  hale_data->nnodes_by_subcell = 8;
  hale_data->nsubcells_by_cell = 8;
  hale_data->nsubcells = umesh->ncells * hale_data->nsubcells_by_cell;

  size_t allocated = allocate_data(&hale_data->pressure0, umesh->ncells);
  allocated += allocate_data(&hale_data->velocity_x0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_z0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_x1, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y1, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_z1, umesh->nnodes);
  allocated += allocate_data(&hale_data->energy1, umesh->ncells);
  allocated += allocate_data(&hale_data->kinetic_energy, umesh->ncells);
  allocated += allocate_data(&hale_data->density1, umesh->ncells);
  allocated += allocate_data(&hale_data->ie_flux, umesh->ncells);
  allocated += allocate_data(&hale_data->ke_flux, umesh->ncells);
  allocated += allocate_data(&hale_data->mass_flux, umesh->ncells);
  allocated += allocate_data(&hale_data->momentum_x_flux, umesh->ncells);
  allocated += allocate_data(&hale_data->momentum_y_flux, umesh->ncells);
  allocated += allocate_data(&hale_data->momentum_z_flux, umesh->ncells);
  allocated += allocate_data(&hale_data->pressure1, umesh->ncells);
  allocated += allocate_data(&hale_data->cell_mass, umesh->ncells);
  allocated += allocate_data(&hale_data->nodal_mass, umesh->nnodes);
  allocated += allocate_data(&hale_data->nodal_volumes, umesh->nnodes);
  allocated += allocate_data(&hale_data->nodal_soundspeed, umesh->nnodes);
  allocated += allocate_data(&hale_data->limiter, umesh->nnodes);
  allocated += allocate_data(&hale_data->rezoned_nodes_x, umesh->nnodes);
  allocated += allocate_data(&hale_data->rezoned_nodes_y, umesh->nnodes);
  allocated += allocate_data(&hale_data->rezoned_nodes_z, umesh->nnodes);
  allocated += allocate_data(&hale_data->cell_volume, umesh->ncells);
  allocated += allocate_data(&hale_data->subcell_force_x, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_force_y, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_force_z, hale_data->nsubcells);

  // In hale, the fundamental principle is that the mass at the cell and
  // sub-cell are conserved, so we can initialise them from the mesh
  // and then only the remapping step will ever adjust them
  init_cell_centroids(umesh->ncells, umesh->cells_offsets,
                      umesh->cells_to_nodes, umesh->nodes_x0, umesh->nodes_y0,
                      umesh->nodes_z0, umesh->cell_centroids_x,
                      umesh->cell_centroids_y, umesh->cell_centroids_z);

  // Initialises the cell mass, sub-cell mass and sub-cell volume
  init_mesh_mass(
      umesh->ncells, umesh->nnodes, hale_data->density0, umesh->nodes_x0,
      umesh->nodes_y0, umesh->nodes_z0, hale_data->nodal_mass,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->faces_cclockwise_cell, umesh->cells_offsets, umesh->cells_to_nodes,
      umesh->nodes_offsets, umesh->nodes_to_cells, hale_data->cell_volume,
      hale_data->nodal_volumes, hale_data->cell_mass);

  store_rezoned_mesh(umesh->nnodes, umesh->nodes_x0, umesh->nodes_y0,
                     umesh->nodes_z0, hale_data->rezoned_nodes_x,
                     hale_data->rezoned_nodes_y, hale_data->rezoned_nodes_z);

  return allocated;
}

// Deallocates all of the hale specific data
void deallocate_hale_data(HaleData* hale_data) {
  // TODO: Populate this correctly !
}

// Writes out unstructured mesh data to visit
void write_unstructured_to_visit_3d(const int nnodes, int ncells,
                                    const int step, double* nodes_x,
                                    double* nodes_y, double* nodes_z,
                                    const int* cells_to_nodes,
                                    const double* arr, const int nodal,
                                    const int quads) {

  double* coords[] = {(double*)nodes_x, (double*)nodes_y, (double*)nodes_z};

  int shapecounts[] = {ncells};
  int shapesize[] = {8};
  int shapetype[] = {DB_ZONETYPE_HEX};

  int ndims = 3;
  int nshapes = 1;

  char filename[MAX_STR_LEN];
  sprintf(filename, "output%04d.silo", step);

  DBfile* dbfile =
      DBCreate(filename, DB_CLOBBER, DB_LOCAL, "simulation time step", DB_HDF5);

  /* Write out connectivity information. */
  DBPutZonelist2(dbfile, "zonelist", ncells, ndims, cells_to_nodes,
                 ncells * shapesize[0], 0, 0, 0, shapetype, shapesize,
                 shapecounts, nshapes, NULL);

  /* Write an unstructured mesh. */
  DBPutUcdmesh(dbfile, "mesh", ndims, NULL, coords, nnodes, ncells, "zonelist",
               NULL, DB_DOUBLE, NULL);

  DBPutUcdvar1(dbfile, "arr", "mesh", arr, (nodal ? nnodes : ncells), NULL, 0,
               DB_DOUBLE, (nodal ? DB_NODECENT : DB_ZONECENT), NULL);

  DBClose(dbfile);
}
