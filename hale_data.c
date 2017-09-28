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
  const int nfaces_by_cell = 6;
  const int nnodes_by_face = 4;
  const int nnodes_by_cell = 8;
  hale_data->nsubcells = umesh->ncells * nfaces_by_cell * nnodes_by_face;
  hale_data->nsubcell_edges = 4;

  size_t allocated = allocate_data(&hale_data->pressure0, umesh->ncells);
  allocated += allocate_data(&hale_data->velocity_x0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_z0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_x1, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y1, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_z1, umesh->nnodes);
  allocated += allocate_data(&hale_data->energy1, umesh->ncells);
  allocated += allocate_data(&hale_data->density1, umesh->ncells);
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

  // TODO: This constant is the number of subcells that might neighbour a
  // subcell, which is the number of subcell faces
  allocated += allocate_int_data(&hale_data->subcells_to_subcells,
                                 hale_data->nsubcells * 4);
  allocated += allocate_int_data(&hale_data->subcell_face_offsets,
                                 umesh->ncells * nfaces_by_cell + 1);
  allocated += allocate_data(&hale_data->subcell_momentum_flux_x,
                             hale_data->nsubcells * 2);
  allocated += allocate_data(&hale_data->subcell_momentum_flux_y,
                             hale_data->nsubcells * 2);
  allocated += allocate_data(&hale_data->subcell_momentum_flux_z,
                             hale_data->nsubcells * 2);
  allocated +=
      allocate_data(&hale_data->subcell_ie_density0, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_mass0, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_ie_mass_flux, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_mass_flux, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_volume, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->corner_force_x, umesh->ncells * nnodes_by_cell);
  allocated +=
      allocate_data(&hale_data->corner_force_y, umesh->ncells * nnodes_by_cell);
  allocated +=
      allocate_data(&hale_data->corner_force_z, umesh->ncells * nnodes_by_cell);
  allocated +=
      allocate_data(&hale_data->subcell_centroids_x, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_centroids_y, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_centroids_z, hale_data->nsubcells);

  // In hale, the fundamental principle is that the mass at the cell and
  // sub-cell are conserved, so we can initialise them from the mesh
  // and then only the remapping step will ever adjust them
  init_cell_centroids(umesh->ncells, umesh->cells_offsets,
                      umesh->cells_to_nodes, umesh->nodes_x0, umesh->nodes_y0,
                      umesh->nodes_z0, umesh->cell_centroids_x,
                      umesh->cell_centroids_y, umesh->cell_centroids_z);

  init_subcells_to_subcells(
      umesh->ncells, umesh->faces_to_cells0, umesh->faces_to_cells1,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->cell_centroids_x, umesh->cell_centroids_y, umesh->cell_centroids_z,
      umesh->cells_to_faces_offsets, umesh->cells_to_faces,
      hale_data->subcells_to_subcells, hale_data->subcell_face_offsets);

  init_mesh_mass(umesh->ncells, umesh->cell_centroids_x,
                 umesh->cell_centroids_y, umesh->cell_centroids_z,
                 hale_data->density0, umesh->nodes_x0, umesh->nodes_y0,
                 umesh->nodes_z0, hale_data->cell_mass,
                 hale_data->subcell_mass0, umesh->cells_to_faces_offsets,
                 umesh->cells_to_faces, umesh->faces_to_nodes_offsets,
                 umesh->faces_to_nodes, hale_data->subcell_face_offsets);

  store_rezoned_mesh(umesh->nnodes, umesh->nodes_x0, umesh->nodes_y0,
                     umesh->nodes_z0, hale_data->rezoned_nodes_x,
                     hale_data->rezoned_nodes_y, hale_data->rezoned_nodes_z);

  return allocated;
}

// This method sets up the subcell nodes and connectivity for a structured mesh
// viewed as an unstructured mesh. This is not intended to be used for
// production purposes, but instead should be used for debugging the code.
size_t init_subcell_data_structures(Mesh* mesh, HaleData* hale_data) {

  const int nx = mesh->local_nx;
  const int ny = mesh->local_ny;
  const int nz = mesh->local_nz;
  const int nsubcells_per_cell = 24;
  const int nsubcell_nodes_per_cell = nsubcells_per_cell * NTET_NODES;

  // Construct the subcell mesh description
  const int subcell_nodes_off = 0;
  const int subcell_face_c_xy_off = (nx + 1) * (ny + 1) * (nz + 1);
  const int subcell_face_c_yz_off = subcell_face_c_xy_off + nx * ny * (nz + 1);
  const int subcell_face_c_zx_off = subcell_face_c_yz_off + (nx + 1) * ny * nz;
  const int subcell_cell_c_off = subcell_face_c_zx_off + nx * (ny + 1) * nz;
  const int nsubcell_nodes = subcell_cell_c_off + nx * ny * nz;

  hale_data->nsubcells_per_cell = nsubcells_per_cell;
  hale_data->nsubcell_nodes = nsubcell_nodes;

  size_t allocated = allocate_data(&hale_data->subcell_data_x, nsubcell_nodes);
  allocated += allocate_data(&hale_data->subcell_data_y, nsubcell_nodes);
  allocated += allocate_data(&hale_data->subcell_data_z, nsubcell_nodes);
  allocated += allocate_int_data(&hale_data->subcells_to_nodes,
                                 nx * ny * nz * nsubcell_nodes_per_cell);
  printf("Allocated %.4lf GB for subcell debugging output\n", allocated / GB);

  // Determine subcell connectivity in a planar fashion
  double dx = 1.0 / nx;
  double dy = 1.0 / ny;
  double dz = 1.0 / nz;

#define NODE_IND(i, j, k)                                                      \
  (subcell_nodes_off + ((i) * (nx + 1) * (ny + 1) + (j) * (nx + 1) + (k)))
#define FACE_C_XY_IND(i, j, k)                                                 \
  (subcell_face_c_xy_off + ((i)*nx * ny + (j)*nx + (k)))
#define FACE_C_YZ_IND(i, j, k)                                                 \
  (subcell_face_c_yz_off + ((i) * (nx + 1) * ny + (j) * (nx + 1) + (k)))
#define FACE_C_ZX_IND(i, j, k)                                                 \
  (subcell_face_c_zx_off + ((i)*nx * (ny + 1) + (j)*nx + (k)))
#define CELL_C_IND(i, j, k) (subcell_cell_c_off + ((i)*nx * ny + (j)*nx + (k)))

  // Construct the nodal positions
  for (int ii = 0; ii < nz + 1; ++ii) {
    for (int jj = 0; jj < ny + 1; ++jj) {
      for (int kk = 0; kk < nx + 1; ++kk) {
        hale_data->subcell_data_x[NODE_IND(ii, jj, kk)] = kk * dx;
        hale_data->subcell_data_y[NODE_IND(ii, jj, kk)] = jj * dy;
        hale_data->subcell_data_z[NODE_IND(ii, jj, kk)] = ii * dz;

        if (kk < nx && jj < ny) {
          hale_data->subcell_data_x[(FACE_C_XY_IND(ii, jj, kk))] =
              0.5 * dx + kk * dx;
          hale_data->subcell_data_y[(FACE_C_XY_IND(ii, jj, kk))] =
              0.5 * dy + jj * dy;
          hale_data->subcell_data_z[(FACE_C_XY_IND(ii, jj, kk))] = ii * dz;
        }
        if (jj < ny && ii < nz) {
          hale_data->subcell_data_x[FACE_C_YZ_IND(ii, jj, kk)] = kk * dx;
          hale_data->subcell_data_y[FACE_C_YZ_IND(ii, jj, kk)] =
              0.5 * dy + jj * dy;
          hale_data->subcell_data_z[FACE_C_YZ_IND(ii, jj, kk)] =
              0.5 * dz + ii * dz;
        }
        if (kk < nx && ii < nz) {
          hale_data->subcell_data_x[FACE_C_ZX_IND(ii, jj, kk)] =
              0.5 * dx + kk * dx;
          hale_data->subcell_data_y[FACE_C_ZX_IND(ii, jj, kk)] = jj * dy;
          hale_data->subcell_data_z[FACE_C_ZX_IND(ii, jj, kk)] =
              0.5 * dz + ii * dz;
        }
        if (kk < nx && jj < ny && ii < nz) {
          hale_data->subcell_data_x[CELL_C_IND(ii, jj, kk)] =
              0.5 * dx + kk * dx;
          hale_data->subcell_data_y[CELL_C_IND(ii, jj, kk)] =
              0.5 * dy + jj * dy;
          hale_data->subcell_data_z[CELL_C_IND(ii, jj, kk)] =
              0.5 * dz + ii * dz;
        }
      }
    }
  }

  for (int ii = 0; ii < nz; ++ii) {
    for (int jj = 0; jj < ny; ++jj) {
      for (int kk = 0; kk < nx; ++kk) {
        const int cell_index = (ii * nx * ny + jj * nx + kk);
        const int c_off = cell_index * nsubcell_nodes_per_cell;

        // Front subcells
        hale_data->subcells_to_nodes[(c_off + 0)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 1)] = NODE_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 2)] = NODE_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 3)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 4)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 5)] = NODE_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 6)] =
            NODE_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 7)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 8)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 9)] =
            NODE_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 10)] = NODE_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 11)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 12)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 13)] = NODE_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 14)] = NODE_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 15)] = CELL_C_IND(ii, jj, kk);

        // Left subcells
        hale_data->subcells_to_nodes[(c_off + 16)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 17)] = NODE_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 18)] = NODE_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 19)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 20)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 21)] = NODE_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 22)] =
            NODE_IND(ii + 1, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 23)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 24)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 25)] =
            NODE_IND(ii + 1, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 26)] = NODE_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 27)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 28)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 29)] = NODE_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 30)] = NODE_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 31)] = CELL_C_IND(ii, jj, kk);

        // Bottom subcells
        hale_data->subcells_to_nodes[(c_off + 32)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 33)] = NODE_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 34)] = NODE_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 35)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 36)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 37)] = NODE_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 38)] =
            NODE_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 39)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 40)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 41)] =
            NODE_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 42)] = NODE_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 43)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 44)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 45)] = NODE_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 46)] = NODE_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 47)] = CELL_C_IND(ii, jj, kk);

        // Right subcells
        hale_data->subcells_to_nodes[(c_off + 48)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 49)] = NODE_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 50)] =
            NODE_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 51)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 52)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 53)] =
            NODE_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 54)] =
            NODE_IND(ii + 1, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 55)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 56)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 57)] =
            NODE_IND(ii + 1, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 58)] =
            NODE_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 59)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 60)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 61)] =
            NODE_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 62)] = NODE_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 63)] = CELL_C_IND(ii, jj, kk);

        // Top subcells
        hale_data->subcells_to_nodes[(c_off + 64)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 65)] = NODE_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 66)] =
            NODE_IND(ii + 1, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 67)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 68)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 69)] =
            NODE_IND(ii + 1, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 70)] =
            NODE_IND(ii + 1, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 71)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 72)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 73)] =
            NODE_IND(ii + 1, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 74)] =
            NODE_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 75)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 76)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 77)] =
            NODE_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 78)] = NODE_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 79)] = CELL_C_IND(ii, jj, kk);

        // Back subcells
        hale_data->subcells_to_nodes[(c_off + 80)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 81)] = NODE_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 82)] =
            NODE_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 83)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 84)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 85)] =
            NODE_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 86)] =
            NODE_IND(ii + 1, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 87)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 88)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 89)] =
            NODE_IND(ii + 1, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 90)] =
            NODE_IND(ii + 1, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 91)] = CELL_C_IND(ii, jj, kk);

        hale_data->subcells_to_nodes[(c_off + 92)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 93)] =
            NODE_IND(ii + 1, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 94)] = NODE_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 95)] = CELL_C_IND(ii, jj, kk);
      }
    }
  }

  return allocated;
}

// Deallocates all of the hale specific data
void deallocate_hale_data(HaleData* hale_data) {
  // TODO: Populate this correctly !
}

// Writes out unstructured triangles to visit
void write_unstructured_to_visit_2d(const int nnodes, int ncells,
                                    const int step, double* nodes_x0,
                                    double* nodes_y0, const int* cells_to_nodes,
                                    const double* arr, const int nodal,
                                    const int quads) {
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

// Writes out unstructured tetrahedral subcells to visit
void subcells_to_visit(const int nnodes, int ncells, const int step,
                       double* nodes_x, double* nodes_y, double* nodes_z,
                       const int* cells_to_nodes, const double* arr,
                       const int nodal, const int quads) {

  double* coords[] = {(double*)nodes_x, (double*)nodes_y, (double*)nodes_z};

  int shapecounts[] = {ncells};
  int shapesize[] = {4};
  int shapetype[] = {DB_ZONETYPE_TET};

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
