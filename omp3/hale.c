#include "hale.h"
#include "../../shared.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_2d(
    Mesh* mesh, const int ncells, const int nnodes, const double visc_coeff1,
    const double visc_coeff2, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_to_nodes,
    int* cells_offsets, int* nodes_to_cells, int* nodes_offsets,
    double* nodes_x0, double* nodes_y0, double* nodes_z0, double* nodes_x1,
    double* nodes_y1, double* nodes_z1, int* boundary_index, int* boundary_type,
    double* boundary_normal_x, double* boundary_normal_y,
    double* boundary_normal_z, double* cell_volume, double* energy0,
    double* energy1, double* density0, double* density1, double* pressure0,
    double* pressure1, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* velocity_x1, double* velocity_y1,
    double* velocity_z1, double* subcell_force_x, double* subcell_force_y,
    double* subcell_force_z, double* cell_mass, double* nodal_mass,
    double* nodal_volumes, double* nodal_soundspeed, double* limiter,
    double* subcell_volume, double* subcell_ie_mass0, double* subcell_mass0,
    double* subcell_ie_mass1, double* subcell_mass1, double* subcell_momentum_x,
    double* subcell_momentum_y, double* subcell_momentum_z,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, double* subcell_kinetic_energy,
    double* rezoned_nodes_x, double* rezoned_nodes_y, double* rezoned_nodes_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* subcell_face_offsets,
    int* subcells_to_subcells) {

  // Describe the subcell node layout
  const int nx = cbrt(ncells);
  const int ny = cbrt(ncells);
  const int nz = cbrt(ncells);
  const int nsubcells_per_cell = 24;
  const int nsubcell_nodes_per_cell = nsubcells_per_cell * NTET_NODES;

  // Construct the subcell mesh description
  int* subcells_to_nodes;
  double* subcell_data_x;
  double* subcell_data_y;
  double* subcell_data_z;

  const int subcell_nodes_off = 0;
  const int subcell_face_c_xy_off = (nx + 1) * (ny + 1) * (nz + 1);
  const int subcell_face_c_yz_off = subcell_face_c_xy_off + nx * ny * (nz + 1);
  const int subcell_face_c_zx_off = subcell_face_c_yz_off + (nx + 1) * ny * nz;
  const int subcell_cell_c_off = subcell_face_c_zx_off + nx * (ny + 1) * nz;
  const int nsubcell_nodes = subcell_cell_c_off + nx * ny * nz;

  size_t allocated = allocate_data(&subcell_data_x, nsubcell_nodes);
  allocated += allocate_data(&subcell_data_y, nsubcell_nodes);
  allocated += allocate_data(&subcell_data_z, nsubcell_nodes);
  allocated +=
      allocate_int_data(&subcells_to_nodes, ncells * nsubcell_nodes_per_cell);
  printf("allocated %.4lf GB for subcell output\n", allocated / GB);

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
        subcell_data_x[NODE_IND(ii, jj, kk)] = kk * dx;
        subcell_data_y[NODE_IND(ii, jj, kk)] = jj * dy;
        subcell_data_z[NODE_IND(ii, jj, kk)] = ii * dz;

        if (kk < nx && jj < ny) {
          subcell_data_x[(FACE_C_XY_IND(ii, jj, kk))] = 0.5 * dx + kk * dx;
          subcell_data_y[(FACE_C_XY_IND(ii, jj, kk))] = 0.5 * dy + jj * dy;
          subcell_data_z[(FACE_C_XY_IND(ii, jj, kk))] = ii * dz;
        }
        if (jj < ny && ii < nz) {
          subcell_data_x[FACE_C_YZ_IND(ii, jj, kk)] = kk * dx;
          subcell_data_y[FACE_C_YZ_IND(ii, jj, kk)] = 0.5 * dy + jj * dy;
          subcell_data_z[FACE_C_YZ_IND(ii, jj, kk)] = 0.5 * dz + ii * dz;
        }
        if (kk < nx && ii < nz) {
          subcell_data_x[FACE_C_ZX_IND(ii, jj, kk)] = 0.5 * dx + kk * dx;
          subcell_data_y[FACE_C_ZX_IND(ii, jj, kk)] = jj * dy;
          subcell_data_z[FACE_C_ZX_IND(ii, jj, kk)] = 0.5 * dz + ii * dz;
        }
        if (kk < nx && jj < ny && ii < nz) {
          subcell_data_x[CELL_C_IND(ii, jj, kk)] = 0.5 * dx + kk * dx;
          subcell_data_y[CELL_C_IND(ii, jj, kk)] = 0.5 * dy + jj * dy;
          subcell_data_z[CELL_C_IND(ii, jj, kk)] = 0.5 * dz + ii * dz;
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
        subcells_to_nodes[(c_off + 0)] = NODE_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 1)] = NODE_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 2)] = FACE_C_XY_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 3)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 4)] = NODE_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 5)] = NODE_IND(ii, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 6)] = FACE_C_XY_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 7)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 8)] = NODE_IND(ii, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 9)] = NODE_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 10)] = FACE_C_XY_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 11)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 12)] = NODE_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 13)] = NODE_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 14)] = FACE_C_XY_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 15)] = CELL_C_IND(ii, jj, kk);

        // Left subcells
        subcells_to_nodes[(c_off + 16)] = NODE_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 17)] = NODE_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 18)] = FACE_C_YZ_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 19)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 20)] = NODE_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 21)] = NODE_IND(ii + 1, jj + 1, kk);
        subcells_to_nodes[(c_off + 22)] = FACE_C_YZ_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 23)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 24)] = NODE_IND(ii + 1, jj + 1, kk);
        subcells_to_nodes[(c_off + 25)] = NODE_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 26)] = FACE_C_YZ_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 27)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 28)] = NODE_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 29)] = NODE_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 30)] = FACE_C_YZ_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 31)] = CELL_C_IND(ii, jj, kk);

        // Bottom subcells
        subcells_to_nodes[(c_off + 32)] = NODE_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 33)] = NODE_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 34)] = FACE_C_ZX_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 35)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 36)] = NODE_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 37)] = NODE_IND(ii + 1, jj, kk + 1);
        subcells_to_nodes[(c_off + 38)] = FACE_C_ZX_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 39)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 40)] = NODE_IND(ii + 1, jj, kk + 1);
        subcells_to_nodes[(c_off + 41)] = NODE_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 42)] = FACE_C_ZX_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 43)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 44)] = NODE_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 45)] = NODE_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 46)] = FACE_C_ZX_IND(ii, jj, kk);
        subcells_to_nodes[(c_off + 47)] = CELL_C_IND(ii, jj, kk);

        // Right subcells
        subcells_to_nodes[(c_off + 48)] = NODE_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 49)] = NODE_IND(ii + 1, jj, kk + 1);
        subcells_to_nodes[(c_off + 50)] = FACE_C_YZ_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 51)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 52)] = NODE_IND(ii + 1, jj, kk + 1);
        subcells_to_nodes[(c_off + 53)] = NODE_IND(ii + 1, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 54)] = FACE_C_YZ_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 55)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 56)] = NODE_IND(ii + 1, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 57)] = NODE_IND(ii, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 58)] = FACE_C_YZ_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 59)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 60)] = NODE_IND(ii, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 61)] = NODE_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 62)] = FACE_C_YZ_IND(ii, jj, kk + 1);
        subcells_to_nodes[(c_off + 63)] = CELL_C_IND(ii, jj, kk);

        // Top subcells
        subcells_to_nodes[(c_off + 64)] = NODE_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 65)] = NODE_IND(ii, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 66)] = FACE_C_ZX_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 67)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 68)] = NODE_IND(ii, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 69)] = NODE_IND(ii + 1, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 70)] = FACE_C_ZX_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 71)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 72)] = NODE_IND(ii + 1, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 73)] = NODE_IND(ii + 1, jj + 1, kk);
        subcells_to_nodes[(c_off + 74)] = FACE_C_ZX_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 75)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 76)] = NODE_IND(ii + 1, jj + 1, kk);
        subcells_to_nodes[(c_off + 77)] = NODE_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 78)] = FACE_C_ZX_IND(ii, jj + 1, kk);
        subcells_to_nodes[(c_off + 79)] = CELL_C_IND(ii, jj, kk);

        // Back subcells
        subcells_to_nodes[(c_off + 80)] = NODE_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 81)] = NODE_IND(ii + 1, jj + 1, kk);
        subcells_to_nodes[(c_off + 82)] = FACE_C_XY_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 83)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 84)] = NODE_IND(ii + 1, jj + 1, kk);
        subcells_to_nodes[(c_off + 85)] = NODE_IND(ii + 1, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 86)] = FACE_C_XY_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 87)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 88)] = NODE_IND(ii + 1, jj + 1, kk + 1);
        subcells_to_nodes[(c_off + 89)] = NODE_IND(ii + 1, jj, kk + 1);
        subcells_to_nodes[(c_off + 90)] = FACE_C_XY_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 91)] = CELL_C_IND(ii, jj, kk);

        subcells_to_nodes[(c_off + 92)] = NODE_IND(ii + 1, jj, kk + 1);
        subcells_to_nodes[(c_off + 93)] = NODE_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 94)] = FACE_C_XY_IND(ii + 1, jj, kk);
        subcells_to_nodes[(c_off + 95)] = CELL_C_IND(ii, jj, kk);
      }
    }
  }

  printf("\nPerforming the Lagrangian Phase\n");

#if 0
  // Perform the Lagrangian phase of the ALE algorithm where the mesh will move
  // due to the pressure (ideal gas) and artificial viscous forces
  lagrangian_phase(
      mesh, ncells, nnodes, visc_coeff1, visc_coeff2, cell_centroids_x,
      cell_centroids_y, cell_centroids_z, cells_to_nodes, cells_offsets,
      nodes_to_cells, nodes_offsets, nodes_x0, nodes_y0, nodes_z0, nodes_x1,
      nodes_y1, nodes_z1, boundary_index, boundary_type, boundary_normal_x,
      boundary_normal_y, boundary_normal_z, energy0, energy1, density0,
      density1, pressure0, pressure1, velocity_x0, velocity_y0, velocity_z0,
      velocity_x1, velocity_y1, velocity_z1, subcell_force_x, subcell_force_y,
      subcell_force_z, cell_mass, nodal_mass, nodal_volumes, nodal_soundspeed,
      limiter, nodes_to_faces_offsets, nodes_to_faces, faces_to_nodes,
      faces_to_nodes_offsets, faces_to_cells0, faces_to_cells1,
      cells_to_faces_offsets, cells_to_faces);

  printf("\nPerforming Gathering Phase\n");

  // Gather the subcell quantities for mass, internal and kinetic energy
  // density, and momentum
  gather_subcell_quantities(
      ncells, nnodes, nodal_volumes, nodal_mass, cell_centroids_x,
      cell_centroids_y, cell_centroids_z, cells_offsets, nodes_x0, nodes_y0,
      nodes_z0, energy0, density0, velocity_x0, velocity_y0, velocity_z0,
      cell_mass, subcell_volume, subcell_ie_mass0, subcell_momentum_x,
      subcell_momentum_y, subcell_momentum_z, subcell_centroids_x,
      subcell_centroids_y, subcell_centroids_z, cell_volume,
      subcell_face_offsets, faces_to_nodes, faces_to_nodes_offsets,
      faces_to_cells0, faces_to_cells1, cells_to_faces_offsets, cells_to_faces,
      cells_to_nodes);
#endif // if 0

  calc_volumes_centroids(ncells, cells_to_faces_offsets, cell_centroids_x,
                         cell_centroids_y, cell_centroids_z, cells_to_faces,
                         faces_to_nodes, faces_to_nodes_offsets,
                         subcell_face_offsets, nodes_x0, nodes_y0, nodes_z0,
                         cell_volume, subcell_centroids_x, subcell_centroids_y,
                         subcell_centroids_z, subcell_volume);

// Calculate the sub-cell internal energies
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    // Calculating the volume dist necessary for the least squares
    // regression
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // Determine the weighted volume dist for neighbouring cells
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // Each face/node pair has two sub-cells
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // The left and right nodes on the face for this anchor node
        const int subcell_index = subcell_off + nn;

        subcell_ie_mass0[(subcell_index)] =
            subcell_centroids_x[(subcell_index)];
      }
    }
  }

  subcells_to_visit(nsubcell_nodes, ncells * nsubcells_per_cell, 10,
                    subcell_data_x, subcell_data_y, subcell_data_z,
                    subcells_to_nodes, subcell_ie_mass0, 0, 1);

  printf("\nPerforming Remap Phase\n");

  remap_phase(ncells, nnodes, cell_centroids_x, cell_centroids_y,
              cell_centroids_z, cells_to_nodes, cells_offsets, nodes_x0,
              nodes_y0, nodes_z0, cell_volume, energy0, energy1, density0,
              velocity_x0, velocity_y0, velocity_z0, cell_mass, nodal_mass,
              subcell_volume, subcell_ie_mass0, subcell_ie_mass1, subcell_mass0,
              subcell_mass1, subcell_momentum_x, subcell_momentum_y,
              subcell_momentum_z, subcell_centroids_x, subcell_centroids_y,
              subcell_centroids_z, rezoned_nodes_x, rezoned_nodes_y,
              rezoned_nodes_z, nodes_to_faces_offsets, nodes_to_faces,
              faces_to_nodes, faces_to_nodes_offsets, faces_to_cells0,
              faces_to_cells1, cells_to_faces_offsets, cells_to_faces,
              subcell_face_offsets, subcells_to_subcells);

  printf("\nEulerian Mesh Rezone\n");
#if 0
  apply_mesh_rezoning(nnodes, rezoned_nodes_x, rezoned_nodes_y, rezoned_nodes_z,
                      nodes_x0, nodes_y0, nodes_z0);
#endif // if 0

  init_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0, nodes_y0,
                      nodes_z0, cell_centroids_x, cell_centroids_y,
                      cell_centroids_z);
}
