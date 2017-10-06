#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <math.h>

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(
    const int ncells, const int nnodes, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const double* density, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, double* cell_mass, double* subcell_mass,
    double* nodal_mass, int* cells_to_faces_offsets, int* cells_to_faces,
    int* faces_to_nodes_offsets, int* faces_to_nodes,
    int* faces_to_subcells_offsets, int* nodes_to_faces_offsets,
    int* nodes_to_faces, int* faces_to_cells0, int* faces_to_cells1) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
  double total_subcell_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass, total_subcell_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double cm = 0.0;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off =
          faces_to_subcells_offsets[(cell_to_faces_off + ff)];

      // Calculate the face center... SHOULD WE PRECOMPUTE?
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                    face_to_nodes_off, &face_c);

      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        // Fetch the nodes attached to our current node on
        // the current face
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn2)];
        const int next_node =
            (nn2 + 1 < nnodes_by_face)
                ? faces_to_nodes[(face_to_nodes_off + nn2 + 1)]
                : faces_to_nodes[(face_to_nodes_off)];

        // Get the halfway point on the right edge
        const double half_edge_x =
            0.5 * (nodes_x[(current_node)] + nodes_x[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y[(current_node)] + nodes_y[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z[(current_node)] + nodes_z[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (half_edge_x - face_c.x);
        const double a_y = (half_edge_y - face_c.y);
        const double a_z = (half_edge_z - face_c.z);
        const double b_x = (cell_centroids_x[(cc)] - face_c.x);
        const double b_y = (cell_centroids_y[(cc)] - face_c.y);
        const double b_z = (cell_centroids_z[(cc)] - face_c.z);

        // Calculate the area vector S using cross product
        const double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        const double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        const double S_z = 0.5 * (a_x * b_y - a_y * b_x);

        double subcell_volume =
            2.0 * fabs(((half_edge_x - nodes_x[(current_node)]) * S_x +
                        (half_edge_y - nodes_y[(current_node)]) * S_y +
                        (half_edge_z - nodes_z[(current_node)]) * S_z) /
                       3.0);

        subcell_mass[(subcell_off + nn2)] = density[(cc)] * subcell_volume;
        total_subcell_mass += subcell_mass[(subcell_off + nn2)];
        cm += density[(cc)] * subcell_volume;
      }
    }

    cell_mass[(cc)] = cm;
    total_mass += cm;
  }
  STOP_PROFILING(&compute_profile, __func__);

  printf("Initial Total Mesh Mass: %.12f\n", total_mass);
  printf("Initial Total Subcell Mass: %.12f\n", total_subcell_mass);

  // Calculate the nodal mass
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;

    vec_t node_c = {nodes_x[(nn)], nodes_y[(nn)], nodes_z[(nn)]};

    // Consider all faces attached to node
    for (int ff = 0; ff < nfaces_by_node; ++ff) {
      const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
      if (face_index == -1) {
        continue;
      }

      // Determine the offset into the list of nodes
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Find node center and location of current node on
      // face
      vec_t face_c = {0.0, 0.0, 0.0};
      int node_in_face_c;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c.x += nodes_x[(node_index)] / nnodes_by_face;
        face_c.y += nodes_y[(node_index)] / nnodes_by_face;
        face_c.z += nodes_z[(node_index)] / nnodes_by_face;

        // Choose the node in the list of nodes attached to
        // the face
        if (nn == node_index) {
          node_in_face_c = nn2;
        }
      }

      // Fetch the nodes attached to our current node on the
      // current face
      int nodes[2];
      nodes[0] = (node_in_face_c - 1 >= 0)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c - 1)]
                     : faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)];
      nodes[1] = (node_in_face_c + 1 < nnodes_by_face)
                     ? faces_to_nodes[(face_to_nodes_off + node_in_face_c + 1)]
                     : faces_to_nodes[(face_to_nodes_off)];

      // Fetch the cells attached to our current face
      int cells[2];
      cells[0] = faces_to_cells0[(face_index)];
      cells[1] = faces_to_cells1[(face_index)];

      // Add contributions from all of the cells attached to
      // the face
      for (int cc = 0; cc < 2; ++cc) {
        if (cells[(cc)] == -1) {
          continue;
        }

        // Add contributions for both edges attached to our
        // current node
        for (int nn2 = 0; nn2 < 2; ++nn2) {
          // Get the halfway point on the right edge
          vec_t half_edge = {0.5 * (nodes_x[(nodes[(nn2)])] + nodes_x[(nn)]),
                             0.5 * (nodes_y[(nodes[(nn2)])] + nodes_y[(nn)]),
                             0.5 * (nodes_z[(nodes[(nn2)])] + nodes_z[(nn)])};

          // Setup basis on plane of tetrahedron
          vec_t a = {(face_c.x - node_c.x), (face_c.y - node_c.y),
                     (face_c.z - node_c.z)};
          vec_t b = {(face_c.x - half_edge.x), (face_c.y - half_edge.y),
                     (face_c.z - half_edge.z)};
          vec_t ab = {(cell_centroids_x[(cells[cc])] - face_c.x),
                      (cell_centroids_y[(cells[cc])] - face_c.y),
                      (cell_centroids_z[(cells[cc])] - face_c.z)};

          // Calculate the area vector S using cross product
          vec_t A = {0.5 * (a.y * b.z - a.z * b.y),
                     -0.5 * (a.x * b.z - a.z * b.x),
                     0.5 * (a.x * b.y - a.y * b.x)};

          const double subcell_volume =
              fabs((ab.x * A.x + ab.y * A.y + ab.z * A.z) / 3.0);

          nodal_mass[(nn)] += density[(cells[(cc)])] * subcell_volume;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, "calc_nodal_mass_vol");
}

// Initialises the centroids for each cell
void init_cell_centroids(const int ncells, const int* cells_offsets,
                         const int* cells_to_nodes, const double* nodes_x,
                         const double* nodes_y, const double* nodes_z,
                         double* cell_centroids_x, double* cell_centroids_y,
                         double* cell_centroids_z) {

  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    vec_t cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cells_off, &cell_centroid);

    cell_centroids_x[(cc)] = cell_centroid.x;
    cell_centroids_y[(cc)] = cell_centroid.y;
    cell_centroids_z[(cc)] = cell_centroid.z;
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const int* faces_to_cells0, const int* faces_to_cells1,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, int* cells_to_faces_offsets,
    int* cells_to_faces, int* subcells_offsets, int* subcells_to_subcells,
    int* subcells_to_subcells_offsets, int* faces_to_subcells_offsets,
    int* cells_offsets, int* nodes_to_faces_offsets, int* nodes_to_faces,
    int* cells_to_nodes) {

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int subcell_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - subcell_off;

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;
      const int subcell_index = subcell_off + nn;

      int nsubcell_neighbours = 0;
      // Consider all faces attached to node
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if (faces_to_cells0[(face_index)] != cc &&
            faces_to_cells1[(face_index)] != cc) {
          continue;
        }

        nsubcell_neighbours++;
      }

      subcells_to_subcells_offsets[(subcell_index)] = nsubcell_neighbours * 2;
    }
  }

  // TODO: PREFIX SUM PARALLELISATION?
  for (int cc = 0; cc < ncells; ++cc) {
    const int subcell_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - subcell_off;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int subcell_index = subcell_off + nn;
      subcells_to_subcells_offsets[(subcell_index + 1)] +=
          subcells_to_subcells_offsets[(subcell_index)];
    }
  }

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int subcell_off = cells_offsets[(cc)];
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - subcell_off;

    vec_t cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_centroid);

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;
      const int subcell_index = subcell_off + nn;
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_to_subcells_off =
          subcells_to_subcells_offsets[(subcell_index)];

      // Consider all faces attached to node
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
        if (faces_to_cells0[(face_index)] != cc &&
            faces_to_cells1[(face_index)] != cc) {
          continue;
        }

        // We can consider two neighbour contributions loosely associated with a
        // face, the external and the internal

        /* INTERNAL NEIGHBOUR */

        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        vec_t face_normal;
        const int face_clockwise =
            calc_surface_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z,
                                &cell_centroid, &face_normal);

        // We have to find our position on the face
        // The nodes will mostly be in cache anyway so this is actually cheap
        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
        const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
        subcells_to_subcells[(subcell_to_subcells_off + ff)] =
            face_clockwise ? prev_node : next_node;

        /* EXTERNAL NEIGHBOUR */

        subcells_to_subcells[(subcell_to_subcells_off + nfaces_by_node + ff)] =
            face_clockwise ? prev_node : next_node;
      }
    }
  }
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
