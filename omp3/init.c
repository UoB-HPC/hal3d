#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <math.h>

void search_for_neighbour_face(const int ff, const int nfaces_by_subcell,
                               const int rnode_index,
                               const int subcell_to_faces_off, int* faces,
                               const int* faces_to_nodes_offsets,
                               const int* faces_to_nodes,
                               int* subcells_to_faces);

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

void init_subcells_to_faces(
    const int ncells, const int nsubcells, const int* cells_offsets,
    const int* nodes_to_faces_offsets, const int* cells_to_nodes,
    const int* faces_to_cells0, const int* faces_to_cells1,
    const int* nodes_to_faces, const int* faces_to_nodes,
    const int* faces_to_nodes_offsets, int* subcells_to_faces,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    int* subcells_to_faces_offsets) {

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;
      const int subcell_index = cell_to_nodes_off + nn;

      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if (faces_to_cells0[(face_index)] == cc ||
            faces_to_cells1[(face_index)] == cc) {
          subcells_to_faces_offsets[(subcell_index + 1)]++;
        }
      }
    }
  }

  // TODO: CAN WE IMPROVE THIS PREFIX SUM?
  for (int ss = 0; ss < nsubcells; ++ss) {
    subcells_to_faces_offsets[(ss + 1)] += subcells_to_faces_offsets[(ss)];
  }

#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_centroid);

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;
      const int subcell_index = cell_to_nodes_off + nn;

      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // Fetch all of the faces that are attached to a subcell
      int f = 0;
      int faces[nfaces_by_subcell];
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if (faces_to_cells0[(face_index)] != cc &&
            faces_to_cells1[(face_index)] != cc) {
          continue;
        }

        if (face_index != -1) {
          faces[(f++)] = face_index;
        }
      }

      subcells_to_faces[(subcell_to_faces_off)] = faces[0];

      for (int ff = 0; ff < nfaces_by_subcell - 1; ++ff) {
        const int face_index = faces[(ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        vec_t face_normal;
        const int face_clockwise =
            calc_surface_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z,
                                &cell_centroid, &face_normal);
        const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
        const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
        const int rnode_index = faces_to_nodes[(
            face_to_nodes_off + (face_clockwise ? prev_node : next_node))];

        subcells_to_faces[(subcell_to_faces_off + ff + 1)] = -1;

        for (int ff2 = 1; ff2 < nfaces_by_subcell; ++ff2) {
          if (ff == ff2) {
            continue;
          }

          const int face_index2 = faces[(ff2)];
          const int face_to_nodes_off2 = faces_to_nodes_offsets[(face_index2)];
          const int nnodes_by_face2 =
              faces_to_nodes_offsets[(face_index2 + 1)] - face_to_nodes_off2;

          for (int nn2 = 0; nn2 < nnodes_by_face2; ++nn2) {
            const int node_index = faces_to_nodes[(face_to_nodes_off2 + nn2)];
            // Check if this face shares an edge
            if (node_index == rnode_index) {
              subcells_to_faces[(subcell_to_faces_off + ff + 1)] = face_index2;
              break;
            }
          }

          if (subcells_to_faces[(subcell_to_faces_off + ff + 1)] != -1) {
            break;
          }
        }
      }

      printf("cc %d nn %d\n", cc, nn);
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        printf("%d\n", subcells_to_faces[(subcell_to_faces_off + ff)]);
      }
    }
  }
}

void search_for_neighbour_face(const int ff, const int nfaces_by_subcell,
                               const int rnode_index,
                               const int subcell_to_faces_off, int* faces,
                               const int* faces_to_nodes_offsets,
                               const int* faces_to_nodes,
                               int* subcells_to_faces) {}

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(const int ncells, const int* faces_to_cells0,
                               const int* faces_to_cells1,
                               const int* faces_to_nodes_offsets,
                               const int* faces_to_nodes, const double* nodes_x,
                               const double* nodes_y, const double* nodes_z,
                               int* subcells_to_subcells,
                               int* subcells_to_subcells_offsets,
                               int* cells_offsets, int* nodes_to_faces_offsets,
                               int* nodes_to_faces, int* cells_to_nodes) {

#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int subcell_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - subcell_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;
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
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;
      const int subcell_index = subcell_off + nn;
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

        const int neighbour_cell_index = (faces_to_cells0[(face_index)] == cc)
                                             ? faces_to_cells1[(face_index)]
                                             : faces_to_cells0[(face_index)];

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
        int neighbour_subcell_index;
        for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
            const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
            const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
            neighbour_subcell_index = faces_to_nodes[(
                face_to_nodes_off + (face_clockwise) ? prev_node : next_node)];
            break;
          }
        }

        // Now we have to work back to find our location in the cell
        // NOTE: Cells to nodes is essentially cells to subcells here
        for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
          if (cells_to_nodes[(cell_to_nodes_off + nn2)] ==
              neighbour_subcell_index) {
            subcells_to_subcells[(subcell_to_subcells_off + ff)] =
                cell_to_nodes_off + nn2;
            break;
          }
        }

        /* EXTERNAL NEIGHBOUR */

        const int neighbour_to_nodes_off =
            cells_offsets[(neighbour_cell_index)];
        const int nnodes_by_neighbour =
            cells_offsets[(neighbour_cell_index + 1)] - neighbour_to_nodes_off;

        // NOTE: Cells to nodes is essentially cells to subcells here
        for (int nn2 = 0; nn2 < nnodes_by_neighbour; ++nn2) {
          if (cells_to_nodes[(neighbour_to_nodes_off + nn2)] ==
              neighbour_subcell_index) {
            const int neighbour_subcell_index = neighbour_to_nodes_off + nn2;
            const int neighbour_subcell_to_subcells_off =
                subcells_to_subcells_offsets[(neighbour_subcell_index)];
            subcells_to_subcells[(neighbour_subcell_to_subcells_off + ff)] =
                neighbour_to_nodes_off + nn2;
            break;
          }
        }
      }
    }
  }
}

// NOTE: This is not intended to be a production device, rather used for
// debugging the code against a well tested description of the subcell mesh.
size_t init_subcell_data_structures(Mesh* mesh, HaleData* hale_data,
                                    UnstructuredMesh* umesh) {

  const int nx = mesh->local_nx;
  const int ny = mesh->local_ny;
  const int nz = mesh->local_nz;

  // Construct the subcell mesh description
  const int subcell_half_nodes_x_off = (nx + 1) * (ny + 1) * (nz + 1);
  const int subcell_half_nodes_y_off =
      subcell_half_nodes_x_off + (nx) * (ny + 1) * (nz + 1);
  const int subcell_half_nodes_z_off =
      subcell_half_nodes_y_off + (nx + 1) * (ny) * (nz + 1);
  const int subcell_face_c_xy_off =
      subcell_half_nodes_z_off + (nx + 1) * (ny + 1) * (nz);
  const int subcell_face_c_yz_off = subcell_face_c_xy_off + nx * ny * (nz + 1);
  const int subcell_face_c_zx_off = subcell_face_c_yz_off + (nx + 1) * ny * nz;
  const int subcell_cell_c_off = subcell_face_c_zx_off + nx * (ny + 1) * nz;
  const int nsubcell_nodes = subcell_cell_c_off + nx * ny * nz;
  hale_data->nsubcell_nodes = nsubcell_nodes;

  size_t allocated = allocate_data(&hale_data->subcell_nodes_x, nsubcell_nodes);
  allocated += allocate_data(&hale_data->subcell_nodes_y, nsubcell_nodes);
  allocated += allocate_data(&hale_data->subcell_nodes_z, nsubcell_nodes);
  allocated += allocate_int_data(&hale_data->subcells_to_nodes,
                                 umesh->ncells * hale_data->nsubcells_per_cell *
                                     hale_data->nnodes_per_subcell);
  printf("Allocated %.4lf GB for subcell debugging output\n", allocated / GB);

// Determine subcell connectivity in a planar fashion
#define NODE_IND(i, j, k) ((i) * (nx + 1) * (ny + 1) + (j) * (nx + 1) + (k))
#define HALF_NODE_X_IND(i, j, k)                                               \
  (subcell_half_nodes_x_off + ((i) * (nx) * (ny + 1) + (j) * (nx) + (k)))
#define HALF_NODE_Y_IND(i, j, k)                                               \
  (subcell_half_nodes_y_off + ((i) * (nx + 1) * (ny) + (j) * (nx + 1) + (k)))
#define HALF_NODE_Z_IND(i, j, k)                                               \
  (subcell_half_nodes_z_off +                                                  \
   ((i) * (nx + 1) * (ny + 1) + (j) * (nx + 1) + (k)))
#define FACE_C_XY_IND(i, j, k)                                                 \
  (subcell_face_c_xy_off + ((i)*nx * ny + (j)*nx + (k)))
#define FACE_C_YZ_IND(i, j, k)                                                 \
  (subcell_face_c_yz_off + ((i) * (nx + 1) * ny + (j) * (nx + 1) + (k)))
#define FACE_C_ZX_IND(i, j, k)                                                 \
  (subcell_face_c_zx_off + ((i)*nx * (ny + 1) + (j)*nx + (k)))
#define CELL_C_IND(i, j, k) (subcell_cell_c_off + ((i)*nx * ny + (j)*nx + (k)))

// Construct the nodal positions
#pragma omp parallel for
  for (int ii = 0; ii < nz + 1; ++ii) {
    for (int jj = 0; jj < ny + 1; ++jj) {
      for (int kk = 0; kk < nx + 1; ++kk) {
        // Corner nodes
        hale_data->subcell_nodes_x[NODE_IND(ii, jj, kk)] =
            umesh->nodes_x0[(NODE_IND(ii, jj, kk))];
        hale_data->subcell_nodes_y[NODE_IND(ii, jj, kk)] =
            umesh->nodes_y0[(NODE_IND(ii, jj, kk))];
        hale_data->subcell_nodes_z[NODE_IND(ii, jj, kk)] =
            umesh->nodes_z0[(NODE_IND(ii, jj, kk))];

        if (kk < nx) {
          hale_data->subcell_nodes_x[HALF_NODE_X_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_x0[(NODE_IND(ii, jj, kk + 1))]);
          hale_data->subcell_nodes_y[HALF_NODE_X_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_y0[(NODE_IND(ii, jj, kk + 1))]);
          hale_data->subcell_nodes_z[HALF_NODE_X_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_z0[(NODE_IND(ii, jj, kk + 1))]);
        }

        if (jj < ny) {
          hale_data->subcell_nodes_x[HALF_NODE_Y_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk))]);
          hale_data->subcell_nodes_y[HALF_NODE_Y_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk))]);
          hale_data->subcell_nodes_z[HALF_NODE_Y_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk))]);
        }

        if (ii < nz) {
          hale_data->subcell_nodes_x[HALF_NODE_Z_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk))]);
          hale_data->subcell_nodes_y[HALF_NODE_Z_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk))]);
          hale_data->subcell_nodes_z[HALF_NODE_Z_IND(ii, jj, kk)] =
              0.5 * (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
                     umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk))]);
        }

        if (kk < nx && jj < ny) {
          hale_data->subcell_nodes_x[(FACE_C_XY_IND(ii, jj, kk))] =
              (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk))]) /
              4.0;
          hale_data->subcell_nodes_y[(FACE_C_XY_IND(ii, jj, kk))] =
              (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk))]) /
              4.0;
          hale_data->subcell_nodes_z[(FACE_C_XY_IND(ii, jj, kk))] =
              (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk))]) /
              4.0;
        }

        if (jj < ny && ii < nz) {
          hale_data->subcell_nodes_x[(FACE_C_YZ_IND(ii, jj, kk))] =
              (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj + 1, kk))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
          hale_data->subcell_nodes_y[(FACE_C_YZ_IND(ii, jj, kk))] =
              (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj + 1, kk))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
          hale_data->subcell_nodes_z[(FACE_C_YZ_IND(ii, jj, kk))] =
              (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj + 1, kk))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
        }

        if (kk < nx && ii < nz) {
          hale_data->subcell_nodes_x[(FACE_C_ZX_IND(ii, jj, kk))] =
              (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
          hale_data->subcell_nodes_y[(FACE_C_ZX_IND(ii, jj, kk))] =
              (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
          hale_data->subcell_nodes_z[(FACE_C_ZX_IND(ii, jj, kk))] =
              (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk))]) /
              4.0;
        }

        if (ii < nz && jj < ny && kk < nx) {
          hale_data->subcell_nodes_x[CELL_C_IND(ii, jj, kk)] =
              (umesh->nodes_x0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj + 1, kk + 1))] +
               umesh->nodes_x0[(NODE_IND(ii + 1, jj + 1, kk))]) /
              8.0;
          hale_data->subcell_nodes_y[CELL_C_IND(ii, jj, kk)] =
              (umesh->nodes_y0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj + 1, kk + 1))] +
               umesh->nodes_y0[(NODE_IND(ii + 1, jj + 1, kk))]) /
              8.0;
          hale_data->subcell_nodes_z[CELL_C_IND(ii, jj, kk)] =
              (umesh->nodes_z0[(NODE_IND(ii, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii, jj + 1, kk))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj + 1, kk + 1))] +
               umesh->nodes_z0[(NODE_IND(ii + 1, jj + 1, kk))]) /
              8.0;
        }
      }
    }
  }

#pragma omp parallel for
  for (int ii = 0; ii < nz; ++ii) {
    for (int jj = 0; jj < ny; ++jj) {
      for (int kk = 0; kk < nx; ++kk) {
        const int cell_index = (ii * nx * ny + jj * nx + kk);
        const int c_off = cell_index * (hale_data->nnodes_per_subcell *
                                        hale_data->nsubcells_per_cell);

        // Bottom left subcell
        hale_data->subcells_to_nodes[(c_off + 0)] = NODE_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 1)] = HALF_NODE_X_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 2)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 3)] = HALF_NODE_Y_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 4)] = HALF_NODE_Z_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 5)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 6)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 7)] = FACE_C_YZ_IND(ii, jj, kk);

        // Bottom right subcell
        hale_data->subcells_to_nodes[(c_off + 8)] = HALF_NODE_X_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 9)] = NODE_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 10)] =
            HALF_NODE_Y_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 11)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 12)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 13)] =
            HALF_NODE_Z_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 14)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 15)] = CELL_C_IND(ii, jj, kk);

        // Top left subcell
        hale_data->subcells_to_nodes[(c_off + 16)] =
            HALF_NODE_Y_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 17)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 18)] =
            HALF_NODE_X_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 19)] = NODE_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 20)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 21)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 22)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 23)] =
            HALF_NODE_Z_IND(ii, jj + 1, kk);

        // Top right subcell
        hale_data->subcells_to_nodes[(c_off + 24)] = FACE_C_XY_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 25)] =
            HALF_NODE_Y_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 26)] =
            NODE_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 27)] =
            HALF_NODE_X_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 28)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 29)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 30)] =
            HALF_NODE_Z_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 31)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);

        // Bottom back left subcell
        hale_data->subcells_to_nodes[(c_off + 32)] =
            HALF_NODE_Z_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 33)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 34)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 35)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 36)] = NODE_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 37)] =
            HALF_NODE_X_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 38)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 39)] =
            HALF_NODE_Y_IND(ii + 1, jj, kk);

        // Bottom back right subcell
        hale_data->subcells_to_nodes[(c_off + 40)] = FACE_C_ZX_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 41)] =
            HALF_NODE_Z_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 42)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 43)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 44)] =
            HALF_NODE_X_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 45)] =
            NODE_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 46)] =
            HALF_NODE_Y_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 47)] =
            FACE_C_XY_IND(ii + 1, jj, kk);

        // Top back left subcell
        hale_data->subcells_to_nodes[(c_off + 48)] = FACE_C_YZ_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 49)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 50)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 51)] =
            HALF_NODE_Z_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 52)] =
            HALF_NODE_Y_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 53)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 54)] =
            HALF_NODE_X_IND(ii + 1, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 55)] =
            NODE_IND(ii + 1, jj + 1, kk);

        // Top back right subcell
        hale_data->subcells_to_nodes[(c_off + 56)] = CELL_C_IND(ii, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 57)] =
            FACE_C_YZ_IND(ii, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 58)] =
            HALF_NODE_Z_IND(ii, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 59)] =
            FACE_C_ZX_IND(ii, jj + 1, kk);
        hale_data->subcells_to_nodes[(c_off + 60)] =
            FACE_C_XY_IND(ii + 1, jj, kk);
        hale_data->subcells_to_nodes[(c_off + 61)] =
            HALF_NODE_Y_IND(ii + 1, jj, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 62)] =
            NODE_IND(ii + 1, jj + 1, kk + 1);
        hale_data->subcells_to_nodes[(c_off + 63)] =
            HALF_NODE_X_IND(ii + 1, jj + 1, kk);
      }
    }
  }

  return allocated;
}
