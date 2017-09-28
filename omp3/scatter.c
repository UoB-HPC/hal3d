#include "../hale_data.h"
#include <stdio.h>

// Perform the scatter step of the ALE remapping algorithm
void scatter_phase(const int ncells, const int nnodes, const double total_mass,
                   const double total_ie, double* cell_volume, double* energy0,
                   double* energy1, double* density0, double* velocity_x0,
                   double* velocity_y0, double* velocity_z0, double* cell_mass,
                   double* nodal_mass, double* subcell_ie_mass0,
                   double* subcell_mass0, double* subcell_ie_mass_flux,
                   double* subcell_mass_flux, double* subcell_momentum_flux_x,
                   double* subcell_momentum_flux_y,
                   double* subcell_momentum_flux_z, int* nodes_to_faces_offsets,
                   int* nodes_to_faces, int* faces_to_nodes,
                   int* faces_to_nodes_offsets, int* faces_to_cells0,
                   int* faces_to_cells1, int* cells_to_faces_offsets,
                   int* cells_to_faces, int* subcell_face_offsets) {

  // Scatter energy and density, and print the conservation of mass
  double rz_total_mass = 0.0;
  double rz_total_ie = 0.0;
#pragma omp parallel for reduction(+ : rz_total_mass, rz_total_ie)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    cell_mass[(cc)] = 0.0;
    energy1[(cc)] = 0.0;

    /* LOOP OVER CELL FACES */
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      /* LOOP OVER FACE NODES */
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int subcell_index = subcell_off + nn;

        // Evaluate the subcell masses
        subcell_mass0[(subcell_index)] -= subcell_mass_flux[(subcell_index)];
        subcell_ie_mass0[(subcell_index)] -=
            subcell_ie_mass_flux[(subcell_index)];

        // Scatter the subcell mass data back to the cell
        cell_mass[(cc)] += subcell_mass0[(subcell_index)];
        energy1[(cc)] += subcell_ie_mass0[(subcell_index)];
      }
    }

    // Scatter the energy and density
    density0[(cc)] = cell_mass[(cc)] / cell_volume[(cc)];
    energy0[(cc)] = energy1[(cc)] / cell_mass[(cc)];

    // Calculate the conservation data
    rz_total_mass += cell_mass[(cc)];
    rz_total_ie += energy1[(cc)];
  }

  printf(
      "Rezoned Total Mass %.12f, Initial Total Mass %.12f, Difference %.12f\n",
      rz_total_mass, total_mass, total_mass - rz_total_mass);
  printf("Rezoned Total Internal Energy %.12f, Initial Total Energy %.12f, "
         "Difference "
         "%.12f\n",
         rz_total_ie, total_ie, total_ie - rz_total_ie);

  // Scattering the momentum
  double total_vx = 0.0;
  double total_vy = 0.0;
  double total_vz = 0.0;
#pragma omp parallel for reduction(+ : total_vx, total_vy, total_vz)
  for (int nn = 0; nn < nnodes; ++nn) {
    velocity_x0[(nn)] = 0.0;
    velocity_y0[(nn)] = 0.0;
    velocity_z0[(nn)] = 0.0;

    const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
    const int nfaces_by_node =
        nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;

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

      // Find node center and location of current node on face
      int node_in_face_c;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        // Choose the node in the list of nodes attached to the face
        if (nn == faces_to_nodes[(face_to_nodes_off + nn2)]) {
          node_in_face_c = nn2;
        }
      }

      // Fetch the cells attached to our current face
      int cells[2];
      cells[0] = faces_to_cells0[(face_index)];
      cells[1] = faces_to_cells1[(face_index)];

      // Add contributions from all of the cells attached to the face
      for (int cc = 0; cc < NSUBSUBCELLS; ++cc) {
        if (cells[(cc)] == -1) {
          continue;
        }

        const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
        const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

        // Add contributions for both edges attached to our current node
        const int nedges_by_node = 2;
        for (int nn2 = 0; nn2 < nedges_by_node; ++nn2) {
          velocity_x0[(nn)] +=
              subcell_momentum_flux_x[(subcell_off + node_in_face_c)];
          velocity_y0[(nn)] +=
              subcell_momentum_flux_y[(subcell_off + node_in_face_c)];
          velocity_z0[(nn)] +=
              subcell_momentum_flux_z[(subcell_off + node_in_face_c)];
        }
      }
    }

    velocity_x0[(nn)] /= nodal_mass[(nn)];
    velocity_y0[(nn)] /= nodal_mass[(nn)];
    velocity_z0[(nn)] /= nodal_mass[(nn)];

    total_vx += velocity_x0[(nn)];
    total_vy += velocity_y0[(nn)];
    total_vz += velocity_z0[(nn)];
  }

  printf("Total Scattered Velocity %.12f %.12f %.12f\n", total_vx, total_vy,
         total_vz);
}
