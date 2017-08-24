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
    double* subcell_volume, double* subcell_ie_density, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* subcell_kinetic_energy, double* rezoned_nodes_x,
    double* rezoned_nodes_y, double* rezoned_nodes_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* subcell_face_offsets,
    int* subcells_to_subcells) {

  double total_mass = 0.0;
  double total_energy = 0.0;
  double total_density = 0.0;
#pragma omp parallel for reduction(+ : total_mass, total_energy, total_density)
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
    total_energy += energy0[(cc)];
    total_density += density0[(cc)];
  }
  printf("Total Mass %.12f\n", total_mass);

  printf("\nPerforming the Lagrangian Phase\n");

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
      cell_mass, subcell_volume, subcell_ie_density, subcell_mass,
      subcell_velocity_x, subcell_velocity_y, subcell_velocity_z,
      subcell_centroids_x, subcell_centroids_y, subcell_centroids_z,
      cell_volume, subcell_face_offsets, faces_to_nodes, faces_to_nodes_offsets,
      faces_to_cells0, faces_to_cells1, cells_to_faces_offsets, cells_to_faces,
      cells_to_nodes);

#if 0
  /*
   * Remapping
   */

  printf("\nPerforming Remap Phase\n");

/* LOOP OVER CELLS */
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // Clear the cell mass as we will redistribute at the subcell
    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    vec_t rz_cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, rezoned_nodes_x, rezoned_nodes_y,
                  rezoned_nodes_z, cells_to_nodes, cell_to_nodes_off,
                  &rz_cell_centroid);

    // Zero out the arrays that will be used to store the remapped quantities
    cell_mass[(cc)] = 0.0;
    energy1[(cc)] = 0.0;
    cell_volume[(cc)] = 0.0;

    /* LOOP OVER CELL FACES */
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x0, nodes_y0, nodes_z0,
                    faces_to_nodes, face_to_nodes_off, &face_c);
      vec_t rz_face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, rezoned_nodes_x, rezoned_nodes_y,
                    rezoned_nodes_z, faces_to_nodes, face_to_nodes_off,
                    &rz_face_c);

      /* LOOP OVER FACE NODES */
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        const int subcell_index = subcell_off + nn;
        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];

        vec_t face_normal;
        const int face_clockwise =
            calc_surface_normal(n0, n1, n2, nodes_x0, nodes_y0, nodes_z0,
                                &cell_centroid, &face_normal);

        int rnode_index;
        if (face_clockwise) {
          rnode_index = faces_to_nodes[(
              face_to_nodes_off + ((nn == 0) ? nnodes_by_face - 1 : nn - 1))];
        } else {
          rnode_index = faces_to_nodes[(
              face_to_nodes_off + ((nn == nnodes_by_face - 1) ? 0 : nn + 1))];
        }

        // Describe the subcell tetrahedron connectivity
        const int subcell_to_faces[] = {0, 1, 2, 1, 3, 2, 0, 2, 3, 0, 3, 1};
        const vec_t subcell[] = {
            {nodes_x0[(node_index)], nodes_y0[(node_index)],
             nodes_z0[(node_index)]},
            {nodes_x0[(rnode_index)], nodes_y0[(rnode_index)],
             nodes_z0[(rnode_index)]},
            {face_c.x, face_c.y, face_c.z},
            {cell_centroid.x, cell_centroid.y, cell_centroid.z}};
        const vec_t rz_subcell[] = {
            {rezoned_nodes_x[(node_index)], rezoned_nodes_y[(node_index)],
             rezoned_nodes_z[(node_index)]},
            {rezoned_nodes_x[(rnode_index)], rezoned_nodes_y[(rnode_index)],
             rezoned_nodes_z[(rnode_index)]},
            {rz_face_c.x, rz_face_c.y, rz_face_c.z},
            {rz_cell_centroid.x, rz_cell_centroid.y, rz_cell_centroid.z}};

        /* LOOP OVER SUBCELL FACES */
        for (int pp = 0; pp < NTET_FACES; ++pp) {

          // Describe the swept edge prism
          const int swept_edge_faces_to_nodes_offsets[] = {0, 3, 7, 11, 15, 18};
          const int swept_edge_faces_to_nodes[] = {0, 1, 2, 1, 4, 5, 2, 0, 2,
                                                   5, 3, 0, 3, 4, 1, 3, 5, 4};
          const int swept_edge_to_faces[] = {0, 1, 2, 3, 4};
          const int sf0 = subcell_to_faces[(pp * NTET_NODES_PER_FACE + 0)];
          const int sf1 = subcell_to_faces[(pp * NTET_NODES_PER_FACE + 1)];
          const int sf2 = subcell_to_faces[(pp * NTET_NODES_PER_FACE + 2)];
          double swept_edge_nodes_x[] = {
              subcell[(sf0)].x,    subcell[(sf1)].x,    subcell[(sf2)].x,
              rz_subcell[(sf0)].x, rz_subcell[(sf1)].x, rz_subcell[(sf2)].x};
          double swept_edge_nodes_y[] = {
              subcell[(sf0)].y,    subcell[(sf1)].y,    subcell[(sf2)].y,
              rz_subcell[(sf0)].y, rz_subcell[(sf1)].y, rz_subcell[(sf2)].y};
          double swept_edge_nodes_z[] = {
              subcell[(sf0)].z,    subcell[(sf1)].z,    subcell[(sf2)].z,
              rz_subcell[(sf0)].z, rz_subcell[(sf1)].z, rz_subcell[(sf2)].z};

          // Determine the swept edge prism's centroid
          vec_t swept_edge_centroid = {0.0, 0.0, 0.0};
          for (int pn = 0; pn < NPRISM_NODES; ++pn) {
            swept_edge_centroid.x += swept_edge_nodes_x[(pn)] / NPRISM_NODES;
            swept_edge_centroid.y += swept_edge_nodes_y[(pn)] / NPRISM_NODES;
            swept_edge_centroid.z += swept_edge_nodes_z[(pn)] / NPRISM_NODES;
          }

          // Calculate the volume of the swept edge prism
          double swept_edge_vol = 0.0;
          calc_volume(0, NPRISM_FACES, swept_edge_to_faces,
                      swept_edge_faces_to_nodes,
                      swept_edge_faces_to_nodes_offsets, swept_edge_nodes_x,
                      swept_edge_nodes_y, swept_edge_nodes_z,
                      &swept_edge_centroid, &swept_edge_vol);

          // Ignore the special case of an empty swept edge region
          if (swept_edge_vol <= 0.0) {
            continue;
          }

          const int is_internal_sweep = !check_normal_orientation(
              0, swept_edge_nodes_x, swept_edge_nodes_y, swept_edge_nodes_z,
              &swept_edge_centroid, &face_normal);

          // Depending upon which subcell we are sweeping into, choose the
          // subcell index with which to reconstruct the density
          const int subcell_neighbour_index =
              subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + pp)];
          const int sweep_subcell_index =
              (is_internal_sweep ? subcell_index : subcell_neighbour_index);

          // Only perform the sweep on the external face if it isn't a
          // boundary
          if (subcell_neighbour_index == -1) {
            continue;
          }

          vec_t inv[3];

          /* CALCULATE INVERSE COEFFICIENT MATRIX */

          // This coefficient matrix is identical for all of the advections
          calc_inverse_coefficient_matrix(
              sweep_subcell_index, subcells_to_subcells, subcell_centroids_x,
              subcell_centroids_y, subcell_centroids_z, subcell_volume,
              NSUBCELL_NEIGHBOURS, sweep_subcell_index * NSUBCELL_NEIGHBOURS,
              &inv);

          /* CALCULATE THE MAXIMUM AND MINIMUM VALUES IN THE NEIGHBOURHOOD */
          double g_ie_min = 0.0;
          double g_ie_max = 0.0;
          double g_mass_min = 0.0;
          double g_mass_max = 0.0;
          for (int ss2 = 0; ss2 < NSUBCELL_NEIGHBOURS; ++ss2) {
            const int neighbour_subcell_index = subcells_to_subcells[(
                sweep_subcell_index * NSUBCELL_NEIGHBOURS + ss2)];

            // Ignore boundary neighbours
            if (neighbour_subcell_index == -1) {
              continue;
            }
            g_mass_max =
                max(g_mass_max, subcell_mass[(neighbour_subcell_index)]);
            g_mass_min =
                min(g_mass_min, subcell_mass[(neighbour_subcell_index)]);
            g_ie_max =
                max(g_ie_max, subcell_ie_density[(neighbour_subcell_index)]);
            g_ie_min =
                min(g_ie_min, subcell_ie_density[(neighbour_subcell_index)]);
          }

          /* ADVECT MASS */

          // For all of the subcell centered quantities, determine the flux.
          vec_t grad_mass = {0.0, 0.0, 0.0};
          calc_gradient(sweep_subcell_index, NSUBCELL_NEIGHBOURS,
                        sweep_subcell_index * NSUBCELL_NEIGHBOURS,
                        subcells_to_subcells, subcell_mass, subcell_centroids_x,
                        subcell_centroids_y, subcell_centroids_z,
                        (const vec_t(*)[3]) & inv, &grad_mass);

          // Calculate and apply limiter for the gradient
          apply_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                        &grad_mass, &cell_centroid, nodes_x0, nodes_y0,
                        nodes_z0, subcell_mass[(cc)], g_mass_max, g_mass_min);

          // Calculate the flux for internal energy density in the subcell
          const double mass_flux =
              swept_edge_vol *
              (subcell_mass[(sweep_subcell_index)] +
               grad_mass.x * (swept_edge_centroid.x -
                              subcell_centroids_x[(sweep_subcell_index)]) +
               grad_mass.y * (swept_edge_centroid.y -
                              subcell_centroids_y[(sweep_subcell_index)]) +
               grad_mass.z * (swept_edge_centroid.z -
                              subcell_centroids_z[(sweep_subcell_index)]));
          subcell_mass[(subcell_index)] -= mass_flux;

          /* ADVECT INTERNAL ENERGY DENSITY */

          // For all of the subcell centered quantities, determine the flux.
          vec_t grad_ie = {0.0, 0.0, 0.0};
          calc_gradient(
              sweep_subcell_index, NSUBCELL_NEIGHBOURS,
              sweep_subcell_index * NSUBCELL_NEIGHBOURS, subcells_to_subcells,
              subcell_ie_density, subcell_centroids_x, subcell_centroids_y,
              subcell_centroids_z, (const vec_t(*)[3]) & inv, &grad_ie);

          // Calculate and apply limiter for the gradient
          apply_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                        &grad_mass, &cell_centroid, nodes_x0, nodes_y0,
                        nodes_z0, energy0[(cc)] * density0[(cc)], g_ie_max,
                        g_ie_min);

          // Calculate the flux for internal energy density in the subcell
          const double ie_flux =
              swept_edge_vol *
              (subcell_ie_density[(sweep_subcell_index)] +
               grad_ie.x * (swept_edge_centroid.x -
                            subcell_centroids_x[(sweep_subcell_index)]) +
               grad_ie.y * (swept_edge_centroid.y -
                            subcell_centroids_y[(sweep_subcell_index)]) +
               grad_ie.z * (swept_edge_centroid.z -
                            subcell_centroids_z[(sweep_subcell_index)]));
          subcell_ie_density[(subcell_index)] -= ie_flux;
        }

        // Gather the value back to the cell
        cell_mass[(cc)] += subcell_mass[(subcell_index)];
        energy1[(cc)] += subcell_ie_density[(subcell_index)];
      }
    }

    // Update the volume of the cell to the new rezoned mesh
    calc_volume(cell_to_faces_off, nfaces_by_cell, cells_to_faces,
                faces_to_nodes, faces_to_nodes_offsets, rezoned_nodes_x,
                rezoned_nodes_y, rezoned_nodes_z, &rz_cell_centroid,
                &cell_volume[(cc)]);
  }

  // Print the conservation of mass
  double rz_total_mass = 0.0;
  double rz_total_density = 0.0;
  double rz_total_energy = 0.0;
#pragma omp parallel for reduction(+ : rz_total_mass, rz_total_density,        \
                                   rz_total_energy)
  for (int cc = 0; cc < ncells; ++cc) {
    density0[(cc)] = cell_mass[(cc)] / cell_volume[(cc)];
    energy0[(cc)] = energy1[(cc)] / cell_mass[(cc)];
    rz_total_mass += cell_mass[(cc)];
    rz_total_energy += energy0[(cc)];
    rz_total_density += density0[(cc)];
  }
  printf("Rezoned Total Mass %.12f, Initial Total Mass %.12f\n", rz_total_mass,
         total_mass);
  printf("Rezoned Total Energy %.12f, Initial Total Energy %.12f\n",
         rz_total_energy, total_energy);
  printf("Rezoned Total Density %.12f, Initial Total Density %.12f\n",
         rz_total_density, total_density);

#if 0
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
      for (int cc = 0; cc < 2; ++cc) {
        if (cells[(cc)] == -1) {
          continue;
        }

        const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
        const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

        // Add contributions for both edges attached to our current node
        for (int nn2 = 0; nn2 < 2; ++nn2) {
          velocity_x0[(nn)] +=
              subcell_velocity_x[(subcell_off + node_in_face_c)];
          velocity_y0[(nn)] +=
              subcell_velocity_y[(subcell_off + node_in_face_c)];
          velocity_z0[(nn)] +=
              subcell_velocity_z[(subcell_off + node_in_face_c)];
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
  printf("%.12f %.12f %.12f\n", total_vx, total_vy, total_vz);
#endif // if 0
#endif // if 0
}
