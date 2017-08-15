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
    double* subcell_velocity_z, double* subcell_centroid_x,
    double* subcell_centroid_y, double* subcell_centroid_z,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, double* subcell_kinetic_energy,
    double* rezoned_nodes_x, double* rezoned_nodes_y, double* rezoned_nodes_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* subcell_face_offsets,
    int* subcells_to_subcells) {

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
#endif // if 0

  // Gather the subcell quantities for mass, internal and kinetic energy
  // density, and momentum
  gather_subcell_quantities(
      ncells, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      cells_to_nodes, cells_offsets, nodes_x0, nodes_y0, nodes_z0, energy0,
      density0, velocity_x0, velocity_y0, velocity_z0, cell_mass,
      subcell_volume, subcell_ie_density, subcell_mass, subcell_velocity_x,
      subcell_velocity_y, subcell_velocity_z, subcell_centroid_x,
      subcell_centroid_y, subcell_centroid_z, cell_volume, subcell_face_offsets,
      nodes_to_faces_offsets, nodes_to_faces, faces_to_nodes,
      faces_to_nodes_offsets, faces_to_cells0, faces_to_cells1,
      cells_to_faces_offsets, cells_to_faces);

#if 0
  for (int nn = 0; nn < nnodes; ++nn) {
    rezoned_nodes_x[(nn)] *= 0.5;
    rezoned_nodes_y[(nn)] *= 0.5;
    rezoned_nodes_z[(nn)] *= 0.5;
  }

  // Calculate all of the subcell centroids, this is precomputed because the
  // reconstruction of the subcell nodes from faces is quite expensive, and will
  // require a significant amount of repetetive compute inside the remapping
  // step due to the fact that the swept regions don't only take from the
  // current subcell, but the neighbouring subcells too
  // Calculates the subcells of all centroids
  calc_subcell_centroids(
      ncells, cell_centroids_x, cell_centroids_y, cell_centroids_z,
      subcell_face_offsets, faces_to_nodes_offsets, faces_to_nodes,
      cells_to_faces_offsets, cells_to_faces, nodes_x0, nodes_y0, nodes_z0,
      subcell_centroids_x, subcell_centroids_y, subcell_centroids_z);

  /* LOOP OVER CELLS */
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

    // The cell mass is zeroed out to gather later
    cell_mass[(cc)] = 0.0;

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

        vec_t normal;
        const int face_clockwise = calc_surface_normal(
            n0, n1, n2, nodes_x0, nodes_y0, nodes_z0, &cell_centroid, &normal);
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
        const int nnodes_on_tet_face = 3;
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
            {cell_centroid.x, cell_centroid.y, cell_centroid.z}};

        /* LOOP OVER SUBCELL FACES */
        for (int pp = 0; pp < NTET_FACES; ++pp) {
          // Describe the subcell prism
          const int prism_faces_to_nodes_offsets[] = {0, 3, 7, 11, 15, 18};
          const int prism_faces_to_nodes[] = {0, 1, 2, 1, 4, 5, 2, 0, 2,
                                              5, 3, 0, 3, 4, 1, 4, 3, 5};
          const int prism_to_faces[] = {0, 1, 2, 3, 4};
          double prism_nodes_x[] = {
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 0)])].x,
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 1)])].x,
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 2)])].x,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 0)])].x,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 1)])].x,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 2)])].x};
          double prism_nodes_y[] = {
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 0)])].y,
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 1)])].y,
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 2)])].y,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 0)])].y,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 1)])].y,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 2)])].y};
          double prism_nodes_z[] = {
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 0)])].z,
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 1)])].z,
              subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 2)])].z,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 0)])].z,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 1)])].z,
              rz_subcell[(subcell_to_faces[(pp * nnodes_on_tet_face + 2)])].z};

          // Determine the prism's centroid
          vec_t prism_centroid = {0.0, 0.0, 0.0};
          calc_prism_centroid(&prism_centroid, prism_nodes_x, prism_nodes_y,
                              prism_nodes_z);

          double swept_edge_vol = 0.0;
          vec_t swept_edge_integrals = {0.0, 0.0, 0.0};
          calc_weighted_volume_integrals(
              0, NPRISM_FACES, prism_to_faces, prism_faces_to_nodes,
              prism_faces_to_nodes_offsets, prism_nodes_x, prism_nodes_y,
              prism_nodes_z, &prism_centroid, &swept_edge_integrals,
              &swept_edge_vol);

          // Ignore faces that haven't changed.
          if (swept_edge_vol <= 0.0) {
            continue;
          }

#if 0
          // Check if we are sweeping into the current subcell
          int is_internal_sweep = ((sweep_direction.x * face_normal.x +
                                    sweep_direction.y * face_normal.y +
                                    sweep_direction.z * face_normal.z) < 0.0);

          // Depending upon which subcell we are sweeping into, choose the
          // subcell index with which to reconstruct the density
          int subcell_neighbour_index =
              subcells_to_subcells[(subcell_index * NSUBCELL_NEIGHBOURS + pp)];
          int sweep_subcell_index =
              (is_internal_sweep ? subcell_index : subcell_neighbour_index);

          // Calculate the inverse coefficient matrix for the least squares
          // regression of the gradient, which is the same for all quantities.
          vec_t inv[3];
          int nsubcells_by_subcell = NSUBCELL_NEIGHBOURS;
          int subcell_to_subcells_off = subcell_index * NSUBCELL_NEIGHBOURS;
          calc_inverse_coefficient_matrix(
              sweep_subcell_index, subcells_to_subcells, subcell_centroid_x,
              subcell_centroid_y, subcell_centroid_z, subcell_centroids_x,
              subcell_centroids_y, subcell_centroids_z, subcell_volume,
              nsubcells_by_subcell, subcell_to_subcells_off, &inv);

          // Only perform the sweep on the external face if it isn't a boundary
          if (subcell_neighbour_index != -1) {

            // For all of the subcell centered quantities, determine the flux.
            vec_t mass_gradient;
            calc_gradient(
                sweep_subcell_index, nsubcells_by_subcell,
                subcell_to_subcells_off, subcells_to_subcells, subcell_mass,
                subcell_centroid_x, subcell_centroid_y, subcell_centroid_z,
                subcell_volume, (const vec_t(*)[3]) & inv, &mass_gradient);

            // Calculate the flux for internal energy density in the subcell
            const double mass_flux =
                subcell_mass[(sweep_subcell_index)] * swept_edge_vol +
                mass_gradient.x * swept_edge_integrals.x +
                mass_gradient.y * swept_edge_integrals.y +
                mass_gradient.z * swept_edge_integrals.z -
                swept_edge_vol *
                    (mass_gradient.x *
                         subcell_centroids_x[(sweep_subcell_index)] +
                     mass_gradient.y *
                         subcell_centroids_y[(sweep_subcell_index)] +
                     mass_gradient.z *
                         subcell_centroids_z[(sweep_subcell_index)]);

            subcell_mass[(subcell_index)] -= mass_flux;
          }

          // Gather the value back to the cell
          cell_mass[(cc)] += subcell_mass[(subcell_index)];
#endif // if 0
        }
      }
    }
  }

  double tm = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    tm += cell_mass[cc];
  }
  printf("rezoned total mass %.12f\n", tm);

  //
  //
  //
  //
  //
  //
  //
  //
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x0[nn] = rezoned_nodes_x[(nn)];
    nodes_y0[nn] = rezoned_nodes_y[(nn)];
    nodes_z0[nn] = rezoned_nodes_z[(nn)];
  }
#endif // if 0
}

#if 0
/*
 *
 * Need to perform the signed volume calculation, not this stupid
 * below.
 *
 */

// Construct a vector from the mesh face to the rezoned face
vec_t sweep_direction = {rz_face_c.x - face_c.x,
  rz_face_c.y - face_c.y,
  rz_face_c.z - face_c.z};

// Choose three points on the current face
const int fn0 = faces_to_nodes[(face_to_nodes_off + 0)];
const int fn1 = faces_to_nodes[(face_to_nodes_off + 1)];
const int fn2 = faces_to_nodes[(face_to_nodes_off + 2)];

// The normal pointing outward from the face
vec_t face_normal;
calc_surface_normal(fn0, fn1, fn2, nodes_x0, nodes_y0, nodes_z0,
    &cell_centroid, &face_normal);

#endif // if 0
