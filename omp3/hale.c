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
    double* boundary_normal_z, double* energy0, double* energy1,
    double* density0, double* density1, double* pressure0, double* pressure1,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* velocity_x1, double* velocity_y1, double* velocity_z1,
    double* subcell_force_x, double* subcell_force_y, double* subcell_force_z,
    double* cell_mass, double* nodal_mass, double* nodal_volumes,
    double* nodal_soundspeed, double* limiter, double* subcell_volume,
    double* subcell_ie_density, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_integrals_x,
    double* subcell_integrals_y, double* subcell_integrals_z,
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
      subcell_velocity_y, subcell_velocity_z, subcell_integrals_x,
      subcell_integrals_y, subcell_integrals_z, nodes_to_faces_offsets,
      nodes_to_faces, faces_to_nodes, faces_to_nodes_offsets, faces_to_cells0,
      faces_to_cells1, cells_to_faces_offsets, cells_to_faces);

  double total_ie = 0.0;
  double total_ie_subcell = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    double subcell_sum = 0.0;
    double subcell_mass_sum = 0.0;
    double volume_by_subcell = 0.0;
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int subcell_index = (cell_to_nodes_off + ss);
      subcell_sum += subcell_ie_density[(subcell_index)];
      volume_by_subcell += subcell_volume[(subcell_index)];
      subcell_mass_sum += subcell_mass[(subcell_index)];
    }
    total_ie += cell_mass[cc] * energy0[cc];
    total_ie_subcell += subcell_sum;
    printf("ie : %.12f = %.12f\n", cell_mass[cc] * energy0[cc], subcell_sum);
  }
  printf("total ie %.12f = %.12f\n", total_ie, total_ie_subcell);

#if 0
  // Calculate all of the subcell centroids, this is precomputed because the
  // reconstruction of the subcell nodes from faces is quite expensive, and will
  // require a significant amount of repetetive compute inside the remapping
  // step due to the fact that the swept regions don't only take from the
  // current subcell, but the neighbouring subcells too
  calc_subcell_centroids(ncells, cells_offsets, cell_centroids_x,
                         cell_centroids_y, cell_centroids_z, cells_to_nodes,
                         subcell_face_offsets, subcells_to_faces,
                         faces_to_nodes_offsets, faces_to_nodes, nodes_x0,
                         nodes_y0, nodes_z0, subcell_centroids_x,
                         subcell_centroids_y, subcell_centroids_z);

  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    vec_t rz_cell_centroid = {0.0, 0.0, 0.0};
    calc_centroid(nsubcells_by_cell, rezoned_nodes_x, rezoned_nodes_y,
                  rezoned_nodes_z, cells_to_nodes, cell_to_nodes_off,
                  &rz_cell_centroid);

    /*
    * Here we are constructing a reference subcell prism for the target face,
    * this reference element is used for all of the faces of the subcell, in
    * fact it's the same for all cells too, so this could be moved into some
    * global space it's going to spill anyway. The reference shape is by
    * construction when considering a swept edge remap.
    */

    const int prism_faces_to_nodes_offsets[] = {0, 4, 8, 12, 16, 20, 24};
    const int prism_faces_to_nodes[] = {0, 1, 2, 3, 0, 1, 5, 4, 0, 4, 7, 3,
                                        1, 5, 6, 2, 4, 5, 6, 7, 3, 2, 6, 7};
    const int prism_to_faces[] = {0, 1, 2, 3, 4, 5};
    double prism_nodes_x[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double prism_nodes_y[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double prism_nodes_z[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // We discover the subcell gradients using a least squares fit for the
    // gradient between the subcell and its neighbours
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int subcell_index = (cell_to_nodes_off + ss);
      const int subcell_node_index = cells_to_nodes[(subcell_index)];
      const int subcell_to_faces_off =
          subcell_face_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcell_face_offsets[(subcell_index + 1)] - subcell_to_faces_off;
      const int subcell_to_subcells_off =
          subcell_face_offsets[(subcell_index)] * 2;

      // We will calculate the swept edge region for the internal and external
      // face here, this relies on the faces being ordered in a ring.
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_index2 =
            (ff == nfaces_by_subcell - 1)
                ? subcells_to_faces[(subcell_to_faces_off)]
                : subcells_to_faces[(subcell_to_faces_off + ff + 1)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
        const int face2_to_nodes_off = faces_to_nodes_offsets[(face_index2)];
        const int nnodes_by_face2 =
            faces_to_nodes_offsets[(face_index2 + 1)] - face2_to_nodes_off;

        /*
         * Determine all of the nodes for the swept edge region inside the
         * current mesh and in the rezoned mesh
         */

        // calculate the face center for the current and rezoned meshes
        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x0, nodes_y0, nodes_z0,
                      faces_to_nodes, face_to_nodes_off, &face_c);
        vec_t face2_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face2, nodes_x0, nodes_y0, nodes_z0,
                      faces_to_nodes, face2_to_nodes_off, &face2_c);

        // calculate the subsequent face center for current and rezoned meshes
        vec_t rz_face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, rezoned_nodes_x, rezoned_nodes_y,
                      rezoned_nodes_z, faces_to_nodes, face_to_nodes_off,
                      &rz_face_c);
        vec_t rz_face2_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face2, rezoned_nodes_x, rezoned_nodes_y,
                      rezoned_nodes_z, faces_to_nodes, face2_to_nodes_off,
                      &rz_face2_c);

        // Determine the offset into the faces_to_nodes array
        int sn_off;
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (subcell_node_index == faces_to_nodes[(face_to_nodes_off + nn)]) {
            sn_off = nn;
          }
        }

        // The half edges are the points between the node at the subcell and
        // the
        // right and left nodes on our current face
        const int l_off = (sn_off == 0) ? nnodes_by_face - 1 : sn_off - 1;
        const int r_off = (sn_off == nnodes_by_face - 1) ? 0 : sn_off + 1;
        const int node_l_index = faces_to_nodes[(face_to_nodes_off + l_off)];
        const int node_r_index = faces_to_nodes[(face_to_nodes_off + r_off)];

        vec_t half_edge_l = {
            0.5 * (nodes_x0[(subcell_node_index)] + nodes_x0[(node_l_index)]),
            0.5 * (nodes_y0[(subcell_node_index)] + nodes_y0[(node_l_index)]),
            0.5 * (nodes_z0[(subcell_node_index)] + nodes_z0[(node_l_index)])};
        vec_t half_edge_r = {
            0.5 * (nodes_x0[(subcell_node_index)] + nodes_x0[(node_r_index)]),
            0.5 * (nodes_y0[(subcell_node_index)] + nodes_y0[(node_r_index)]),
            0.5 * (nodes_z0[(subcell_node_index)] + nodes_z0[(node_r_index)])};

        vec_t rz_half_edge_l = {0.5 * (rezoned_nodes_x[(subcell_node_index)] +
                                       rezoned_nodes_x[(node_l_index)]),
                                0.5 * (rezoned_nodes_y[(subcell_node_index)] +
                                       rezoned_nodes_y[(node_l_index)]),
                                0.5 * (rezoned_nodes_z[(subcell_node_index)] +
                                       rezoned_nodes_z[(node_l_index)])};
        vec_t rz_half_edge_r = {0.5 * (rezoned_nodes_x[(subcell_node_index)] +
                                       rezoned_nodes_x[(node_r_index)]),
                                0.5 * (rezoned_nodes_y[(subcell_node_index)] +
                                       rezoned_nodes_y[(node_r_index)]),
                                0.5 * (rezoned_nodes_z[(subcell_node_index)] +
                                       rezoned_nodes_z[(node_r_index)])};

        /*
         * Construct the swept edge prism for the internal and external face
         * that is described by the above nodes, and determine the weighted
         * volume swept_edge_integrals
         */

        // Firstly we will determine the external swept region
        vec_t nodes = {nodes_x0[(subcell_node_index)],
                       nodes_y0[(subcell_node_index)],
                       nodes_z0[(subcell_node_index)]};
        vec_t rz_nodes = {rezoned_nodes_x[(subcell_node_index)],
                          rezoned_nodes_y[(subcell_node_index)],
                          rezoned_nodes_z[(subcell_node_index)]};

        // Fill in the node list for the swept region's prism
        vec_t prism_centroid = {0.0, 0.0, 0.0};
        construct_external_swept_region(
            &nodes, &rz_nodes, &half_edge_l, &half_edge_r, &rz_half_edge_l,
            &rz_half_edge_r, &face_c, &rz_face_c, &prism_centroid,
            prism_nodes_x, prism_nodes_y, prism_nodes_z);

        // Determine the volume integrals for the prism
        vec_t swept_edge_integrals = {0.0, 0.0, 0.0};
        double swept_edge_vol = 0.0;
        calc_weighted_volume_integrals(
            0, NSUBCELL_FACES, prism_to_faces, prism_faces_to_nodes,
            prism_faces_to_nodes_offsets, prism_nodes_x, prism_nodes_y,
            prism_nodes_z, &prism_centroid, &swept_edge_integrals,
            &swept_edge_vol);

        if (swept_edge_vol > 0.0) {
          printf("swept_edge_vol %.12f\n", swept_edge_vol);

          // Choose three points on the current face
          int fn0 = faces_to_nodes[(face_to_nodes_off + 0)];
          int fn1 = faces_to_nodes[(face_to_nodes_off + 1)];
          int fn2 = faces_to_nodes[(face_to_nodes_off + 2)];

          // Construct a vector from the mesh face to the rezoned face
          vec_t sweep_direction = {rz_face_c.x - face_c.x,
                                   rz_face_c.y - face_c.y,
                                   rz_face_c.z - face_c.z};

          // The normal pointing outward from the face
          vec_t face_normal;
          calc_surface_normal(fn0, fn1, fn2, nodes_x0, nodes_y0, nodes_z0,
                              &cell_centroid, &face_normal);

          // Check if we are sweeping into the current subcell
          int is_internal_sweep = ((sweep_direction.x * face_normal.x +
                                    sweep_direction.y * face_normal.y +
                                    sweep_direction.z * face_normal.z) < 0.0);

          // Depending upon which subcell we are sweeping into, choose the
          // subcell
          // index with which to reconstruct the density
          int sweep_subcell_index =
              (is_internal_sweep ? subcell_index
                                 : subcells_to_subcells[(
                                       subcell_to_subcells_off + ff * 2 + 1)]);

          printf("%.12f %.12f %.12f\n", swept_edge_integrals.x,
                 swept_edge_integrals.y, swept_edge_integrals.z);

          // Calculate the inverse coefficient matrix for the least squares
          // regression of the gradient, which is the same for all quantities.
          vec_t inv[3];
          int nsubcells_by_subcell;
          int subcell_to_subcells_off;
          calc_inverse_coefficient_matrix(
              sweep_subcell_index, subcell_face_offsets,
              subcells_to_subcells, subcell_integrals_x, subcell_integrals_y,
              subcell_integrals_z, subcell_centroids_x, subcell_centroids_y,
              subcell_centroids_z, subcell_volume, &nsubcells_by_subcell,
              &subcell_to_subcells_off, &inv);

          for (int ii = 0; ii < 3; ++ii) {
            printf("inv (%.6f %.6f %.6f)\n", inv[ii].x, inv[ii].y, inv[ii].z);
          }

          // For all of the subcell centered quantities, determine the flux.
          vec_t mass_gradient;
          calc_gradient(sweep_subcell_index, nsubcells_by_subcell,
                        subcell_to_subcells_off, subcells_to_subcells,
                        subcell_mass, subcell_integrals_x, subcell_integrals_y,
                        subcell_integrals_z, subcell_volume,
                        (const vec_t(*)[3]) & inv, &mass_gradient);

          printf("mass gradient %.12f %.12f %.12f\n", mass_gradient.x,
                 mass_gradient.y, mass_gradient.z);

          // Calculate the flux for internal energy density in the subcell
          const double mass_flux =
              subcell_mass[(sweep_subcell_index)] * swept_edge_vol +
              mass_gradient.x * swept_edge_integrals.x +
              mass_gradient.y * swept_edge_integrals.y +
              mass_gradient.z * swept_edge_integrals.z -
              swept_edge_vol * (mass_gradient.x *
                                    subcell_centroids_x[(sweep_subcell_index)] +
                                mass_gradient.y *
                                    subcell_centroids_y[(sweep_subcell_index)] +
                                mass_gradient.z *
                                    subcell_centroids_z[(sweep_subcell_index)]);

          subcell_mass[(subcell_index)] -= mass_flux;

          printf("subcell %d face %d mass flux %.12f\n", subcell_index,
                 face_index, mass_flux);

#if 0
          // For all of the subcell centered quantities, determine the flux.
          vec_t ie_gradient;
          calc_gradient(
              sweep_subcell_index, nsubcells_by_subcell,
              subcell_to_subcells_off, subcells_to_subcells, subcell_ie_density,
              subcell_integrals_x, subcell_integrals_y, subcell_integrals_z,
              subcell_volume, (const vec_t(*)[3]) & inv, &ie_gradient);

          // Calculate the flux for internal energy density in the subcell
          const double ie_flux =
              subcell_ie_density[(sweep_subcell_index)] * swept_edge_vol +
              ie_gradient.x * swept_edge_integrals.x -
              ie_gradient.x * subcell_centroids_x[(sweep_subcell_index)] +
              ie_gradient.y * swept_edge_integrals.y -
              ie_gradient.x * subcell_centroids_y[(sweep_subcell_index)] +
              ie_gradient.z * swept_edge_integrals.z -
              ie_gradient.x * subcell_centroids_z[(sweep_subcell_index)];

          subcell_ie_density[(subcell_index)] -= ie_flux;


        vec_t ke_gradient;
        calc_gradient(sweep_subcell_index, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_kinetic_energy, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &ke_gradient);

        vec_t xmomentum_gradient;
        calc_gradient(sweep_subcell_index, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_velocity_x, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &xmomentum_gradient);

        vec_t ymomentum_gradient;
        calc_gradient(sweep_subcell_index, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_velocity_y, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &ymomentum_gradient);

        vec_t zmomentum_gradient;
        calc_gradient(sweep_subcell_index, nsubcells_by_subcell,
                      subcell_to_subcells_off, subcells_to_subcells,
                      subcell_velocity_z, subcell_integrals_x,
                      subcell_integrals_y, subcell_integrals_z, subcell_volume,
                      &inv, &zmomentum_gradient);
#endif // if 0
        }

/*
 * Perform the swept edge remaps for the subcells...
 */

#if 0
        // Secondly we will determine the internal swept region

        // TODO: Fairly sure that this returns the opposite value to what we
        // are expecting later on...
        // This is duplicated work
        vec_t normal;
        calc_normal(fn0, fn1, fn2, nodes_x0, nodes_y0, nodes_z0, &normal);
        const int face_clockwise = check_normal_orientation(
            fn0, nodes_x0, nodes_y0, nodes_z0, &cell_centroid, &normal);

        construct_internal_swept_region(
            face_clockwise, &half_edge_l, &half_edge_r, &rz_half_edge_l,
            &rz_half_edge_r, &face_c, &face2_c, &rz_face_c, &rz_face2_c,
            &cell_centroid, &rz_cell_centroid, &prism_centroid, prism_nodes_x,
            prism_nodes_y, prism_nodes_z);
        calc_weighted_volume_integrals(
            0, NSUBCELL_FACES, prism_to_faces, prism_faces_to_nodes,
            prism_faces_to_nodes_offsets, prism_nodes_x, prism_nodes_y,
            prism_nodes_z, &prism_centroid, &swept_edge_integrals,
            &swept_edge_vol);

        // Choose three points on the face
        fn0 = faces_to_nodes[(face2_to_nodes_off + 0)];
        fn1 = faces_to_nodes[(face2_to_nodes_off + 1)];
        fn2 = faces_to_nodes[(face2_to_nodes_off + 2)];

        // Construct a vector from the mesh face to the rezoned face
        sweep_direction.x = rz_face_c.x - face_c.x;
        sweep_direction.y = rz_face_c.y - face_c.y;
        sweep_direction.z = rz_face_c.z - face_c.z;

        // The normal pointing outward from the face
        calc_surface_normal(fn0, fn1, fn2, nodes_x0, nodes_y0, nodes_z0,
                            &cell_centroid, &face_normal);

        is_internal_sweep = ((sweep_direction.x * face_normal.x +
                              sweep_direction.y * face_normal.y +
                              sweep_direction.z * face_normal.z) > 0.0);

        sweep_subcell_index =
            (is_internal_sweep
                 ? subcell_index
                 : subcells_to_subcells[(subcell_to_subcells_off + ff * 2)]);
#endif // if 0
      }
    }
  }
#endif // if 0
}

#if 0
const int tet_faces_to_nodes_offsets[] = {0, 4, 8, 12, 16, 20, 24};
const int tet_faces_to_nodes[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4,
  6, 2, 3, 7, 1, 2, 6, 5, 0, 4, 7, 3};
const int tet_to_faces[] = {0, 1, 2, 3, 4, 5};
double tet_nodes_x[] = {1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0};
double tet_nodes_y[] = {1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0};
double tet_nodes_z[] = {1.1, 1.1, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0};

vec_t tet_integrals;
vec_t tet_centroid = {1.5, 1.5, 1.5};
double vol;
calc_weighted_volume_integrals(0, NSUBCELL_FACES, tet_to_faces,
    tet_faces_to_nodes, tet_faces_to_nodes_offsets,
    tet_nodes_x, tet_nodes_y, tet_nodes_z,
    &tet_centroid, &tet_integrals, &vol);

printf("%.12f %.12f %.12f %.12f\n", tet_integrals.x, tet_integrals.y,
    tet_integrals.z, vol);

return;
#if 0
  nodes_x0[0] += 0.3;
  nodes_x0[1] += 0.3;
  nodes_x0[2] += 0.3;
  nodes_x0[3] += 0.3;

  init_cell_centroids(ncells, cells_offsets, cells_to_nodes, nodes_x0, nodes_y0,
                      nodes_z0, cell_centroids_x, cell_centroids_y,
                      cell_centroids_z);

  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    vec_t integrals = {0.0, 0.0, 0.0};
    double vol = 0.0;
    calc_weighted_volume_integrals(cell_to_faces_off, nfaces_by_cell,
                                   cells_to_faces, faces_to_nodes,
                                   faces_to_nodes_offsets, nodes_x0, nodes_y0,
                                   nodes_z0, &cell_centroid, &integrals, &vol);

    density0[(cc)] = cell_mass[(cc)] / vol;
  }
#endif // if 0

#endif // if 0
