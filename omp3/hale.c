#include "hale.h"
#include "../../shared.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_3d(Mesh* mesh, HaleData* hale_data,
                                 UnstructuredMesh* umesh, const int timestep) {

#if 0
  // Describe the subcell node layout
  printf("\nPerforming the Lagrangian Phase\n");

  // Perform the Lagrangian phase of the ALE algorithm where the mesh will move
  // due to the pressure (ideal gas) and artificial viscous forces
  lagrangian_phase(
      mesh, umesh->ncells, umesh->nnodes, hale_data->visc_coeff1,
      hale_data->visc_coeff2, umesh->cell_centroids_x, umesh->cell_centroids_y,
      umesh->cell_centroids_z, umesh->cells_to_nodes, umesh->cells_offsets,
      umesh->nodes_to_cells, umesh->nodes_offsets, umesh->nodes_x0,
      umesh->nodes_y0, umesh->nodes_z0, umesh->nodes_x1, umesh->nodes_y1,
      umesh->nodes_z1, umesh->boundary_index, umesh->boundary_type,
      umesh->boundary_normal_x, umesh->boundary_normal_y,
      umesh->boundary_normal_z, hale_data->energy0, hale_data->energy1,
      hale_data->density0, hale_data->density1, hale_data->pressure0,
      hale_data->pressure1, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0, hale_data->velocity_x1, hale_data->velocity_y1,
      hale_data->velocity_z1, hale_data->subcell_force_x,
      hale_data->subcell_force_y, hale_data->subcell_force_z,
      hale_data->cell_mass, hale_data->nodal_mass, hale_data->nodal_volumes,
      hale_data->nodal_soundspeed, hale_data->limiter,
      umesh->nodes_to_faces_offsets, umesh->nodes_to_faces,
      umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
      umesh->faces_to_cells0, umesh->faces_to_cells1,
      umesh->cells_to_faces_offsets, umesh->cells_to_faces);

  if (hale_data->perform_remap) {
    printf("\nPerforming Gathering Phase\n");

    double total_ie = 0.0;
    double total_mass = 0.0;
    vec_t initial_momentum = {0.0, 0.0, 0.0};

    // gathers all of the subcell quantities on the mesh
    gather_subcell_quantities(
        umesh->ncells, umesh->nnodes, hale_data->nnodes_by_subcell,
        hale_data->nodal_volumes, hale_data->nodal_mass,
        umesh->cell_centroids_x, umesh->cell_centroids_y,
        umesh->cell_centroids_z, umesh->nodes_to_cells, umesh->nodes_x0,
        umesh->nodes_y0, umesh->nodes_z0, hale_data->energy0,
        hale_data->density0, hale_data->velocity_x0, hale_data->velocity_y0,
        hale_data->velocity_z0, hale_data->cell_mass, hale_data->subcell_volume,
        hale_data->subcell_ie_mass, hale_data->subcell_momentum_x,
        hale_data->subcell_momentum_y, hale_data->subcell_momentum_z,
        hale_data->subcell_centroids_x, hale_data->subcell_centroids_y,
        hale_data->subcell_centroids_z, hale_data->cell_volume,
        hale_data->subcells_to_faces_offsets, umesh->faces_to_nodes,
        umesh->faces_to_nodes_offsets, umesh->faces_to_cells0,
        umesh->faces_to_cells1, umesh->cells_to_faces_offsets,
        umesh->cells_to_faces, hale_data->subcells_to_faces,
        umesh->nodes_offsets, umesh->cells_offsets, umesh->cells_to_nodes,
        umesh->nodes_to_nodes_offsets, umesh->nodes_to_nodes,
        &initial_momentum);

    printf("\nPerforming Remap Phase\n");

    // Performs a remap and some scattering of the subcell values
    remap_phase(
        umesh->ncells, umesh->cells_offsets, umesh->nodes_x0, umesh->nodes_y0,
        umesh->nodes_z0, hale_data->rezoned_nodes_x, hale_data->rezoned_nodes_y,
        hale_data->rezoned_nodes_z, hale_data->subcell_momentum_x,
        hale_data->subcell_momentum_y, hale_data->subcell_momentum_z,
        umesh->cells_to_nodes, umesh->faces_to_nodes_offsets,
        umesh->faces_to_nodes, hale_data->subcells_to_faces_offsets,
        hale_data->subcells_to_faces, hale_data->subcells_to_subcells_offsets,
        hale_data->subcells_to_subcells, umesh->faces_to_cells0,
        umesh->faces_to_cells1, hale_data->subcell_momentum_flux_x,
        hale_data->subcell_momentum_flux_y, hale_data->subcell_momentum_flux_z,
        hale_data->subcell_centroids_x, hale_data->subcell_centroids_y,
        hale_data->subcell_centroids_z, hale_data->subcell_volume,
        hale_data->subcell_mass, hale_data->subcell_mass_flux,
        hale_data->subcell_ie_mass, hale_data->subcell_ie_mass_flux);

    printf("\nEulerian Mesh Rezone\n");

    // Finalise the mesh rezone
    apply_mesh_rezoning(umesh->nnodes, hale_data->rezoned_nodes_x,
                        hale_data->rezoned_nodes_y, hale_data->rezoned_nodes_z,
                        umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0);

    // Determine the new cell centroids
    init_cell_centroids(umesh->ncells, umesh->cells_offsets,
                        umesh->cells_to_nodes, umesh->nodes_x0, umesh->nodes_y0,
                        umesh->nodes_z0, umesh->cell_centroids_x,
                        umesh->cell_centroids_y, umesh->cell_centroids_z);

    printf("\nPerforming the Scattering Phase\n");

    init_subcell_data_structures(mesh, hale_data, umesh);
    write_unstructured_to_visit_3d(
        hale_data->nsubcell_nodes, umesh->ncells * hale_data->nsubcells_by_cell,
        timestep * 2 + 1, hale_data->subcell_nodes_x,
        hale_data->subcell_nodes_y, hale_data->subcell_nodes_z,
        hale_data->subcells_to_nodes, hale_data->subcell_mass_flux, 0, 1);
    // Perform the scatter step of the ALE remapping algorithm
    scatter_phase(
        umesh->ncells, umesh->nnodes, &initial_momentum,
        hale_data->rezoned_nodes_x, hale_data->rezoned_nodes_y,
        hale_data->rezoned_nodes_z, hale_data->cell_volume, hale_data->energy0,
        hale_data->density0, hale_data->velocity_x0, hale_data->velocity_y0,
        hale_data->velocity_z0, hale_data->cell_mass, hale_data->nodal_mass,
        hale_data->subcell_ie_mass, hale_data->subcell_mass,
        hale_data->subcell_ie_mass_flux, hale_data->subcell_mass_flux,
        hale_data->subcell_momentum_x, hale_data->subcell_momentum_y,
        hale_data->subcell_momentum_z, hale_data->subcell_momentum_flux_x,
        hale_data->subcell_momentum_flux_y, hale_data->subcell_momentum_flux_z,
        umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
        umesh->cells_to_faces_offsets, umesh->cells_to_faces,
        umesh->nodes_offsets, umesh->nodes_to_cells, umesh->cells_offsets,
        umesh->cells_to_nodes, &total_mass, &total_ie);
  }
  write_unstructured_to_visit_3d(
      umesh->nnodes, umesh->ncells, timestep, umesh->nodes_x0, umesh->nodes_y0,
      umesh->nodes_z0, umesh->cells_to_nodes, hale_data->density0, 0, 1);
#endif // if 0

  double a = 0.0;
  int c_to_n[] = {0, 1, 2, 3, 4, 5, 6, 7};
  const int swept_edge_to_faces[] = {0, 1, 2, 3, 4, 5};
  const int swept_edge_faces_to_nodes_offsets[] = {0, 4, 8, 12, 16, 20, 24};
  const int swept_edge_faces_to_nodes[] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 4, 7, 3,
                                           7, 6, 2, 3, 1, 2, 6, 5, 0, 1, 5, 4};
  double nodes_x[] = {6.10529301274715e-01, 6.56414250371847e-01,
                      6.56414250796353e-01, 6.10529302391724e-01,
                      6.00000000000000e-01, 6.50000000000000e-01,
                      6.50000000000000e-01, 6.00000000000000e-01};

  double nodes_y[] = {8.00000000015110e-01, 7.99999999979950e-01,
                      7.99999999978640e-01, 8.00000000015553e-01,
                      8.00000000000000e-01, 8.00000000000000e-01,
                      8.00000000000000e-01, 8.00000000000000e-01};

  double nodes_z[] = {8.99999999824102e-01, 8.99999999085645e-01,
                      8.49999999531488e-01, 8.49999999920049e-01,
                      9.00000000000000e-01, 9.00000000000000e-01,
                      8.50000000000000e-01, 8.50000000000000e-01};

  double swept_edge_vol = 0.0;
  vec_t swept_edge_c = {0.0, 0.0, 0.0};
  calc_centroid(2 * NNODES_BY_SUBCELL_FACE, nodes_x, nodes_y, nodes_z,
                swept_edge_faces_to_nodes, 0, &swept_edge_c);
  calc_volume(0, 2 + NNODES_BY_SUBCELL_FACE, swept_edge_to_faces,
              swept_edge_faces_to_nodes, swept_edge_faces_to_nodes_offsets,
              nodes_x, nodes_y, nodes_z, &swept_edge_c, &swept_edge_vol);
  printf("sec %.12e %.12e %.12e\n", swept_edge_c.x, swept_edge_c.y,
         swept_edge_c.z);

  write_unstructured_to_visit_3d(8, 1, 10000, nodes_x, nodes_y, nodes_z, c_to_n,
                                 &a, 0, 1);
#if 0
  // Contributes a face to the volume of some cell
  int nfaces_by_cell = 2 + NNODES_BY_SUBCELL_FACE;
  for (int ff = 0; ff < nfaces_by_cell; ++ff) {
    const int face_index = swept_edge_to_faces[(ff)];
    const int face_to_nodes_off =
        swept_edge_faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        swept_edge_faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    // Determine the outward facing unit normal vector
    vec_t face_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z,
                  swept_edge_faces_to_nodes, face_to_nodes_off, &face_c);

    double tn_x[3];
    double tn_y[3];
    double tn_z[3];

    vec_t face_normal = {0.0, 0.0, 0.0};

    // We have a triangle per edge, which are per node on the face
    double face_vol = 0.0;
    for (int tt = 0; tt < nnodes_by_face; ++tt) {
      const int next_node = (tt == nnodes_by_face - 1) ? 0 : tt + 1;
      const int n0 = swept_edge_faces_to_nodes[(face_to_nodes_off + tt)];
      const int n1n =
          swept_edge_faces_to_nodes[(face_to_nodes_off + next_node)];
      const int faces_to_nodes_tri[3] = {0, 1, 2};

      // Construct the face triangle associated with the node
      tn_x[0] = nodes_x[(n0)];
      tn_y[0] = nodes_y[(n0)];
      tn_z[0] = nodes_z[(n0)];
      tn_x[1] = nodes_x[(n1n)];
      tn_y[1] = nodes_y[(n1n)];
      tn_z[1] = nodes_z[(n1n)];
      tn_x[2] = face_c.x;
      tn_y[2] = face_c.y;
      tn_z[2] = face_c.z;

      vec_t tnormal = {0.0, 0.0, 0.0};
      calc_unit_normal(0, 1, 2, tn_x, tn_y, tn_z, &tnormal);

      printf("triangle %d face %d\n", tt, ff);
      for (int ii = 0; ii < 3; ++ii)
        printf("%.12e %.12e %.12e\n", tn_x[ii], tn_y[ii], tn_z[ii]);

      printf("%.12e %.12e %.12e\n", tnormal.x, tnormal.y, tnormal.z);

      // Sum the normal
      face_normal.x += tnormal.x;
      face_normal.y += tnormal.y;
      face_normal.z += tnormal.z;

      // The projection of the normal vector onto a point on the face
      double omega = -(tnormal.x * tn_x[(2)] + tnormal.y * tn_y[(2)] +
                       tnormal.z * tn_z[(2)]);
      printf("omega %.12e\n", omega);

      // Select a basis maximising gamma
      int basis;
      if (fabs(tnormal.x) > fabs(tnormal.y)) {
        basis = (fabs(tnormal.x) > fabs(tnormal.z)) ? YZX : XYZ;
      } else {
        basis = (fabs(tnormal.z) > fabs(tnormal.y)) ? XYZ : ZXY;
      }

      if (basis == XYZ) {
        calc_face_integrals(3, 0, omega, faces_to_nodes_tri, tn_x, tn_y,
                            tnormal, &face_vol);
      } else if (basis == YZX) {
        dswap(tnormal.x, tnormal.y);
        dswap(tnormal.y, tnormal.z);
        calc_face_integrals(3, 0, omega, faces_to_nodes_tri, tn_y, tn_z,
                            tnormal, &face_vol);
      } else if (basis == ZXY) {
        dswap(tnormal.x, tnormal.y);
        dswap(tnormal.x, tnormal.z);
        calc_face_integrals(3, 0, omega, faces_to_nodes_tri, tn_z, tn_x,
                            tnormal, &face_vol);
      }
      printf("%.12e\n", face_vol);
    }

    face_normal.x /= nnodes_by_face;
    face_normal.y /= nnodes_by_face;
    face_normal.z /= nnodes_by_face;
    vec_t face_to_cell_c = {swept_edge_c.x - face_c.x,
                            swept_edge_c.y - face_c.y,
                            swept_edge_c.z - face_c.z};

    double dot =
        (face_to_cell_c.x * face_normal.x + face_to_cell_c.y * face_normal.y +
         face_to_cell_c.z * face_normal.z);
    printf("dot %.12e\n", dot);
    const int face_clockwise = dot > 0.0;
    printf("face clockwise %d\n", face_clockwise);
    swept_edge_vol += (face_clockwise) ? -face_vol : face_vol;
  }

#if 0
    double nodes_x[] = {4.56458682715267e-01, 4.56458682731202e-01,
                        4.56458682714134e-01, 4.56458682697067e-01,
                        4.50000000000000e-01, 4.50000000000000e-01,
                        4.50000000000000e-01, 4.50000000000000e-01};

    double nodes_y[] = {9.99999998757641e-02, 9.99999998854356e-02,
                        1.49999999942718e-01, 1.49999999937882e-01,
                        1.00000000000000e-01, 1.00000000000000e-01,
                        1.50000000000000e-01, 1.50000000000000e-01};

    double nodes_z[] = {8.00000000000000e-01, 8.50000000052446e-01,
                        8.50000000057282e-01, 8.00000000000000e-01,
                        8.00000000000000e-01, 8.50000000000000e-01,
                        8.50000000000000e-01, 8.00000000000000e-01};

#if 0
  double nodes_x[] = {5.12163493437608e-01, 5.12163493535731e-01,
                      4.56458682697067e-01, 4.56458682715267e-01,
                      5.00000000000000e-01, 5.00000000000000e-01,
                      4.50000000000000e-01, 4.50000000000000e-01};
  double nodes_y[] = {9.99999998254011e-02, 1.49999999912701e-01,
                      1.49999999937882e-01, 9.99999998757641e-02,
                      1.00000000000000e-01, 1.50000000000000e-01,
                      1.50000000000000e-01, 1.00000000000000e-01};
  double nodes_z[] = {8.00000000000000e-01, 8.00000000000000e-01,
                      8.00000000000000e-01, 8.00000000000000e-01,
                      8.00000000000000e-01, 8.00000000000000e-01,
                      8.00000000000000e-01, 8.00000000000000e-01};
#endif // if 0

#endif // if 0
#endif // if 0
  printf("swept_edge_vol %.12e\n", swept_edge_vol);
}
