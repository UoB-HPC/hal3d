#include "../../comms.h"
#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>

// Performs a remap and some scattering of the subcell values
void remap_phase(UnstructuredMesh* umesh, HaleData* hale_data) {

  // Advects mass and energy through the subcell faces using swept edge approx
  perform_advection(
      umesh->ncells, umesh->cells_offsets, umesh->nodes_x0, umesh->nodes_y0,
      umesh->nodes_z0, hale_data->rezoned_nodes_x, hale_data->rezoned_nodes_y,
      hale_data->rezoned_nodes_z, umesh->cells_to_nodes,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->faces_cclockwise_cell, hale_data->subcells_to_faces_offsets,
      hale_data->subcells_to_faces, hale_data->subcells_to_subcells_offsets,
      hale_data->subcells_to_subcells, hale_data->subcell_centroids_x,
      hale_data->subcell_centroids_y, hale_data->subcell_centroids_z,
      umesh->faces_to_cells0, umesh->faces_to_cells1, hale_data->subcell_volume,
      hale_data->subcell_momentum_flux_x, hale_data->subcell_momentum_flux_y,
      hale_data->subcell_momentum_flux_z, hale_data->subcell_momentum_x,
      hale_data->subcell_momentum_y, hale_data->subcell_momentum_z,
      hale_data->subcell_mass, hale_data->subcell_mass_flux,
      hale_data->subcell_ie_mass, hale_data->subcell_ie_mass_flux);
}

// Advects mass and energy through the subcell faces using swept edge approx
void perform_advection(
    const int ncells, const int* cells_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const double* rezoned_nodes_x,
    const double* rezoned_nodes_y, const double* rezoned_nodes_z,
    const int* cells_to_nodes, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const int* faces_cclockwise_cell,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* subcells_to_subcells_offsets, const int* subcells_to_subcells,
    const double* subcell_centroids_x, const double* subcell_centroids_y,
    const double* subcell_centroids_z, const int* faces_to_cells0,
    const int* faces_to_cells1, double* subcell_volume,
    double* subcell_momentum_flux_x, double* subcell_momentum_flux_y,
    double* subcell_momentum_flux_z, const double* subcell_momentum_x,
    const double* subcell_momentum_y, const double* subcell_momentum_z,
    double* subcell_mass, double* subcell_mass_flux, double* subcell_ie_mass,
    double* subcell_ie_mass_flux) {

#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_c);
    vec_t rz_cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, rezoned_nodes_x, rezoned_nodes_y,
                  rezoned_nodes_z, cells_to_nodes, cell_to_nodes_off,
                  &rz_cell_c);

    // Looping over corner subcells here
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      vec_t subcell_c = {subcell_centroids_x[(subcell_index)],
                         subcell_centroids_y[(subcell_index)],
                         subcell_centroids_z[(subcell_index)]};

      // Consider all faces attached to node
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
        const int neighbour_cc = (faces_to_cells0[(face_index)] == cc)
                                     ? faces_to_cells1[(face_index)]
                                     : faces_to_cells0[(face_index)];

        // The face centroid is the same for all nodes on the face
        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                      face_to_nodes_off, &face_c);
        vec_t rz_face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, rezoned_nodes_x, rezoned_nodes_y,
                      rezoned_nodes_z, faces_to_nodes, face_to_nodes_off,
                      &rz_face_c);

        // Determine the position of the node in the face list of nodes
        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);
        const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
        const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
        const int rnode_off = (face_clockwise ? prev_node : next_node);
        const int lnode_off = (face_clockwise ? next_node : prev_node);
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];
        const int lnode_index = faces_to_nodes[(face_to_nodes_off + lnode_off)];
        const int swept_edge_to_faces[] = {0, 1, 2, 3, 4, 5};
        const int swept_edge_faces_to_nodes[] = {0, 1, 2, 3, 4, 5, 6, 7,
                                                 0, 3, 7, 4, 7, 6, 2, 3,
                                                 1, 5, 6, 2, 0, 4, 5, 1};
        const int swept_edge_faces_to_nodes_offsets[] = {0,  4,  8, 12,
                                                         16, 20, 24};

        /* INTERNAL FACE */

        const int r_face_off = (ff == nfaces_by_subcell - 1) ? 0 : ff + 1;
        const int lface_off = (ff == 0) ? nfaces_by_subcell - 1 : ff - 1;
        const int r_face_index =
            subcells_to_faces[(subcell_to_faces_off + r_face_off)];
        const int lface_index =
            subcells_to_faces[(subcell_to_faces_off + lface_off)];
        const int r_face_to_nodes_off = faces_to_nodes_offsets[(r_face_index)];
        const int lface_to_nodes_off = faces_to_nodes_offsets[(lface_index)];
        const int nnodes_by_r_face =
            faces_to_nodes_offsets[(r_face_index + 1)] - r_face_to_nodes_off;
        const int nnodes_by_lface =
            faces_to_nodes_offsets[(lface_index + 1)] - lface_to_nodes_off;

        vec_t r_iface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_r_face, nodes_x, nodes_y, nodes_z,
                      faces_to_nodes, r_face_to_nodes_off, &r_iface_c);

        const int r_face_clockwise =
            (faces_cclockwise_cell[(r_face_index)] != cc);

        // Determine the position of the node in the face list of nodes
        for (nn2 = 0; nn2 < nnodes_by_r_face; ++nn2) {
          if (faces_to_nodes[(r_face_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

        const int r_face_next_node =
            (nn2 == nnodes_by_r_face - 1) ? 0 : nn2 + 1;
        const int r_face_prev_node =
            (nn2 == 0) ? nnodes_by_r_face - 1 : nn2 - 1;
        const int r_face_rnode_off =
            (r_face_clockwise ? r_face_prev_node : r_face_next_node);
        const int r_face_rnode_index =
            faces_to_nodes[(r_face_to_nodes_off + r_face_rnode_off)];

        vec_t l_iface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_lface, nodes_x, nodes_y, nodes_z,
                      faces_to_nodes, lface_to_nodes_off, &l_iface_c);
        vec_t rz_r_iface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_r_face, rezoned_nodes_x, rezoned_nodes_y,
                      rezoned_nodes_z, faces_to_nodes, r_face_to_nodes_off,
                      &rz_r_iface_c);
        vec_t rz_l_iface_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_lface, rezoned_nodes_x, rezoned_nodes_y,
                      rezoned_nodes_z, faces_to_nodes, lface_to_nodes_off,
                      &rz_l_iface_c);

        double inodes_x[2 * NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_x[(node_index)] + nodes_x[(r_face_rnode_index)]),
            r_iface_c.x, cell_c.x, l_iface_c.x,
            0.5 * (rezoned_nodes_x[(node_index)] +
                   rezoned_nodes_x[(r_face_rnode_index)]),
            rz_r_iface_c.x, rz_cell_c.x, rz_l_iface_c.x};
        double inodes_y[2 * NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_y[(node_index)] + nodes_y[(r_face_rnode_index)]),
            r_iface_c.y, cell_c.y, l_iface_c.y,
            0.5 * (rezoned_nodes_y[(node_index)] +
                   rezoned_nodes_y[(r_face_rnode_index)]),
            rz_r_iface_c.y, rz_cell_c.y, rz_l_iface_c.y};
        double inodes_z[2 * NNODES_BY_SUBCELL_FACE] = {
            0.5 * (nodes_z[(node_index)] + nodes_z[(r_face_rnode_index)]),
            r_iface_c.z, cell_c.z, l_iface_c.z,
            0.5 * (rezoned_nodes_z[(node_index)] +
                   rezoned_nodes_z[(r_face_rnode_index)]),
            rz_r_iface_c.z, rz_cell_c.z, rz_l_iface_c.z};

        // Contributes the local mass and energy flux for a given subcell face
        contribute_mass_and_energy_flux(
            cc, neighbour_cc, ff, node_index, subcell_index, &subcell_c,
            &cell_c, inodes_x, inodes_y, inodes_z, subcell_mass,
            subcell_mass_flux, subcell_ie_mass, subcell_ie_mass_flux,
            subcell_volume, swept_edge_faces_to_nodes, subcell_centroids_x,
            subcell_centroids_y, subcell_centroids_z, swept_edge_to_faces,
            swept_edge_faces_to_nodes_offsets, subcells_to_subcells_offsets,
            subcells_to_subcells, subcells_to_faces_offsets, subcells_to_faces,
            faces_to_nodes_offsets, faces_to_nodes, cells_offsets,
            cells_to_nodes, nodes_x, nodes_y, nodes_z, 1);

        // Contributes the local mass and energy flux for a given subcell face
        contribute_momentum_flux(
            cc, neighbour_cc, ff, node_index, subcell_index, &subcell_c,
            &cell_c, inodes_x, inodes_y, inodes_z, subcell_volume,
            subcell_momentum_flux_x, subcell_momentum_flux_y,
            subcell_momentum_flux_z, subcell_momentum_x, subcell_momentum_y,
            subcell_momentum_z, swept_edge_faces_to_nodes, subcell_centroids_x,
            subcell_centroids_y, subcell_centroids_z, swept_edge_to_faces,
            swept_edge_faces_to_nodes_offsets, subcells_to_subcells_offsets,
            subcells_to_subcells, subcells_to_faces_offsets, subcells_to_faces,
            faces_to_nodes_offsets, faces_to_nodes, cells_offsets,
            cells_to_nodes, nodes_x, nodes_y, nodes_z, 1);

        /* EXTERNAL FACE */

        // We explicitly disallow flux on the boundary, this could be disable
        // for testing purposes in order to ensure that no flux is inadvertently
        // accumulating on the boundaries
        if (neighbour_cc == -1) {
          continue;
        }

        double enodes_x[2 * NNODES_BY_SUBCELL_FACE] = {
            nodes_x[(node_index)],
            0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]), face_c.x,
            0.5 * (nodes_x[(node_index)] + nodes_x[(lnode_index)]),
            rezoned_nodes_x[(node_index)],
            0.5 * (rezoned_nodes_x[(node_index)] +
                   rezoned_nodes_x[(rnode_index)]),
            rz_face_c.x, 0.5 * (rezoned_nodes_x[(node_index)] +
                                rezoned_nodes_x[(lnode_index)])};
        double enodes_y[2 * NNODES_BY_SUBCELL_FACE] = {
            nodes_y[(node_index)],
            0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]), face_c.y,
            0.5 * (nodes_y[(node_index)] + nodes_y[(lnode_index)]),
            rezoned_nodes_y[(node_index)],
            0.5 * (rezoned_nodes_y[(node_index)] +
                   rezoned_nodes_y[(rnode_index)]),
            rz_face_c.y, 0.5 * (rezoned_nodes_y[(node_index)] +
                                rezoned_nodes_y[(lnode_index)])};
        double enodes_z[2 * NNODES_BY_SUBCELL_FACE] = {
            nodes_z[(node_index)],
            0.5 * (nodes_z[(node_index)] + nodes_z[(rnode_index)]), face_c.z,
            0.5 * (nodes_z[(node_index)] + nodes_z[(lnode_index)]),
            rezoned_nodes_z[(node_index)],
            0.5 * (rezoned_nodes_z[(node_index)] +
                   rezoned_nodes_z[(rnode_index)]),
            rz_face_c.z, 0.5 * (rezoned_nodes_z[(node_index)] +
                                rezoned_nodes_z[(lnode_index)])};

        // Contributes the local mass and energy flux for a given subcell face
        contribute_mass_and_energy_flux(
            cc, neighbour_cc, ff, node_index, subcell_index, &subcell_c,
            &cell_c, enodes_x, enodes_y, enodes_z, subcell_mass,
            subcell_mass_flux, subcell_ie_mass, subcell_ie_mass_flux,
            subcell_volume, swept_edge_faces_to_nodes, subcell_centroids_x,
            subcell_centroids_y, subcell_centroids_z, swept_edge_to_faces,
            swept_edge_faces_to_nodes_offsets, subcells_to_subcells_offsets,
            subcells_to_subcells, subcells_to_faces_offsets, subcells_to_faces,
            faces_to_nodes_offsets, faces_to_nodes, cells_offsets,
            cells_to_nodes, nodes_x, nodes_y, nodes_z, 0);

        // Contributes the local mass and energy flux for a given subcell face
        contribute_momentum_flux(
            cc, neighbour_cc, ff, node_index, subcell_index, &subcell_c,
            &cell_c, enodes_x, enodes_y, enodes_z, subcell_volume,
            subcell_momentum_flux_x, subcell_momentum_flux_y,
            subcell_momentum_flux_z, subcell_momentum_x, subcell_momentum_y,
            subcell_momentum_z, swept_edge_faces_to_nodes, subcell_centroids_x,
            subcell_centroids_y, subcell_centroids_z, swept_edge_to_faces,
            swept_edge_faces_to_nodes_offsets, subcells_to_subcells_offsets,
            subcells_to_subcells, subcells_to_faces_offsets, subcells_to_faces,
            faces_to_nodes_offsets, faces_to_nodes, cells_offsets,
            cells_to_nodes, nodes_x, nodes_y, nodes_z, 0);
      }
    }
  }
}

// Contributes the local mass and energy flux for a given subcell face
void contribute_mass_and_energy_flux(
    const int cc, const int neighbour_cc, const int ff, const int node_index,
    const int subcell_index, vec_t* subcell_c, vec_t* cell_c,
    const double* se_nodes_x, const double* se_nodes_y,
    const double* se_nodes_z, const double* subcell_mass,
    double* subcell_mass_flux, const double* subcell_ie_mass,
    double* subcell_ie_mass_flux, const double* subcell_volume,
    const int* swept_edge_faces_to_nodes, const double* subcell_centroids_x,
    const double* subcell_centroids_y, const double* subcell_centroids_z,
    const int* swept_edge_to_faces,
    const int* swept_edge_faces_to_nodes_offsets,
    const int* subcells_to_subcells_offsets, const int* subcells_to_subcells,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const int* cells_offsets, const int* cells_to_nodes, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const int internal) {

  // Get the centroids for the swept edge prism and faces
  vec_t face_c = {0.0, 0.0, 0.0};
  vec_t rz_face_c = {0.0, 0.0, 0.0};
  vec_t swept_edge_c = {0.0, 0.0, 0.0};
  calc_centroid(NNODES_BY_SUBCELL_FACE, se_nodes_x, se_nodes_y, se_nodes_z,
                swept_edge_faces_to_nodes, 0, &face_c);
  calc_centroid(NNODES_BY_SUBCELL_FACE, se_nodes_x, se_nodes_y, se_nodes_z,
                swept_edge_faces_to_nodes,
                swept_edge_faces_to_nodes_offsets[(1)], &rz_face_c);
  calc_centroid(2 * NNODES_BY_SUBCELL_FACE, se_nodes_x, se_nodes_y, se_nodes_z,
                swept_edge_faces_to_nodes, 0, &swept_edge_c);

  // Calculate the volume of the swept edge prism
  double swept_edge_vol = 0.0;
  calc_volume(0, 2 + NNODES_BY_SUBCELL_FACE, swept_edge_to_faces,
              swept_edge_faces_to_nodes, swept_edge_faces_to_nodes_offsets,
              se_nodes_x, se_nodes_y, se_nodes_z, &swept_edge_c,
              &swept_edge_vol);

  // Ignore the special case of an empty swept edge region
  if (swept_edge_vol < EPS) {
    if (swept_edge_vol < -EPS) {
      printf("Negative swept edge volume %d %.12f\n", cc, swept_edge_vol);
    }
    return;
  }

  // current sub cell
  vec_t ab = {rz_face_c.x - face_c.x, rz_face_c.y - face_c.y,
              rz_face_c.z - face_c.z};
  vec_t ac = {subcell_c->x - face_c.x, subcell_c->y - face_c.y,
              subcell_c->z - face_c.z};
  const int is_outflux = (ab.x * ac.x + ab.y * ac.y + ab.z * ac.z > 0.0);

  // Depending upon which subcell we are sweeping into, choose the
  // subcell index with which to reconstruct the density
  const int subcell_to_subcells_off =
      subcells_to_subcells_offsets[(subcell_index)];
  const int internal_offset = (internal ? 0 : 1);
  const int subcell_neighbour_index = subcells_to_subcells[(
      subcell_to_subcells_off + 2 * ff + internal_offset)];

  // Only perform the sweep on the external face if it isn't a
  // boundary
  if (subcell_neighbour_index == -1) {
    TERMINATE(
        "We should not be attempting to flux from boundary. Volume: %.12f.",
        swept_edge_vol);
  }

  // The sweep subcell index is where we will reconstruct the value of the
  // swept edge region from
  const int sweep_subcell_index =
      (is_outflux ? subcell_index : subcell_neighbour_index);

  // Get the cell center of the sweep cell
  vec_t sweep_cell_c = {0.0, 0.0, 0.0};
  if (is_outflux) {
    sweep_cell_c = *cell_c;
  } else {
    const int cell_to_nodes_off = cells_offsets[(neighbour_cc)];
    const int nnodes_by_cell =
        cells_offsets[(neighbour_cc + 1)] - cell_to_nodes_off;
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &sweep_cell_c);
  }

  /* CALCULATE THE SWEEP SUBCELL GRADIENTS FOR MASS AND ENERGY */

  vec_t inv[3] = {{0.0, 0.0, 0.0}};
  vec_t coeff[3] = {{0.0, 0.0, 0.0}};
  vec_t m_rhs = {0.0, 0.0, 0.0};
  vec_t ie_rhs = {0.0, 0.0, 0.0};

  double gmax_m = -DBL_MAX;
  double gmin_m = DBL_MAX;
  double gmax_ie = -DBL_MAX;
  double gmin_ie = DBL_MAX;

  vec_t sweep_subcell_c = {subcell_centroids_x[(sweep_subcell_index)],
                           subcell_centroids_y[(sweep_subcell_index)],
                           subcell_centroids_z[(sweep_subcell_index)]};

  const double sweep_subcell_density = subcell_mass[(sweep_subcell_index)] /
                                       subcell_volume[(sweep_subcell_index)];
  const double sweep_subcell_ie_density =
      subcell_ie_mass[(sweep_subcell_index)] /
      subcell_volume[(sweep_subcell_index)];

  const int sweep_subcell_to_subcells_off =
      subcells_to_subcells_offsets[(sweep_subcell_index)];
  const int nsubcell_neighbours =
      subcells_to_subcells_offsets[(sweep_subcell_index + 1)] -
      sweep_subcell_to_subcells_off;

  for (int ss = 0; ss < nsubcell_neighbours; ++ss) {
    const int sweep_neighbour_index =
        subcells_to_subcells[(sweep_subcell_to_subcells_off + ss)];

    // Ignore boundary neighbours
    if (sweep_neighbour_index == -1) {
      continue;
    }

    double neighbour_vol = subcell_volume[(sweep_neighbour_index)];
    vec_t i = {subcell_centroids_x[(sweep_neighbour_index)] * neighbour_vol -
                   sweep_subcell_c.x * neighbour_vol,
               subcell_centroids_y[(sweep_neighbour_index)] * neighbour_vol -
                   sweep_subcell_c.y * neighbour_vol,
               subcell_centroids_z[(sweep_neighbour_index)] * neighbour_vol -
                   sweep_subcell_c.z * neighbour_vol};

#if 0
    if (subcell_index == 50728) {
      printf("i %.12e %.12e %.12e\n", i.x / neighbour_vol, i.y / neighbour_vol,
             i.z / neighbour_vol);
    }
#endif // if 0

    // Store the neighbouring cell's contribution to the coefficients
    coeff[0].x += 2.0 * (i.x * i.x) / (neighbour_vol * neighbour_vol);
    coeff[0].y += 2.0 * (i.x * i.y) / (neighbour_vol * neighbour_vol);
    coeff[0].z += 2.0 * (i.x * i.z) / (neighbour_vol * neighbour_vol);
    coeff[1].x += 2.0 * (i.y * i.x) / (neighbour_vol * neighbour_vol);
    coeff[1].y += 2.0 * (i.y * i.y) / (neighbour_vol * neighbour_vol);
    coeff[1].z += 2.0 * (i.y * i.z) / (neighbour_vol * neighbour_vol);
    coeff[2].x += 2.0 * (i.z * i.x) / (neighbour_vol * neighbour_vol);
    coeff[2].y += 2.0 * (i.z * i.y) / (neighbour_vol * neighbour_vol);
    coeff[2].z += 2.0 * (i.z * i.z) / (neighbour_vol * neighbour_vol);

    // Calculate the RHS for mass
    const double neighbour_m_density =
        subcell_mass[(sweep_neighbour_index)] / neighbour_vol;
    const double dneighbour_m_density =
        neighbour_m_density - sweep_subcell_density;

    m_rhs.x += 2.0 * dneighbour_m_density * i.x / neighbour_vol;
    m_rhs.y += 2.0 * dneighbour_m_density * i.y / neighbour_vol;
    m_rhs.z += 2.0 * dneighbour_m_density * i.z / neighbour_vol;

    // Calculate the RHS for energy
    const double neighbour_ie_density =
        subcell_ie_mass[(sweep_neighbour_index)] / neighbour_vol;
    const double dneighbour_ie_density =
        neighbour_ie_density - sweep_subcell_ie_density;

    ie_rhs.x += 2.0 * dneighbour_ie_density * i.x / neighbour_vol;
    ie_rhs.y += 2.0 * dneighbour_ie_density * i.y / neighbour_vol;
    ie_rhs.z += 2.0 * dneighbour_ie_density * i.z / neighbour_vol;

    // Store the maximum / minimum values for rho in the neighbourhood
    gmax_m = max(gmax_m, neighbour_m_density);
    gmin_m = min(gmin_m, neighbour_m_density);
    gmax_ie = max(gmax_ie, neighbour_ie_density);
    gmin_ie = min(gmin_ie, neighbour_ie_density);
  }

  calc_3x3_inverse(&coeff, &inv);

  // Calculate the gradient for the mass density
  vec_t grad_m = {inv[0].x * m_rhs.x + inv[0].y * m_rhs.y + inv[0].z * m_rhs.z,
                  inv[1].x * m_rhs.x + inv[1].y * m_rhs.y + inv[1].z * m_rhs.z,
                  inv[2].x * m_rhs.x + inv[2].y * m_rhs.y + inv[2].z * m_rhs.z};

  // Calculate the gradient for the internal energy density
  vec_t grad_ie = {
      inv[0].x * ie_rhs.x + inv[0].y * ie_rhs.y + inv[0].z * ie_rhs.z,
      inv[1].x * ie_rhs.x + inv[1].y * ie_rhs.y + inv[1].z * ie_rhs.z,
      inv[2].x * ie_rhs.x + inv[2].y * ie_rhs.y + inv[2].z * ie_rhs.z};

  /* LIMIT THE GRADIENT */

  // Performing the limiting actually requires the sweep subcell's nodes
  double m_limiter = 1.0;
  double ie_limiter = 1.0;

  m_limiter =
      min(m_limiter,
          calc_cell_limiter(sweep_subcell_density, gmax_m, gmin_m, &grad_m,
                            nodes_x[(node_index)], nodes_y[(node_index)],
                            nodes_z[(node_index)], &sweep_subcell_c));
  m_limiter =
      min(m_limiter, calc_cell_limiter(sweep_subcell_density, gmax_m, gmin_m,
                                       &grad_m, sweep_cell_c.x, sweep_cell_c.y,
                                       sweep_cell_c.z, &sweep_subcell_c));
  ie_limiter = min(ie_limiter,
                   calc_cell_limiter(sweep_subcell_ie_density, gmax_ie, gmin_ie,
                                     &grad_ie, nodes_x[(node_index)],
                                     nodes_y[(node_index)],
                                     nodes_z[(node_index)], &sweep_subcell_c));
  ie_limiter = min(ie_limiter,
                   calc_cell_limiter(sweep_subcell_ie_density, gmax_ie, gmin_ie,
                                     &grad_ie, sweep_cell_c.x, sweep_cell_c.y,
                                     sweep_cell_c.z, &sweep_subcell_c));

  // Consider all faces attached to node
  const int subcell_to_faces_off =
      subcells_to_faces_offsets[(sweep_subcell_index)];
  const int nfaces_by_subcell =
      subcells_to_faces_offsets[(sweep_subcell_index + 1)] -
      subcell_to_faces_off;
  for (int ff2 = 0; ff2 < nfaces_by_subcell; ++ff2) {
    const int face_index = subcells_to_faces[(subcell_to_faces_off + ff2)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    vec_t face_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                  face_to_nodes_off, &face_c);

    // Determine the position of the node in the face list of nodes
    int nn2;
    for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
      if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
        break;
      }
    }

    const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
    const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
    const int rnode_index = faces_to_nodes[(face_to_nodes_off + next_node)];
    const int lnode_index = faces_to_nodes[(face_to_nodes_off + prev_node)];

    vec_t r_half_edge = {0.5 * (nodes_x[(rnode_index)] + nodes_x[(node_index)]),
                         0.5 * (nodes_y[(rnode_index)] + nodes_y[(node_index)]),
                         0.5 *
                             (nodes_z[(rnode_index)] + nodes_z[(node_index)])};
    vec_t l_half_edge = {0.5 * (nodes_x[(lnode_index)] + nodes_x[(node_index)]),
                         0.5 * (nodes_y[(lnode_index)] + nodes_y[(node_index)]),
                         0.5 *
                             (nodes_z[(lnode_index)] + nodes_z[(node_index)])};

    m_limiter =
        min(m_limiter,
            calc_cell_limiter(sweep_subcell_density, gmax_m, gmin_m, &grad_m,
                              face_c.x, face_c.y, face_c.z, &sweep_subcell_c));
#if 0
    if (subcell_index == 50728) {
      printf("limiter %.12e\n", m_limiter);
    }
#endif // if 0
    m_limiter =
        min(m_limiter, calc_cell_limiter(sweep_subcell_density, gmax_m, gmin_m,
                                         &grad_m, r_half_edge.x, r_half_edge.y,
                                         r_half_edge.z, &sweep_subcell_c));
#if 0
    if (subcell_index == 50728) {
      printf("limiter %.12e\n", m_limiter);
    }
#endif // if 0
    m_limiter =
        min(m_limiter, calc_cell_limiter(sweep_subcell_density, gmax_m, gmin_m,
                                         &grad_m, l_half_edge.x, l_half_edge.y,
                                         l_half_edge.z, &sweep_subcell_c));
#if 0
    if (subcell_index == 50728) {
      printf("limiter %.12e\n", m_limiter);
    }
#endif // if 0

    ie_limiter =
        min(ie_limiter, calc_cell_limiter(sweep_subcell_ie_density, gmax_ie,
                                          gmin_ie, &grad_ie, face_c.x, face_c.y,
                                          face_c.z, &sweep_subcell_c));
    ie_limiter =
        min(ie_limiter,
            calc_cell_limiter(sweep_subcell_ie_density, gmax_ie, gmin_ie,
                              &grad_ie, r_half_edge.x, r_half_edge.y,
                              r_half_edge.z, &sweep_subcell_c));
    ie_limiter =
        min(ie_limiter,
            calc_cell_limiter(sweep_subcell_ie_density, gmax_ie, gmin_ie,
                              &grad_ie, l_half_edge.x, l_half_edge.y,
                              l_half_edge.z, &sweep_subcell_c));

    // OPTION: CALCULATE THE ORIENTATION OF EVERY FACE AND REDUCE THE NUMBER OF
    // TIMES THAT YOU HAVE TO CALC THE CELL LIMITER... AS SOME OF THE WORK IS
    // REDUNDANT. I FEEL WE SHOULD JUST KEEP REAPPLYING FOR PERFORMANCE.
  }

  grad_m.x *= m_limiter;
  grad_m.y *= m_limiter;
  grad_m.z *= m_limiter;
  grad_ie.x *= ie_limiter;
  grad_ie.y *= ie_limiter;
  grad_ie.z *= ie_limiter;

#if 0
  if (subcell_index == 50728) {
    printf("grad_m limited %.12e %.12e %.12e limiter %.12e\n", grad_m.x,
           grad_m.y, grad_m.z, m_limiter);
  }
#endif // if 0

  // Calculate the flux for the mass
  const double local_mass_flux =
      swept_edge_vol *
      (sweep_subcell_density + grad_m.x * (swept_edge_c.x - sweep_subcell_c.x) +
       grad_m.y * (swept_edge_c.y - sweep_subcell_c.y) +
       grad_m.z * (swept_edge_c.z - sweep_subcell_c.z));

  // Calculate the flux for internal energy density in the subcell
  const double local_energy_flux =
      swept_edge_vol * (sweep_subcell_ie_density +
                        grad_ie.x * (swept_edge_c.x - sweep_subcell_c.x) +
                        grad_ie.y * (swept_edge_c.y - sweep_subcell_c.y) +
                        grad_ie.z * (swept_edge_c.z - sweep_subcell_c.z));

#if 0
  if (subcell_index == 50728) { // || subcell_index == 50731) {
#if 0
    printf("cell_c %.12e %.12e %.12e\n", swept_edge_c.x, swept_edge_c.y,
           swept_edge_c.z);
    printf("face_c %.12e %.12e %.12e\n", face_c.x, face_c.y, face_c.z);
#endif // if 0
#if 0
    printf("double nodes_x[]={");
    for (int nn = 0; nn < 2 * NNODES_BY_SUBCELL_FACE; ++nn) {
      printf("%.16e,", se_nodes_x[nn]);
    }
    printf("};\n");
    printf("double nodes_y[]={");
    for (int nn = 0; nn < 2 * NNODES_BY_SUBCELL_FACE; ++nn) {
      printf("%.16e,", se_nodes_y[nn]);
    }
    printf("};\n");
    printf("double nodes_z[]={");
    for (int nn = 0; nn < 2 * NNODES_BY_SUBCELL_FACE; ++nn) {
      printf("%.16e,", se_nodes_z[nn]);
    }
    printf("};\n");
#endif // if 0
    printf("subcell_index %d\n", subcell_index);
    printf("swept_edge_vol %.15e\n", swept_edge_vol);
    printf("neighbour_cc %d\n", neighbour_cc);
    printf("sweep_subcell %d\n", sweep_subcell_index);
    printf("%.12e flowing %s\n", local_mass_flux, is_outflux ? "out" : "in");
    double res = (sweep_subcell_density +
                  grad_m.x * (swept_edge_c.x - sweep_subcell_c.x) +
                  grad_m.y * (swept_edge_c.y - sweep_subcell_c.y) +
                  grad_m.z * (swept_edge_c.z - sweep_subcell_c.z));
    printf("dist %.12e %.12e %.12e\n", (swept_edge_c.x - sweep_subcell_c.x),
           (swept_edge_c.y - sweep_subcell_c.y),
           (swept_edge_c.z - sweep_subcell_c.z));
    printf("%.12e %.12e %.12e\n\n", res, gmax_m, gmin_m);
  }
#if 0
  if (local_mass_flux < 0.0 || local_energy_flux < 0.0) {
    printf("Encountered negative swept edge region flux in subcell %d: mass "
           "%.12e energy %.12e.\n",
           subcell_index, local_mass_flux, local_energy_flux);
    printf("swept_edge_vol %.12e mass %.12e energy %.12e\n", swept_edge_vol,
           subcell_mass[(sweep_subcell_index)], sweep_subcell_ie_density);
  }
#endif // if 0
#endif // if 0

  // Mass and energy are either flowing into or out of the subcell
  if (is_outflux) {
    subcell_mass_flux[(subcell_index)] += local_mass_flux;
    subcell_ie_mass_flux[(subcell_index)] += local_energy_flux;
  } else {
    subcell_mass_flux[(subcell_index)] -= local_mass_flux;
    subcell_ie_mass_flux[(subcell_index)] -= local_energy_flux;
  }

#if 0
  if (subcell_mass[(subcell_index)] - subcell_mass_flux[(subcell_index)] <
      0.0) {

    printf("Subcell mass going negative %d: mass %.12e energy %.12e.\n",
           subcell_index, local_mass_flux, local_energy_flux);
    printf("swept_edge_vol %.12e mass %.12e energy %.12e\n", swept_edge_vol,
           subcell_mass[(sweep_subcell_index)], sweep_subcell_ie_density);
    printf("swept_edge_c %.12e %.12e %.12e\n", swept_edge_c.x, swept_edge_c.y,
           swept_edge_c.z);
    printf("subcell_c %.12e %.12e %.12e\n", subcell_c->x, subcell_c->y,
           subcell_c->z);
    printf("face_c %.12e %.12e %.12e\n", face_c.x, face_c.y, face_c.z);
  }
#endif // if 0
}

// Contributes the local mass and energy flux for a given subcell face
void contribute_momentum_flux(
    const int cc, const int neighbour_cc, const int ff, const int node_index,
    const int subcell_index, vec_t* subcell_c, vec_t* cell_c,
    const double* se_nodes_x, const double* se_nodes_y,
    const double* se_nodes_z, const double* subcell_volume,
    double* subcell_momentum_flux_x, double* subcell_momentum_flux_y,
    double* subcell_momentum_flux_z, const double* subcell_momentum_x,
    const double* subcell_momentum_y, const double* subcell_momentum_z,
    const int* swept_edge_faces_to_nodes, const double* subcell_centroids_x,
    const double* subcell_centroids_y, const double* subcell_centroids_z,
    const int* swept_edge_to_faces,
    const int* swept_edge_faces_to_nodes_offsets,
    const int* subcells_to_subcells_offsets, const int* subcells_to_subcells,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const int* cells_offsets, const int* cells_to_nodes, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const int internal) {

  // Get the centroids for the swept edge prism and faces
  vec_t face_c = {0.0, 0.0, 0.0};
  vec_t rz_face_c = {0.0, 0.0, 0.0};
  vec_t swept_edge_c = {0.0, 0.0, 0.0};
  calc_centroid(NNODES_BY_SUBCELL_FACE, se_nodes_x, se_nodes_y, se_nodes_z,
                swept_edge_faces_to_nodes, 0, &face_c);
  calc_centroid(NNODES_BY_SUBCELL_FACE, se_nodes_x, se_nodes_y, se_nodes_z,
                swept_edge_faces_to_nodes,
                swept_edge_faces_to_nodes_offsets[(1)], &rz_face_c);
  calc_centroid(2 * NNODES_BY_SUBCELL_FACE, se_nodes_x, se_nodes_y, se_nodes_z,
                swept_edge_faces_to_nodes, 0, &swept_edge_c);

  // Calculate the volume of the swept edge prism
  double swept_edge_vol = 0.0;
  calc_volume(0, 2 + NNODES_BY_SUBCELL_FACE, swept_edge_to_faces,
              swept_edge_faces_to_nodes, swept_edge_faces_to_nodes_offsets,
              se_nodes_x, se_nodes_y, se_nodes_z, &swept_edge_c,
              &swept_edge_vol);

  // Ignore the special case of an empty swept edge region
  if (swept_edge_vol < EPS) {
    if (swept_edge_vol < -EPS) {
      printf("Negative swept edge volume %d %.12f\n", cc, swept_edge_vol);
    }
    return;
  }

  // Determine whether the swept edge region is flowing into or out of the
  // current sub cell
  vec_t ab = {rz_face_c.x - face_c.x, rz_face_c.y - face_c.y,
              rz_face_c.z - face_c.z};
  vec_t ac = {subcell_c->x - face_c.x, subcell_c->y - face_c.y,
              subcell_c->z - face_c.z};
  const int is_outflux = (ab.x * ac.x + ab.y * ac.y + ab.z * ac.z > 0.0);

  // Depending upon which subcell we are sweeping into, choose the
  // subcell index with which to reconstruct the density
  const int subcell_to_subcells_off =
      subcells_to_subcells_offsets[(subcell_index)];
  const int internal_offset = (internal ? 0 : 1);
  const int subcell_neighbour_index = subcells_to_subcells[(
      subcell_to_subcells_off + 2 * ff + internal_offset)];

  // Only perform the sweep on the external face if it isn't a
  // boundary
  if (subcell_neighbour_index == -1) {
    TERMINATE(
        "We should not be attempting to flux from boundary. Volume: %.12f.",
        swept_edge_vol);
  }

  // The sweep subcell index is where we will reconstruct the value of the
  // swept edge region from
  const int sweep_subcell_index =
      (is_outflux ? subcell_index : subcell_neighbour_index);

  // Get the cell center of the sweep cell
  vec_t sweep_cell_c;
  if (is_outflux) {
    sweep_cell_c = *cell_c;
  } else {
    const int cell_to_nodes_off = cells_offsets[(neighbour_cc)];
    const int nnodes_by_cell =
        cells_offsets[(neighbour_cc + 1)] - cell_to_nodes_off;
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &sweep_cell_c);
  }

  /* CALCULATE THE SWEEP SUBCELL GRADIENTS FOR MOMENTUM */

  vec_t inv[3] = {{0.0, 0.0, 0.0}};
  vec_t coeff[3] = {{0.0, 0.0, 0.0}};
  vec_t vx_rhs = {0.0, 0.0, 0.0};
  vec_t vy_rhs = {0.0, 0.0, 0.0};
  vec_t vz_rhs = {0.0, 0.0, 0.0};

  double gmax_vx = -DBL_MAX;
  double gmin_vx = DBL_MAX;
  double gmax_vy = -DBL_MAX;
  double gmin_vy = DBL_MAX;
  double gmax_vz = -DBL_MAX;
  double gmin_vz = DBL_MAX;

  vec_t sweep_subcell_c = {subcell_centroids_x[(sweep_subcell_index)],
                           subcell_centroids_y[(sweep_subcell_index)],
                           subcell_centroids_z[(sweep_subcell_index)]};

  const double subcell_vx = subcell_momentum_x[(sweep_subcell_index)] /
                            subcell_volume[(sweep_subcell_index)];
  const double subcell_vy = subcell_momentum_y[(sweep_subcell_index)] /
                            subcell_volume[(sweep_subcell_index)];
  const double subcell_vz = subcell_momentum_z[(sweep_subcell_index)] /
                            subcell_volume[(sweep_subcell_index)];

  const int sweep_subcell_to_subcells_off =
      subcells_to_subcells_offsets[(sweep_subcell_index)];
  const int nsubcell_neighbours =
      subcells_to_subcells_offsets[(sweep_subcell_index + 1)] -
      sweep_subcell_to_subcells_off;

  for (int ss = 0; ss < nsubcell_neighbours; ++ss) {
    const int sweep_neighbour_index =
        subcells_to_subcells[(sweep_subcell_to_subcells_off + ss)];

    // Ignore boundary neighbours
    if (sweep_neighbour_index == -1) {
      continue;
    }

    double neighbour_vol = subcell_volume[(sweep_neighbour_index)];
    vec_t i = {subcell_centroids_x[(sweep_neighbour_index)] * neighbour_vol -
                   sweep_subcell_c.x * neighbour_vol,
               subcell_centroids_y[(sweep_neighbour_index)] * neighbour_vol -
                   sweep_subcell_c.y * neighbour_vol,
               subcell_centroids_z[(sweep_neighbour_index)] * neighbour_vol -
                   sweep_subcell_c.z * neighbour_vol};

    // Store the neighbouring cell's contribution to the coefficients
    coeff[0].x += 2.0 * (i.x * i.x) / (neighbour_vol * neighbour_vol);
    coeff[0].y += 2.0 * (i.x * i.y) / (neighbour_vol * neighbour_vol);
    coeff[0].z += 2.0 * (i.x * i.z) / (neighbour_vol * neighbour_vol);
    coeff[1].x += 2.0 * (i.y * i.x) / (neighbour_vol * neighbour_vol);
    coeff[1].y += 2.0 * (i.y * i.y) / (neighbour_vol * neighbour_vol);
    coeff[1].z += 2.0 * (i.y * i.z) / (neighbour_vol * neighbour_vol);
    coeff[2].x += 2.0 * (i.z * i.x) / (neighbour_vol * neighbour_vol);
    coeff[2].y += 2.0 * (i.z * i.y) / (neighbour_vol * neighbour_vol);
    coeff[2].z += 2.0 * (i.z * i.z) / (neighbour_vol * neighbour_vol);

    // Prepare differential
    const double dneighbour_vx =
        (subcell_momentum_x[(sweep_neighbour_index)] / neighbour_vol -
         subcell_vx);
    const double dneighbour_vy =
        (subcell_momentum_y[(sweep_neighbour_index)] / neighbour_vol -
         subcell_vy);
    const double dneighbour_vz =
        (subcell_momentum_z[(sweep_neighbour_index)] / neighbour_vol -
         subcell_vz);

    vx_rhs.x += 2.0 * dneighbour_vx * i.x / neighbour_vol;
    vx_rhs.y += 2.0 * dneighbour_vx * i.y / neighbour_vol;
    vx_rhs.z += 2.0 * dneighbour_vx * i.z / neighbour_vol;

    vy_rhs.x += 2.0 * dneighbour_vy * i.x / neighbour_vol;
    vy_rhs.y += 2.0 * dneighbour_vy * i.y / neighbour_vol;
    vy_rhs.z += 2.0 * dneighbour_vy * i.z / neighbour_vol;

    vz_rhs.x += 2.0 * dneighbour_vz * i.x / neighbour_vol;
    vz_rhs.y += 2.0 * dneighbour_vz * i.y / neighbour_vol;
    vz_rhs.z += 2.0 * dneighbour_vz * i.z / neighbour_vol;

    gmax_vx = max(gmax_vx, dneighbour_vx);
    gmin_vx = min(gmin_vx, dneighbour_vx);
    gmax_vy = max(gmax_vy, dneighbour_vy);
    gmin_vy = min(gmin_vy, dneighbour_vy);
    gmax_vz = max(gmax_vz, dneighbour_vz);
    gmin_vz = min(gmin_vz, dneighbour_vz);
  }

  calc_3x3_inverse(&coeff, &inv);

  vec_t grad_vx = {
      inv[0].x * vx_rhs.x + inv[0].y * vx_rhs.y + inv[0].z * vx_rhs.z,
      inv[1].x * vx_rhs.x + inv[1].y * vx_rhs.y + inv[1].z * vx_rhs.z,
      inv[2].x * vx_rhs.x + inv[2].y * vx_rhs.y + inv[2].z * vx_rhs.z};
  vec_t grad_vy = {
      inv[0].x * vy_rhs.x + inv[0].y * vy_rhs.y + inv[0].z * vy_rhs.z,
      inv[1].x * vy_rhs.x + inv[1].y * vy_rhs.y + inv[1].z * vy_rhs.z,
      inv[2].x * vy_rhs.x + inv[2].y * vy_rhs.y + inv[2].z * vy_rhs.z};
  vec_t grad_vz = {
      inv[0].x * vz_rhs.x + inv[0].y * vz_rhs.y + inv[0].z * vz_rhs.z,
      inv[1].x * vz_rhs.x + inv[1].y * vz_rhs.y + inv[1].z * vz_rhs.z,
      inv[2].x * vz_rhs.x + inv[2].y * vz_rhs.y + inv[2].z * vz_rhs.z};

  /* LIMIT THE GRADIENT */

  // Performing the limiting actually requires the sweep subcell's nodes
  double vx_limiter = 1.0;
  double vy_limiter = 1.0;
  double vz_limiter = 1.0;

  // Performing the limiting actually requires the sweep subcell's nodes
  vx_limiter =
      min(vx_limiter,
          calc_cell_limiter(subcell_vx, gmax_vx, gmin_vx, &grad_vx,
                            nodes_x[(node_index)], nodes_y[(node_index)],
                            nodes_z[(node_index)], &sweep_subcell_c));
  vx_limiter =
      min(vx_limiter, calc_cell_limiter(subcell_vx, gmax_vx, gmin_vx, &grad_vx,
                                        sweep_cell_c.x, sweep_cell_c.y,
                                        sweep_cell_c.z, &sweep_subcell_c));
  vy_limiter =
      min(vy_limiter,
          calc_cell_limiter(subcell_vy, gmax_vy, gmin_vy, &grad_vy,
                            nodes_x[(node_index)], nodes_y[(node_index)],
                            nodes_z[(node_index)], &sweep_subcell_c));
  vy_limiter =
      min(vy_limiter, calc_cell_limiter(subcell_vy, gmax_vy, gmin_vy, &grad_vy,
                                        sweep_cell_c.x, sweep_cell_c.y,
                                        sweep_cell_c.z, &sweep_subcell_c));
  vz_limiter =
      min(vz_limiter,
          calc_cell_limiter(subcell_vz, gmax_vz, gmin_vz, &grad_vz,
                            nodes_x[(node_index)], nodes_y[(node_index)],
                            nodes_z[(node_index)], &sweep_subcell_c));
  vz_limiter =
      min(vz_limiter, calc_cell_limiter(subcell_vz, gmax_vz, gmin_vz, &grad_vz,
                                        sweep_cell_c.x, sweep_cell_c.y,
                                        sweep_cell_c.z, &sweep_subcell_c));

  // Consider all faces attached to node
  const int subcell_to_faces_off =
      subcells_to_faces_offsets[(sweep_subcell_index)];
  const int nfaces_by_subcell =
      subcells_to_faces_offsets[(sweep_subcell_index + 1)] -
      subcell_to_faces_off;
  for (int ff2 = 0; ff2 < nfaces_by_subcell; ++ff2) {
    const int face_index = subcells_to_faces[(subcell_to_faces_off + ff2)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    vec_t face_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                  face_to_nodes_off, &face_c);

    // Determine the position of the node in the face list of nodes
    int nn2;
    for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
      if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
        break;
      }
    }

    const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
    const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
    const int rnode_index = faces_to_nodes[(face_to_nodes_off + next_node)];
    const int lnode_index = faces_to_nodes[(face_to_nodes_off + prev_node)];

    vec_t r_half_edge = {0.5 * (nodes_x[(rnode_index)] + nodes_x[(node_index)]),
                         0.5 * (nodes_y[(rnode_index)] + nodes_y[(node_index)]),
                         0.5 *
                             (nodes_z[(rnode_index)] + nodes_z[(node_index)])};
    vec_t l_half_edge = {0.5 * (nodes_x[(lnode_index)] + nodes_x[(node_index)]),
                         0.5 * (nodes_y[(lnode_index)] + nodes_y[(node_index)]),
                         0.5 *
                             (nodes_z[(lnode_index)] + nodes_z[(node_index)])};

    vx_limiter = min(vx_limiter, calc_cell_limiter(subcell_vx, gmax_vx, gmin_vx,
                                                   &grad_vx, face_c.x, face_c.y,
                                                   face_c.z, &sweep_subcell_c));
    vx_limiter = min(vx_limiter,
                     calc_cell_limiter(subcell_vx, gmax_vx, gmin_vx, &grad_vx,
                                       r_half_edge.x, r_half_edge.y,
                                       r_half_edge.z, &sweep_subcell_c));
    vx_limiter = min(vx_limiter,
                     calc_cell_limiter(subcell_vx, gmax_vx, gmin_vx, &grad_vx,
                                       l_half_edge.x, l_half_edge.y,
                                       l_half_edge.z, &sweep_subcell_c));

    vy_limiter = min(vy_limiter, calc_cell_limiter(subcell_vy, gmax_vy, gmin_vy,
                                                   &grad_vy, face_c.x, face_c.y,
                                                   face_c.z, &sweep_subcell_c));
    vy_limiter = min(vy_limiter,
                     calc_cell_limiter(subcell_vy, gmax_vy, gmin_vy, &grad_vy,
                                       r_half_edge.x, r_half_edge.y,
                                       r_half_edge.z, &sweep_subcell_c));
    vy_limiter = min(vy_limiter,
                     calc_cell_limiter(subcell_vy, gmax_vy, gmin_vy, &grad_vy,
                                       l_half_edge.x, l_half_edge.y,
                                       l_half_edge.z, &sweep_subcell_c));

    vz_limiter = min(vz_limiter, calc_cell_limiter(subcell_vz, gmax_vz, gmin_vz,
                                                   &grad_vz, face_c.x, face_c.y,
                                                   face_c.z, &sweep_subcell_c));
    vz_limiter = min(vz_limiter,
                     calc_cell_limiter(subcell_vz, gmax_vz, gmin_vz, &grad_vz,
                                       r_half_edge.x, r_half_edge.y,
                                       r_half_edge.z, &sweep_subcell_c));
    vz_limiter = min(vz_limiter,
                     calc_cell_limiter(subcell_vz, gmax_vz, gmin_vz, &grad_vz,
                                       l_half_edge.x, l_half_edge.y,
                                       l_half_edge.z, &sweep_subcell_c));

    // OPTION: CALCULATE THE ORIENTATION OF EVERY FACE AND REDUCE THE NUMBER OF
    // TIMES THAT YOU HAVE TO CALC THE CELL LIMITER... AS SOME OF THE WORK IS
    // REDUNDANT. I FEEL WE SHOULD JUST KEEP REAPPLYING FOR PERFORMANCE.
  }

  grad_vx.x *= vx_limiter;
  grad_vx.y *= vx_limiter;
  grad_vx.z *= vx_limiter;
  grad_vy.x *= vy_limiter;
  grad_vy.y *= vy_limiter;
  grad_vy.z *= vy_limiter;
  grad_vz.x *= vz_limiter;
  grad_vz.y *= vz_limiter;
  grad_vz.z *= vz_limiter;

  // Calculate the flux for internal energy density in the subcell
  const double local_x_momentum_flux =
      swept_edge_vol *
      (subcell_vx + grad_vx.x * (swept_edge_c.x - sweep_subcell_c.x) +
       grad_vx.y * (swept_edge_c.y - sweep_subcell_c.y) +
       grad_vx.z * (swept_edge_c.z - sweep_subcell_c.z));
  const double local_y_momentum_flux =
      swept_edge_vol *
      (subcell_vy + grad_vy.x * (swept_edge_c.x - sweep_subcell_c.x) +
       grad_vy.y * (swept_edge_c.y - sweep_subcell_c.y) +
       grad_vy.z * (swept_edge_c.z - sweep_subcell_c.z));
  const double local_z_momentum_flux =
      swept_edge_vol *
      (subcell_vz + grad_vz.x * (swept_edge_c.x - sweep_subcell_c.x) +
       grad_vz.y * (swept_edge_c.y - sweep_subcell_c.y) +
       grad_vz.z * (swept_edge_c.z - sweep_subcell_c.z));

  // Either the momentum is flowing in or out
  if (is_outflux) {
    subcell_momentum_flux_x[(subcell_index)] += local_x_momentum_flux;
    subcell_momentum_flux_y[(subcell_index)] += local_y_momentum_flux;
    subcell_momentum_flux_z[(subcell_index)] += local_z_momentum_flux;
  } else {
    subcell_momentum_flux_x[(subcell_index)] -= local_x_momentum_flux;
    subcell_momentum_flux_y[(subcell_index)] -= local_y_momentum_flux;
    subcell_momentum_flux_z[(subcell_index)] -= local_z_momentum_flux;
  }
}

// Calculate the normal vector from the provided nodes
void calc_unit_normal(const int n0, const int n1, const int n2,
                      const double* nodes_x, const double* nodes_y,
                      const double* nodes_z, vec_t* normal) {

  // Calculate the normal
  calc_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, normal);

  // Normalise the normal
  double len = sqrt(normal->x * normal->x + normal->y * normal->y +
                    normal->z * normal->z);

  // Force propagation of zero length normal
  if (len < EPS) {
    len = 0.0;
  }

  normal->x /= len;
  normal->y /= len;
  normal->z /= len;
}

// Calculate the normal for a plane
void calc_normal(const int n0, const int n1, const int n2,
                 const double* nodes_x, const double* nodes_y,
                 const double* nodes_z, vec_t* normal) {
  // Get two vectors on the face plane
  vec_t dn0 = {0.0, 0.0, 0.0};
  vec_t dn1 = {0.0, 0.0, 0.0};

  // Outwards facing normal for clockwise ordering
  dn0.x = nodes_x[(n0)] - nodes_x[(n1)];
  dn0.y = nodes_y[(n0)] - nodes_y[(n1)];
  dn0.z = nodes_z[(n0)] - nodes_z[(n1)];
  dn1.x = nodes_x[(n2)] - nodes_x[(n1)];
  dn1.y = nodes_y[(n2)] - nodes_y[(n1)];
  dn1.z = nodes_z[(n2)] - nodes_z[(n1)];

  // Cross product to get the normal
  normal->x = (dn0.y * dn1.z - dn1.y * dn0.z);
  normal->y = (dn0.z * dn1.x - dn1.z * dn0.x);
  normal->z = (dn0.x * dn1.y - dn1.x * dn0.y);
}

// Contributes a face to the volume of some cell
// Expects a non-overlapping polyhedra, allowing non-planar faces
void contribute_face_volume(const int nnodes_by_face, const int* faces_to_nodes,
                            const double* nodes_x, const double* nodes_y,
                            const double* nodes_z, const vec_t* cell_c,
                            double* vol) {

  double local_vol = 0.0;
  vec_t face_c = {0.0, 0.0, 0.0};
  calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes, 0,
                &face_c);

  for (int nn = 0; nn < nnodes_by_face; ++nn) {
    // Fetch the nodes attached to our current node on the current face
    const int current_node = faces_to_nodes[(nn)];
    const int next_node = (nn + 1 < nnodes_by_face) ? faces_to_nodes[(nn + 1)]
                                                    : faces_to_nodes[(0)];

    // Get the halfway point on the right edge
    vec_t half_edge;
    half_edge.x = 0.5 * (nodes_x[(current_node)] + nodes_x[(next_node)]);
    half_edge.y = 0.5 * (nodes_y[(current_node)] + nodes_y[(next_node)]);
    half_edge.z = 0.5 * (nodes_z[(current_node)] + nodes_z[(next_node)]);

    // Setup basis on plane of tetrahedron
    vec_t a;
    a.x = (half_edge.x - face_c.x);
    a.y = (half_edge.y - face_c.y);
    a.z = (half_edge.z - face_c.z);
    vec_t b;
    b.x = (cell_c->x - face_c.x);
    b.y = (cell_c->y - face_c.y);
    b.z = (cell_c->z - face_c.z);

    // Calculate the area vector S using cross product
    vec_t S;
    S.x = 0.5 * (a.y * b.z - a.z * b.y);
    S.y = -0.5 * (a.x * b.z - a.z * b.x);
    S.z = 0.5 * (a.x * b.y - a.y * b.x);

    // TODO: WE MULTIPLY BY 2 HERE BECAUSE WE ARE ADDING THE VOLUME TO BOTH
    // THE CURRENT AND NEXT NODE, OTHERWISE WE ONLY ACCOUNT FOR HALF OF THE
    // 'HALF' TETRAHEDRONS
    local_vol += fabs(2.0 * ((half_edge.x - nodes_x[(current_node)]) * S.x +
                             (half_edge.y - nodes_y[(current_node)]) * S.y +
                             (half_edge.z - nodes_z[(current_node)]) * S.z) /
                      3.0);
  }

  *vol += local_vol;
}

// Calculates the weighted volume dist for a provided cell along x-y-z
void calc_volume(const int cell_to_faces_off, const int nfaces_by_cell,
                 const int* cells_to_faces, const int* faces_to_nodes,
                 const int* faces_to_nodes_offsets, const double* nodes_x,
                 const double* nodes_y, const double* nodes_z,
                 const vec_t* cell_c, double* vol) {

  // Prepare to accumulate the volume
  *vol = 0.0;

  for (int ff = 0; ff < nfaces_by_cell; ++ff) {
    const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    contribute_face_volume(nnodes_by_face, &faces_to_nodes[(face_to_nodes_off)],
                           nodes_x, nodes_y, nodes_z, cell_c, vol);

    if (isnan(*vol)) {
      *vol = 0.0;
      return;
    }
  }

  *vol = fabs(*vol);
}

// Stores the rezoned mesh specification as the original mesh. Until we
// determine a reasonable rezoning algorithm, this makes us Eulerian
void store_rezoned_mesh(const int nnodes, const double* nodes_x,
                        const double* nodes_y, const double* nodes_z,
                        double* rezoned_nodes_x, double* rezoned_nodes_y,
                        double* rezoned_nodes_z) {

// Store the rezoned nodes
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    rezoned_nodes_x[(nn)] = nodes_x[(nn)];
    rezoned_nodes_y[(nn)] = nodes_y[(nn)];
    rezoned_nodes_z[(nn)] = nodes_z[(nn)];
  }
}

// Calculate the centroid
void calc_centroid(const int nnodes, const double* nodes_x,
                   const double* nodes_y, const double* nodes_z,
                   const int* indirection, const int offset, vec_t* centroid) {

  centroid->x = 0.0;
  centroid->y = 0.0;
  centroid->z = 0.0;
  for (int nn2 = 0; nn2 < nnodes; ++nn2) {
    const int node_index = indirection[(offset + nn2)];
    centroid->x += nodes_x[(node_index)] / nnodes;
    centroid->y += nodes_y[(node_index)] / nnodes;
    centroid->z += nodes_z[(node_index)] / nnodes;
  }
}

// Calculates the inverse of a 3x3 matrix, out-of-place
void calc_3x3_inverse(vec_t (*a)[3], vec_t (*inv)[3]) {
  // Calculate the determinant of the 3x3
  const double det =
      (*a)[0].x * ((*a)[1].y * (*a)[2].z - (*a)[1].z * (*a)[2].y) -
      (*a)[0].y * ((*a)[1].x * (*a)[2].z - (*a)[1].z * (*a)[2].x) +
      (*a)[0].z * ((*a)[1].x * (*a)[2].y - (*a)[1].y * (*a)[2].x);

  // Check if the matrix is singular
  if (det == 0.0) {
    TERMINATE("singular coefficient matrix");
  } else {
    // Perform the simple and fast 3x3 matrix inverstion
    (*inv)[0].x = ((*a)[1].y * (*a)[2].z - (*a)[1].z * (*a)[2].y) / det;
    (*inv)[0].y = ((*a)[0].z * (*a)[2].y - (*a)[0].y * (*a)[2].z) / det;
    (*inv)[0].z = ((*a)[0].y * (*a)[1].z - (*a)[0].z * (*a)[1].y) / det;

    (*inv)[1].x = ((*a)[1].z * (*a)[2].x - (*a)[1].x * (*a)[2].z) / det;
    (*inv)[1].y = ((*a)[0].x * (*a)[2].z - (*a)[0].z * (*a)[2].x) / det;
    (*inv)[1].z = ((*a)[0].z * (*a)[1].x - (*a)[0].x * (*a)[1].z) / det;

    (*inv)[2].x = ((*a)[1].x * (*a)[2].y - (*a)[1].y * (*a)[2].x) / det;
    (*inv)[2].y = ((*a)[0].y * (*a)[2].x - (*a)[0].x * (*a)[2].y) / det;
    (*inv)[2].z = ((*a)[0].x * (*a)[1].y - (*a)[0].y * (*a)[1].x) / det;
  }
}

// Calculates the local limiter for a cell
double calc_cell_limiter(const double rho, const double gmax, const double gmin,
                         vec_t* grad, const double node_x, const double node_y,
                         const double node_z, const vec_t* cell_c) {

  double g_unlimited = rho + grad->x * (node_x - cell_c->x) +
                       grad->y * (node_y - cell_c->y) +
                       grad->z * (node_z - cell_c->z);

  double limiter = 1.0;
  if (g_unlimited - rho > 0.0) {
    limiter = min(limiter, ((gmax - rho) / (g_unlimited - rho)));
  } else if (g_unlimited - rho < 0.0) {
    limiter = min(limiter, ((gmin - rho) / (g_unlimited - rho)));
  }
  return limiter;
}

// Calculates the limiter for the provided gradient
double apply_cell_limiter(const int nnodes_by_cell, const int cell_to_nodes_off,
                          const int* cells_to_nodes, vec_t* grad,
                          const vec_t* cell_c, const double* nodes_x,
                          const double* nodes_y, const double* nodes_z,
                          const double rho, const double gmax,
                          const double gmin) {

  // Calculate the limiter for the gradient
  double limiter = 1.0;
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
    limiter = min(limiter, calc_cell_limiter(rho, gmax, gmin, grad,
                                             nodes_x[(node_index)],
                                             nodes_y[(node_index)],
                                             nodes_z[(node_index)], cell_c));
  }

  grad->x *= limiter;
  grad->y *= limiter;
  grad->z *= limiter;

  return limiter;
}

// Calculates the limiter for the provided gradient
double apply_node_limiter(const int ncells_by_node, const int node_to_cells_off,
                          const int* nodes_to_cells, vec_t* grad,
                          const vec_t* node, const double* cell_centroids_x,
                          const double* cell_centroids_y,
                          const double* cell_centroids_z, const double rho,
                          const double gmax, const double gmin) {

  // Calculate the limiter for the gradient
  double limiter = DBL_MAX;
  for (int cc = 0; cc < ncells_by_node; ++cc) {
    const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
    limiter = calc_node_limiter(
        rho, gmax, gmin, grad, cell_centroids_x[(cell_index)],
        cell_centroids_y[(cell_index)], cell_centroids_z[(cell_index)], node);
  }

  grad->x *= limiter;
  grad->y *= limiter;
  grad->z *= limiter;

  return limiter;
}

// Calculates the local limiter for a node
double calc_node_limiter(const double rho, const double gmax, const double gmin,
                         vec_t* grad, const double cell_x, const double cell_y,
                         const double cell_z, const vec_t* node) {
  double g_unlimited = rho + grad->x * (cell_x - node->x) +
                       grad->y * (cell_y - node->y) +
                       grad->z * (cell_z - node->z);

  double limiter = 1.0;
  if (g_unlimited - rho > 0.0) {
    limiter = min(1.0, ((gmax - rho) / (g_unlimited - rho)));
  } else if (g_unlimited - rho < 0.0) {
    limiter = min(1.0, ((gmin - rho) / (g_unlimited - rho)));
  }
  return limiter;
}

// Applies the mesh rezoning strategy. This is a pure Eulerian strategy.
void apply_mesh_rezoning(const int nnodes, const double* rezoned_nodes_x,
                         const double* rezoned_nodes_y,
                         const double* rezoned_nodes_z, double* nodes_x,
                         double* nodes_y, double* nodes_z) {

// Apply the rezoned mesh into the main mesh
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x[(nn)] = rezoned_nodes_x[(nn)];
    nodes_y[(nn)] = rezoned_nodes_y[(nn)];
    nodes_z[(nn)] = rezoned_nodes_z[(nn)];
  }
}
