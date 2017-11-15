
__global__ void call_cell_subcell_mass(
    const int ncells, const int* cells_to_nodes_offsets, double* subcell_mass, 
    const double* density, const double* subcell_volume, double* cell_mass) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
  const int nnodes_by_cell = cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;

  // looping over corner subcells here
  double total_mass = 0.0;
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    const int subcell_index = cell_to_nodes_off + nn;
    subcell_mass[(subcell_index)] =
      density[(cc)] * subcell_volume[(subcell_index)];

#if 0
    total_mass += subcell_mass[(subcell_index)];
    total_mass_in_subcells += subcell_mass[(subcell_index)];
#endif // if 0
  }

  cell_mass[(cc)] = total_mass;
  total_mass_in_cells += cell_mass[(cc)];
}

__global__ void calc_nodal_mass(
    const int nnodes, const int* nodes_to_cells_offsets, const int* nodes_to_cells, 
    const int* cells_to_nodes_offsets, const int* cells_to_nodes, 
    const double* subcell_mass, double* nodal_mass) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
  const int ncells_by_node = nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

  nodal_mass[(nn)] = 0.0;

  for (int cc = 0; cc < ncells_by_node; ++cc) {
    const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
    const int cell_to_nodes_off = cells_to_nodes_offsets[(cell_index)];
    const int nnodes_by_cell =
      cells_to_nodes_offsets[(cell_index + 1)] - cell_to_nodes_off;

    for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
      if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
        const int subcell_index = cell_to_nodes_off + nn2;
        nodal_mass[(nn)] += subcell_mass[(subcell_index)];
        break;
      }
    }
  }
}

__global__ void calc_subcell_vol_centroids(
    const int ncells, const int nnodes, const int nnodes_by_subcell,
    const int* cells_to_nodes_offsets, const int* cells_to_nodes,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes, const int* faces_to_nodes_offsets,
    const int* faces_cclockwise_cell, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* subcell_volume, double* cell_volume, double* nodal_volumes,
    int* nodes_to_cells_offsets, int* nodes_to_cells) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
  const int nnodes_by_cell = cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;

  // Calculates the weighted volume dist for a provided cell along x-y-z
  vec_t cell_c = {0.0, 0.0, 0.0};
  calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
      cell_to_nodes_off, &cell_c);

  // Looping over corner subcells here
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
    const int subcell_index = cell_to_nodes_off + nn;
    const int subcell_to_faces_off =
      subcells_to_faces_offsets[(subcell_index)];
    const int nfaces_by_subcell =
      subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

    subcell_centroids_x[(subcell_index)] = 0.0;
    subcell_centroids_y[(subcell_index)] = 0.0;
    subcell_centroids_z[(subcell_index)] = 0.0;

    // Consider all faces attached to node
    for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
      const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
          face_to_nodes_off, &face_c);

      const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);

      // Determine the position of the node in the face list of nodes
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
          break;
        }
      }

      const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
      const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
      const int rnode_off = (face_clockwise ? prev_node : next_node);
      const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];

      subcell_centroids_x[(subcell_index)] +=
        0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]) + face_c.x;
      subcell_centroids_y[(subcell_index)] +=
        0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]) + face_c.y;
      subcell_centroids_z[(subcell_index)] +=
        0.5 * (nodes_z[(node_index)] + nodes_z[(rnode_index)]) + face_c.z;
    }

    subcell_centroids_x[(subcell_index)] =
      (subcell_centroids_x[(subcell_index)] + cell_c.x +
       nodes_x[(node_index)]) /
      nnodes_by_subcell;
    subcell_centroids_y[(subcell_index)] =
      (subcell_centroids_y[(subcell_index)] + cell_c.y +
       nodes_y[(node_index)]) /
      nnodes_by_subcell;
    subcell_centroids_z[(subcell_index)] =
      (subcell_centroids_z[(subcell_index)] + cell_c.z +
       nodes_z[(node_index)]) /
      nnodes_by_subcell;

    vec_t subcell_c = {subcell_centroids_x[(subcell_index)],
      subcell_centroids_y[(subcell_index)],
      subcell_centroids_z[(subcell_index)]};

    subcell_volume[(subcell_index)] = 0.0;

    // Consider all faces attached to node
    for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
      const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // The face centroid is the same for all nodes on the face
      vec_t face_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
          face_to_nodes_off, &face_c);

      const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);

      // Determine the position of the node in the face list of nodes
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        if (faces_to_nodes[(face_to_nodes_off + nn2)] == node_index) {
          break;
        }
      }

      const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
      const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
      const int rnode_off = (face_clockwise ? prev_node : next_node);
      const int lnode_off = (face_clockwise ? next_node : prev_node);
      const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];
      const int lnode_index = faces_to_nodes[(face_to_nodes_off + lnode_off)];

      /* EXTERNAL FACE */

      const int subcell_faces_to_nodes[NNODES_BY_SUBCELL_FACE] = {0, 1, 2, 3};

      double enodes_x[NNODES_BY_SUBCELL_FACE] = {
        nodes_x[(node_index)],
        0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]), face_c.x,
        0.5 * (nodes_x[(node_index)] + nodes_x[(lnode_index)])};
      double enodes_y[NNODES_BY_SUBCELL_FACE] = {
        nodes_y[(node_index)],
        0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]), face_c.y,
        0.5 * (nodes_y[(node_index)] + nodes_y[(lnode_index)])};
      double enodes_z[NNODES_BY_SUBCELL_FACE] = {
        nodes_z[(node_index)],
        0.5 * (nodes_z[(node_index)] + nodes_z[(rnode_index)]), face_c.z,
        0.5 * (nodes_z[(node_index)] + nodes_z[(lnode_index)])};

      contribute_face_volume(NNODES_BY_SUBCELL_FACE, subcell_faces_to_nodes,
          enodes_x, enodes_y, enodes_z, &subcell_c,
          &subcell_volume[(subcell_index)]);

      /* INTERNAL FACE */

      const int r_face_off = (ff == nfaces_by_subcell - 1) ? 0 : ff + 1;
      const int l_face_off = (ff == 0) ? nfaces_by_subcell - 1 : ff - 1;
      const int r_face_index =
        subcells_to_faces[(subcell_to_faces_off + r_face_off)];
      const int l_face_index =
        subcells_to_faces[(subcell_to_faces_off + l_face_off)];
      const int r_face_to_nodes_off = faces_to_nodes_offsets[(r_face_index)];
      const int l_face_to_nodes_off = faces_to_nodes_offsets[(l_face_index)];
      const int nnodes_by_rface =
        faces_to_nodes_offsets[(r_face_index + 1)] - r_face_to_nodes_off;
      const int nnodes_by_lface =
        faces_to_nodes_offsets[(l_face_index + 1)] - l_face_to_nodes_off;

      vec_t rface_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_rface, nodes_x, nodes_y, nodes_z,
          faces_to_nodes, r_face_to_nodes_off, &rface_c);

      const int r_face_clockwise =
        (faces_cclockwise_cell[(r_face_index)] != cc);

      // Determine the position of the node in the face list of nodes
      for (nn2 = 0; nn2 < nnodes_by_rface; ++nn2) {
        if (faces_to_nodes[(r_face_to_nodes_off + nn2)] == node_index) {
          break;
        }
      }

      const int rface_next_node = (nn2 == nnodes_by_rface - 1) ? 0 : nn2 + 1;
      const int rface_prev_node = (nn2 == 0) ? nnodes_by_rface - 1 : nn2 - 1;
      const int rface_rnode_off =
        (r_face_clockwise ? rface_prev_node : rface_next_node);
      const int rface_rnode_index =
        faces_to_nodes[(r_face_to_nodes_off + rface_rnode_off)];

      vec_t lface_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_lface, nodes_x, nodes_y, nodes_z,
          faces_to_nodes, l_face_to_nodes_off, &lface_c);

      double inodes_x[NNODES_BY_SUBCELL_FACE] = {
        0.5 * (nodes_x[(node_index)] + nodes_x[(rface_rnode_index)]),
        rface_c.x, cell_c.x, lface_c.x};
      double inodes_y[NNODES_BY_SUBCELL_FACE] = {
        0.5 * (nodes_y[(node_index)] + nodes_y[(rface_rnode_index)]),
        rface_c.y, cell_c.y, lface_c.y};
      double inodes_z[NNODES_BY_SUBCELL_FACE] = {
        0.5 * (nodes_z[(node_index)] + nodes_z[(rface_rnode_index)]),
        rface_c.z, cell_c.z, lface_c.z};

      contribute_face_volume(NNODES_BY_SUBCELL_FACE, subcell_faces_to_nodes,
          inodes_x, inodes_y, inodes_z, &subcell_c,
          &subcell_volume[(subcell_index)]);

      if (isnan(subcell_volume[(subcell_index)])) {
        subcell_volume[(subcell_index)] = 0.0;
        break;
      }
    }

    subcell_volume[(subcell_index)] = fabs(subcell_volume[(subcell_index)]);
#if 0
    total_subcell_volume += subcell_volume[(subcell_index)];
#endif // if 0
  }
}

__global__ void calc_nodal_volumes(
    const int nnodes, const int* nodes_to_cells_offsets, const int* nodes_to_cells,
    const int* cells_to_nodes_offsets, const int* cells_to_nodes, 
    const double* subcell_volume, double* nodal_volumes) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
  const int ncells_by_node = nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

  nodal_volumes[(nn)] = 0.0;

  for (int cc = 0; cc < ncells_by_node; ++cc) {
    const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
    const int cell_to_nodes_off = cells_to_nodes_offsets[(cell_index)];
    const int nnodes_by_cell =
      cells_to_nodes_offsets[(cell_index + 1)] - cell_to_nodes_off;

    for (int nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
      if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
        const int subcell_index = cell_to_nodes_off + nn2;
        nodal_volumes[(nn)] += subcell_volume[(subcell_index)];
        break;
      }
    }
  }
}

__global__ void calc_cell_centroids(const int ncells, const int* cells_to_nodes_offsets,
    const int* cells_to_nodes, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cells_off = cells_to_nodes_offsets[(cc)];
  const int nnodes_by_cell = cells_to_nodes_offsets[(cc + 1)] - cells_off;

  vec_t cell_c = {0.0, 0.0, 0.0};
  calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
      cells_off, &cell_c);

  cell_centroids_x[(cc)] = cell_c.x;
  cell_centroids_y[(cc)] = cell_c.y;
  cell_centroids_z[(cc)] = cell_c.z;
}
