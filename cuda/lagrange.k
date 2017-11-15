#include <float.h>
#include "../../cuda/reduction.k"

// A simple ideal gas equation of state
__global__ void equation_of_state(const int ncells, const double* energy,
                                  const double* density, double* pressure) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  pressure[(cc)] = (GAM - 1.0) * energy[(cc)] * density[(cc)];
}

// Calculates the volume of a subsubcell
__device__ double calc_subsubcell_volume(
    const int cc, const int rnode_index, const int node_index, vec_t face_c,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z) {

  // Construct the vectors describing an edge tetrahedron
  const vec_t ad = {(face_c.x - nodes_x[(node_index)]),
                    (face_c.y - nodes_y[(node_index)]),
                    (face_c.z - nodes_z[(node_index)])};
  const vec_t bd = {nodes_x[(rnode_index)] - nodes_x[(node_index)],
                    nodes_y[(rnode_index)] - nodes_y[(node_index)],
                    nodes_z[(rnode_index)] - nodes_z[(node_index)]};
  const vec_t cd = {cell_centroids_x[(cc)] - nodes_x[(node_index)],
                    cell_centroids_y[(cc)] - nodes_y[(node_index)],
                    cell_centroids_z[(cc)] - nodes_z[(node_index)]};

  // Fetch the are vector of one of the faces of the tetrahedron
  const vec_t area = {0.5 * (ad.y * bd.z - ad.z * bd.y),
                      -0.5 * (ad.x * bd.z - ad.z * bd.x),
                      0.5 * (ad.x * bd.y - ad.y * bd.x)};

  // Determine the volume using standard irregular tetrahedron formula
  const double edge_subcell_vol =
      fabs(cd.x * area.x + cd.y * area.y + cd.z * area.z) / 3.0;

  // The subsubcell is half the volume of the edge subcell
  return 0.5 * edge_subcell_vol;
}

// Calculates the volume in a cell by tetrahedral decomposition
__device__ double calc_cell_volume(
    const int cc, const int nfaces_by_cell, const int cell_to_faces_off,
    const int* cells_to_faces, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z) {

  double cell_vol = 0.0;

  // Look at all of the faces attached to the cell
  for (int ff = 0; ff < nfaces_by_cell; ++ff) {
    const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    vec_t face_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                  face_to_nodes_off, &face_c);

    // Now we will sum the contributions at each of the nodes
    for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
      // Fetch the nodes attached to our current node on the current face
      const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
      const int rnode_index =
          (nn2 + 1 < nnodes_by_face)
              ? faces_to_nodes[(face_to_nodes_off + nn2 + 1)]
              : faces_to_nodes[(face_to_nodes_off)];

      cell_vol += 2.0 * calc_subsubcell_volume(
                            cc, rnode_index, node_index, face_c, nodes_x,
                            nodes_y, nodes_z, cell_centroids_x,
                            cell_centroids_y, cell_centroids_z);
    }
  }

  return cell_vol;
}

// Calculate the centroid
__device__ void calc_centroid(const int nnodes, const double* nodes_x,
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

// Calculates the nodal volume and sound speed
__global__ void calc_nodal_vol_and_c(
    const int nnodes, const int* nodes_to_faces_offsets,
    const int* nodes_to_faces, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const int* faces_to_cells0,
    const int* faces_to_cells1, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const double* energy, double* nodal_volumes, double* nodal_soundspeed) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  const int node_to_faces_off = nodes_to_faces_offsets[(nn)];
  const int nfaces_by_node =
      nodes_to_faces_offsets[(nn + 1)] - node_to_faces_off;

  nodal_volumes[(nn)] = 0.0;
  nodal_soundspeed[(nn)] = 0.0;

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
    vec_t face_c = {0.0, 0.0, 0.0};
    int node_in_face_c;
    for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
      const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
      face_c.x += nodes_x[(node_index)] / nnodes_by_face;
      face_c.y += nodes_y[(node_index)] / nnodes_by_face;
      face_c.z += nodes_z[(node_index)] / nnodes_by_face;

      // Choose the node in the list of nodes attached to the face
      if (nn == node_index) {
        node_in_face_c = nn2;
      }
    }

    // Fetch the nodes attached to our current node on the current face
    int local_nodes[2];
    local_nodes[0] =
        (node_in_face_c - 1 >= 0)
            ? faces_to_nodes[(face_to_nodes_off + node_in_face_c - 1)]
            : faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)];
    local_nodes[1] =
        (node_in_face_c + 1 < nnodes_by_face)
            ? faces_to_nodes[(face_to_nodes_off + node_in_face_c + 1)]
            : faces_to_nodes[(face_to_nodes_off)];

    // Fetch the cells attached to our current face
    int local_cells[2];
    local_cells[0] = faces_to_cells0[(face_index)];
    local_cells[1] = faces_to_cells1[(face_index)];

    // Add contributions from both of the cells attached to the face
    for (int cc = 0; cc < 2; ++cc) {
      const int cell_index = local_cells[(cc)];
      if (cell_index == -1) {
        continue;
      }

      // Add contributions for both edges attached to our current node
      for (int nn2 = 0; nn2 < 2; ++nn2) {
        const double subsubcell_vol = calc_subsubcell_volume(
            cell_index, local_nodes[(nn2)], nn, face_c, nodes_x, nodes_y,
            nodes_z, cell_centroids_x, cell_centroids_y, cell_centroids_z);
        nodal_soundspeed[(nn)] +=
            sqrt(GAM * (GAM - 1.0) * energy[(cell_index)]) * subsubcell_vol;
        nodal_volumes[(nn)] += subsubcell_vol;
      }
    }
  }
}

// Sets all of the subcell forces to 0
__global__ void zero_subcell_forces(const int ncells,
                                    const int* cells_to_nodes_offsets,
                                    double* subcell_force_x,
                                    double* subcell_force_y,
                                    double* subcell_force_z) {

  // TODO: Obvious optimisations here and in the CPU code, where we don't
  // actually need to use the indirections if we are simply stepping through the
  // subcells in order...

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
  const int nnodes_by_cell =
      cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    const int subcell_index = cell_to_nodes_off + nn;
    subcell_force_x[(subcell_index)] = 0.0;
    subcell_force_y[(subcell_index)] = 0.0;
    subcell_force_z[(subcell_index)] = 0.0;
  }
}

// Calculate the subcell force from pressure gradients
__global__ void calc_subcell_force_from_pressure(
    const int ncells, const int* cells_to_faces_offsets,
    const int* cells_to_nodes_offsets, const int* cells_to_faces,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const int* cells_to_nodes, const int* faces_cclockwise_cell,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    const double* pressure, double* subcell_force_x, double* subcell_force_y,
    double* subcell_force_z) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
  const int nfaces_by_cell =
      cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
  const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
  const int nnodes_by_cell =
      cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;

  // Look at all of the faces attached to the cell
  for (int ff = 0; ff < nfaces_by_cell; ++ff) {
    const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    // Calculate the face center... SHOULD WE PRECOMPUTE?
    vec_t face_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                  face_to_nodes_off, &face_c);

    // Now we will sum the contributions at each of the nodes
    // TODO: THERE IS SOME SYMMETRY HERE THAT MEANS WE MIGHT BE ABLE TO
    // OPTIMISE
    for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
      // Fetch the nodes attached to our current node on the current face
      const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
      const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);
      const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
      const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
      const int rnode_off = (face_clockwise ? prev_node : next_node);
      const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];

      // Get the halfway point on the right edge
      vec_t half_edge = {0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]),
                         0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]),
                         0.5 *
                             (nodes_z[(node_index)] + nodes_z[(rnode_index)])};

      // Setup basis on plane of tetrahedron
      vec_t a = {(nodes_x[(node_index)] - half_edge.x),
                 (nodes_y[(node_index)] - half_edge.y),
                 (nodes_z[(node_index)] - half_edge.z)};
      vec_t b = {(face_c.x - half_edge.x), (face_c.y - half_edge.y),
                 (face_c.z - half_edge.z)};

      // Calculate the area vector A using cross product
      vec_t A = {0.5 * (a.y * b.z - a.z * b.y), -0.5 * (a.x * b.z - a.z * b.x),
                 0.5 * (a.x * b.y - a.y * b.x)};

      int subcell_index;
      int rsubcell_index;
      for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
        if (cells_to_nodes[(cell_to_nodes_off + nn3)] == node_index) {
          subcell_index = cell_to_nodes_off + nn3;
        } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == rnode_index) {
          rsubcell_index = cell_to_nodes_off + nn3;
        }
      }

      subcell_force_x[(subcell_index)] += pressure[(cc)] * A.x;
      subcell_force_y[(subcell_index)] += pressure[(cc)] * A.y;
      subcell_force_z[(subcell_index)] += pressure[(cc)] * A.z;
      subcell_force_x[(rsubcell_index)] += pressure[(cc)] * A.x;
      subcell_force_y[(rsubcell_index)] += pressure[(cc)] * A.y;
      subcell_force_z[(rsubcell_index)] += pressure[(cc)] * A.z;
    }
  }
}

// Scale the soundspeed by the inverse of the nodal volume
__global__ void scale_soundspeed(const int nnodes, const double* nodal_volumes,
                                 double* nodal_soundspeed) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
}

// Calculate the time centered evolved velocities, by calculating the predicted
// values at the new timestep and averaging with current velocity
__global__ void
calc_new_velocity(const int nnodes, const double dt,
                  const int* nodes_to_cells_offsets, const int* nodes_to_cells,
                  const int* cells_to_nodes_offsets, const int* cells_to_nodes,
                  const double* subcell_force_x, const double* subcell_force_y,
                  const double* subcell_force_z, const double* nodal_mass,
                  const double* velocity_x0, const double* velocity_y0,
                  const double* velocity_z0, double* velocity_x1,
                  double* velocity_y1, double* velocity_z1) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
  const int ncells_by_node =
      nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

  // Accumulate the force at this node
  vec_t node_force = {0.0, 0.0, 0.0};
  for (int cc = 0; cc < ncells_by_node; ++cc) {
    const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
    const int cell_to_nodes_off = cells_to_nodes_offsets[(cell_index)];
    const int nnodes_by_cell =
        cells_to_nodes_offsets[(cell_index + 1)] - cell_to_nodes_off;

    // ARRGHHHH
    int nn2;
    for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
      if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
        break;
      }
    }

    const int subcell_index = cell_to_nodes_off + nn2;
    node_force.x += subcell_force_x[(subcell_index)];
    node_force.y += subcell_force_y[(subcell_index)];
    node_force.z += subcell_force_z[(subcell_index)];
  }

  // Determine the predicted velocity
  velocity_x1[(nn)] = velocity_x0[(nn)] + dt * node_force.x / nodal_mass[(nn)];
  velocity_y1[(nn)] = velocity_y0[(nn)] + dt * node_force.y / nodal_mass[(nn)];
  velocity_z1[(nn)] = velocity_z0[(nn)] + dt * node_force.z / nodal_mass[(nn)];

  // Calculate the time centered velocity
  velocity_x1[(nn)] = 0.5 * (velocity_x0[(nn)] + velocity_x1[(nn)]);
  velocity_y1[(nn)] = 0.5 * (velocity_y0[(nn)] + velocity_y1[(nn)]);
  velocity_z1[(nn)] = 0.5 * (velocity_z0[(nn)] + velocity_z1[(nn)]);
}

// Moves the nodes to the next time level
__global__ void move_nodes(const int nnodes, const double dt,
                           const double* nodes_x0, const double* nodes_y0,
                           const double* nodes_z0, const double* velocity_x1,
                           const double* velocity_y1, const double* velocity_z1,
                           double* nodes_x1, double* nodes_y1,
                           double* nodes_z1) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  nodes_x1[(nn)] = nodes_x0[(nn)] + dt * velocity_x1[(nn)];
  nodes_y1[(nn)] = nodes_y0[(nn)] + dt * velocity_y1[(nn)];
  nodes_z1[(nn)] = nodes_z0[(nn)] + dt * velocity_z1[(nn)];
}

// calculates a new density from the pressure gradients
__global__ void calc_predicted_density(
    const int ncells, const int* cells_to_faces_offsets,
    const int* cells_to_faces, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const double* nodes_x1, const double* nodes_y1,
    const double* nodes_z1, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const double* cell_mass, double* density1) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
  const int nfaces_by_cell =
      cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

  const double cell_volume = calc_cell_volume(
      cc, nfaces_by_cell, cell_to_faces_off, cells_to_faces,
      faces_to_nodes_offsets, faces_to_nodes, nodes_x1, nodes_y1, nodes_z1,
      cell_centroids_x, cell_centroids_y, cell_centroids_z);

  density1[(cc)] = cell_mass[(cc)] / cell_volume;
}

// Time centers the pressure
__global__ void time_center_pressure(const int ncells, const double* energy1,
                                     const double* density1,
                                     const double* pressure0,
                                     double* pressure1) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  // Calculate the predicted pressure from the equation of state
  pressure1[(cc)] = (GAM - 1.0) * energy1[(cc)] * density1[(cc)];

  // Determine the time centered pressure
  pressure1[(cc)] = 0.5 * (pressure0[(cc)] + pressure1[(cc)]);
}

// Time centers the nodal positions
__global__ void time_center_nodes(const int nnodes, const double* nodes_x0,
                                  const double* nodes_y0,
                                  const double* nodes_z0, double* nodes_x1,
                                  double* nodes_y1, double* nodes_z1) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  nodes_x1[(nn)] = 0.5 * (nodes_x1[(nn)] + nodes_x0[(nn)]);
  nodes_y1[(nn)] = 0.5 * (nodes_y1[(nn)] + nodes_y0[(nn)]);
  nodes_z1[(nn)] = 0.5 * (nodes_z1[(nn)] + nodes_z0[(nn)]);
}

// Updates and time center velocity in the corrector step
__global__ void update_and_time_center_velocity(
    const int nnodes, const double dt, const int* nodes_to_cells_offsets,
    const int* nodes_to_cells, const int* cells_to_nodes_offsets,
    const int* cells_to_nodes, const double* nodal_mass,
    const double* subcell_force_x, const double* subcell_force_y,
    const double* subcell_force_z, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* velocity_x1, double* velocity_y1,
    double* velocity_z1) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
  const int ncells_by_node =
      nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

  // Consider all faces attached to node
  vec_t node_force = {0.0, 0.0, 0.0};
  for (int cc = 0; cc < ncells_by_node; ++cc) {
    const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
    const int cell_to_nodes_off = cells_to_nodes_offsets[(cell_index)];
    const int nnodes_by_cell =
        cells_to_nodes_offsets[(cell_index + 1)] - cell_to_nodes_off;

    int nn2;
    for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
      if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
        break;
      }
    }

    node_force.x += subcell_force_x[(cell_to_nodes_off + nn2)];
    node_force.y += subcell_force_y[(cell_to_nodes_off + nn2)];
    node_force.z += subcell_force_z[(cell_to_nodes_off + nn2)];
  }

  // TODO: Do we actually need to update the velocities back here??
  // Calculate the new velocities
  velocity_x1[(nn)] += dt * node_force.x / nodal_mass[(nn)];
  velocity_y1[(nn)] += dt * node_force.y / nodal_mass[(nn)];
  velocity_z1[(nn)] += dt * node_force.z / nodal_mass[(nn)];

  // Calculate the corrected time centered velocities
  velocity_x0[(nn)] = 0.5 * (velocity_x1[(nn)] + velocity_x0[(nn)]);
  velocity_y0[(nn)] = 0.5 * (velocity_y1[(nn)] + velocity_y0[(nn)]);
  velocity_z0[(nn)] = 0.5 * (velocity_z1[(nn)] + velocity_z0[(nn)]);
}

// Advances the nodes using the corrected velocity
__global__ void advance_nodes_corrected(const int nnodes, const double dt,
                                        const double* velocity_x0,
                                        const double* velocity_y0,
                                        const double* velocity_z0,
                                        double* nodes_x0, double* nodes_y0,
                                        double* nodes_z0) {

  const int nn = blockIdx.x * blockDim.x + threadIdx.x;
  if (nn >= nnodes) {
    return;
  }

  nodes_x0[(nn)] += dt * velocity_x0[(nn)];
  nodes_y0[(nn)] += dt * velocity_y0[(nn)];
  nodes_z0[(nn)] += dt * velocity_z0[(nn)];
}

// Calculate the new energy base on subcell forces
__global__ void calc_predicted_energy(
    const int ncells, const double dt, const int* cells_to_nodes_offsets,
    const int* cells_to_nodes, const double* velocity_x1,
    const double* velocity_y1, const double* velocity_z1,
    const double* subcell_force_x, const double* subcell_force_y,
    const double* subcell_force_z, const double* energy0,
    const double* cell_mass, double* energy1) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
  const int nnodes_by_cell =
      cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;

  double cell_force = 0.0;
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
    const int subcell_index = cell_to_nodes_off + nn;
    cell_force +=
        (velocity_x1[(node_index)] * subcell_force_x[(subcell_index)] +
         velocity_y1[(node_index)] * subcell_force_y[(subcell_index)] +
         velocity_z1[(node_index)] * subcell_force_z[(subcell_index)]);
  }
  energy1[(cc)] = energy0[(cc)] - dt * cell_force / cell_mass[(cc)];
}

// Calculates the energy from the correct subcell pressures and velocity
__global__ void calc_corrected_energy(
    const int ncells, const double dt, const int* cells_to_nodes_offsets,
    const int* cells_to_nodes, const double* velocity_x0,
    const double* velocity_y0, const double* velocity_z0,
    const double* subcell_force_x, const double* subcell_force_y,
    const double* subcell_force_z, const double* cell_mass, double* energy0) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
  const int nnodes_by_cell =
      cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;

  double cell_force = 0.0;
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
    const int subcell_index = cell_to_nodes_off + nn;
    cell_force +=
        (velocity_x0[(node_index)] * subcell_force_x[(subcell_index)] +
         velocity_y0[(node_index)] * subcell_force_y[(subcell_index)] +
         velocity_z0[(node_index)] * subcell_force_z[(subcell_index)]);
  }

  energy0[(cc)] -= dt * cell_force / cell_mass[(cc)];
}

// Calculates the density from the corrected volume
__global__ void calc_corrected_density(
    const int ncells, const int* cells_to_faces_offsets,
    const int* cells_to_faces, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const double* cell_mass, double* cell_volume, double* density) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
  const int nfaces_by_cell =
      cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

  cell_volume[(cc)] = calc_cell_volume(
      cc, nfaces_by_cell, cell_to_faces_off, cells_to_faces,
      faces_to_nodes_offsets, faces_to_nodes, nodes_x, nodes_y, nodes_z,
      cell_centroids_x, cell_centroids_y, cell_centroids_z);

  // Update the density using the new volume
  density[(cc)] = cell_mass[(cc)] / cell_volume[(cc)];
}

// Controls the timestep for the simulation
__global__ void calc_timestep(const int ncells, const double* nodes_x,
                              const double* nodes_y, const double* nodes_z,
                              const double* energy, double* dt,
                              int* cells_to_faces_offsets, int* cells_to_faces,
                              int* faces_to_nodes_offsets, int* faces_to_nodes,
                              double* reduce_array) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;

  __shared__ double sdata[NTHREADS];
  sdata[tid] = DBL_MAX;

  if (cc < ncells) {

    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double shortest_edge = DBL_MAX;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // Fetch the nodes attached to our current node on the current face
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];

        const int rnode_index =
            (nn + 1 < nnodes_by_face)
                ? faces_to_nodes[(face_to_nodes_off + nn + 1)]
                : faces_to_nodes[(face_to_nodes_off)];
        const double x_component =
            nodes_x[(node_index)] - nodes_x[(rnode_index)];
        const double y_component =
            nodes_y[(node_index)] - nodes_y[(rnode_index)];
        const double z_component =
            nodes_z[(node_index)] - nodes_z[(rnode_index)];

        // Find the shortest edge of this cell
        shortest_edge = min(shortest_edge, sqrt(x_component * x_component +
                                                y_component * y_component +
                                                z_component * z_component));
      }
    }

    const double soundspeed = sqrt(GAM * (GAM - 1.0) * energy[(cc)]);
    sdata[tid] = shortest_edge / soundspeed;
  }
  __syncthreads();

  min_reduce_in_shared<double, NTHREADS>(tid, sdata);
  if (tid == 0)
    reduce_array[blockIdx.x] = sdata[0];
}

// Calculates the artificial viscous forces for momentum acceleration
__global__ void calc_artificial_viscosity(
    const int ncells, const double visc_coeff1, const double visc_coeff2,
    const int* cells_to_nodes_offsets, const int* cells_to_nodes,
    const int* faces_cclockwise_cell, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* velocity_x,
    const double* velocity_y, const double* velocity_z,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter, double* subcell_force_x,
    double* subcell_force_y, double* subcell_force_z,
    int* faces_to_nodes_offsets, int* faces_to_nodes,
    int* cells_to_faces_offsets, int* cells_to_faces) {

  const int cc = blockIdx.x * blockDim.x + threadIdx.x;
  if (cc >= ncells) {
    return;
  }

  const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
  const int nfaces_by_cell =
      cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
  const int cell_to_nodes_off = cells_to_nodes_offsets[(cc)];
  const int nnodes_by_cell =
      cells_to_nodes_offsets[(cc + 1)] - cell_to_nodes_off;

  // Look at all of the faces attached to the cell
  for (int ff = 0; ff < nfaces_by_cell; ++ff) {
    const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    vec_t face_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                  face_to_nodes_off, &face_c);

    // Now we will sum the contributions at each of the nodes
    for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
      const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
      const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);
      const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
      const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
      const int rnode_off = (face_clockwise ? prev_node : next_node);
      const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];

      // Get the halfway point on the right edge
      vec_t half_edge = {0.5 * (nodes_x[(node_index)] + nodes_x[(rnode_index)]),
                         0.5 * (nodes_y[(node_index)] + nodes_y[(rnode_index)]),
                         0.5 *
                             (nodes_z[(node_index)] + nodes_z[(rnode_index)])};

      // Setup basis on plane of tetrahedron
      vec_t a = {(cell_centroids_x[(cc)] - face_c.x),
                 (cell_centroids_y[(cc)] - face_c.y),
                 (cell_centroids_z[(cc)] - face_c.z)};
      vec_t b = {(half_edge.x - face_c.x), (half_edge.y - face_c.y),
                 (half_edge.z - face_c.z)};

      vec_t S = {0.5 * (a.y * b.z - a.z * b.y), -0.5 * (a.x * b.z - a.z * b.x),
                 0.5 * (a.x * b.y - a.y * b.x)};

      // Calculate the velocity gradients
      vec_t dvel = {velocity_x[(node_index)] - velocity_x[(rnode_index)],
                    velocity_y[(node_index)] - velocity_y[(rnode_index)],
                    velocity_z[(node_index)] - velocity_z[(rnode_index)]};

      const double dvel_mag =
          sqrt(dvel.x * dvel.x + dvel.y * dvel.y + dvel.z * dvel.z);

      // Calculate the unit vectors of the velocity gradients
      vec_t dvel_unit = {(dvel_mag != 0.0) ? dvel.x / dvel_mag : 0.0,
                         (dvel_mag != 0.0) ? dvel.y / dvel_mag : 0.0,
                         (dvel_mag != 0.0) ? dvel.z / dvel_mag : 0.0};

      // Get the edge-centered density
      double nodal_density =
          nodal_mass[(node_index)] / nodal_volumes[(node_index)];
      double rnodal_density =
          nodal_mass[(rnode_index)] / nodal_volumes[(rnode_index)];
      const double density_edge = (2.0 * nodal_density * rnodal_density) /
                                  (nodal_density + rnodal_density);

      // Calculate the artificial viscous force term for the edge
      double expansion_term = (dvel.x * S.x + dvel.y * S.y + dvel.z * S.z);

      // If the cell is compressing, calculate the edge forces and add
      // their contributions to the node forces
      if (expansion_term <= 0.0) {
        // Calculate the minimum soundspeed
        const double cs = min(nodal_soundspeed[(node_index)],
                              nodal_soundspeed[(rnode_index)]);
        const double t = 0.25 * (GAM + 1.0);
        const double edge_visc_force_x =
            density_edge *
            (visc_coeff2 * t * fabs(dvel.x) +
             sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel.x * dvel.x +
                  visc_coeff1 * visc_coeff1 * cs * cs)) *
            (1.0 - limiter[(node_index)]) * expansion_term * dvel_unit.x;
        const double edge_visc_force_y =
            density_edge *
            (visc_coeff2 * t * fabs(dvel.y) +
             sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel.y * dvel.y +
                  visc_coeff1 * visc_coeff1 * cs * cs)) *
            (1.0 - limiter[(node_index)]) * expansion_term * dvel_unit.y;
        const double edge_visc_force_z =
            density_edge *
            (visc_coeff2 * t * fabs(dvel.z) +
             sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel.z * dvel.z +
                  visc_coeff1 * visc_coeff1 * cs * cs)) *
            (1.0 - limiter[(node_index)]) * expansion_term * dvel_unit.z;

        int subcell_index;
        int rsubcell_index;
        for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
          if (cells_to_nodes[(cell_to_nodes_off + nn3)] == node_index) {
            subcell_index = cell_to_nodes_off + nn3;
          } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == rnode_index) {
            rsubcell_index = cell_to_nodes_off + nn3;
          }
        }

        // Add the contributions of the edge based artifical viscous terms
        // to the main force terms
        subcell_force_x[(subcell_index)] += edge_visc_force_x;
        subcell_force_y[(subcell_index)] += edge_visc_force_y;
        subcell_force_z[(subcell_index)] += edge_visc_force_z;
        subcell_force_x[(rsubcell_index)] -= edge_visc_force_x;
        subcell_force_y[(rsubcell_index)] -= edge_visc_force_y;
        subcell_force_z[(rsubcell_index)] -= edge_visc_force_z;
      }
    }
  }
}
