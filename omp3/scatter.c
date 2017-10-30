#include "../../shared.h"
#include "../hale_data.h"
#include "hale.h"
#include <float.h>
#include <stdio.h>

// Limits all of the gradients during flux determination
void limit_momentum_gradients(
    vec_t nodes, vec_t* subcell_c, const double subcell_vx,
    const double subcell_vy, const double subcell_vz, const double gmax_vx,
    const double gmin_vx, const double gmax_vy, const double gmin_vy,
    const double gmax_vz, const double gmin_vz, vec_t* grad_vx, vec_t* grad_vy,
    vec_t* grad_vz, double* vx_limiter, double* vy_limiter, double* vz_limiter);

// Scatter the subcell energy and mass quantities back to the cell centers
void scatter_energy_and_mass(
    const int ncells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, double* cell_volume, double* energy, double* density,
    double* kinetic_energy, double* velocity_x, double* velocity_y,
    double* velocity_z, double* cell_mass, double* subcell_mass,
    double* subcell_ie_mass, double* subcell_ke_mass, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* cells_to_faces_offsets,
    int* cells_to_faces, int* cells_offsets, int* cells_to_nodes,
    int* cells_to_nodes_offsets, double initial_mass, double initial_ie_mass,
    double initial_ke_mass);

// Scatter the subcell momentum to the node centered velocities
void scatter_momentum(
    const int nnodes, vec_t* initial_momentum, int* nodes_to_cells_offsets,
    int* nodes_to_cells, int* cells_to_nodes_offsets, int* cells_to_nodes,
    const int* subcells_to_subcells_offsets, const int* subcells_to_subcells,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const int* faces_cclockwise_cell, const double* nodal_volumes,
    double* velocity_x, double* velocity_y, double* velocity_z,
    double* nodal_mass, double* subcell_mass, const double* subcell_centroids_x,
    const double* subcell_centroids_y, const double* subcell_centroids_z,
    const double* subcell_volume, double* nodes_x, double* nodes_y,
    double* nodes_z, double* subcell_momentum_x, double* subcell_momentum_y,
    double* subcell_momentum_z);

// Perform the scatter step of the ALE remapping algorithm
void scatter_phase(UnstructuredMesh* umesh, HaleData* hale_data,
                   vec_t* initial_momentum, double initial_mass,
                   double initial_ie_mass, double initial_ke_mass) {

  // Calculates the cell volume, subcell volume and the subcell centroids
  calc_volumes_centroids(
      umesh->ncells, umesh->nnodes, hale_data->nnodes_by_subcell,
      umesh->cells_offsets, umesh->cells_to_nodes,
      hale_data->subcells_to_faces_offsets, hale_data->subcells_to_faces,
      umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
      umesh->faces_cclockwise_cell, umesh->nodes_x0, umesh->nodes_y0,
      umesh->nodes_z0, hale_data->subcell_centroids_x,
      hale_data->subcell_centroids_y, hale_data->subcell_centroids_z,
      hale_data->subcell_volume, hale_data->cell_volume,
      hale_data->nodal_volumes, umesh->nodes_offsets, umesh->nodes_to_cells);

#if 0
  // Scatter the subcell momentum to the node centered velocities
  scatter_momentum(
      umesh->nnodes, initial_momentum, umesh->nodes_offsets,
      umesh->nodes_to_cells, umesh->cells_offsets, umesh->cells_to_nodes,
      hale_data->subcells_to_subcells_offsets, hale_data->subcells_to_subcells,
      hale_data->subcells_to_faces_offsets, hale_data->subcells_to_faces,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->faces_cclockwise_cell, hale_data->nodal_volumes,
      hale_data->velocity_x0, hale_data->velocity_y0, hale_data->velocity_z0,
      hale_data->nodal_mass, hale_data->subcell_mass,
      hale_data->subcell_centroids_x, hale_data->subcell_centroids_y,
      hale_data->subcell_centroids_z, hale_data->subcell_volume,
      umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
      hale_data->subcell_momentum_x, hale_data->subcell_momentum_y,
      hale_data->subcell_momentum_z);
#endif // if 0

  // Scatter the subcell energy and mass quantities back to the cell centers
  scatter_energy_and_mass(
      umesh->ncells, umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
      hale_data->cell_volume, hale_data->energy0, hale_data->density0,
      hale_data->ke_mass, hale_data->velocity_x0, hale_data->velocity_y0,
      hale_data->velocity_z0, hale_data->cell_mass, hale_data->subcell_mass,
      hale_data->subcell_ie_mass, hale_data->subcell_ke_mass,
      umesh->faces_to_nodes, umesh->faces_to_nodes_offsets,
      umesh->cells_to_faces_offsets, umesh->cells_to_faces,
      umesh->cells_offsets, umesh->cells_to_nodes, umesh->cells_offsets,
      initial_mass, initial_ie_mass, initial_ke_mass);
}

// Scatter the subcell energy and mass quantities back to the cell centers
void scatter_energy_and_mass(
    const int ncells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, double* cell_volume, double* energy, double* density,
    double* ke_mass, double* velocity_x, double* velocity_y, double* velocity_z,
    double* cell_mass, double* subcell_mass, double* subcell_ie_mass,
    double* subcell_ke_mass, int* faces_to_nodes, int* faces_to_nodes_offsets,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_offsets,
    int* cells_to_nodes, int* cells_to_nodes_offsets, double initial_mass,
    double initial_ie_mass, double initial_ke_mass) {

  // Scatter energy and density, and print the conservation of mass
  double rz_total_mass = 0.0;
  double rz_total_e_mass = 0.0;
#pragma omp parallel for reduction(+ : rz_total_mass, rz_total_e_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double new_ke_mass = 0.0;
    double total_mass = 0.0;
    double total_ie_mass = 0.0;
    double total_ke_mass = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
      const int subcell_index = cell_to_nodes_off + nn;
      total_mass += subcell_mass[(subcell_index)];
      total_ie_mass += subcell_ie_mass[(subcell_index)];
      total_ke_mass += subcell_ke_mass[(subcell_index)];
      new_ke_mass += subcell_mass[(subcell_index)] *
                     (velocity_x[(node_index)] * velocity_x[(node_index)] +
                      velocity_y[(node_index)] * velocity_y[(node_index)] +
                      velocity_z[(node_index)] * velocity_z[(node_index)]);
    }

    // Update the volume of the cell to the new rezoned mesh
    vec_t cell_c = {0.0, 0.0, 0.0};
    calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                  cell_to_nodes_off, &cell_c);
    calc_volume(cell_to_faces_off, nfaces_by_cell, cells_to_faces,
                faces_to_nodes, faces_to_nodes_offsets, nodes_x, nodes_y,
                nodes_z, &cell_c, &cell_volume[(cc)]);

    // Scatter the energy and density
    cell_mass[(cc)] = total_mass;
    density[(cc)] = cell_mass[(cc)] / cell_volume[(cc)];

    double total_e_mass = total_ie_mass; // + (total_ke_mass - new_ke_mass);
    energy[(cc)] = total_e_mass / cell_mass[(cc)];

    // Calculate the conservation data
    rz_total_mass += total_mass;
    rz_total_e_mass += total_e_mass;
  }

  printf("Initial Total Mass %.12f\n", initial_mass);
  printf("Rezoned Total Mass %.12f\n", rz_total_mass);
  printf("Difference         %.12f\n\n", rz_total_mass - initial_mass);

  printf("Initial Total Energy          %.12f\n",
         (initial_ie_mass + initial_ke_mass));
  printf("Rezoned Total Internal Energy %.12f\n", rz_total_e_mass);
  printf("Difference                    %.12f\n\n",
         rz_total_e_mass - (initial_ie_mass + initial_ke_mass));
}

#if 0
// Scatter the subcell momentum to the node centered velocities
void scatter_momentum(const int nnodes, vec_t* initial_momentum,
                      int* nodes_to_cells_offsets, int* nodes_to_cells,
                      int* cells_to_nodes_offsets, int* cells_to_nodes,
                      double* velocity_x, double* velocity_y,
                      double* velocity_z, double* nodal_mass,
                      double* subcell_mass, double* subcell_momentum_x,
                      double* subcell_momentum_y, double* subcell_momentum_z) {

  double total_momentum_x = 0.0;
  double total_momentum_y = 0.0;
  double total_momentum_z = 0.0;

#pragma omp parallel for reduction(+ : total_momentum_x, total_momentum_y,     \
                                   total_momentum_z)
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
    const int ncells_by_node =
        nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

    double mass_at_node = 0.0;
    double node_momentum_x = 0.0;
    double node_momentum_y = 0.0;
    double node_momentum_z = 0.0;

    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_to_nodes_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_to_nodes_offsets[(cell_index + 1)] - cell_to_nodes_off;

      // Determine the position of the node in the cell
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
          break;
        }
      }

      const int subcell_index = cell_to_nodes_off + nn2;
      node_momentum_x += subcell_momentum_x[(subcell_index)];
      node_momentum_y += subcell_momentum_y[(subcell_index)];
      node_momentum_z += subcell_momentum_z[(subcell_index)];
      mass_at_node += subcell_mass[(subcell_index)];
    }

    nodal_mass[(nn)] = mass_at_node;

    total_momentum_x += node_momentum_x;
    total_momentum_y += node_momentum_y;
    total_momentum_z += node_momentum_z;

    velocity_x[(nn)] = node_momentum_x / nodal_mass[(nn)];
    velocity_y[(nn)] = node_momentum_y / nodal_mass[(nn)];
    velocity_z[(nn)] = node_momentum_z / nodal_mass[(nn)];
  }

  printf("Initial total momentum %.12f %.12f %.12f\n", initial_momentum->x,
         initial_momentum->y, initial_momentum->z);
  printf("Rezoned total momentum %.12f %.12f %.12f\n", total_momentum_x,
         total_momentum_y, total_momentum_z);
  printf("Difference             %.12f %.12f %.12f\n\n",
         initial_momentum->x - total_momentum_x,
         initial_momentum->y - total_momentum_y,
         initial_momentum->z - total_momentum_z);
}
#endif // if 0

// Scatter the subcell momentum to the node centered velocities
void scatter_momentum(
    const int nnodes, vec_t* initial_momentum, int* nodes_to_cells_offsets,
    int* nodes_to_cells, int* cells_to_nodes_offsets, int* cells_to_nodes,
    const int* subcells_to_subcells_offsets, const int* subcells_to_subcells,
    const int* subcells_to_faces_offsets, const int* subcells_to_faces,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const int* faces_cclockwise_cell, const double* nodal_volumes,
    double* velocity_x, double* velocity_y, double* velocity_z,
    double* nodal_mass, double* subcell_mass, const double* subcell_centroids_x,
    const double* subcell_centroids_y, const double* subcell_centroids_z,
    const double* subcell_volume, double* nodes_x, double* nodes_y,
    double* nodes_z, double* subcell_momentum_x, double* subcell_momentum_y,
    double* subcell_momentum_z) {

  double total_momentum_x = 0.0;
  double total_momentum_y = 0.0;
  double total_momentum_z = 0.0;

#if 0
#pragma omp parallel for reduction(+ : total_momentum_x, total_momentum_y,     \
                                   total_momentum_z)
#endif // if 0
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_cells_off = nodes_to_cells_offsets[(nn)];
    const int ncells_by_node =
        nodes_to_cells_offsets[(nn + 1)] - node_to_cells_off;

    vec_t node = {nodes_x[(nn)], nodes_y[(nn)], nodes_z[(nn)]};
    vec_t node_momentum = {0.0, 0.0, 0.0};

    for (int cc = 0; cc < ncells_by_node; ++cc) {
      const int cell_index = nodes_to_cells[(node_to_cells_off + cc)];
      const int cell_to_nodes_off = cells_to_nodes_offsets[(cell_index)];
      const int nnodes_by_cell =
          cells_to_nodes_offsets[(cell_index + 1)] - cell_to_nodes_off;

      // Determine the position of the node in the cell
      int nn2;
      for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
        if (cells_to_nodes[(cell_to_nodes_off + nn2)] == nn) {
          break;
        }
      }

      const int subcell_index = cell_to_nodes_off + nn2;
      const int subcell_to_subcells_off =
          subcells_to_subcells_offsets[(subcell_index)];
      const int nsubcells_by_subcell =
          subcells_to_subcells_offsets[(subcell_index + 1)] -
          subcell_to_subcells_off;

      /* CALCULATE THE SWEEP SUBCELL GRADIENTS FOR MASS AND ENERGY */

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

      vec_t subcell_c = {subcell_centroids_x[(subcell_index)],
                         subcell_centroids_y[(subcell_index)],
                         subcell_centroids_z[(subcell_index)]};

      const double subcell_vol = subcell_volume[(subcell_index)];
      vec_t subcell_v = {subcell_momentum_x[(subcell_index)] / subcell_vol,
                         subcell_momentum_y[(subcell_index)] / subcell_vol,
                         subcell_momentum_z[(subcell_index)] / subcell_vol};

      for (int ss = 0; ss < nsubcells_by_subcell; ++ss) {
        const int subcell_neighbour_index =
            subcells_to_subcells[(subcell_to_subcells_off + ss)];

        // Only perform the sweep on the external face if it isn't a
        // boundary
        if (subcell_neighbour_index == -1) {
          continue;
        }

        const double neighbour_vol = subcell_volume[(subcell_neighbour_index)];
        vec_t i = {
            (subcell_centroids_x[(subcell_neighbour_index)] - subcell_c.x) *
                neighbour_vol,
            (subcell_centroids_y[(subcell_neighbour_index)] - subcell_c.y) *
                neighbour_vol,
            (subcell_centroids_z[(subcell_neighbour_index)] - subcell_c.z) *
                neighbour_vol};

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

        vec_t neighbour_v = {
            subcell_momentum_x[(subcell_neighbour_index)] / neighbour_vol,
            subcell_momentum_y[(subcell_neighbour_index)] / neighbour_vol,
            subcell_momentum_z[(subcell_neighbour_index)] / neighbour_vol};

        // Prepare differential
        const double dneighbour_vx = (neighbour_v.x - subcell_v.x);
        const double dneighbour_vy = (neighbour_v.y - subcell_v.y);
        const double dneighbour_vz = (neighbour_v.z - subcell_v.z);

        vx_rhs.x += 2.0 * dneighbour_vx * i.x / neighbour_vol;
        vx_rhs.y += 2.0 * dneighbour_vx * i.y / neighbour_vol;
        vx_rhs.z += 2.0 * dneighbour_vx * i.z / neighbour_vol;

        vy_rhs.x += 2.0 * dneighbour_vy * i.x / neighbour_vol;
        vy_rhs.y += 2.0 * dneighbour_vy * i.y / neighbour_vol;
        vy_rhs.z += 2.0 * dneighbour_vy * i.z / neighbour_vol;

        vz_rhs.x += 2.0 * dneighbour_vz * i.x / neighbour_vol;
        vz_rhs.y += 2.0 * dneighbour_vz * i.y / neighbour_vol;
        vz_rhs.z += 2.0 * dneighbour_vz * i.z / neighbour_vol;

        gmax_vx = max(gmax_vx, neighbour_v.x);
        gmin_vx = min(gmin_vx, neighbour_v.x);
        gmax_vy = max(gmax_vy, neighbour_v.y);
        gmin_vy = min(gmin_vy, neighbour_v.y);
        gmax_vz = max(gmax_vz, neighbour_v.z);
        gmin_vz = min(gmin_vz, neighbour_v.z);
      }

      calc_3x3_inverse(&coeff, &inv);

      // Calculate the gradient for momentum
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

      double vx_limiter = 1.0;
      double vy_limiter = 1.0;
      double vz_limiter = 1.0;

      // Limit at node
      limit_momentum_gradients(node, &subcell_c, subcell_v.x, subcell_v.y,
                               subcell_v.z, gmax_vx, gmin_vx, gmax_vy, gmin_vy,
                               gmax_vz, gmin_vz, &grad_vx, &grad_vy, &grad_vz,
                               &vx_limiter, &vy_limiter, &vz_limiter);

      // Limit at cell center
      vec_t cell_c = {0.0, 0.0, 0.0};
      calc_centroid(nnodes_by_cell, nodes_x, nodes_y, nodes_z, cells_to_nodes,
                    cell_to_nodes_off, &cell_c);
      limit_momentum_gradients(cell_c, &subcell_c, subcell_v.x, subcell_v.y,
                               subcell_v.z, gmax_vx, gmin_vx, gmax_vy, gmin_vy,
                               gmax_vz, gmin_vz, &grad_vx, &grad_vy, &grad_vz,
                               &vx_limiter, &vy_limiter, &vz_limiter);

      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // Limit at half edges and face centers
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // The face centroid is the same for all nodes on the face
        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x, nodes_y, nodes_z, faces_to_nodes,
                      face_to_nodes_off, &face_c);

        limit_momentum_gradients(
            face_c, &subcell_c, subcell_v.x, subcell_v.y, subcell_v.z, gmax_vx,
            gmin_vx, gmax_vy, gmin_vy, gmax_vz, gmin_vz, &grad_vx, &grad_vy,
            &grad_vz, &vx_limiter, &vy_limiter, &vz_limiter);

        // Determine the position of the node in the face list of nodes
        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          if (faces_to_nodes[(face_to_nodes_off + nn2)] == nn) {
            break;
          }
        }

        const int face_clockwise = (faces_cclockwise_cell[(face_index)] != cc);
        const int next_node = (nn2 == nnodes_by_face - 1) ? 0 : nn2 + 1;
        const int prev_node = (nn2 == 0) ? nnodes_by_face - 1 : nn2 - 1;
        const int rnode_off = (face_clockwise ? prev_node : next_node);
        const int rnode_index = faces_to_nodes[(face_to_nodes_off + rnode_off)];

        // Get the halfway point on the right edge
        vec_t half_edge = {0.5 * (node.x + nodes_x[(rnode_index)]),
                           0.5 * (node.y + nodes_y[(rnode_index)]),
                           0.5 * (node.z + nodes_z[(rnode_index)])};

        limit_momentum_gradients(
            half_edge, &subcell_c, subcell_v.x, subcell_v.y, subcell_v.z,
            gmax_vx, gmin_vx, gmax_vy, gmin_vy, gmax_vz, gmin_vz, &grad_vx,
            &grad_vy, &grad_vz, &vx_limiter, &vy_limiter, &vz_limiter);
      }

      const double dx = node.x - subcell_c.x;
      const double dy = node.y - subcell_c.y;
      const double dz = node.z - subcell_c.z;
      node_momentum.x += subcell_vol * (subcell_v.x + grad_vx.x * dx +
                                        grad_vx.y * dy + grad_vx.z * dz);
      node_momentum.y += subcell_vol * (subcell_v.y + grad_vy.x * dx +
                                        grad_vy.y * dy + grad_vy.z * dz);
      node_momentum.z += subcell_vol * (subcell_v.z + grad_vz.x * dx +
                                        grad_vz.y * dy + grad_vz.z * dz);
    }

    velocity_x[(nn)] = node_momentum.x / nodal_mass[(nn)];
    velocity_y[(nn)] = node_momentum.y / nodal_mass[(nn)];
    velocity_z[(nn)] = node_momentum.z / nodal_mass[(nn)];
  }

  printf("Initial total momentum %.12f %.12f %.12f\n", initial_momentum->x,
         initial_momentum->y, initial_momentum->z);
  printf("Rezoned total momentum %.12f %.12f %.12f\n", total_momentum_x,
         total_momentum_y, total_momentum_z);
  printf("Difference             %.12f %.12f %.12f\n\n",
         initial_momentum->x - total_momentum_x,
         initial_momentum->y - total_momentum_y,
         initial_momentum->z - total_momentum_z);
}

// Limits all of the gradients during flux determination
void limit_momentum_gradients(vec_t nodes, vec_t* subcell_c,
                              const double subcell_vx, const double subcell_vy,
                              const double subcell_vz, const double gmax_vx,
                              const double gmin_vx, const double gmax_vy,
                              const double gmin_vy, const double gmax_vz,
                              const double gmin_vz, vec_t* grad_vx,
                              vec_t* grad_vy, vec_t* grad_vz,
                              double* vx_limiter, double* vy_limiter,
                              double* vz_limiter) {

  *vx_limiter =
      min(*vx_limiter, calc_cell_limiter(subcell_vx, gmax_vx, gmin_vx, grad_vx,
                                         nodes.x, nodes.y, nodes.z, subcell_c));
  *vy_limiter =
      min(*vy_limiter, calc_cell_limiter(subcell_vy, gmax_vy, gmin_vy, grad_vy,
                                         nodes.x, nodes.y, nodes.z, subcell_c));
  *vz_limiter =
      min(*vz_limiter, calc_cell_limiter(subcell_vz, gmax_vz, gmin_vz, grad_vz,
                                         nodes.x, nodes.y, nodes.z, subcell_c));
}
