#include "../../comms.h"
#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>

// Gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(
    const int ncells, double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_to_nodes, int* cells_offsets,
    double* nodes_x0, double* nodes_y0, double* nodes_z0, double* energy0,
    double* density0, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* cell_mass, double* subcell_volume,
    double* subcell_ie_density, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_integrals_x,
    double* subcell_integrals_y, double* subcell_integrals_z,
    int* nodes_to_faces_offsets, int* nodes_to_faces, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces) {

// Collect the sub-cell centered velocities
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // Calculate the weighted velocity at the sub-cell center
    vec_t uc = {0.0, 0.0, 0.0};
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = (cells_to_nodes[(cell_to_nodes_off + nn)]);
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;

      int Sn = 0;
      vec_t b = {0.0, 0.0, 0.0};
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if ((faces_to_cells0[(face_index)] != cc &&
             faces_to_cells1[(face_index)] != cc) ||
            face_index == -1) {
          continue;
        }

        // We have encountered a true face
        Sn += 2;

        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Look at all of the nodes around a face
        int node;
        vec_t f = {0.0, 0.0, 0.0};
        double face_mass = 0.0;
        for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
          const int node_index0 = faces_to_nodes[(face_to_nodes_off + nn2)];
          const int node_l_index =
              (nn2 == 0)
                  ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                  : faces_to_nodes[(face_to_nodes_off + nn2 - 1)];
          const int node_r_index =
              (nn2 == nnodes_by_face - 1)
                  ? faces_to_nodes[(face_to_nodes_off)]
                  : faces_to_nodes[(face_to_nodes_off + nn2 + 1)];

          // Add the face center contributions
          double mass = subcell_mass[(cell_to_nodes_off + nn2)];
          f.x += mass * (2.0 * velocity_x0[(node_index0)] -
                         0.5 * velocity_x0[(node_l_index)] -
                         0.5 * velocity_x0[(node_r_index)]);
          f.y += mass * (2.0 * velocity_y0[(node_index0)] -
                         0.5 * velocity_y0[(node_l_index)] -
                         0.5 * velocity_y0[(node_r_index)]);
          f.z += mass * (2.0 * velocity_z0[(node_index0)] -
                         0.5 * velocity_z0[(node_l_index)] -
                         0.5 * velocity_z0[(node_r_index)]);
          face_mass += mass;
          if (node_index0 == node_index) {
            node = nn2;
          }
        }

        const int node_l_index =
            (node == 0)
                ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                : faces_to_nodes[(face_to_nodes_off + node - 1)];
        const int node_r_index =
            (node == nnodes_by_face - 1)
                ? faces_to_nodes[(face_to_nodes_off)]
                : faces_to_nodes[(face_to_nodes_off + node + 1)];

        // Add contributions for right, left, and face center
        b.x += 0.5 * velocity_x0[(node_l_index)] +
               0.5 * velocity_x0[(node_r_index)] + 2.0 * f.x / face_mass;
        b.y += 0.5 * velocity_y0[(node_l_index)] +
               0.5 * velocity_y0[(node_r_index)] + 2.0 * f.y / face_mass;
        b.z += 0.5 * velocity_z0[(node_l_index)] +
               0.5 * velocity_z0[(node_r_index)] + 2.0 * f.z / face_mass;
      }

      double mass = subcell_mass[(cell_to_nodes_off + nn)];
      uc.x += mass * (2.5 * velocity_x0[(node_index)] - (b.x / Sn)) /
              cell_mass[(cc)];
      uc.y += mass * (2.5 * velocity_y0[(node_index)] - (b.y / Sn)) /
              cell_mass[(cc)];
      uc.z += mass * (2.5 * velocity_z0[(node_index)] - (b.z / Sn)) /
              cell_mass[(cc)];
    }

    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = (cells_to_nodes[(cell_to_nodes_off + nn)]);
      const int node_to_faces_off = nodes_to_faces_offsets[(node_index)];
      const int nfaces_by_node =
          nodes_to_faces_offsets[(node_index + 1)] - node_to_faces_off;

      int Sn = 0;
      vec_t b = {0.0, 0.0, 0.0};
      for (int ff = 0; ff < nfaces_by_node; ++ff) {
        const int face_index = nodes_to_faces[(node_to_faces_off + ff)];
        if ((faces_to_cells0[(face_index)] != cc &&
             faces_to_cells1[(face_index)] != cc) ||
            face_index == -1) {
          continue;
        }

        // We have encountered a true face
        Sn += 2;

        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        int node;
        vec_t f = {0.0, 0.0, 0.0};
        double face_mass = 0.0;
        for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {

          const int node_index0 = faces_to_nodes[(face_to_nodes_off + nn2)];
          const int node_l_index =
              (nn2 == 0)
                  ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                  : faces_to_nodes[(face_to_nodes_off + nn2 - 1)];
          const int node_r_index =
              (nn2 == nnodes_by_face - 1)
                  ? faces_to_nodes[(face_to_nodes_off)]
                  : faces_to_nodes[(face_to_nodes_off + nn2 + 1)];

          // Add the face center contributions
          double mass = subcell_mass[(cell_to_nodes_off + nn2)];
          f.x += mass * (2.0 * velocity_x0[(node_index0)] -
                         0.5 * velocity_x0[(node_l_index)] -
                         0.5 * velocity_x0[(node_r_index)]);
          f.y += mass * (2.0 * velocity_y0[(node_index0)] -
                         0.5 * velocity_y0[(node_l_index)] -
                         0.5 * velocity_y0[(node_r_index)]);
          f.z += mass * (2.0 * velocity_z0[(node_index0)] -
                         0.5 * velocity_z0[(node_l_index)] -
                         0.5 * velocity_z0[(node_r_index)]);
          face_mass += mass;

          if (node_index0 == node_index) {
            node = nn2;
          }
        }

        const int node_l_index =
            (node == 0)
                ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                : faces_to_nodes[(face_to_nodes_off + node - 1)];
        const int node_r_index =
            (node == nnodes_by_face - 1)
                ? faces_to_nodes[(face_to_nodes_off)]
                : faces_to_nodes[(face_to_nodes_off + node + 1)];

        // Add right and left node contributions
        b.x += 0.5 * velocity_x0[(node_l_index)] +
               0.5 * velocity_x0[(node_r_index)] + 2.0 * f.x / face_mass;
        b.y += 0.5 * velocity_y0[(node_l_index)] +
               0.5 * velocity_y0[(node_r_index)] + 2.0 * f.y / face_mass;
        b.z += 0.5 * velocity_z0[(node_l_index)] +
               0.5 * velocity_z0[(node_r_index)] + 2.0 * f.z / face_mass;
      }

      // Calculate the final sub-cell velocities
      subcell_velocity_x[(cell_to_nodes_off + nn)] =
          0.25 * (1.5 * velocity_x0[(node_index)] + uc.x + b.x / Sn);
      subcell_velocity_y[(cell_to_nodes_off + nn)] =
          0.25 * (1.5 * velocity_y0[(node_index)] + uc.y + b.y / Sn);
      subcell_velocity_z[(cell_to_nodes_off + nn)] =
          0.25 * (1.5 * velocity_z0[(node_index)] + uc.z + b.z / Sn);
    }
  }

/*
*      GATHERING STAGE OF THE REMAP
*/

// Calculate the sub-cell internal energies
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    // Calculating the volume integrals necessary for the least squares
    // regression
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    // The coefficients of the 3x3 gradient coefficient matrix
    vec_t coeff[3] = {{0.0, 0.0, 0.0}};
    vec_t rhs = {0.0, 0.0, 0.0};

    // Determine the weighted volume integrals for neighbouring cells
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int neighbour_index = (faces_to_cells0[(face_index)] == cc)
                                      ? faces_to_cells1[(face_index)]
                                      : faces_to_cells0[(face_index)];
      // Check if boundary face
      if (neighbour_index == -1) {
        continue;
      }

      const int neighbour_to_faces_off =
          cells_to_faces_offsets[(neighbour_index)];
      const int nfaces_by_neighbour =
          cells_to_faces_offsets[(neighbour_index + 1)] -
          neighbour_to_faces_off;

      // Calculate the weighted volume integral coefficients
      double vol = 0.0;
      vec_t integrals = {0.0, 0.0, 0.0};
      vec_t neighbour_centroid = {0.0, 0.0, 0.0};
      neighbour_centroid.x = cell_centroids_x[(neighbour_index)];
      neighbour_centroid.y = cell_centroids_y[(neighbour_index)];
      neighbour_centroid.z = cell_centroids_z[(neighbour_index)];
      calc_weighted_volume_integrals(
          neighbour_to_faces_off, nfaces_by_neighbour, cells_to_faces,
          faces_to_nodes, faces_to_nodes_offsets, nodes_x0, nodes_y0, nodes_z0,
          &neighbour_centroid, &integrals, &vol);

      // Complete the integral coefficient as a distance
      integrals.x -= cell_centroid.x * vol;
      integrals.y -= cell_centroid.y * vol;
      integrals.z -= cell_centroid.z * vol;

      // Store the neighbouring cell's contribution to the coefficients
      coeff[0].x += (2.0 * integrals.x * integrals.x) / (vol * vol);
      coeff[0].y += (2.0 * integrals.x * integrals.y) / (vol * vol);
      coeff[0].z += (2.0 * integrals.x * integrals.z) / (vol * vol);

      coeff[1].x += (2.0 * integrals.y * integrals.x) / (vol * vol);
      coeff[1].y += (2.0 * integrals.y * integrals.y) / (vol * vol);
      coeff[1].z += (2.0 * integrals.y * integrals.z) / (vol * vol);

      coeff[2].x += (2.0 * integrals.z * integrals.x) / (vol * vol);
      coeff[2].y += (2.0 * integrals.z * integrals.y) / (vol * vol);
      coeff[2].z += (2.0 * integrals.z * integrals.z) / (vol * vol);

      // Prepare the RHS, which includes energy differential
      const double de = (energy0[(neighbour_index)] - energy0[(cc)]);
      rhs.x += (2.0 * integrals.x * de / vol);
      rhs.y += (2.0 * integrals.y * de / vol);
      rhs.z += (2.0 * integrals.z * de / vol);
    }

    // Determine the inverse of the coefficient matrix
    vec_t inv[3];
    calc_3x3_inverse(&coeff, &inv);

    // Solve for the energy gradient
    vec_t grad_energy;
    grad_energy.x = inv[0].x * rhs.x + inv[0].y * rhs.y + inv[0].z * rhs.z;
    grad_energy.y = inv[1].x * rhs.x + inv[1].y * rhs.y + inv[1].z * rhs.z;
    grad_energy.z = inv[2].x * rhs.x + inv[2].y * rhs.y + inv[2].z * rhs.z;

    // Describe the connectivity for a simple tetrahedron, the sub-cell shape
    const int subcell_faces_to_nodes_offsets[NTET_FACES + 1] = {0, 3, 6, 9, 12};
    const int subcell_faces_to_nodes[NTET_FACES * NTET_NODES_PER_FACE] = {
        0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};
    const int subcell_to_faces[NTET_FACES] = {0, 1, 2, 3};
    double subcell_nodes_x[NTET_NODES] = {0.0};
    double subcell_nodes_y[NTET_NODES] = {0.0};
    double subcell_nodes_z[NTET_NODES] = {0.0};

    // The centroid remains a component of all sub-cells
    subcell_nodes_x[3] = cell_centroid.x;
    subcell_nodes_y[3] = cell_centroid.y;
    subcell_nodes_z[3] = cell_centroid.z;

    // Determine the weighted volume integrals for neighbouring cells
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // TODO: SHOULD WE PRECOMPUTE THE FACE CENTROID???
      // The face centroid is the same for all nodes on the face
      subcell_nodes_x[2] = 0.0;
      subcell_nodes_y[2] = 0.0;
      subcell_nodes_z[2] = 0.0;
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        subcell_nodes_x[2] += nodes_x0[(node_index)] / nnodes_by_face;
        subcell_nodes_y[2] += nodes_y0[(node_index)] / nnodes_by_face;
        subcell_nodes_z[2] += nodes_z0[(node_index)] / nnodes_by_face;
      }

      // Each face/node pair has two sub-cells
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];

        // TODO: HAVE MOVED THIS, CHECK IT WORKS....
        // Find the node offset in the cell
        int nn2;
        for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
          if (cells_to_nodes[(cell_to_nodes_off + nn2)] == node_index) {
            break;
          }
        }

// The left and right nodes on the face for this anchor node
#define NNEIGHBOUR_NODES 2
        int nodes_a[NNEIGHBOUR_NODES];
        nodes_a[0] =
            (nn == 0) ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                      : faces_to_nodes[(face_to_nodes_off + nn - 1)];
        nodes_a[1] = (nn == nnodes_by_face - 1)
                         ? faces_to_nodes[(face_to_nodes_off)]
                         : faces_to_nodes[(face_to_nodes_off + nn + 1)];

        // Loop over both of the neighbour nodes
        for (int ss = 0; ss < NNEIGHBOUR_NODES; ++ss) {
          // Store the right and left nodes
          subcell_nodes_x[1] =
              0.5 * (nodes_x0[nodes_a[ss]] + nodes_x0[(node_index)]);
          subcell_nodes_y[1] =
              0.5 * (nodes_y0[nodes_a[ss]] + nodes_y0[(node_index)]);
          subcell_nodes_z[1] =
              0.5 * (nodes_z0[nodes_a[ss]] + nodes_z0[(node_index)]);

          // Store the anchor node
          subcell_nodes_x[0] = nodes_x0[(node_index)];
          subcell_nodes_y[0] = nodes_y0[(node_index)];
          subcell_nodes_z[0] = nodes_z0[(node_index)];

          // Determine the sub-cell centroid
          vec_t subcell_centroid = {0.0, 0.0, 0.0};
          for (int ii = 0; ii < NTET_NODES; ++ii) {
            subcell_centroid.x += subcell_nodes_x[ii] / NTET_NODES;
            subcell_centroid.y += subcell_nodes_y[ii] / NTET_NODES;
            subcell_centroid.z += subcell_nodes_z[ii] / NTET_NODES;
          }

          // Calculate the weighted volume integral coefficients
          double vol = 0.0;
          vec_t integrals = {0.0, 0.0, 0.0};
          calc_weighted_volume_integrals(
              0, NTET_FACES, subcell_to_faces, subcell_faces_to_nodes,
              subcell_faces_to_nodes_offsets, subcell_nodes_x, subcell_nodes_y,
              subcell_nodes_z, &subcell_centroid, &integrals, &vol);

          // TODO: THIS MIGHT BE A STUPID WAY TO DO THIS.
          // WE ARE LOOKING AT ALL OF THE SUBCELL TETRAHEDRONS, WHEN WE COULD BE
          // LOOKING AT A SINGLE CORNER SUBCELL PER NODE

          // Store the weighted integrals
          subcell_integrals_x[(cell_to_nodes_off + nn2)] += integrals.x;
          subcell_integrals_y[(cell_to_nodes_off + nn2)] += integrals.y;
          subcell_integrals_z[(cell_to_nodes_off + nn2)] += integrals.z;
          subcell_volume[(cell_to_nodes_off + nn2)] += vol;

          // Determine subcell energy from linear function at cell
          subcell_ie_density[(cell_to_nodes_off + nn2)] +=
              vol * (density0[(cc)] * energy0[(cc)] -
                     (grad_energy.x * cell_centroid.x +
                      grad_energy.y * cell_centroid.y +
                      grad_energy.z * cell_centroid.z)) +
              grad_energy.x * integrals.x + grad_energy.y * integrals.y +
              grad_energy.z * integrals.z;
        }
      }
    }
  }
}

// Calculates the face integral for the provided face, projected onto
// the two-dimensional basis
void calc_projections(const int nnodes_by_face, const int face_to_nodes_off,
                      const int* faces_to_nodes, const double* alpha,
                      const double* beta, pi_t* pi) {

  double pione = 0.0;
  double pialpha = 0.0;
  double pialpha2 = 0.0;
  double pibeta = 0.0;
  double pibeta2 = 0.0;
  double pialphabeta = 0.0;

  // Calculate the coefficients for the projected face integral
  for (int nn = 0; nn < nnodes_by_face; ++nn) {
    const int n0 = faces_to_nodes[(face_to_nodes_off + nn)];
    const int n1 = (nn == nnodes_by_face - 1)
                       ? faces_to_nodes[(face_to_nodes_off)]
                       : faces_to_nodes[(face_to_nodes_off + nn + 1)];

    // Calculate all of the coefficients
    const double a0 = alpha[(n0)];
    const double a1 = alpha[(n1)];
    const double b0 = beta[(n0)];
    const double b1 = beta[(n1)];
    const double dalpha = a1 - a0;
    const double dbeta = b1 - b0;
    const double Calpha = a1 * (a1 + a0) + a0 * a0;
    const double Cbeta = b1 * b1 + b1 * b0 + b0 * b0;
    const double Calphabeta = 3.0 * a1 * a1 + 2.0 * a1 * a0 + a0 * a0;
    const double Kalphabeta = a1 * a1 + 2.0 * a1 * a0 + 3.0 * a0 * a0;

    // Accumulate the projection integrals
    pione += dbeta * (a1 + a0) / 2.0;
    pialpha += dbeta * (Calpha) / 6.0;
    pialpha2 += dbeta * (a1 * Calpha + a0 * a0 * a0) / 12.0;
    pibeta -= dalpha * (Cbeta) / 6.0;
    pibeta2 -= dalpha * (b1 * Cbeta + b0 * b0 * b0) / 12.0;
    pialphabeta += dbeta * (b1 * Calphabeta + b0 * Kalphabeta) / 24.0;
  }

  // Store the final coefficients, flipping all results if we went through
  // in a clockwise order and got a negative area
  const double flip = (pione > 0.0 ? 1.0 : -1.0);
  pi->one += flip * pione;
  pi->alpha += flip * pialpha;
  pi->alpha2 += flip * pialpha2;
  pi->beta += flip * pibeta;
  pi->beta2 += flip * pibeta2;
  pi->alpha_beta += flip * pialphabeta;
}

// Resolves the volume integrals in alpha-beta-gamma basis
void calc_face_integrals(const int nnodes_by_face, const int face_to_nodes_off,
                         const int orientation, const int n0,
                         const int* faces_to_nodes, const double* nodes_alpha,
                         const double* nodes_beta, const double* nodes_gamma,
                         vec_t normal, vec_t* T, double* vol) {

  pi_t pi = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  calc_projections(nnodes_by_face, face_to_nodes_off, faces_to_nodes,
                   nodes_alpha, nodes_beta, &pi);

  // The projection of the normal vector onto a point on the face
  double omega = -(normal.x * nodes_alpha[(n0)] + normal.y * nodes_beta[(n0)] +
                   normal.z * nodes_gamma[(n0)]);

  // Finalise the weighted face integrals
  const double Falpha = pi.alpha / fabs(normal.z);
  const double Fbeta = pi.beta / fabs(normal.z);
  const double Fgamma =
      -(normal.x * pi.alpha + normal.y * pi.beta + omega * pi.one) /
      (fabs(normal.z) * normal.z);

  const double Falpha2 = pi.alpha2 / fabs(normal.z);
  const double Fbeta2 = pi.beta2 / fabs(normal.z);
  const double Fgamma2 =
      (normal.x * normal.x * pi.alpha2 +
       2.0 * normal.x * normal.y * pi.alpha_beta +
       normal.y * normal.y * pi.beta2 + 2.0 * normal.x * omega * pi.alpha +
       2.0 * normal.y * omega * pi.beta + omega * omega * pi.one) /
      (fabs(normal.z) * normal.z * normal.z);

  // TODO: STUPID HACK UNTIL I FIND THE CULPRIT!
  // x-y-z and the volumes are in the wrong order..

  // Accumulate the weighted volume integrals
  if (orientation == XYZ) {
    T->y += 0.5 * normal.x * Falpha2;
    T->x += 0.5 * normal.y * Fbeta2;
    T->z += 0.5 * normal.z * Fgamma2;
    *vol += normal.y * Fbeta;
  } else if (orientation == YZX) {
    T->y += 0.5 * normal.y * Fbeta2;
    T->x += 0.5 * normal.z * Fgamma2;
    T->z += 0.5 * normal.x * Falpha2;
    *vol += normal.x * Falpha;
  } else if (orientation == ZXY) {
    T->y += 0.5 * normal.z * Fgamma2;
    T->x += 0.5 * normal.x * Falpha2;
    T->z += 0.5 * normal.y * Fbeta2;
    *vol += normal.z * Fgamma;
  }
}

// Checks if the normal vector is pointing inward or outward
// n0 is just a point on the plane
int check_normal_orientation(const int n0, const double* nodes_x,
                             const double* nodes_y, const double* nodes_z,
                             const vec_t* cell_centroid, vec_t* normal) {

  // Calculate a vector from face to cell centroid
  vec_t ab;
  ab.x = (cell_centroid->x - nodes_x[(n0)]);
  ab.y = (cell_centroid->y - nodes_y[(n0)]);
  ab.z = (cell_centroid->z - nodes_z[(n0)]);

  return (ab.x * normal->x + ab.y * normal->y + ab.z * normal->z > 0.0);
}

// Calculates the surface normal of a vector pointing outwards
void calc_surface_normal(const int n0, const int n1, const int n2,
                         const double* nodes_x, const double* nodes_y,
                         const double* nodes_z, const vec_t* cell_centroid,
                         vec_t* normal) {

  // Calculate the unit normal vector
  calc_unit_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, normal);

  // Determine the orientation of the normal
  const int flip = check_normal_orientation(n0, nodes_x, nodes_y, nodes_z,
                                            cell_centroid, normal);

  // Flip the vector if necessary
  normal->x *= (flip ? -1.0 : 1.0);
  normal->y *= (flip ? -1.0 : 1.0);
  normal->z *= (flip ? -1.0 : 1.0);
}

// Calculate the normal vector from the provided nodes
void calc_unit_normal(const int n0, const int n1, const int n2,
                      const double* nodes_x, const double* nodes_y,
                      const double* nodes_z, vec_t* normal) {

  // Calculate the normal
  calc_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, normal);

  // Normalise the normal
  normalise(normal);
}

// Normalise a vector
void normalise(vec_t* a) {
  const double a_inv_mag = 1.0 / sqrt(a->x * a->x + a->y * a->y + a->z * a->z);
  a->x *= a_inv_mag;
  a->y *= a_inv_mag;
  a->z *= a_inv_mag;
}

// Calculate the normal for a plane
void calc_normal(const int n0, const int n1, const int n2,
                 const double* nodes_x, const double* nodes_y,
                 const double* nodes_z, vec_t* normal) {
  // Get two vectors on the face plane
  vec_t dn0 = {0.0, 0.0, 0.0};
  vec_t dn1 = {0.0, 0.0, 0.0};
  dn0.x = nodes_x[(n2)] - nodes_x[(n1)];
  dn0.y = nodes_y[(n2)] - nodes_y[(n1)];
  dn0.z = nodes_z[(n2)] - nodes_z[(n1)];
  dn1.x = nodes_x[(n1)] - nodes_x[(n0)];
  dn1.y = nodes_y[(n1)] - nodes_y[(n0)];
  dn1.z = nodes_z[(n1)] - nodes_z[(n0)];

  // Cross product to get the normal
  normal->x = (dn0.y * dn1.z - dn0.z * dn1.y);
  normal->y = (dn0.z * dn1.x - dn0.x * dn1.z);
  normal->z = (dn0.x * dn1.y - dn0.y * dn1.x);
}

// Calculates the weighted volume integrals for a provided cell along x-y-z
void calc_weighted_volume_integrals(
    const int cell_to_faces_off, const int nfaces_by_cell,
    const int* cells_to_faces, const int* faces_to_nodes,
    const int* faces_to_nodes_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const vec_t* cell_centroid,
    vec_t* T, double* vol) {

  // Zero as we are reducing into this container
  T->x = 0.0;
  T->y = 0.0;
  T->z = 0.0;
  *vol = 0.0;

  // The weighted volume integrals are calculated over the polyhedral faces
  for (int ff = 0; ff < nfaces_by_cell; ++ff) {
    const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
    const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
    const int nnodes_by_face =
        faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

    // Choosing three nodes for calculating the unit normal
    // We can obviously assume there are at least three nodes
    const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
    const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
    const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];

    // Determine the outward facing unit normal vector
    vec_t normal = {0.0, 0.0, 0.0};
    calc_surface_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, cell_centroid,
                        &normal);

    // Select the orientation based on the face area
    int orientation;
    if (fabs(normal.x) > fabs(normal.y)) {
      orientation = (fabs(normal.x) > fabs(normal.z)) ? YZX : XYZ;
    } else {
      orientation = (fabs(normal.z) > fabs(normal.y)) ? XYZ : ZXY;
    }

    // The orientation determines which order we pass the nodes by axes
    // We calculate the individual face integrals and the unit normal to the
    // face in the alpha-beta-gamma basis
    // The weighted integrals essentially provide the center of mass
    // coordinates
    // for the polyhedra
    if (orientation == XYZ) {
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, orientation, n0,
                          faces_to_nodes, nodes_x, nodes_y, nodes_z, normal, T,
                          vol);
    } else if (orientation == YZX) {
      dswap(normal.x, normal.y);
      dswap(normal.y, normal.z);
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, orientation, n0,
                          faces_to_nodes, nodes_y, nodes_z, nodes_x, normal, T,
                          vol);
    } else if (orientation == ZXY) {
      dswap(normal.x, normal.y);
      dswap(normal.x, normal.z);
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, orientation, n0,
                          faces_to_nodes, nodes_z, nodes_x, nodes_y, normal, T,
                          vol);
    }
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
    (*inv)[2].y = ((*a)[0].x * (*a)[2].x - (*a)[0].x * (*a)[2].y) / det;
    (*inv)[2].z = ((*a)[0].x * (*a)[1].y - (*a)[0].y * (*a)[1].x) / det;
  }
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

// Constructs the prism for swept region of a subcell face external to a cell
void construct_external_swept_region(
    const vec_t* nodes, const vec_t* rz_nodes, const vec_t* half_edge_l,
    const vec_t* half_edge_r, const vec_t* rz_half_edge_l,
    const vec_t* rz_half_edge_r, const vec_t* face_c, const vec_t* rz_face_c,
    vec_t* prism_centroid, double* prism_nodes_x, double* prism_nodes_y,
    double* prism_nodes_z) {

  prism_nodes_x[(0)] = nodes->x;
  prism_nodes_y[(0)] = nodes->y;
  prism_nodes_z[(0)] = nodes->z;
  prism_nodes_x[(1)] = half_edge_r->x;
  prism_nodes_y[(1)] = half_edge_r->y;
  prism_nodes_z[(1)] = half_edge_r->z;
  prism_nodes_x[(2)] = face_c->x;
  prism_nodes_y[(2)] = face_c->y;
  prism_nodes_z[(2)] = face_c->z;
  prism_nodes_x[(3)] = half_edge_l->x;
  prism_nodes_y[(3)] = half_edge_l->y;
  prism_nodes_z[(3)] = half_edge_l->z;
  prism_nodes_x[(4)] = rz_nodes->x;
  prism_nodes_y[(4)] = rz_nodes->y;
  prism_nodes_z[(4)] = rz_nodes->z;
  prism_nodes_x[(5)] = rz_half_edge_r->x;
  prism_nodes_y[(5)] = rz_half_edge_r->y;
  prism_nodes_z[(5)] = rz_half_edge_r->z;
  prism_nodes_x[(6)] = rz_face_c->x;
  prism_nodes_y[(6)] = rz_face_c->y;
  prism_nodes_z[(6)] = rz_face_c->z;
  prism_nodes_x[(7)] = rz_half_edge_l->x;
  prism_nodes_y[(7)] = rz_half_edge_l->y;
  prism_nodes_z[(7)] = rz_half_edge_l->z;

  prism_centroid->x = 0.0;
  prism_centroid->y = 0.0;
  prism_centroid->z = 0.0;
  for (int pp = 0; pp < NSUBCELL_NODES; ++pp) {
    prism_centroid->x += prism_nodes_x[(pp)] / NSUBCELL_NODES;
    prism_centroid->y += prism_nodes_y[(pp)] / NSUBCELL_NODES;
    prism_centroid->z += prism_nodes_z[(pp)] / NSUBCELL_NODES;
  }
}

// Constructs the prism for swept region of a subcell face internal to a cell
void construct_internal_swept_region(
    const int face_rorientation, const vec_t* half_edge_l,
    const vec_t* half_edge_r, const vec_t* rz_half_edge_l,
    const vec_t* rz_half_edge_r, const vec_t* face_c, const vec_t* face2_c,
    const vec_t* rz_face_c, const vec_t* rz_face2_c, const vec_t* cell_centroid,
    const vec_t* rz_cell_centroid, vec_t* prism_centroid, double* prism_nodes_x,
    double* prism_nodes_y, double* prism_nodes_z) {

  // Constructing a prism from all known points in mesh and rezoned mesh
  prism_nodes_x[(0)] = face_rorientation ? half_edge_r->x : half_edge_l->x;
  prism_nodes_y[(0)] = face_rorientation ? half_edge_r->y : half_edge_l->y;
  prism_nodes_z[(0)] = face_rorientation ? half_edge_r->z : half_edge_l->z;
  prism_nodes_x[(1)] = face2_c->x;
  prism_nodes_y[(1)] = face2_c->y;
  prism_nodes_z[(1)] = face2_c->z;
  prism_nodes_x[(2)] = cell_centroid->x;
  prism_nodes_y[(2)] = cell_centroid->y;
  prism_nodes_z[(2)] = cell_centroid->z;
  prism_nodes_x[(3)] = face_c->x;
  prism_nodes_y[(3)] = face_c->y;
  prism_nodes_z[(3)] = face_c->z;
  prism_nodes_x[(4)] =
      face_rorientation ? rz_half_edge_r->x : rz_half_edge_l->x;
  prism_nodes_y[(4)] =
      face_rorientation ? rz_half_edge_r->y : rz_half_edge_l->y;
  prism_nodes_z[(4)] =
      face_rorientation ? rz_half_edge_r->z : rz_half_edge_l->z;
  prism_nodes_x[(5)] = rz_face2_c->x;
  prism_nodes_y[(5)] = rz_face2_c->y;
  prism_nodes_z[(5)] = rz_face2_c->z;
  prism_nodes_x[(6)] = rz_cell_centroid->x;
  prism_nodes_y[(6)] = rz_cell_centroid->y;
  prism_nodes_z[(6)] = rz_cell_centroid->z;
  prism_nodes_x[(7)] = rz_face_c->x;
  prism_nodes_y[(7)] = rz_face_c->y;
  prism_nodes_z[(7)] = rz_face_c->z;

  // Determine the prism's centroid
  prism_centroid->x = 0.0;
  prism_centroid->y = 0.0;
  prism_centroid->z = 0.0;
  for (int pp = 0; pp < NSUBCELL_NODES; ++pp) {
    prism_centroid->x += prism_nodes_x[(pp)] / NSUBCELL_NODES;
    prism_centroid->y += prism_nodes_y[(pp)] / NSUBCELL_NODES;
    prism_centroid->z += prism_nodes_z[(pp)] / NSUBCELL_NODES;
  }
}

// Calculate the inverse coefficient matrix for a subcell, in order to determine
// the gradients of the subcell quantities using least squares.
void calc_inverse_coefficient_matrix(
    const int subcell_index, const int* subcells_to_faces_offsets,
    const int* subcells_to_subcells, const double* subcell_integrals_x,
    const double* subcell_integrals_y, const double* subcell_integrals_z,
    const double* subcell_volume, int* nsubcells_by_subcell,
    int* subcell_to_subcells_off, vec_t (*inv)[3]) {

  *subcell_to_subcells_off = subcells_to_faces_offsets[(subcell_index)] * 2;
  *nsubcells_by_subcell = (subcells_to_faces_offsets[(subcell_index + 1)] * 2) -
                          *subcell_to_subcells_off;

  // The coefficients of the 3x3 gradient coefficient matrix
  vec_t coeff[3] = {{0.0, 0.0, 0.0}};
  for (int ss2 = 0; ss2 < *nsubcells_by_subcell; ++ss2) {
    const int neighbour_subcell_index =
        subcells_to_subcells[(*subcell_to_subcells_off + ss2)];
    if (neighbour_subcell_index == -1) {
      continue;
    }

    const double ix = subcell_integrals_x[(neighbour_subcell_index)];
    const double iy = subcell_integrals_y[(neighbour_subcell_index)];
    const double iz = subcell_integrals_z[(neighbour_subcell_index)];
    const double vol = subcell_volume[(neighbour_subcell_index)];

    // Store the neighbouring cell's contribution to the coefficients
    coeff[0].x += (2.0 * ix * ix) / (vol * vol);
    coeff[0].y += (2.0 * ix * iy) / (vol * vol);
    coeff[0].z += (2.0 * ix * iz) / (vol * vol);
    coeff[1].x += (2.0 * iy * ix) / (vol * vol);
    coeff[1].y += (2.0 * iy * iy) / (vol * vol);
    coeff[1].z += (2.0 * iy * iz) / (vol * vol);
    coeff[2].x += (2.0 * iz * ix) / (vol * vol);
    coeff[2].y += (2.0 * iz * iy) / (vol * vol);
    coeff[2].z += (2.0 * iz * iz) / (vol * vol);
  }

  for (int ii = 0; ii < 3; ++ii) {
    printf("coeff (%.6f %.6f %.6f)\n", coeff[ii].x, coeff[ii].y, coeff[ii].z);
  }

  calc_3x3_inverse(&coeff, inv);
}

// Calculate the gradient for the
void calc_gradient(const int subcell_index, const int nsubcells_by_subcell,
                   const int subcell_to_subcells_off,
                   const int* subcells_to_subcells, const double* phi,
                   const double* subcell_integrals_x,
                   const double* subcell_integrals_y,
                   const double* subcell_integrals_z,
                   const double* subcell_volume, const vec_t (*inv)[3],
                   vec_t* gradient) {

  // Calculate the gradient for the internal energy density
  vec_t rhs = {0.0, 0.0, 0.0};
  for (int ss2 = 0; ss2 < nsubcells_by_subcell; ++ss2) {
    const int neighbour_subcell_index =
        subcells_to_subcells[(subcell_to_subcells_off + ss2)];

    // Prepare differential
    const double de = (phi[(neighbour_subcell_index)] - phi[(subcell_index)]);

    // Calculate the subcell gradients for all of the variables
    rhs.x += (2.0 * subcell_integrals_x[(subcell_index)] * de /
              subcell_volume[(subcell_index)]);
    rhs.y += (2.0 * subcell_integrals_y[(subcell_index)] * de /
              subcell_volume[(subcell_index)]);
    rhs.z += (2.0 * subcell_integrals_z[(subcell_index)] * de /
              subcell_volume[(subcell_index)]);
  }

  gradient->x = (*inv)[0].x * rhs.x + (*inv)[0].y * rhs.y + (*inv)[0].z * rhs.z;
  gradient->y = (*inv)[1].x * rhs.x + (*inv)[1].y * rhs.y + (*inv)[1].z * rhs.z;
  gradient->z = (*inv)[2].x * rhs.x + (*inv)[2].y * rhs.y + (*inv)[2].z * rhs.z;
}

// Calculates the subcells of all centroids
void calc_subcell_centroids(
    const int ncells, const int* cells_offsets, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const int* cells_to_nodes, const int* subcells_to_faces_offsets,
    const int* subcells_to_faces, const int* faces_to_nodes_offsets,
    const int* faces_to_nodes, const double* nodes_x0, const double* nodes_y0,
    const double* nodes_z0, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z) {

// This is a lot of heavy lifting to repetetively perform so precomputing
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int subcell_index = (cell_to_nodes_off + ss);
      const int subcell_node_index = cells_to_nodes[(subcell_index)];
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Sum the face center of all subcell faces
        vec_t face_c = {0.0, 0.0, 0.0};
        calc_centroid(nnodes_by_face, nodes_x0, nodes_y0, nodes_z0,
                      faces_to_nodes, face_to_nodes_off, &face_c);
        subcell_centroids_x[(subcell_index)] += face_c.x;
        subcell_centroids_y[(subcell_index)] += face_c.y;
        subcell_centroids_z[(subcell_index)] += face_c.z;

        // calculate the subsequent face center for current and rezoned meshes
        int sn_off;
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (subcell_node_index == faces_to_nodes[(face_to_nodes_off + nn)]) {
            sn_off = nn;
          }
        }

        // Choose threee points on the planar face
        const int fn0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int fn1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int fn2 = faces_to_nodes[(face_to_nodes_off + 2)];

        vec_t normal;
        calc_normal(fn0, fn1, fn2, nodes_x0, nodes_y0, nodes_z0, &normal);
        const int face_rorientation = check_normal_orientation(
            fn0, nodes_x0, nodes_y0, nodes_z0, &cell_centroid, &normal);

        const int l_off = (sn_off == 0) ? nnodes_by_face - 1 : sn_off - 1;
        const int r_off = (sn_off == nnodes_by_face - 1) ? 0 : sn_off + 1;
        const int next_node_index =
            face_rorientation ? faces_to_nodes[(face_to_nodes_off + r_off)]
                              : faces_to_nodes[(face_to_nodes_off + l_off)];

        subcell_centroids_x[(subcell_index)] +=
            0.5 *
            (nodes_x0[(subcell_node_index)] + nodes_x0[(next_node_index)]);
        subcell_centroids_y[(subcell_index)] +=
            0.5 *
            (nodes_y0[(subcell_node_index)] + nodes_y0[(next_node_index)]);
        subcell_centroids_z[(subcell_index)] +=
            0.5 *
            (nodes_z0[(subcell_node_index)] + nodes_z0[(next_node_index)]);
      }

      // Add corner node
      subcell_centroids_x[(subcell_index)] += nodes_x0[(subcell_node_index)];
      subcell_centroids_y[(subcell_index)] += nodes_y0[(subcell_node_index)];
      subcell_centroids_z[(subcell_index)] += nodes_z0[(subcell_node_index)];

      // Add cell centroid
      subcell_centroids_x[(subcell_index)] += cell_centroid.x;
      subcell_centroids_y[(subcell_index)] += cell_centroid.y;
      subcell_centroids_z[(subcell_index)] += cell_centroid.z;

      // Take the average of all the subcell nodes
      subcell_centroids_x[(subcell_index)] /= (nfaces_by_subcell * 2 + 2);
      subcell_centroids_y[(subcell_index)] /= (nfaces_by_subcell * 2 + 2);
      subcell_centroids_z[(subcell_index)] /= (nfaces_by_subcell * 2 + 2);
    }
  }
}
