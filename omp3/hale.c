#include "hale.h"
#include "../../comms.h"
#include "../../params.h"
#include "../../shared.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// TODO: At this stage, there are so many additional fields required
// to handle the sub-cell data for the remapping phase, there will be some use
// in considering whether some of the fields could be shared or whether
// adaptations to the algorithm are even necessary for this particular point

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_2d(
    Mesh* mesh, const int ncells, const int nnodes,
    const int nsubcell_neighbours, const double visc_coeff1,
    const double visc_coeff2, double* cell_centroids_x,
    double* cell_centroids_y, double* cell_centroids_z, int* cells_to_nodes,
    int* cells_offsets, int* nodes_to_cells, int* cells_to_cells,
    int* nodes_offsets, double* nodes_x0, double* nodes_y0, double* nodes_z0,
    double* nodes_x1, double* nodes_y1, double* nodes_z1, int* boundary_index,
    int* boundary_type, const double* original_nodes_x,
    const double* original_nodes_y, const double* original_nodes_z,
    double* boundary_normal_x, double* boundary_normal_y,
    double* boundary_normal_z, double* energy0, double* energy1,
    double* density0, double* density1, double* pressure0, double* pressure1,
    double* velocity_x0, double* velocity_y0, double* velocity_z0,
    double* velocity_x1, double* velocity_y1, double* velocity_z1,
    double* subcell_force_x, double* subcell_force_y, double* subcell_force_z,
    double* node_force_x, double* node_force_y, double* node_force_z,
    double* cell_mass, double* nodal_mass, double* nodal_volumes,
    double* nodal_soundspeed, double* limiter, double* subcell_volume,
    double* subcell_internal_energy, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_kinetic_energy,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, double* subcell_grad_x, double* subcell_grad_y,
    double* subcell_grad_z, int* nodes_to_faces_offsets, int* nodes_to_faces,
    int* faces_to_nodes, int* faces_to_nodes_offsets, int* faces_to_cells0,
    int* faces_to_cells1, int* cells_to_faces_offsets, int* cells_to_faces) {

  double total_mass = 0.0;
  for (int cc = 0; cc < ncells; ++cc) {
    total_mass += cell_mass[(cc)];
  }
  printf("total mass %.12f\n", total_mass);

  // The idea of the algorithm in 3d is to calculate the face centered and cell
  // centered velocities using the 2d and 3d equations
  //
  //
  // TODO: Could we just describe the density as a function and then interpolate
  // it?

  // TODO: LOADS OF OPTIMISATIONS HERE

  // Calculate the sub-cell internal energies
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
    vec_t coeff[3] = {0.0};
    vec_t rhs = {0.0};

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
      vec_t integrals;
      vec_t neighbour_centroid = {0.0};
      neighbour_centroid.x = cell_centroids_x[(neighbour_index)];
      neighbour_centroid.y = cell_centroids_y[(neighbour_index)];
      neighbour_centroid.z = cell_centroids_z[(neighbour_index)];
      calc_weighted_volume_integrals(
          neighbour_to_faces_off, nfaces_by_neighbour, cells_to_faces,
          faces_to_nodes, faces_to_nodes_offsets, nodes_x0, nodes_y0, nodes_z0,
          neighbour_centroid, &integrals, &vol);

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
#define NSUBCELL_FACES 4
#define NSUBCELL_NODES 4
    const int subcell_faces_to_nodes_offsets[NSUBCELL_NODES + 1] = {0, 3, 6, 9,
                                                                    12};
    const int subcell_faces_to_nodes[NSUBCELL_FACES * 3] = {0, 1, 2, 0, 1, 3,
                                                            0, 2, 3, 1, 2, 3};
    const int subcell_to_faces[NSUBCELL_FACES] = {0, 1, 2, 3};
    double subcell_nodes_x[NSUBCELL_NODES] = {0.0};
    double subcell_nodes_y[NSUBCELL_NODES] = {0.0};
    double subcell_nodes_z[NSUBCELL_NODES] = {0.0};

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

        // The left and right nodes on the face for this anchor node
        int nodes_a[2];
        nodes_a[0] =
            (nn == 0) ? faces_to_nodes[(face_to_nodes_off + nnodes_by_face - 1)]
                      : faces_to_nodes[(face_to_nodes_off + nn - 1)];
        nodes_a[1] = (nn == nnodes_by_face - 1)
                         ? faces_to_nodes[(face_to_nodes_off)]
                         : faces_to_nodes[(face_to_nodes_off + nn + 1)];

        for (int ss = 0; ss < 2; ++ss) {
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
          vec_t subcell_centroid = {0.0};
          for (int ii = 0; ii < NSUBCELL_NODES; ++ii) {
            subcell_centroid.x += subcell_nodes_x[ii] / NSUBCELL_NODES;
            subcell_centroid.y += subcell_nodes_y[ii] / NSUBCELL_NODES;
            subcell_centroid.z += subcell_nodes_z[ii] / NSUBCELL_NODES;
          }
#if 0
          printf("%.12f %.12f %.12f\n", subcell_centroid.x, subcell_centroid.y,
                 subcell_centroid.z);

          for (int aa = 0; aa < 4; ++aa) {
            printf("(%.4f %.4f %.4f) ", subcell_nodes_x[aa],
                   subcell_nodes_y[aa], subcell_nodes_z[aa]);
          }
          printf("\n");
#endif // if 0

          // Calculate the weighted volume integral coefficients
          double vol = 0.0;
          vec_t integrals = {0.0};
          calc_weighted_volume_integrals(
              0, NSUBCELL_FACES, subcell_to_faces, subcell_faces_to_nodes,
              subcell_faces_to_nodes_offsets, subcell_nodes_x, subcell_nodes_y,
              subcell_nodes_z, subcell_centroid, &integrals, &vol);

#if 0
          printf("%.12f\n", vol);
#endif // if 0

          int nn2;
          for (nn2 = 0; nn2 < nnodes_by_cell; ++nn2) {
            if (cells_to_nodes[(cell_to_nodes_off + nn2)] == node_index) {
              break;
            }
          }

          // TODO: THIS MIGHT BE A STUPID WAY TO DO THIS.
          // WE ARE LOOKING AT ALL OF THE SUBCELL TETRAHEDRONS, WHEN WE COULD BE
          // LOOKING AT A SINGLE CORNER SUBCELL PER NODE
          //
          // Determine subcell energy from linear function determined at cell
          subcell_internal_energy[(cell_to_nodes_off + nn2)] +=
              vol * (density0[(cc)] * energy0[(cc)] -
                     (grad_energy.x * cell_centroid.x +
                      grad_energy.y * cell_centroid.y +
                      grad_energy.z * cell_centroid.z)) +
              grad_energy.x * integrals.x + grad_energy.y * integrals.y +
              grad_energy.z * integrals.z;
        }
      }
    }

#if 0
    printf("subcell ie ");
    for (int nn = 0; nn < 8; ++nn) {
      printf("%.12f ", subcell_internal_energy[(cell_to_nodes_off + nn)]);
    }
    printf("\n");
#endif // if 0
  }
}

// Calculates the weighted volume integrals for a provided cell along x-y-z
void calc_weighted_volume_integrals(
    const int cell_to_faces_off, const int nfaces_by_cell,
    const int* cells_to_faces, const int* faces_to_nodes,
    const int* faces_to_nodes_offsets, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, const vec_t cell_centroid,
    vec_t* T, double* vol) {

  // Zero as we are reducing into this container
  T->x = 0.0;
  T->y = 0.0;
  T->z = 0.0;
  *vol = 0.0;

#if 0
  printf(
      "(%.4f %.4f %.4f) (%.4f %.4f %.4f) (%.4f %.4f %.4f) (%.4f %.4f %.4f)\n",
      nodes_x[0], nodes_y[0], nodes_z[0], nodes_x[1], nodes_y[1], nodes_z[1],
      nodes_x[2], nodes_y[2], nodes_z[2], nodes_x[3], nodes_y[3], nodes_z[3]);
#endif // if 0

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
    vec_t normal = {0.0};
    calc_unit_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, cell_centroid,
                     &normal);

#if 0
    printf("normal %.12f %.12f %.12f\n", normal.x, normal.y, normal.z);
#endif // if 0

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

// Resolves the volume integrals in alpha-beta-gamma basis
void calc_face_integrals(const int nnodes_by_face, const int face_to_nodes_off,
                         const int orientation, const int n0,
                         const int* faces_to_nodes, const double* nodes_alpha,
                         const double* nodes_beta, const double* nodes_gamma,
                         vec_t normal, vec_t* T, double* vol) {

  pi_t pi = {0.0};
  calc_projections(nnodes_by_face, face_to_nodes_off, faces_to_nodes,
                   nodes_alpha, nodes_beta, nodes_gamma, &pi, normal);

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

  // Accumulate the weighted volume integrals
  if (orientation == XYZ) {
    T->y += 0.5 * normal.x * Falpha2;
    T->x += 0.5 * normal.y * Fbeta2;
    T->z += 0.5 * normal.z * Fgamma2;
    *vol += normal.x * Falpha;
  } else if (orientation == YZX) {
    T->y += 0.5 * normal.y * Fbeta2;
    T->x += 0.5 * normal.z * Fgamma2;
    T->z += 0.5 * normal.x * Falpha2;
    *vol += normal.y * Fbeta;
  } else if (orientation == ZXY) {
    T->y += 0.5 * normal.z * Fgamma2;
    T->x += 0.5 * normal.x * Falpha2;
    T->z += 0.5 * normal.y * Fbeta2;
    *vol += normal.z * Fgamma;
  }

#if 0
  // Accumulate the weighted volume integrals
  if (orientation == XYZ) {
    T->x += 0.5 * normal.x * Falpha2;
    T->y += 0.5 * normal.y * Fbeta2;
    T->z += 0.5 * normal.z * Fgamma2;
    *vol += normal.x * Falpha;
  } else if (orientation == YZX) {
    T->x += 0.5 * normal.y * Fbeta2;
    T->y += 0.5 * normal.z * Fgamma2;
    T->z += 0.5 * normal.x * Falpha2;
    *vol += normal.y * Fbeta;
  } else if (orientation == ZXY) {
    T->x += 0.5 * normal.z * Fgamma2;
    T->y += 0.5 * normal.x * Falpha2;
    T->z += 0.5 * normal.y * Fbeta2;
    *vol += normal.z * Fgamma;
  }
#endif // if 0
}

// Calculates the face integral for the provided face, projected onto
// the two-dimensional basis
void calc_projections(const int nnodes_by_face, const int face_to_nodes_off,
                      const int* faces_to_nodes, const double* alpha,
                      const double* beta, const double* gamma, pi_t* pi,
                      vec_t normal) {

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

// Calculate the normal vector from the provided nodes
void calc_unit_normal(const int n0, const int n1, const int n2,
                      const double* nodes_x, const double* nodes_y,
                      const double* nodes_z, vec_t cell_centroid,
                      vec_t* normal) {

  // Get two vectors on the face plane
  vec_t dn0 = {0.0};
  vec_t dn1 = {0.0};
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

  // Calculate a vector from face to cell centroid
  vec_t ab;
  ab.x = (cell_centroid.x - nodes_x[(n0)]);
  ab.y = (cell_centroid.y - nodes_y[(n0)]);
  ab.z = (cell_centroid.z - nodes_z[(n0)]);

  // Normalise the vector, and flip if necessary
  const int flip =
      (ab.x * normal->x + ab.y * normal->y + ab.z * normal->z > 0.0);
  const double normal_mag = sqrt(normal->x * normal->x + normal->y * normal->y +
                                 normal->z * normal->z);
  normal->x /= (flip ? -1.0 : 1.0) * normal_mag;
  normal->y /= (flip ? -1.0 : 1.0) * normal_mag;
  normal->z /= (flip ? -1.0 : 1.0) * normal_mag;
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

// Controls the timestep for the simulation
void set_timestep(const int ncells, const int* cells_to_nodes,
                  const int* cells_offsets, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes) {

  // TODO: THIS IS A GOOD EXAMPLE OF WHERE WE SHOULD MARRY FACES TO EDGES
  // RATHER THAN DIRECTLY TO NODES.... WE ARE CURRENTLY PERFORMING TWICE
  // AS MANY CALCULATIONS AS WE NEED TO

  // Calculate the timestep based on the computational mesh and CFL
  // condition
  double local_dt = DBL_MAX;
  START_PROFILING(&compute_profile);
#pragma omp parallel for reduction(min : local_dt)
  for (int cc = 0; cc < ncells; ++cc) {
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
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn)];

        const int next_node = (nn + 1 < nnodes_by_face)
                                  ? faces_to_nodes[(face_to_nodes_off + nn + 1)]
                                  : faces_to_nodes[(face_to_nodes_off)];
        const double x_component =
            nodes_x[(current_node)] - nodes_x[(next_node)];
        const double y_component =
            nodes_y[(current_node)] - nodes_y[(next_node)];
        const double z_component =
            nodes_z[(current_node)] - nodes_z[(next_node)];

        // Find the shortest edge of this cell
        shortest_edge = min(shortest_edge, sqrt(x_component * x_component +
                                                y_component * y_component +
                                                z_component * z_component));
      }
    }

    const double soundspeed = sqrt(GAM * (GAM - 1.0) * energy[(cc)]);
    local_dt = min(local_dt, shortest_edge / soundspeed);
  }
  STOP_PROFILING(&compute_profile, __func__);

  *dt = CFL * local_dt;
  printf("Timestep %.8fs\n", *dt);
}

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(
    const int ncells, const int* cells_offsets, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const int* cells_to_nodes, const double* density, const double* nodes_x,
    const double* nodes_y, const double* nodes_z, double* cell_mass,
    double* subcell_volume, double* subcell_mass, int* cells_to_faces_offsets,
    int* cells_to_faces, int* faces_to_nodes_offsets, int* faces_to_nodes) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
  double total_mass = 0.0;
#pragma omp parallel for reduction(+ : total_mass)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Calculate the face center... SHOULD WE PRECOMPUTE?
      double face_c_x = 0.0;
      double face_c_y = 0.0;
      double face_c_z = 0.0;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c_x += nodes_x[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z[(node_index)] / nnodes_by_face;
      }

      // Now we will sum the contributions at each of the nodes
      // TODO: THERE IS SOME SYMMETRY HERE THAT MEANS WE MIGHT BE ABLE TO
      // OPTIMISE
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        // Fetch the nodes attached to our current node on the current face
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn2)];
        const int next_node =
            (nn2 + 1 < nnodes_by_face)
                ? faces_to_nodes[(face_to_nodes_off + nn2 + 1)]
                : faces_to_nodes[(face_to_nodes_off)];

        // Get the halfway point on the right edge
        const double half_edge_x =
            0.5 * (nodes_x[(current_node)] + nodes_x[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y[(current_node)] + nodes_y[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z[(current_node)] + nodes_z[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (half_edge_x - face_c_x);
        const double a_y = (half_edge_y - face_c_y);
        const double a_z = (half_edge_z - face_c_z);
        const double b_x = (cell_centroids_x[(cc)] - face_c_x);
        const double b_y = (cell_centroids_y[(cc)] - face_c_y);
        const double b_z = (cell_centroids_z[(cc)] - face_c_z);

        // Calculate the area vector S using cross product
        const double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        const double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        const double S_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: WE MULTIPLY BY 2 HERE BECAUSE WE ARE ADDING THE VOLUME TO
        // BOTH
        // THE CURRENT AND NEXT NODE, OTHERWISE WE ONLY ACCOUNT FOR HALF OF
        // THE
        // 'HALF' TETRAHEDRONS
        double subcell_volume =
            fabs(2.0 * ((half_edge_x - nodes_x[(current_node)]) * S_x +
                        (half_edge_y - nodes_y[(current_node)]) * S_y +
                        (half_edge_z - nodes_z[(current_node)]) * S_z) /
                 3.0);

        // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER CLOSED
        // FORM SOLUTION?
        for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
          if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
            subcell_mass[(cell_to_nodes_off + nn3)] +=
                density[(cc)] * subcell_volume;
          }
        }

        cell_mass[(cc)] += density[(cc)] * subcell_volume;
      }
    }

    total_mass += cell_mass[(cc)];
  }
  STOP_PROFILING(&compute_profile, __func__);

  printf("Initial total mesh mash: %.15f\n", total_mass);
}

// Initialises the centroids for each cell
void init_cell_centroids(const int ncells, const int* cells_offsets,
                         const int* cells_to_nodes, const double* nodes_x,
                         const double* nodes_y, const double* nodes_z,
                         double* cell_centroids_x, double* cell_centroids_y,
                         double* cell_centroids_z) {
  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cells_off;

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int node_index = cells_to_nodes[(cells_off + nn)];
      cx += nodes_x[(node_index)];
      cy += nodes_y[(node_index)];
      cz += nodes_z[(node_index)];
    }

    cell_centroids_x[(cc)] = cx / (double)nnodes_by_cell;
    cell_centroids_y[(cc)] = cy / (double)nnodes_by_cell;
    cell_centroids_z[(cc)] = cz / (double)nnodes_by_cell;
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Initialises the centroids for each cell
void init_subcell_centroids(
    const int ncells, const int* cells_offsets, const int* cells_to_nodes,
    const double* nodes_x, const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z) {
  // Calculate the cell centroids
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cells_off = cells_offsets[(cc)];
    const int nsubcells = cells_offsets[(cc + 1)] - cells_off;

    const double cell_c_x = cell_centroids_x[(cc)];
    const double cell_c_y = cell_centroids_y[(cc)];
    const double cell_c_z = cell_centroids_z[(cc)];

    for (int ss = 0; ss < nsubcells; ++ss) {
      // TODO: GET THE NODES AROUND A SUB-CELL

      subcell_centroids_x[(cc)] = 0.0;
      subcell_centroids_y[(cc)] = 0.0;
      subcell_centroids_z[(cc)] = 0.0;
    }
  }
  STOP_PROFILING(&compute_profile, __func__);
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

// Calculates the artificial viscous forces for momentum acceleration
void calc_artificial_viscosity(
    const int ncells, const int nnodes, const double visc_coeff1,
    const double visc_coeff2, const int* cells_offsets,
    const int* cells_to_nodes, const int* nodes_offsets,
    const int* nodes_to_cells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const double* cell_centroids_x,
    const double* cell_centroids_y, const double* cell_centroids_z,
    const double* velocity_x, const double* velocity_y,
    const double* velocity_z, const double* nodal_soundspeed,
    const double* nodal_mass, const double* nodal_volumes,
    const double* limiter, double* node_force_x, double* node_force_y,
    double* node_force_z, int* nodes_to_faces_offsets, int* nodes_to_faces,
    int* faces_to_nodes_offsets, int* faces_to_nodes, int* faces_to_cells0,
    int* faces_to_cells1, int* cells_to_faces_offsets, int* cells_to_faces) {

  // Calculate the predicted energy
  START_PROFILING(&compute_profile);
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // Look at all of the faces attached to the cell
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

      // Calculate the face center... SHOULD WE PRECOMPUTE?
      double face_c_x = 0.0;
      double face_c_y = 0.0;
      double face_c_z = 0.0;
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn2)];
        face_c_x += nodes_x[(node_index)] / nnodes_by_face;
        face_c_y += nodes_y[(node_index)] / nnodes_by_face;
        face_c_z += nodes_z[(node_index)] / nnodes_by_face;
      }

      // Now we will sum the contributions at each of the nodes
      // TODO: THERE IS SOME SYMMETRY HERE THAT MEANS WE MIGHT BE ABLE TO
      // OPTIMISE
      for (int nn2 = 0; nn2 < nnodes_by_face; ++nn2) {
        // Fetch the nodes attached to our current node on the current face
        const int current_node = faces_to_nodes[(face_to_nodes_off + nn2)];
        const int next_node =
            (nn2 + 1 < nnodes_by_face)
                ? faces_to_nodes[(face_to_nodes_off + nn2 + 1)]
                : faces_to_nodes[(face_to_nodes_off)];

        // Get the halfway point on the right edge
        const double half_edge_x =
            0.5 * (nodes_x[(current_node)] + nodes_x[(next_node)]);
        const double half_edge_y =
            0.5 * (nodes_y[(current_node)] + nodes_y[(next_node)]);
        const double half_edge_z =
            0.5 * (nodes_z[(current_node)] + nodes_z[(next_node)]);

        // Setup basis on plane of tetrahedron
        const double a_x = (half_edge_x - face_c_x);
        const double a_y = (half_edge_y - face_c_y);
        const double a_z = (half_edge_z - face_c_z);
        const double b_x = (cell_centroids_x[(cc)] - face_c_x);
        const double b_y = (cell_centroids_y[(cc)] - face_c_y);
        const double b_z = (cell_centroids_z[(cc)] - face_c_z);
        const double ab_x = (nodes_x[(current_node)] - half_edge_x);
        const double ab_y = (nodes_y[(current_node)] - half_edge_y);
        const double ab_z = (nodes_z[(current_node)] - half_edge_z);

        // Calculate the area vector S using cross product
        double S_x = 0.5 * (a_y * b_z - a_z * b_y);
        double S_y = -0.5 * (a_x * b_z - a_z * b_x);
        double S_z = 0.5 * (a_x * b_y - a_y * b_x);

        // TODO: I HAVENT WORKED OUT A REASONABLE WAY TO ORDER THE NODES SO
        // THAT THIS COMES OUT CORRECTLY, SO NEED TO FIXUP AFTER THE
        // CALCULATION
        if ((ab_x * S_x + ab_y * S_y + ab_z * S_z) > 0.0) {
          S_x *= -1.0;
          S_y *= -1.0;
          S_z *= -1.0;
        }

        // Calculate the velocity gradients
        const double dvel_x =
            velocity_x[(next_node)] - velocity_x[(current_node)];
        const double dvel_y =
            velocity_y[(next_node)] - velocity_y[(current_node)];
        const double dvel_z =
            velocity_z[(next_node)] - velocity_z[(current_node)];
        const double dvel_mag =
            sqrt(dvel_x * dvel_x + dvel_y * dvel_y + dvel_z * dvel_z);

        // Calculate the unit vectors of the velocity gradients
        const double dvel_unit_x = (dvel_mag != 0.0) ? dvel_x / dvel_mag : 0.0;
        const double dvel_unit_y = (dvel_mag != 0.0) ? dvel_y / dvel_mag : 0.0;
        const double dvel_unit_z = (dvel_mag != 0.0) ? dvel_z / dvel_mag : 0.0;

        // Get the edge-centered density
        double nodal_density0 =
            nodal_mass[(current_node)] / nodal_volumes[(current_node)];
        double nodal_density1 =
            nodal_mass[(next_node)] / nodal_volumes[(next_node)];
        const double density_edge = (2.0 * nodal_density0 * nodal_density1) /
                                    (nodal_density0 + nodal_density1);

        // Calculate the artificial viscous force term for the edge
        double expansion_term = (dvel_x * S_x + dvel_y * S_y + dvel_z * S_z);

        // If the cell is compressing, calculate the edge forces and add
        // their
        // contributions to the node forces
        if (expansion_term <= 0.0) {
          // Calculate the minimum soundspeed
          const double cs = min(nodal_soundspeed[(current_node)],
                                nodal_soundspeed[(next_node)]);
          const double t = 0.25 * (GAM + 1.0);
          const double edge_visc_force_x =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_x) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_x * dvel_x +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_x;
          const double edge_visc_force_y =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_y) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_y * dvel_y +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_y;
          const double edge_visc_force_z =
              density_edge *
              (visc_coeff2 * t * fabs(dvel_z) +
               sqrt(visc_coeff2 * visc_coeff2 * t * t * dvel_z * dvel_z +
                    visc_coeff1 * visc_coeff1 * cs * cs)) *
              (1.0 - limiter[(current_node)]) * expansion_term * dvel_unit_z;

          // TODO: I HATE SEARCHES LIKE THIS... CAN WE FIND SOME BETTER
          // CLOSED
          // FORM SOLUTION?
          int node_off;
          int next_node_off;
          for (int nn3 = 0; nn3 < nnodes_by_cell; ++nn3) {
            if (cells_to_nodes[(cell_to_nodes_off + nn3)] == current_node) {
              node_off = nn3;
            } else if (cells_to_nodes[(cell_to_nodes_off + nn3)] == next_node) {
              next_node_off = nn3;
            }
          }

          // Add the contributions of the edge based artifical viscous terms
          // to the main force terms
          node_force_x[(cell_to_nodes_off + node_off)] -= edge_visc_force_x;
          node_force_y[(cell_to_nodes_off + node_off)] -= edge_visc_force_y;
          node_force_z[(cell_to_nodes_off + node_off)] -= edge_visc_force_z;
          node_force_x[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_x;
          node_force_y[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_y;
          node_force_z[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_z;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, __func__);
}
