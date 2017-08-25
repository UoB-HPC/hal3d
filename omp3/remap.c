#include "../../comms.h"
#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>

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
int calc_surface_normal(const int n0, const int n1, const int n2,
                        const double* nodes_x, const double* nodes_y,
                        const double* nodes_z, const vec_t* cell_centroid,
                        vec_t* normal) {

  // Calculate the unit normal vector
  calc_unit_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, normal);

  // Determine the orientation of the normal
  const int face_clockwise = check_normal_orientation(
      n0, nodes_x, nodes_y, nodes_z, cell_centroid, normal);

  // Flip the vector if necessary
  normal->x *= (face_clockwise ? -1.0 : 1.0);
  normal->y *= (face_clockwise ? -1.0 : 1.0);
  normal->z *= (face_clockwise ? -1.0 : 1.0);

  return face_clockwise;
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
  if (sqrt(a->x * a->x + a->y * a->y + a->z * a->z) == 0.0) {
    a->x = 0.0;
    a->y = 0.0;
    a->z = 0.0;
  } else {
    a->x *= a_inv_mag;
    a->y *= a_inv_mag;
    a->z *= a_inv_mag;
  }
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
  normal->x = (dn0.y * dn1.z - dn0.z * dn1.y);
  normal->y = (dn0.z * dn1.x - dn0.x * dn1.z);
  normal->z = (dn0.x * dn1.y - dn0.y * dn1.x);
}

// Resolves the volume dist in alpha-beta-gamma basis
void calc_face_integrals(const int nnodes_by_face, const int face_to_nodes_off,
                         const int basis, const int face_clockwise,
                         const double omega, const int* faces_to_nodes,
                         const double* nodes_alpha, const double* nodes_beta,
                         vec_t normal, double* vol) {

  double pione = 0.0;
  double pialpha = 0.0;
  double pibeta = 0.0;

  // Calculate the coefficients for the projected face integral
  for (int nn = 0; nn < nnodes_by_face; ++nn) {
    const int n0 = faces_to_nodes[(face_to_nodes_off + nn)];
    const int n1 = (nn == nnodes_by_face - 1)
                       ? faces_to_nodes[(face_to_nodes_off)]
                       : faces_to_nodes[(face_to_nodes_off + nn + 1)];

    // Calculate all of the coefficients
    const double a0 = nodes_alpha[(n0)];
    const double a1 = nodes_alpha[(n1)];
    const double b0 = nodes_beta[(n0)];
    const double b1 = nodes_beta[(n1)];
    const double dalpha = a1 - a0;
    const double dbeta = b1 - b0;
    const double Calpha = a1 * (a1 + a0) + a0 * a0;
    const double Cbeta = b1 * (b1 + b0) + b0 * b0;

    // Accumulate the projection dist
    pione += dbeta * (a1 + a0) / 2.0;
    pialpha += dbeta * (Calpha) / 6.0;
    pibeta -= dalpha * (Cbeta) / 6.0;
  }

  // Store the final coefficients, flipping all results if we went through
  // in a clockwise order and got a negative area
  const double flip = (face_clockwise ? 1.0 : -1.0);
  pione *= flip;
  pialpha *= flip;
  pibeta *= flip;

  // Finalise the weighted face dist
  const double Falpha = pialpha / normal.z;
  const double Fbeta = pibeta / normal.z;
  const double Fgamma =
      -(normal.x * pialpha + normal.y * pibeta + omega * pione) /
      (normal.z * normal.z);

  // Accumulate the weighted volume dist
  if (basis == XYZ) {
    *vol += normal.x * Falpha;
  } else if (basis == YZX) {
    *vol += normal.z * Fgamma;
  } else if (basis == ZXY) {
    *vol += normal.y * Fbeta;
  }
}

// Calculates the weighted volume dist for a provided cell along x-y-z
void calc_volume(const int cell_to_faces_off, const int nfaces_by_cell,
                 const int* cells_to_faces, const int* faces_to_nodes,
                 const int* faces_to_nodes_offsets, const double* nodes_x,
                 const double* nodes_y, const double* nodes_z,
                 const vec_t* cell_centroid, double* vol) {

  // Prepare to reduce accumulate the volume
  *vol = 0.0;

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
    const int face_clockwise = calc_surface_normal(
        n0, n1, n2, nodes_x, nodes_y, nodes_z, cell_centroid, &normal);

    // I have observed that under certain combinations of translation and
    // rotation the swept edge region can be tetrahedral rather than a prism,
    // where an edge is shared between the faces of the subcell and the
    // rezoned
    // subcell. If this happens, which should be rare, due to numerical
    // imprecision, I ignore the contribution of the face, and
    // continue as if we were a tetrahedron.
    if (normal.x == 0.0 && normal.y == 0.0 && normal.z == 0.0) {
      continue;
    }

    // The projection of the normal vector onto a point on the face
    double omega = -(normal.x * nodes_x[(n0)] + normal.y * nodes_y[(n0)] +
                     normal.z * nodes_z[(n0)]);

    // Select the orientation based on the face area
    int basis;
    if (fabs(normal.x) > fabs(normal.y)) {
      basis = (fabs(normal.x) > fabs(normal.z)) ? YZX : XYZ;
    } else {
      basis = (fabs(normal.z) > fabs(normal.y)) ? XYZ : ZXY;
    }

    // The basis ensures that gamma is always maximised
    if (basis == XYZ) {
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, basis,
                          face_clockwise, omega, faces_to_nodes, nodes_x,
                          nodes_y, normal, vol);
    } else if (basis == YZX) {
      dswap(normal.x, normal.y);
      dswap(normal.y, normal.z);
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, basis,
                          face_clockwise, omega, faces_to_nodes, nodes_y,
                          nodes_z, normal, vol);
    } else if (basis == ZXY) {
      dswap(normal.x, normal.y);
      dswap(normal.x, normal.z);
      calc_face_integrals(nnodes_by_face, face_to_nodes_off, basis,
                          face_clockwise, omega, faces_to_nodes, nodes_z,
                          nodes_x, normal, vol);
    }
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

// Calculate the inverse coefficient matrix for a subcell, in order to
// determine the gradients of the subcell quantities using least squares.
void calc_inverse_coefficient_matrix(
    const int subcell_index, const int* subcells_to_subcells,
    const double* subcell_centroids_x, const double* subcell_centroids_y,
    const double* subcell_centroids_z, const double* subcell_volume,
    const int nsubcells_by_subcell, const int subcell_to_subcells_off,
    vec_t (*inv)[3]) {

  // The coefficients of the 3x3 gradient coefficient matrix
  vec_t coeff[3] = {{0.0, 0.0, 0.0}};

  for (int ss2 = 0; ss2 < nsubcells_by_subcell; ++ss2) {
    const int neighbour_subcell_index =
        subcells_to_subcells[(subcell_to_subcells_off + ss2)];

    // Ignore boundary neighbours
    if (neighbour_subcell_index == -1) {
      continue;
    }

    const double vol = subcell_volume[(neighbour_subcell_index)];
    const double ix = subcell_centroids_x[(neighbour_subcell_index)] * vol -
                      subcell_centroids_x[(subcell_index)] * vol;
    const double iy = subcell_centroids_y[(neighbour_subcell_index)] * vol -
                      subcell_centroids_y[(subcell_index)] * vol;
    const double iz = subcell_centroids_z[(neighbour_subcell_index)] * vol -
                      subcell_centroids_z[(subcell_index)] * vol;

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

  calc_3x3_inverse(&coeff, inv);
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

// Calculate the gradient for the
void calc_gradient(const int subcell_index, const int nsubcells_by_subcell,
                   const int subcell_to_subcells_off,
                   const int* subcells_to_subcells, const double* phi,
                   const double* subcell_centroids_x,
                   const double* subcell_centroids_y,
                   const double* subcell_centroids_z, const vec_t (*inv)[3],
                   vec_t* gradient) {

  // Calculate the gradient for the internal energy density
  vec_t rhs = {0.0, 0.0, 0.0};
  for (int ss2 = 0; ss2 < nsubcells_by_subcell; ++ss2) {
    const int neighbour_subcell_index =
        subcells_to_subcells[(subcell_to_subcells_off + ss2)];

    // Prepare differential
    const double dphi = (phi[(neighbour_subcell_index)] - phi[(subcell_index)]);

    // Calculate the subcell gradients for all of the variables
    rhs.x += (2.0 * subcell_centroids_x[(neighbour_subcell_index)] * dphi);
    rhs.y += (2.0 * subcell_centroids_y[(neighbour_subcell_index)] * dphi);
    rhs.z += (2.0 * subcell_centroids_z[(neighbour_subcell_index)] * dphi);
  }

  gradient->x = (*inv)[0].x * rhs.x + (*inv)[0].y * rhs.y + (*inv)[0].z * rhs.z;
  gradient->y = (*inv)[1].x * rhs.x + (*inv)[1].y * rhs.y + (*inv)[1].z * rhs.z;
  gradient->z = (*inv)[2].x * rhs.x + (*inv)[2].y * rhs.y + (*inv)[2].z * rhs.z;
}

// Calculates the limiter for the provided gradient
double apply_limiter(const int nnodes_by_cell, const int cell_to_nodes_off,
                     const int* cells_to_nodes, vec_t* grad,
                     const vec_t* cell_centroid, const double* nodes_x0,
                     const double* nodes_y0, const double* nodes_z0,
                     const double phi, const double gmax, const double gmin) {

  // Calculate the limiter for the gradient
  double limiter = DBL_MAX;
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    const int node_index = cells_to_nodes[(cell_to_nodes_off + nn)];
    double g_unlimited = grad->x * (nodes_x0[(node_index)] - cell_centroid->x) +
                         grad->y * (nodes_y0[(node_index)] - cell_centroid->y) +
                         grad->z * (nodes_z0[(node_index)] - cell_centroid->z);

    double node_limiter = 1.0;
    if (g_unlimited > 0.0) {
      if (g_unlimited > EPS) {
        node_limiter = min(1.0, ((gmax - phi) / (g_unlimited)));
      }
    } else if (g_unlimited < 0.0) {
      if (g_unlimited > EPS) {
        node_limiter = min(1.0, ((gmin - phi) / (g_unlimited)));
      }
    }
    limiter = min(limiter, node_limiter);
  }

  grad->x *= limiter;
  grad->y *= limiter;
  grad->z *= limiter;

  return limiter;
}

// Calculates the cell volume, subcell volume and the subcell centroids
void calc_volumes_centroids(
    const int ncells, const int* cells_to_faces_offsets,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const int* cells_to_faces,
    const int* faces_to_nodes, const int* faces_to_nodes_offsets,
    const int* subcell_face_offsets, const double* nodes_x0,
    const double* nodes_y0, const double* nodes_z0, double* cell_volume,
    double* subcell_centroids_x, double* subcell_centroids_y,
    double* subcell_centroids_z, double* subcell_volume) {

  double total_volume = 0.0;
  double total_subcell_volume = 0.0;
#pragma omp parallel for reduction(+ : total_subcell_volume, total_volume)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    vec_t cell_centroid = {cell_centroids_x[(cc)], cell_centroids_y[(cc)],
                           cell_centroids_z[(cc)]};

    // Precompute the volume of the cell
    calc_volume(cell_to_faces_off, nfaces_by_cell, cells_to_faces,
                faces_to_nodes, faces_to_nodes_offsets, nodes_x0, nodes_y0,
                nodes_z0, &cell_centroid, &cell_volume[(cc)]);

    total_volume += cell_volume[(cc)];

    // Describe the connectivity for a simple tetrahedron, the sub-cell shape
    const int subcell_faces_to_nodes_offsets[] = {0, 3, 6, 9, 12};
    const int subcell_faces_to_nodes[] = {0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 2, 3};
    const int subcell_to_faces[] = {0, 1, 2, 3};
    double subcell_nodes_x[] = {0.0, 0.0, 0.0, 0.0};
    double subcell_nodes_y[] = {0.0, 0.0, 0.0, 0.0};
    double subcell_nodes_z[] = {0.0, 0.0, 0.0, 0.0};

    // The centroid remains a component of all sub-cells
    subcell_nodes_x[3] = cell_centroid.x;
    subcell_nodes_y[3] = cell_centroid.y;
    subcell_nodes_z[3] = cell_centroid.z;

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

      subcell_nodes_x[2] = face_c.x;
      subcell_nodes_y[2] = face_c.y;
      subcell_nodes_z[2] = face_c.z;

      // Each face/node pair has two sub-cells
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // The left and right nodes on the face for this anchor node
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];
        const int subcell_index = subcell_off + nn;

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

        // Store the right and left nodes
        subcell_nodes_x[1] = nodes_x0[(rnode_index)];
        subcell_nodes_y[1] = nodes_y0[(rnode_index)];
        subcell_nodes_z[1] = nodes_z0[(rnode_index)];
        subcell_nodes_x[0] = nodes_x0[(node_index)];
        subcell_nodes_y[0] = nodes_y0[(node_index)];
        subcell_nodes_z[0] = nodes_z0[(node_index)];

        // Determine the sub-cell centroid
        vec_t subcell_centroid = {0.0, 0.0, 0.0};
        for (int ii = 0; ii < NTET_NODES; ++ii) {
          subcell_centroid.x += subcell_nodes_x[(ii)] / NTET_NODES;
          subcell_centroid.y += subcell_nodes_y[(ii)] / NTET_NODES;
          subcell_centroid.z += subcell_nodes_z[(ii)] / NTET_NODES;
        }
        subcell_centroids_x[(subcell_index)] = subcell_centroid.x;
        subcell_centroids_y[(subcell_index)] = subcell_centroid.y;
        subcell_centroids_z[(subcell_index)] = subcell_centroid.z;

        // Precompute the volume of the subcell
        calc_volume(0, NTET_FACES, subcell_to_faces, subcell_faces_to_nodes,
                    subcell_faces_to_nodes_offsets, subcell_nodes_x,
                    subcell_nodes_y, subcell_nodes_z, &subcell_centroid,
                    &subcell_volume[(subcell_index)]);

        total_subcell_volume += subcell_volume[(subcell_index)];
      }
    }
  }

  printf("Total Cell Volume %.12f\n", total_volume);
  printf("Total Subcell Volume %.12f\n", total_subcell_volume);
}

void apply_mesh_rezoning(const int nnodes, const double* rezoned_nodes_x,
                         const double* rezoned_nodes_y,
                         const double* rezoned_nodes_z, double* nodes_x0,
                         double* nodes_y0, double* nodes_z0) {

// Apply the rezoned mesh into the main mesh
#pragma omp parallel for
  for (int nn = 0; nn < nnodes; ++nn) {
    nodes_x0[nn] = rezoned_nodes_x[(nn)];
    nodes_y0[nn] = rezoned_nodes_y[(nn)];
    nodes_z0[nn] = rezoned_nodes_z[(nn)];
  }
}
