#include "../../comms.h"
#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <math.h>

// Gathers all of the subcell quantities on the mesh
void gather_subcell_quantities(
    const int ncells, double* cell_centroids_x, double* cell_centroids_y,
    double* cell_centroids_z, int* cells_offsets, const double* nodes_x0,
    const double* nodes_y0, const double* nodes_z0, double* energy0,
    double* density0, double* velocity_x0, double* velocity_y0,
    double* velocity_z0, double* cell_mass, double* subcell_volume,
    double* subcell_ie_density, double* subcell_mass,
    double* subcell_velocity_x, double* subcell_velocity_y,
    double* subcell_velocity_z, double* subcell_centroids_x,
    double* subcell_centroids_y, double* subcell_centroids_z,
    double* cell_volume, int* subcell_face_offsets, int* faces_to_nodes,
    int* faces_to_nodes_offsets, int* faces_to_cells0, int* faces_to_cells1,
    int* cells_to_faces_offsets, int* cells_to_faces, int* cells_to_nodes) {

  // TODO: This is a highly innaccurate solution, but I'm not sure what the
  // right way to go about this is with arbitrary polyhedrals using tetrahedral
  // subcells.
  double total_momentum_in_subcells_x = 0.0;
  double total_momentum_in_subcells_y = 0.0;
  double total_momentum_in_subcells_z = 0.0;
#pragma omp parallel for reduction(+ : total_momentum_in_subcells_x,           \
                                   total_momentum_in_subcells_y,               \
                                   total_momentum_in_subcells_z)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // Calculate the face center value
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // Choosing three nodes for calculating the unit normal
        // We can obviously assume there are at least three nodes
        const int n0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int n1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int n2 = faces_to_nodes[(face_to_nodes_off + 2)];

        // Determine the outward facing unit normal vector
        vec_t normal = {0.0, 0.0, 0.0};
        const int face_clockwise = calc_surface_normal(
            n0, n1, n2, nodes_x0, nodes_y0, nodes_z0, &cell_centroid, &normal);

        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        const int rnode = (nn == nnodes_by_face - 1) ? 0 : nn + 1;
        const int lnode = (nn == 0) ? nnodes_by_face - 1 : nn - 1;
        const int rnode_index = faces_to_nodes[(
            face_to_nodes_off + (face_clockwise ? lnode : rnode))];

        const double m_s = subcell_mass[(subcell_off + nn)];

        subcell_velocity_x[(subcell_off + nn)] =
            0.5 * m_s *
            (velocity_x0[(node_index)] + velocity_x0[(rnode_index)]);
        subcell_velocity_y[(subcell_off + nn)] =
            0.5 * m_s *
            (velocity_y0[(node_index)] + velocity_y0[(rnode_index)]);
        subcell_velocity_z[(subcell_off + nn)] =
            0.5 * m_s *
            (velocity_z0[(node_index)] + velocity_z0[(rnode_index)]);

        total_momentum_in_subcells_x += subcell_velocity_x[(subcell_off + nn)];
        total_momentum_in_subcells_y += subcell_velocity_y[(subcell_off + nn)];
        total_momentum_in_subcells_z += subcell_velocity_z[(subcell_off + nn)];
      }
    }
  }

  // Calculate the total momentum between nodes and subcell masses
  double total_momentum_in_cells_x = 0.0;
  double total_momentum_in_cells_y = 0.0;
  double total_momentum_in_cells_z = 0.0;
#pragma omp parallel for reduction(+ : total_momentum_in_cells_x,              \
                                   total_momentum_in_cells_y,                  \
                                   total_momentum_in_cells_z)
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
      const int nnodes_by_face =
          faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;
      const int subcell_off = subcell_face_offsets[(cell_to_faces_off + ff)];

      // Calculate the face center value
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        const int node_index = faces_to_nodes[(face_to_nodes_off + nn)];
        const double sm = subcell_mass[(subcell_off + nn)];
        total_momentum_in_cells_x += sm * velocity_x0[(node_index)];
        total_momentum_in_cells_y += sm * velocity_y0[(node_index)];
        total_momentum_in_cells_z += sm * velocity_z0[(node_index)];
      }
    }
  }

  printf("Subcell Gathering Conservation\n");
  printf("Total Momentum Cells    (%.6f %.6f %.6f)\n",
         total_momentum_in_cells_x, total_momentum_in_cells_y,
         total_momentum_in_cells_z);
  printf("Total Momentum Subcells (%.6f %.6f %.6f)\n",
         total_momentum_in_subcells_x, total_momentum_in_subcells_y,
         total_momentum_in_subcells_z);

/*
*      GATHERING STAGE OF THE REMAP
*/

#pragma omp parallel for
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
          subcell_centroid.x += subcell_nodes_x[ii] / NTET_NODES;
          subcell_centroid.y += subcell_nodes_y[ii] / NTET_NODES;
          subcell_centroid.z += subcell_nodes_z[ii] / NTET_NODES;
        }
        subcell_centroids_x[(subcell_index)] = subcell_centroid.x;
        subcell_centroids_y[(subcell_index)] = subcell_centroid.y;
        subcell_centroids_z[(subcell_index)] = subcell_centroid.z;

        // Precompute the volume of the subcell
        calc_volume(0, NTET_FACES, subcell_to_faces, subcell_faces_to_nodes,
                    subcell_faces_to_nodes_offsets, subcell_nodes_x,
                    subcell_nodes_y, subcell_nodes_z, &subcell_centroid,
                    &subcell_volume[(subcell_index)]);
      }
    }
  }

  double total_ie_in_subcells = 0.0;
// Calculate the sub-cell internal energies
#pragma omp parallel for reduction(+ : total_ie_in_subcells)
  for (int cc = 0; cc < ncells; ++cc) {
    // Calculating the volume center_of_mass necessary for the least squares
    // regression
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    const double cell_ie = density0[(cc)] * energy0[(cc)];
    vec_t cell_centroid = {cell_centroids_x[(cc)], cell_centroids_y[(cc)],
                           cell_centroids_z[(cc)]};

    vec_t rhs = {0.0, 0.0, 0.0};
    vec_t coeff[3] = {{0.0, 0.0, 0.0}};

    // Determine the weighted volume center_of_mass for neighbouring cells
    double gmax = -DBL_MAX;
    double gmin = DBL_MAX;
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int neighbour_index = (faces_to_cells0[(face_index)] == cc)
                                      ? faces_to_cells1[(face_index)]
                                      : faces_to_cells0[(face_index)];

      // Check if boundary face
      if (neighbour_index == -1) {
        continue;
      }

      const double neighbour_ie =
          density0[(neighbour_index)] * energy0[(neighbour_index)];

      // Calculate the weighted volume integral coefficients
      vec_t neighbour_centroid = {cell_centroids_x[(neighbour_index)],
                                  cell_centroids_y[(neighbour_index)],
                                  cell_centroids_z[(neighbour_index)]};

      double vol = cell_volume[(neighbour_index)];

      // Calculate the center of mass
      vec_t center_of_mass = {vol * neighbour_centroid.x,
                              vol * neighbour_centroid.y,
                              vol * neighbour_centroid.z};

      // Complete the integral coefficient as a distance
      center_of_mass.x -= cell_centroid.x * vol;
      center_of_mass.y -= cell_centroid.y * vol;
      center_of_mass.z -= cell_centroid.z * vol;

      // Store the neighbouring cell's contribution to the coefficients
      coeff[0].x += (2.0 * center_of_mass.x * center_of_mass.x) / (vol * vol);
      coeff[0].y += (2.0 * center_of_mass.x * center_of_mass.y) / (vol * vol);
      coeff[0].z += (2.0 * center_of_mass.x * center_of_mass.z) / (vol * vol);
      coeff[1].x += (2.0 * center_of_mass.y * center_of_mass.x) / (vol * vol);
      coeff[1].y += (2.0 * center_of_mass.y * center_of_mass.y) / (vol * vol);
      coeff[1].z += (2.0 * center_of_mass.y * center_of_mass.z) / (vol * vol);
      coeff[2].x += (2.0 * center_of_mass.z * center_of_mass.x) / (vol * vol);
      coeff[2].y += (2.0 * center_of_mass.z * center_of_mass.y) / (vol * vol);
      coeff[2].z += (2.0 * center_of_mass.z * center_of_mass.z) / (vol * vol);

      gmax = max(gmax, neighbour_ie);
      gmin = min(gmin, neighbour_ie);

      // Prepare the RHS, which includes energy differential
      const double de = (neighbour_ie - cell_ie);
      rhs.x += (2.0 * center_of_mass.x * de / vol);
      rhs.y += (2.0 * center_of_mass.y * de / vol);
      rhs.z += (2.0 * center_of_mass.z * de / vol);
    }

    // Determine the inverse of the coefficient matrix
    vec_t inv[3];
    calc_3x3_inverse(&coeff, &inv);

    // Solve for the energy gradient
    vec_t grad_energy;
    grad_energy.x = inv[0].x * rhs.x + inv[0].y * rhs.y + inv[0].z * rhs.z;
    grad_energy.y = inv[1].x * rhs.x + inv[1].y * rhs.y + inv[1].z * rhs.z;
    grad_energy.z = inv[2].x * rhs.x + inv[2].y * rhs.y + inv[2].z * rhs.z;

    apply_limiter(nnodes_by_cell, cell_to_nodes_off, cells_to_nodes,
                  &grad_energy, &cell_centroid, nodes_x0, nodes_y0, nodes_z0,
                  cell_ie, gmax, gmin);

    // Determine the weighted volume center_of_mass for neighbouring cells
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

      // Each face/node pair has two sub-cells
      for (int nn = 0; nn < nnodes_by_face; ++nn) {
        // The left and right nodes on the face for this anchor node
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

        double vol = subcell_volume[(subcell_index)];

        // Calculate the center of mass
        vec_t center_of_mass = {subcell_centroids_x[(subcell_index)] * vol,
                                subcell_centroids_y[(subcell_index)] * vol,
                                subcell_centroids_z[(subcell_index)] * vol};

        // Determine subcell energy from linear function at cell
        subcell_ie_density[(subcell_off + nn)] =
            cell_ie * vol +
            grad_energy.x * (center_of_mass.x - cell_centroid.x * vol) +
            grad_energy.y * (center_of_mass.y - cell_centroid.y * vol) +
            grad_energy.z * (center_of_mass.z - cell_centroid.z * vol);

        total_ie_in_subcells += subcell_ie_density[(subcell_off + nn)];
      }
    }
  }

  // Print out the conservation of energy following the gathering
  double total_ie_in_cells = 0.;
#pragma omp parallel for reduction(+ : total_ie_in_cells)
  for (int cc = 0; cc < ncells; ++cc) {
    total_ie_in_cells += cell_mass[cc] * energy0[(cc)];
  }
  printf("Total Energy in Cells    %.12f\nTotal Energy in Subcells %.12f \n",
         total_ie_in_cells, total_ie_in_subcells);
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

// Resolves the volume center_of_mass in alpha-beta-gamma basis
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

    // Accumulate the projection center_of_mass
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

  // Finalise the weighted face center_of_mass
  const double Falpha = pialpha / normal.z;
  const double Fbeta = pibeta / normal.z;
  const double Fgamma =
      -(normal.x * pialpha + normal.y * pibeta + omega * pione) /
      (normal.z * normal.z);

  // Accumulate the weighted volume center_of_mass
  if (basis == XYZ) {
    *vol += normal.x * Falpha;
  } else if (basis == YZX) {
    *vol += normal.z * Fgamma;
  } else if (basis == ZXY) {
    *vol += normal.y * Fbeta;
  }
}

// Calculates the weighted volume center_of_mass for a provided cell along x-y-z
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
    // where an edge is shared between the faces of the subcell and the rezoned
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
                     const int* cell_to_nodes, vec_t* grad,
                     const vec_t* cell_centroid, const double* nodes_x0,
                     const double* nodes_y0, const double* nodes_z0,
                     const double phi, const double gmax, const double gmin) {

  // Calculate the limiter for the gradient
  double limiter = DBL_MAX;
  for (int nn = 0; nn < nnodes_by_cell; ++nn) {
    double g_unlimited =
        phi +
        grad->x * (nodes_x0[cell_to_nodes[(cell_to_nodes_off + nn)]] -
                   cell_centroid->x) +
        grad->y * (nodes_y0[cell_to_nodes[(cell_to_nodes_off + nn)]] -
                   cell_centroid->y) +
        grad->z * (nodes_z0[cell_to_nodes[(cell_to_nodes_off + nn)]] -
                   cell_centroid->z);

    double node_limiter = 1.0;
    if (g_unlimited > phi) {
      node_limiter = min(1.0, (gmax - phi) / (g_unlimited - phi));
    } else if (g_unlimited < phi) {
      node_limiter = min(1.0, (gmin - phi) / (g_unlimited - phi));
    }
    limiter = min(limiter, node_limiter);
  }

  grad->x *= limiter;
  grad->y *= limiter;
  grad->z *= limiter;

  return limiter;
}
