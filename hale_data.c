#include "hale_data.h"
#include "../mesh.h"
#include "../params.h"
#include "../shared.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <silo.h>
#include <stdlib.h>

// Initialises the shared_data variables for two dimensional applications
size_t init_hale_data(HaleData* hale_data, UnstructuredMesh* umesh) {
  hale_data->nsubcells = umesh->ncells * umesh->nnodes_by_cell;
  hale_data->nsubcell_edges = 4;

  size_t allocated = allocate_data(&hale_data->pressure0, umesh->ncells);
  allocated += allocate_data(&hale_data->velocity_x0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_z0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_x1, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y1, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_z1, umesh->nnodes);
  allocated += allocate_data(&hale_data->energy1, umesh->ncells);
  allocated += allocate_data(&hale_data->density1, umesh->ncells);
  allocated += allocate_data(&hale_data->pressure1, umesh->ncells);
  allocated += allocate_data(&hale_data->cell_mass, umesh->ncells);
  allocated += allocate_data(&hale_data->nodal_mass, umesh->nnodes);
  allocated += allocate_data(&hale_data->nodal_volumes, umesh->nnodes);
  allocated += allocate_data(&hale_data->nodal_soundspeed, umesh->nnodes);
  allocated += allocate_data(&hale_data->limiter, umesh->nnodes);
  allocated += allocate_data(&hale_data->rezoned_nodes_x, umesh->nnodes);
  allocated += allocate_data(&hale_data->rezoned_nodes_y, umesh->nnodes);
  allocated += allocate_data(&hale_data->rezoned_nodes_z, umesh->nnodes);

  // TODO: This constant is the number of subcells that might neighbour a
  // subcell, which is at most 8 for a prism!
  allocated += allocate_int_data(&hale_data->subcells_to_subcells,
                                 hale_data->nsubcells * 8);
  // TODO: This constant is essentially making a guess about the maximum number
  // of faces we will see attached to a subcell. The number is determined from
  // the prism shape, which connects a single node with four on the base.
  allocated += allocate_int_data(&hale_data->subcells_to_faces,
                                 hale_data->nsubcells * 4);
  allocated += allocate_int_data(&hale_data->subcells_to_faces_offsets,
                                 hale_data->nsubcells);

  allocated +=
      allocate_data(&hale_data->subcell_velocity_x, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_velocity_y, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_velocity_z, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_integrals_x, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_integrals_y, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_integrals_z, hale_data->nsubcells);
  allocated +=
      allocate_data(&hale_data->subcell_ie_density, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_volume, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_mass, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_force_x, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_force_y, hale_data->nsubcells);
  allocated += allocate_data(&hale_data->subcell_force_z, hale_data->nsubcells);

  // In hale, the fundamental principle is that the mass at the cell and
  // sub-cell are conserved, so we can initialise them from the mesh
  // and then only the remapping step will ever adjust them
  init_cell_centroids(umesh->ncells, umesh->cells_offsets,
                      umesh->cells_to_nodes, umesh->nodes_x0, umesh->nodes_y0,
                      umesh->nodes_z0, umesh->cell_centroids_x,
                      umesh->cell_centroids_y, umesh->cell_centroids_z);

  init_mesh_mass(umesh->ncells, umesh->cells_offsets, umesh->cell_centroids_x,
                 umesh->cell_centroids_y, umesh->cell_centroids_z,
                 umesh->cells_to_nodes, hale_data->density0, umesh->nodes_x0,
                 umesh->nodes_y0, umesh->nodes_z0, hale_data->cell_mass,
                 hale_data->subcell_mass, umesh->cells_to_faces_offsets,
                 umesh->cells_to_faces, umesh->faces_to_nodes_offsets,
                 umesh->faces_to_nodes);

  init_subcells_to_faces(
      umesh->ncells, umesh->cells_offsets, umesh->cells_to_nodes,
      umesh->nodes_to_faces_offsets, umesh->nodes_to_faces,
      umesh->cells_to_faces_offsets, umesh->cells_to_faces,
      umesh->faces_to_cells0, umesh->faces_to_cells1,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->cell_centroids_x, umesh->cell_centroids_y, umesh->cell_centroids_z,
      umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
      hale_data->subcells_to_faces_offsets, hale_data->subcells_to_faces);

  init_subcells_to_subcells(
      umesh->ncells, umesh->nodes_x0, umesh->nodes_y0, umesh->nodes_z0,
      umesh->cells_offsets, umesh->cells_to_nodes,
      umesh->nodes_to_faces_offsets, umesh->nodes_to_faces,
      umesh->faces_to_cells0, umesh->faces_to_cells1,
      umesh->faces_to_nodes_offsets, umesh->faces_to_nodes,
      umesh->cell_centroids_x, umesh->cell_centroids_y, umesh->cell_centroids_z,
      hale_data->subcells_to_faces_offsets, hale_data->subcells_to_faces,
      hale_data->subcells_to_subcells);

  store_rezoned_mesh(umesh->nnodes, umesh->nodes_x0, umesh->nodes_y0,
                     umesh->nodes_z0, hale_data->rezoned_nodes_x,
                     hale_data->rezoned_nodes_y, hale_data->rezoned_nodes_z);

  return allocated;
}

// Deallocates all of the hale specific data
void deallocate_hale_data(HaleData* hale_data) {
  // TODO: Populate this correctly !
}

// Writes out unstructured triangles to visit
void write_unstructured_to_visit_2d(const int nnodes, int ncells,
                                    const int step, double* nodes_x0,
                                    double* nodes_y0, const int* cells_to_nodes,
                                    const double* arr, const int nodal,
                                    const int quads) {
  // Only triangles
  double* coords[] = {(double*)nodes_x0, (double*)nodes_y0};
  int shapesize[] = {(quads ? 4 : 3)};
  int shapecounts[] = {ncells};
  int shapetype[] = {(quads ? DB_ZONETYPE_QUAD : DB_ZONETYPE_TRIANGLE)};
  int ndims = 2;
  int nshapes = 1;

  char filename[MAX_STR_LEN];
  sprintf(filename, "output%04d.silo", step);

  DBfile* dbfile =
      DBCreate(filename, DB_CLOBBER, DB_LOCAL, "simulation time step", DB_HDF5);

  DBPutZonelist2(dbfile, "zonelist", ncells, ndims, cells_to_nodes,
                 ncells * shapesize[0], 0, 0, 0, shapetype, shapesize,
                 shapecounts, nshapes, NULL);
  DBPutUcdmesh(dbfile, "mesh", ndims, NULL, coords, nnodes, ncells, "zonelist",
               NULL, DB_DOUBLE, NULL);
  DBPutUcdvar1(dbfile, "arr", "mesh", arr, (nodal ? nnodes : ncells), NULL, 0,
               DB_DOUBLE, (nodal ? DB_NODECENT : DB_ZONECENT), NULL);
  DBClose(dbfile);
}

// Writes out unstructured triangles to visit
void write_unstructured_to_visit_3d(const int nnodes, int ncells,
                                    const int step, double* nodes_x,
                                    double* nodes_y, double* nodes_z,
                                    const int* cells_to_nodes,
                                    const double* arr, const int nodal,
                                    const int quads) {

  double* coords[] = {(double*)nodes_x, (double*)nodes_y, (double*)nodes_z};

  int shapesize[] = {8};
  int shapecounts[] = {ncells};
  int shapetype[] = {DB_ZONETYPE_HEX};
  int ndims = 3;
  int nshapes = 1;

  char filename[MAX_STR_LEN];
  sprintf(filename, "output%04d.silo", step);

  DBfile* dbfile =
      DBCreate(filename, DB_CLOBBER, DB_LOCAL, "simulation time step", DB_HDF5);

  /* Write out connectivity information. */
  DBPutZonelist2(dbfile, "zonelist", ncells, ndims, cells_to_nodes,
                 ncells * shapesize[0], 0, 0, 0, shapetype, shapesize,
                 shapecounts, nshapes, NULL);

  /* Write an unstructured mesh. */
  DBPutUcdmesh(dbfile, "mesh", ndims, NULL, coords, nnodes, ncells, "zonelist",
               NULL, DB_DOUBLE, NULL);

  DBPutUcdvar1(dbfile, "arr", "mesh", arr, (nodal ? nnodes : ncells), NULL, 0,
               DB_DOUBLE, (nodal ? DB_NODECENT : DB_ZONECENT), NULL);

  DBClose(dbfile);
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

// Calculate the normal vector from the provided nodes
void calc_unit_normal(const int n0, const int n1, const int n2,
                      const double* nodes_x, const double* nodes_y,
                      const double* nodes_z, vec_t cell_centroid,
                      vec_t* normal) {

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
    calc_unit_normal(n0, n1, n2, nodes_x, nodes_y, nodes_z, cell_centroid,
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

// Initialises the cell mass, sub-cell mass and sub-cell volume
void init_mesh_mass(const int ncells, const int* cells_offsets,
                    const double* cell_centroids_x,
                    const double* cell_centroids_y,
                    const double* cell_centroids_z, const int* cells_to_nodes,
                    const double* density, const double* nodes_x,
                    const double* nodes_y, const double* nodes_z,
                    double* cell_mass, double* subcell_mass,
                    int* cells_to_faces_offsets, int* cells_to_faces,
                    int* faces_to_nodes_offsets, int* faces_to_nodes) {

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
    const int ncells, const double visc_coeff1, const double visc_coeff2,
    const int* cells_offsets, const int* cells_to_nodes, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* velocity_x,
    const double* velocity_y, const double* velocity_z,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter, double* subcell_force_x,
    double* subcell_force_y, double* subcell_force_z,
    int* faces_to_nodes_offsets, int* faces_to_nodes,
    int* cells_to_faces_offsets, int* cells_to_faces) {

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
          subcell_force_x[(cell_to_nodes_off + node_off)] -= edge_visc_force_x;
          subcell_force_y[(cell_to_nodes_off + node_off)] -= edge_visc_force_y;
          subcell_force_z[(cell_to_nodes_off + node_off)] -= edge_visc_force_z;
          subcell_force_x[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_x;
          subcell_force_y[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_y;
          subcell_force_z[(cell_to_nodes_off + next_node_off)] +=
              edge_visc_force_z;
        }
      }
    }
  }
  STOP_PROFILING(&compute_profile, __func__);
}

// Controls the timestep for the simulation
void set_timestep(const int ncells, const double* nodes_x,
                  const double* nodes_y, const double* nodes_z,
                  const double* energy, double* dt, int* cells_to_faces_offsets,
                  int* cells_to_faces, int* faces_to_nodes_offsets,
                  int* faces_to_nodes) {

  // TODO: THIS IS SOO BAD, WE NEED TO CORRECTLY CALCULATE THE CHARACTERISTIC
  // LENGTHS AND DETERMINE THE FULL CONDITION

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

// Initialise the subcells to faces connectivity list
void init_subcells_to_faces(
    const int ncells, const int* cells_offsets, const int* cells_to_nodes,
    const int* nodes_to_faces_offsets, const int* nodes_to_faces,
    const int* cells_to_faces_offsets, const int* cells_to_faces,
    const int* faces_to_cells0, const int* faces_to_cells1,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const double* nodes_x,
    const double* nodes_y, const double* nodes_z,
    int* subcells_to_faces_offsets, int* subcells_to_faces) {

// NOTE: Some of these steps might be mergable, but I feel liek the current
// implementation leads to a better read through of the code
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // We will calculate the flux at every face of the subcells
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + ss)];

      // Find all of the faces that share the node (was slightly easier to
      // understand if this and subsequent step were separated)
      int nfaces_on_node = 0;
      for (int ff = 0; ff < nfaces_by_cell; ++ff) {
        const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Attempt to the node on the face
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (faces_to_nodes[(face_to_nodes_off + nn)] == node_index) {
            // We found a touching face
            nfaces_on_node++;
            break;
          }
        }
      }

      subcells_to_faces_offsets[(cell_to_nodes_off + ss + 1)] = nfaces_on_node;
    }
  }

  // TODO: This is another serial conversion from counts to offsets. Need to
  // find a way of paralellising these.
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_subcells_off = cells_offsets[(cc)];
    const int nsubcells_by_cell =
        cells_offsets[(cc + 1)] - cell_to_subcells_off;

    // We will calculate the flux at every face of the subcells
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      subcells_to_faces_offsets[(cell_to_subcells_off + ss + 1)] +=
          subcells_to_faces_offsets[(cell_to_subcells_off + ss)];
    }
  }

// NOTE: Some of these steps might be mergable, but I feel liek the current
// implementation leads to a better read through of the code
// We also do too much work in this, as we have knowledge about those faces
// that have already been processed, but this should be quite minor overall
// and it's and initialisation function so just keep an eye on the
// initialisation performance.
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    // The list of face indices attached to a node
    int faces_on_node[] = {-1, -1, -1, -1};
    int face_rorientation[] = {0, 0, 0, 0};

    // This is a map between one element of faces_on_node and another, with
    // the unique pairings, which is essentially a ring of faces
    int faces_to_faces[] = {-1, -1, -1, -1, -1, -1, -1, -1};

    // We will calculate the flux at every face of the subcells
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int node_index = cells_to_nodes[(cell_to_nodes_off + ss)];

      // Find all of the faces that share the node (was slightly easier to
      // understand if this and subsequent step were separated)
      int nfaces_on_node = 0;
      for (int ff = 0; ff < nfaces_by_cell; ++ff) {
        const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Attempt to find the node on the face
        int found = 0;
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (faces_to_nodes[(face_to_nodes_off + nn)] == node_index) {
            // TODO: This is duplicate, and all of this just to determine
            // orientation is annoying ><

            // Get two vectors on the face plane
            vec_t dn0 = {0.0, 0.0, 0.0};
            vec_t dn1 = {0.0, 0.0, 0.0};
            dn0.x = nodes_x[(faces_to_nodes[(face_to_nodes_off + 2)])] -
                    nodes_x[faces_to_nodes[(face_to_nodes_off + 1)]];
            dn0.y = nodes_y[(faces_to_nodes[(face_to_nodes_off + 2)])] -
                    nodes_y[faces_to_nodes[(face_to_nodes_off + 1)]];
            dn0.z = nodes_z[(faces_to_nodes[(face_to_nodes_off + 2)])] -
                    nodes_z[faces_to_nodes[(face_to_nodes_off + 1)]];
            dn1.x = nodes_x[(faces_to_nodes[(face_to_nodes_off + 1)])] -
                    nodes_x[faces_to_nodes[(face_to_nodes_off + 0)]];
            dn1.y = nodes_y[(faces_to_nodes[(face_to_nodes_off + 1)])] -
                    nodes_y[faces_to_nodes[(face_to_nodes_off + 0)]];
            dn1.z = nodes_z[(faces_to_nodes[(face_to_nodes_off + 1)])] -
                    nodes_z[faces_to_nodes[(face_to_nodes_off + 0)]];

            // Calculate a vector from face to cell centroid
            vec_t ab;
            ab.x = (cell_centroid.x -
                    nodes_x[(faces_to_nodes[(face_to_nodes_off)])]);
            ab.y = (cell_centroid.y -
                    nodes_y[(faces_to_nodes[(face_to_nodes_off)])]);
            ab.z = (cell_centroid.z -
                    nodes_z[(faces_to_nodes[(face_to_nodes_off)])]);

            // Cross product to get the normal
            vec_t normal;
            normal.x = (dn0.y * dn1.z - dn0.z * dn1.y);
            normal.y = (dn0.z * dn1.x - dn0.x * dn1.z);
            normal.z = (dn0.x * dn1.y - dn0.y * dn1.x);
            face_rorientation[(nfaces_on_node)] =
                (ab.x * normal.x + ab.y * normal.y + ab.z * normal.z < 0.0);
            faces_on_node[(nfaces_on_node++)] = face_index;
            break;
          }
        }
      }

      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(cell_to_nodes_off + ss)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(cell_to_nodes_off + ss + 1)] -
          subcell_to_faces_off;

      // Look at all of the faces we have discovered so far and see if
      // there is a connection between the faces
      subcells_to_faces[(subcell_to_faces_off)] = faces_on_node[(0)];
      int previous_fn = 0;
      for (int fn = 0; fn < nfaces_by_subcell - 1; ++fn) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + fn)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Re-find the node on the face
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (faces_to_nodes[(face_to_nodes_off + nn)] == node_index) {
            int side_node;
            if (face_rorientation[(previous_fn)]) {
              side_node =
                  faces_to_nodes[(face_to_nodes_off +
                                  ((nn == nnodes_by_face - 1) ? 0 : nn + 1))];
            } else {
              side_node =
                  faces_to_nodes[(face_to_nodes_off +
                                  ((nn == 0) ? nnodes_by_face - 1 : nn - 1))];
            }

            // Find all of the faces that connect to this face in the list
            int connect_face_index = 0;
            for (int fn2 = 0; fn2 < nfaces_by_subcell; ++fn2) {
              const int face_index2 = faces_on_node[(fn2)];

              // No self connectivity
              if (face_index2 == face_index) {
                continue;
              }

              const int face_to_nodes_off2 =
                  faces_to_nodes_offsets[(face_index2)];
              const int nnodes_by_face2 =
                  faces_to_nodes_offsets[(face_index2 + 1)] -
                  face_to_nodes_off2;

              // Check whether the face is connected
              for (int nn2 = 0; nn2 < nnodes_by_face2; ++nn2) {
                if (faces_to_nodes[(face_to_nodes_off2 + nn2)] == side_node) {
                  subcells_to_faces[(subcell_to_faces_off + fn + 1)] =
                      face_index2;
                  previous_fn = fn2;
                  break;
                }
              }
            }

            break;
          }
        }
      }
    }
  }
#if 0

  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    // We will calculate the flux at every face of the subcells
    for (int ss = 0; ss < nsubcells_by_cell; ++ss) {
      const int subcell_index = cell_to_nodes_off + ss;
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      printf("subcell %d : ", subcell_index);
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        printf("%d ", face_index);
      }
      printf("\n");
    }
  }
#endif // if 0
}

// Initialises the list of neighbours to a subcell
void init_subcells_to_subcells(
    const int ncells, const double* nodes_x, const double* nodes_y,
    const double* nodes_z, const int* cells_offsets, const int* cells_to_nodes,
    const int* nodes_to_faces_offsets, const int* nodes_to_faces,
    const int* faces_to_cells0, const int* faces_to_cells1,
    const int* faces_to_nodes_offsets, const int* faces_to_nodes,
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* cell_centroids_z, const int* subcells_to_faces_offsets,
    const int* subcells_to_faces, int* subcells_to_subcells) {

// This routine uses the previously described subcell faces to determine a ring
// ordering for the subcell neighbours. Essentially, each subcell has a pair of
// neighbours for every external face, and those neighbours are stored in the
// same order as the faces.
#pragma omp parallel for
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nsubcells_in_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    vec_t cell_centroid;
    cell_centroid.x = cell_centroids_x[(cc)];
    cell_centroid.y = cell_centroids_y[(cc)];
    cell_centroid.z = cell_centroids_z[(cc)];

    // For every face on a subcell we have a pair of neighbours that are
    // attached
    for (int ss = 0; ss < nsubcells_in_cell; ++ss) {
      const int subcell_index = (cell_to_nodes_off + ss);
      const int subcell_to_faces_off =
          subcells_to_faces_offsets[(subcell_index)];
      const int nfaces_by_subcell =
          subcells_to_faces_offsets[(subcell_index + 1)] - subcell_to_faces_off;

      // For every face there are two faces
      const int subcell_to_subcells_off = subcell_to_faces_off * 2;

      // Look at all of the faces to find the pair of neighbours associated with
      // each of them
      int neighbour_index = 0;
      for (int ff = 0; ff < nfaces_by_subcell; ++ff) {
        const int face_index = subcells_to_faces[(subcell_to_faces_off + ff)];
        const int face_to_nodes_off = faces_to_nodes_offsets[(face_index)];
        const int nnodes_by_face =
            faces_to_nodes_offsets[(face_index + 1)] - face_to_nodes_off;

        // Determine the neighbouring cell
        const int neighbour_cell_index = (faces_to_cells0[(face_index)] == cc)
                                             ? faces_to_cells1[(face_index)]
                                             : faces_to_cells0[(face_index)];

        // Pick the neighbouring subcell from the adjoining cell
        if (neighbour_cell_index != -1) {
          const int neighbour_to_nodes_off =
              cells_offsets[(neighbour_cell_index)];
          const int nnodes_by_neighbour_cell =
              cells_offsets[(neighbour_cell_index + 1)] -
              neighbour_to_nodes_off;
          for (int nn = 0; nn < nnodes_by_neighbour_cell; ++nn) {
            if (cells_to_nodes[(neighbour_to_nodes_off + nn)] ==
                cells_to_nodes[(subcell_index)]) {
              subcells_to_subcells[(subcell_to_subcells_off +
                                    neighbour_index++)] =
                  neighbour_to_nodes_off + nn;
              break;
            }
          }
        } else {
          subcells_to_subcells[(subcell_to_subcells_off + neighbour_index++)] =
              -1;
        }

        // We again need to determine the orientation in order to calculate the
        // correct right handed node
        vec_t dn0 = {0.0, 0.0, 0.0};
        vec_t dn1 = {0.0, 0.0, 0.0};
        const int fn_off0 = faces_to_nodes[(face_to_nodes_off + 0)];
        const int fn_off1 = faces_to_nodes[(face_to_nodes_off + 1)];
        const int fn_off2 = faces_to_nodes[(face_to_nodes_off + 2)];
        dn0.x = nodes_x[(fn_off2)] - nodes_x[(fn_off1)];
        dn0.y = nodes_y[(fn_off2)] - nodes_y[(fn_off1)];
        dn0.z = nodes_z[(fn_off2)] - nodes_z[(fn_off1)];
        dn1.x = nodes_x[(fn_off1)] - nodes_x[(fn_off0)];
        dn1.y = nodes_y[(fn_off1)] - nodes_y[(fn_off0)];
        dn1.z = nodes_z[(fn_off1)] - nodes_z[(fn_off0)];

        // Calculate a vector from face to cell centroid
        vec_t ab;
        ab.x = (cell_centroid.x - nodes_x[(fn_off0)]);
        ab.y = (cell_centroid.y - nodes_y[(fn_off0)]);
        ab.z = (cell_centroid.z - nodes_z[(fn_off0)]);

        // Cross product to get the normal
        vec_t normal;
        normal.x = (dn0.y * dn1.z - dn0.z * dn1.y);
        normal.y = (dn0.z * dn1.x - dn0.x * dn1.z);
        normal.z = (dn0.x * dn1.y - dn0.y * dn1.x);
        const int face_rorientation =
            (ab.x * normal.x + ab.y * normal.y + ab.z * normal.z < 0.0);

        // Determine the right oriented next node after the subcell node on the
        // face that represents the subcell node in the cell
        int rnode;
        for (int nn = 0; nn < nnodes_by_face; ++nn) {
          if (faces_to_nodes[(face_to_nodes_off + nn)] ==
              cells_to_nodes[(subcell_index)]) {
            const int loff = ((nn == 0) ? nnodes_by_face - 1 : nn - 1);
            const int roff = ((nn == nnodes_by_face - 1) ? 0 : nn + 1);
            rnode = (face_rorientation)
                        ? faces_to_nodes[(face_to_nodes_off + roff)]
                        : faces_to_nodes[(face_to_nodes_off + loff)];
            break;
          }
        }

        // Search for the node (subcell) in the cell
        for (int nn = 0; nn < nsubcells_in_cell; ++nn) {
          const int subcell_index2 = cell_to_nodes_off + nn;
          if (cells_to_nodes[(subcell_index2)] == rnode) {
            subcells_to_subcells[(subcell_to_subcells_off +
                                  neighbour_index++)] = subcell_index2;
            break;
          }
        }
      }
    }
  }
}
