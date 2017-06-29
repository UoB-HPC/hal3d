#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <silo.h>
#include "hale_data.h"
#include "../shared.h"
#include "../mesh.h"
#include "../params.h"

// Initialises the shared_data variables for two dimensional applications
size_t initialise_hale_data_2d(
    const int local_nx, const int local_ny, HaleData* hale_data, 
    UnstructuredMesh* umesh)
{
  size_t allocated = allocate_data(&hale_data->energy0, umesh->ncells);
  allocated += allocate_data(&hale_data->density0, umesh->ncells);
  allocated += allocate_data(&hale_data->pressure0, umesh->ncells);
  allocated += allocate_data(&hale_data->velocity_x0, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y0, umesh->nnodes);
  allocated += allocate_data(&hale_data->energy1, umesh->ncells);
  allocated += allocate_data(&hale_data->density1, umesh->ncells);
  allocated += allocate_data(&hale_data->pressure1, umesh->ncells);
  allocated += allocate_data(&hale_data->velocity_x1, umesh->nnodes);
  allocated += allocate_data(&hale_data->velocity_y1, umesh->nnodes);
  allocated += allocate_data(&hale_data->cell_force_x, umesh->ncells*umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->cell_force_y, umesh->ncells*umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->node_force_x, umesh->nnodes);
  allocated += allocate_data(&hale_data->node_force_y, umesh->nnodes);
  allocated += allocate_data(&hale_data->cell_mass, umesh->ncells);
  allocated += allocate_data(&hale_data->nodal_mass, umesh->nnodes);
  allocated += allocate_data(&hale_data->nodal_volumes, umesh->nnodes);
  allocated += allocate_data(&hale_data->nodal_soundspeed, umesh->nnodes);
  allocated += allocate_data(&hale_data->limiter, umesh->nnodes);

#if 0
  // Set the density and the energy for all of the relevant cells
  for(int cc = 0; cc < umesh->ncells; ++cc) {
    const int nodes_off = umesh->cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = umesh->cells_to_nodes_off[(cc+1)]-nodes_off;
    const double inv_Np = 1.0/(double)nnodes_around_cell;

    double cell_centroids_x = 0.0;
    double cell_centroids_y = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = umesh->cells_to_nodes[(nodes_off)+(nn)];
      cell_centroids_x += umesh->nodes_x0[node_index]*inv_Np;
      cell_centroids_y += umesh->nodes_y0[node_index]*inv_Np;
    }

    if(sqrt(cell_centroids_x*cell_centroids_x+cell_centroids_y*cell_centroids_y) > 0.5) {
      hale_data->density0[(cc)] = 0.125;
      hale_data->energy0[(cc)] = 2.0;
    }
    else {
      hale_data->density0[(cc)] = 1.0;
      hale_data->energy0[(cc)] = 2.5;
    }
  }
#endif // if 0

  // Set the density and the energy for all of the relevant cells
  for(int cc = 0; cc < umesh->ncells; ++cc) {
    const int nodes_off = umesh->cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = umesh->cells_to_nodes_off[(cc+1)]-nodes_off;
    const double inv_Np = 1.0/(double)nnodes_around_cell;

    double cell_centroids_x = 0.0;
    double cell_centroids_y = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = umesh->cells_to_nodes[(nodes_off)+(nn)];
      cell_centroids_x += umesh->nodes_x0[node_index]*inv_Np;
      cell_centroids_y += umesh->nodes_y0[node_index]*inv_Np;
    }

    if(sqrt((cell_centroids_x-0.5)*(cell_centroids_x-0.5)+cell_centroids_y*cell_centroids_y) > 0.2) {
      hale_data->density0[(cc)] = 0.125;
      hale_data->energy0[(cc)] = 2.0;
    }
    else {
      hale_data->density0[(cc)] = 1.0;
      hale_data->energy0[(cc)] = 2.5;
    }
  }

  return allocated;
}

void deallocate_hale_data(
    HaleData* hale_data)
{
  // TODO: Populate this correctly !
  deallocate_data(hale_data->energy1);
  deallocate_data(hale_data->density1);
  deallocate_data(hale_data->pressure1);
  deallocate_data(hale_data->velocity_x1);
  deallocate_data(hale_data->velocity_y1);
  deallocate_data(hale_data->cell_force_x);
  deallocate_data(hale_data->cell_force_y);
  deallocate_data(hale_data->node_force_x);
  deallocate_data(hale_data->node_force_y);
}

// Reads an unstructured mesh from an input file
size_t read_unstructured_mesh(
    UnstructuredMesh* umesh)
{
  // Open the files
  FILE* node_fp = fopen(umesh->node_filename, "r");
  FILE* ele_fp = fopen(umesh->ele_filename, "r");
  if(!node_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", umesh->node_filename);
  }
  if(!ele_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", umesh->ele_filename);
  }

  // Fetch the first line of the nodes file
  char buf[MAX_STR_LEN];
  char* temp = buf;

  // Read the number of nodes, for allocation
  fgets(temp, MAX_STR_LEN, node_fp);
  skip_whitespace(&temp);
  sscanf(temp, "%d", &umesh->nnodes);

  // Read the number of cells
  fgets(temp, MAX_STR_LEN, ele_fp);
  skip_whitespace(&temp);
  sscanf(temp, "%d%d", &umesh->ncells, &umesh->nnodes_by_cell);
  umesh->ncells_by_node = umesh->nnodes_by_cell;

  // Allocate the data structures that we now know the sizes of
  size_t allocated = allocate_data(&umesh->nodes_x0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y0, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_x1, umesh->nnodes);
  allocated += allocate_data(&umesh->nodes_y1, umesh->nnodes);
  allocated += allocate_int_data(&umesh->cells_to_nodes, 
      umesh->ncells*umesh->nnodes_by_cell);
  allocated += allocate_int_data(&umesh->cells_to_nodes_off, umesh->ncells+1);
  allocated += allocate_data(&umesh->cell_centroids_x, umesh->ncells);
  allocated += allocate_data(&umesh->cell_centroids_y, umesh->ncells);
  allocated += allocate_int_data(&umesh->boundary_index, umesh->nnodes);

  int nboundary_cells = 0;

  // Loop through the node file, storing all of the nodes in our data structure
  while(fgets(temp, MAX_STR_LEN, node_fp)) {
    int index;
    int is_boundary;

    sscanf(temp, "%d", &index); 

    int discard;
    sscanf(temp, "%d%lf%lf%d", 
        &discard, 
        &umesh->nodes_x0[(index)], 
        &umesh->nodes_y0[(index)],
        &is_boundary);

    umesh->boundary_index[(index)] = 
      (is_boundary) ? nboundary_cells++ : IS_INTERIOR_NODE;
  }

  int* boundary_edge_list;
  int boundary_edge_index = 0;
  allocated += allocate_int_data(&boundary_edge_list, nboundary_cells*2);
  allocated += allocate_data(&umesh->boundary_normal_x, nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, nboundary_cells);

  // Loop through the element file and flatten into data structure
  while(fgets(temp, MAX_STR_LEN, ele_fp)) {
    int index;
    sscanf(temp, "%d", &index); 

    int discard;
    int node[umesh->nnodes_by_cell];
    sscanf(temp, "%d%d%d%d", &discard, &node[0], &node[1], &node[2]);

    umesh->cells_to_nodes[(index*umesh->nnodes_by_cell)+0] = node[0];
    umesh->cells_to_nodes[(index*umesh->nnodes_by_cell)+1] = node[1];
    umesh->cells_to_nodes[(index*umesh->nnodes_by_cell)+2] = node[2];
    umesh->cells_to_nodes_off[(index+1)] = 
      umesh->cells_to_nodes_off[(index)] + umesh->nnodes_by_cell;

    // Determine whether this cell touches a boundary edge
    int nboundary_nodes = 0;
    for(int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
      nboundary_nodes += (umesh->boundary_index[(node[nn])] != IS_INTERIOR_NODE);
    }

    // Only store edges that are on the boundary, maintaining the 
    // counter-clockwise order
    if(nboundary_nodes == 2) {
      for(int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
        const int next_node_index = (nn+1 == umesh->nnodes_by_cell ? 0 : nn+1);
        if(umesh->boundary_index[(node[nn])] != IS_INTERIOR_NODE && 
           umesh->boundary_index[(node[next_node_index])] != IS_INTERIOR_NODE) {
          boundary_edge_list[boundary_edge_index++] = node[nn];
          boundary_edge_list[boundary_edge_index++] = node[next_node_index];
          break;
        }
      }
    }

    // Check that we are storing the nodes in the correct order
    double A = 0.0;
    for(int ii = 0; ii < umesh->nnodes_by_cell; ++ii) {
      const int ii2 = (ii+1) % umesh->nnodes_by_cell; 
      A += (umesh->nodes_x0[node[ii]]+umesh->nodes_x0[node[ii2]])*
        (umesh->nodes_y0[node[ii2]]-umesh->nodes_y0[node[ii]]);
    }
    assert(A > 0.0 && "Nodes are not stored in counter-clockwise order.\n");
  }

  // Loop through all of the boundary cells and find their normals
  for(int nn = 0; nn < umesh->nnodes; ++nn) {
    const int boundary_index = umesh->boundary_index[(nn)];
    if(boundary_index == IS_INTERIOR_NODE) {
      continue;
    }

    double normal_x = 0.0;
    double normal_y = 0.0;

    for(int bb1 = 0; bb1 < nboundary_cells; ++bb1) {
      const int node0 = boundary_edge_list[bb1*2];
      const int node1 = boundary_edge_list[bb1*2+1];

      if(node0 == nn || node1 == nn) {
        const double node0_x = umesh->nodes_x0[(node0)];
        const double node0_y = umesh->nodes_y0[(node0)];
        const double node1_x = umesh->nodes_x0[(node1)];
        const double node1_y = umesh->nodes_y0[(node1)];

        normal_x += node0_y-node1_y;
        normal_y += -(node0_x-node1_x);
      }
    }

    // We are fixed if we are one of the four corners
    if((umesh->nodes_x0[(nn)] == 0.0 || umesh->nodes_x0[(nn)] == 1.0) &&
       (umesh->nodes_y0[(nn)] == 0.0 || umesh->nodes_y0[(nn)] == 1.0)) {
      umesh->boundary_type[(boundary_index)] = IS_FIXED;
    }
    else {
      umesh->boundary_type[(boundary_index)] = IS_BOUNDARY;
    }

    const double normal_mag = sqrt(normal_x*normal_x+normal_y*normal_y);
    umesh->boundary_normal_x[(boundary_index)] = normal_x/normal_mag;
    umesh->boundary_normal_y[(boundary_index)] = normal_y/normal_mag;
  }

  return allocated;
}

// Reflect the node centered velocities on the boundary
void handle_unstructured_reflect(
    const int nnodes, const int* boundary_index, const int* boundary_type,
    const double* boundary_normal_x, const double* boundary_normal_y, 
    double* velocity_x, double* velocity_y)
{
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    const int index = boundary_index[(nn)];
    if(index == IS_INTERIOR_NODE) {
      continue;
    }

    if(boundary_type[(index)] == IS_BOUNDARY) {
      // Project the velocity onto the face direction
      const double boundary_parallel_x = boundary_normal_y[(index)];
      const double boundary_parallel_y = -boundary_normal_x[(index)];
      const double vel_dot_parallel =
        (velocity_x[(nn)]*boundary_parallel_x+velocity_y[(nn)]*boundary_parallel_y);
      velocity_x[(nn)] = boundary_parallel_x*vel_dot_parallel;
      velocity_y[(nn)] = boundary_parallel_y*vel_dot_parallel;
    }
    else if(boundary_type[(index)] == IS_FIXED) {
      velocity_x[(nn)] = 0.0; 
      velocity_y[(nn)] = 0.0;
    }
  }
}

// Writes out unstructured triangles to visit
void write_unstructured_to_visit(
    const int nnodes, int ncells, const int step, double* nodes_x0, 
    double* nodes_y0, const int* cells_to_nodes, const double* arr, 
    const int nodal, const int quads)
{
  // Only triangles
  double* coords[] = { (double*)nodes_x0, (double*)nodes_y0 };
  int shapesize[] = { (quads ? 4 : 3) };
  int shapecounts[] = { ncells };
  int shapetype[] = { (quads ? DB_ZONETYPE_QUAD : DB_ZONETYPE_TRIANGLE) };
  int ndims = 2;
  int nshapes = 1;

  char filename[MAX_STR_LEN];
  sprintf(filename, "output%04d.silo", step);

  DBfile *dbfile = DBCreate(
      filename, DB_CLOBBER, DB_LOCAL, "simulation time step", DB_HDF5);

  DBPutZonelist2(dbfile, "zonelist", ncells, ndims, cells_to_nodes, 
      ncells*shapesize[0], 0, 0, 0, shapetype, shapesize, shapecounts, nshapes, NULL);
  DBPutUcdmesh(dbfile, "mesh", ndims, NULL, coords, nnodes, 
      ncells, "zonelist", NULL, DB_DOUBLE, NULL);
  DBPutUcdvar1(dbfile, "arr", "mesh", arr, (nodal ? nnodes : ncells), NULL, 0,
      DB_DOUBLE, (nodal ? DB_NODECENT : DB_ZONECENT), NULL);
  DBClose(dbfile);
}

