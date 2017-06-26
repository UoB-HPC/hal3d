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
    UnstructuredMesh* unstructured_mesh)
{
  size_t allocated = allocate_data(&hale_data->energy1, (local_nx)*(local_ny));
  allocated = allocate_data(&hale_data->density1, (local_nx)*(local_ny));
  allocated = allocate_data(&hale_data->pressure1, (local_nx)*(local_ny));
  allocated = allocate_data(&hale_data->velocity_x1, (local_nx+1)*(local_ny+1));
  allocated = allocate_data(&hale_data->velocity_y1, (local_nx+1)*(local_ny+1));
  allocated = allocate_data(&hale_data->cell_force_x, 
      (local_nx)*(local_ny)*unstructured_mesh->nnodes_by_cell);
  allocated = allocate_data(&hale_data->cell_force_y, 
      (local_nx)*(local_ny)*unstructured_mesh->nnodes_by_cell);
  allocated = allocate_data(&hale_data->node_force_x, (local_nx+1)*(local_ny+1));
  allocated = allocate_data(&hale_data->node_force_y, (local_nx+1)*(local_ny+1));
  allocated = allocate_data(&hale_data->cell_mass, (local_nx)*(local_ny));
  allocated = allocate_data(&hale_data->nodal_mass, (local_nx+1)*(local_ny+1));
  allocated = allocate_data(&hale_data->nodal_volumes, (local_nx+1)*(local_ny+1));
  allocated = allocate_data(&hale_data->nodal_soundspeed, (local_nx+1)*(local_ny+1));
  allocated = allocate_data(&hale_data->limiter, (local_nx+1)*(local_ny+1));
  return allocated;
}

void deallocate_hale_data(
    HaleData* hale_data)
{
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

// Builds an unstructured mesh with an nx by ny rectilinear layout
size_t initialise_unstructured_mesh(
    Mesh* mesh, UnstructuredMesh* unstructured_mesh)
{
  // Just setting all cells to have same number of nodes
  const int nx = mesh->local_nx;
  const int ny = mesh->local_nx;
  const int global_nx = mesh->global_nx;
  const int global_ny = mesh->global_nx;
  const double width = mesh->width;
  const double height = mesh->height;
  unstructured_mesh->nnodes_by_cell = 4;
  unstructured_mesh->ncells_by_node = 4;
  unstructured_mesh->ncells = nx*ny;
  unstructured_mesh->nnodes = (nx+1)*(nx+1);

  const int nboundary_cells = 2*(nx+ny);

  size_t allocated = allocate_data(&unstructured_mesh->nodes_x0, unstructured_mesh->nnodes);
  allocated += allocate_data(&unstructured_mesh->nodes_y0, unstructured_mesh->nnodes);
  allocated = allocate_data(&unstructured_mesh->nodes_x1, unstructured_mesh->nnodes);
  allocated += allocate_data(&unstructured_mesh->nodes_y1, unstructured_mesh->nnodes);
  allocated += allocate_int_data(&unstructured_mesh->cells_to_nodes, 
      unstructured_mesh->ncells*unstructured_mesh->nnodes_by_cell);
  allocated += allocate_int_data(&unstructured_mesh->cells_to_nodes_off, unstructured_mesh->ncells+1);
  allocated += allocate_data(&unstructured_mesh->cell_centroids_x, unstructured_mesh->ncells);
  allocated += allocate_data(&unstructured_mesh->cell_centroids_y, unstructured_mesh->ncells);
  allocated += allocate_int_data(&unstructured_mesh->boundary_index, unstructured_mesh->nnodes);
  allocated += allocate_data(&unstructured_mesh->boundary_normal_x, nboundary_cells);
  allocated += allocate_data(&unstructured_mesh->boundary_normal_y, nboundary_cells);
  allocated += allocate_int_data(&unstructured_mesh->boundary_type, nboundary_cells);

  // Construct the list of nodes contiguously, currently Cartesian
  for(int ii = 0; ii < (ny+1); ++ii) {
    for(int jj = 0; jj < (nx+1); ++jj) {
      const int index = (ii)*(nx+1)+(jj);
      const double cell_width = (width/(double)global_nx);
      const double cell_height = (height/(double)global_ny);
      unstructured_mesh->nodes_x0[index] = (double)((jj)-mesh->pad)*cell_width;
      unstructured_mesh->nodes_y0[index] = (double)((ii)-mesh->pad)*cell_height;
    }
  }

  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      const int offset = unstructured_mesh->cells_to_nodes_off[(ii)*nx+(jj)];
      unstructured_mesh->cells_to_nodes[(offset)+0] = (ii)*(nx+1)+(jj);
      unstructured_mesh->cells_to_nodes[(offset)+1] = (ii)*(nx+1)+(jj+1);
      unstructured_mesh->cells_to_nodes[(offset)+2] = (ii+1)*(nx+1)+(jj+1);
      unstructured_mesh->cells_to_nodes[(offset)+3] = (ii+1)*(nx+1)+(jj);
      unstructured_mesh->cells_to_nodes_off[(ii)*nx+(jj)+1] = 
        unstructured_mesh->cells_to_nodes_off[(ii)*nx+(jj)] +
        unstructured_mesh->nnodes_by_cell;
    }
  }

  // TODO: Currently serial only, could do some work to parallelise this if
  // needed later on...
  // Store the halo node's neighbour, and normal
  int index = 0;
  for(int ii = 0; ii < (ny+1); ++ii) {
    for(int jj = 0; jj < (nx+1); ++jj) {
      unstructured_mesh->boundary_index[(ii)*(nx+1)+(jj)] = IS_INTERIOR_NODE;

      if(ii == 0) { 
        if(jj == 0) { 
          unstructured_mesh->boundary_type[(index)] = IS_FIXED;
        }
        else if(jj == (nx+1)-1) {
          unstructured_mesh->boundary_type[(index)] = IS_FIXED;
        }
        else {
          unstructured_mesh->boundary_type[(index)] = IS_BOUNDARY;
          unstructured_mesh->boundary_normal_x[(index)] = 0.0;
          unstructured_mesh->boundary_normal_y[(index)] = 1.0;
        }
        unstructured_mesh->boundary_index[(ii)*(nx+1)+(jj)] = index++;
      }
      else if(ii == (ny+1)-1) {
        if(jj == 0) { 
          unstructured_mesh->boundary_type[(index)] = IS_FIXED;
        }
        else if(jj == (nx+1)-1) {
          unstructured_mesh->boundary_type[(index)] = IS_FIXED;
        }
        else {
          unstructured_mesh->boundary_type[(index)] = IS_BOUNDARY;
          unstructured_mesh->boundary_normal_x[(index)] = 0.0;
          unstructured_mesh->boundary_normal_y[(index)] = -1.0;
        }
        unstructured_mesh->boundary_index[(ii)*(nx+1)+(jj)] = index++;
      } 
      else if(jj == 0) { 
        unstructured_mesh->boundary_type[(index)] = IS_BOUNDARY;
        unstructured_mesh->boundary_normal_x[(index)] = 1.0;
        unstructured_mesh->boundary_normal_y[(index)] = 0.0;
        unstructured_mesh->boundary_index[(ii)*(nx+1)+(jj)] = index++;
      }
      else if(jj == (nx+1)-1) {
        unstructured_mesh->boundary_type[(index)] = IS_BOUNDARY;
        unstructured_mesh->boundary_normal_x[(index)] = -1.0;
        unstructured_mesh->boundary_normal_y[(index)] = 0.0;
        unstructured_mesh->boundary_index[(ii)*(nx+1)+(jj)] = index++;
      }
    }
  }

  return allocated;
}

// Reads an unstructured mesh from an input file
size_t read_unstructured_mesh(
    Mesh* mesh, UnstructuredMesh* unstructured_mesh)
{
  // Just setting all cells to have same number of nodes
  unstructured_mesh->nnodes_by_cell = 3;
  unstructured_mesh->ncells_by_node = 3;

  // Open the files
  FILE* node_fp = fopen(unstructured_mesh->node_filename, "r");
  FILE* ele_fp = fopen(unstructured_mesh->ele_filename, "r");
  if(!node_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", unstructured_mesh->node_filename);
  }
  if(!ele_fp) {
    TERMINATE("Could not open the parameter file: %s.\n", unstructured_mesh->ele_filename);
  }

  // Fetch the first line of the nodes file
  char buf[MAX_STR_LEN];
  char* temp = buf;

  // Read the number of nodes, for allocation
  fgets(temp, MAX_STR_LEN, node_fp);
  skip_whitespace(&temp);
  sscanf(temp, "%d", &unstructured_mesh->nnodes);

  // Read the number of cells
  fgets(temp, MAX_STR_LEN, ele_fp);
  skip_whitespace(&temp);
  sscanf(temp, "%d", &unstructured_mesh->ncells);

  // Allocate the data structures that we now know the sizes of
  size_t allocated = allocate_data(&unstructured_mesh->nodes_x0, unstructured_mesh->nnodes);
  allocated += allocate_data(&unstructured_mesh->nodes_y0, unstructured_mesh->nnodes);
  allocated = allocate_data(&unstructured_mesh->nodes_x1, unstructured_mesh->nnodes);
  allocated += allocate_data(&unstructured_mesh->nodes_y1, unstructured_mesh->nnodes);
  allocated += allocate_int_data(&unstructured_mesh->cells_to_nodes, 
      unstructured_mesh->ncells*unstructured_mesh->nnodes_by_cell);
  allocated += allocate_int_data(&unstructured_mesh->cells_to_nodes_off, unstructured_mesh->ncells+1);
  allocated += allocate_data(&unstructured_mesh->cell_centroids_x, unstructured_mesh->ncells);
  allocated += allocate_data(&unstructured_mesh->cell_centroids_y, unstructured_mesh->ncells);
  allocated += allocate_int_data(&unstructured_mesh->boundary_index, unstructured_mesh->nnodes);

  int nboundary_cells = 0;

  // Loop through the node file, storing all of the nodes in our data structure
  while(fgets(temp, MAX_STR_LEN, node_fp)) {
    int index;
    int is_boundary;

    sscanf(temp, "%d", &index); 

    int discard;
    sscanf(temp, "%d%lf%lf%d", 
        &discard, 
        &unstructured_mesh->nodes_x0[(index)], 
        &unstructured_mesh->nodes_y0[(index)],
        &is_boundary);

    if(is_boundary) {
      unstructured_mesh->boundary_index[(index)] = 1;
      nboundary_cells++;
    }
  }

  int* boundary_edge_list;
  int boundary_edge_index = 0;
  allocate_int_data(&boundary_edge_list, nboundary_cells*2);

  // Loop through the element file and flatten into data structure
  while(fgets(temp, MAX_STR_LEN, ele_fp)) {
    int index;
    sscanf(temp, "%d", &index); 

    int discard;
    int node0;
    int node1;
    int node2;
    sscanf(temp, "%d%d%d%d", &discard, &node0, &node1, &node2);

    unstructured_mesh->cells_to_nodes[(index*unstructured_mesh->nnodes_by_cell)+0] = node0;
    unstructured_mesh->cells_to_nodes[(index*unstructured_mesh->nnodes_by_cell)+1] = node1;
    unstructured_mesh->cells_to_nodes[(index*unstructured_mesh->nnodes_by_cell)+2] = node2;
    unstructured_mesh->cells_to_nodes_off[(index+1)] = 
      unstructured_mesh->cells_to_nodes_off[(index)] + unstructured_mesh->nnodes_by_cell;

    // Store edge information about the connected boundary nodes
    if(unstructured_mesh->boundary_index[node0] == IS_BOUNDARY) {
      boundary_edge_list[boundary_edge_index++] = node0;
    }
    if(unstructured_mesh->boundary_index[node1] == IS_BOUNDARY) {
      boundary_edge_list[boundary_edge_index++] = node1;
    }
    if(unstructured_mesh->boundary_index[node2] == IS_BOUNDARY) {
      boundary_edge_list[boundary_edge_index++] = node2;
    }

    const double A = 
      (unstructured_mesh->nodes_x0[node0]*unstructured_mesh->nodes_y0[node1]-
       unstructured_mesh->nodes_x0[node1]*unstructured_mesh->nodes_y0[node0]+
       unstructured_mesh->nodes_x0[node1]*unstructured_mesh->nodes_y0[node2]-
       unstructured_mesh->nodes_x0[node2]*unstructured_mesh->nodes_y0[node1]+
       unstructured_mesh->nodes_x0[node2]*unstructured_mesh->nodes_y0[node0]-
       unstructured_mesh->nodes_x0[node0]*unstructured_mesh->nodes_y0[node2]);
    assert(A > 0.0 && "Nodes are not stored in counter-clockwise order.\n");
  }

  allocated += allocate_data(&unstructured_mesh->boundary_normal_x, nboundary_cells);
  allocated += allocate_data(&unstructured_mesh->boundary_normal_y, nboundary_cells);
  allocated += allocate_int_data(&unstructured_mesh->boundary_type, nboundary_cells);
  allocated += allocate_data(&unstructured_mesh->boundary_normal_x, nboundary_cells);
  allocated += allocate_data(&unstructured_mesh->boundary_normal_y, nboundary_cells);

  // Loop through all of the boundary cells and find their normals
  for(int bb0 = 0; bb0 < nboundary_cells; ++bb0) {
    double normal_x = 0.0;
    double normal_y = 0.0;
    for(int bb1 = 0; bb1 < nboundary_cells; ++bb1) {
      const int node0 = boundary_edge_list[bb1*2];
      const int node1 = boundary_edge_list[bb1*2+1];

      if(node0 == bb0 || node1 == bb0) {
        const double node0_x = unstructured_mesh->nodes_x0[(node0)];
        const double node0_y = unstructured_mesh->nodes_y0[(node0)];
        const double node1_x = unstructured_mesh->nodes_x0[(node1)];
        const double node1_y = unstructured_mesh->nodes_y0[(node1)];

        unstructured_mesh->boundary_normal_x[bb0] += node1_x-node0_x;
        unstructured_mesh->boundary_normal_y[bb0] += node1_y-node0_y;
      }
    }

    const double normal_mag = sqrt(normal_x*normal_x+normal_y*normal_y);
    unstructured_mesh->boundary_normal_x[bb0] = normal_x/normal_mag;
    unstructured_mesh->boundary_normal_y[bb0] = normal_y/normal_mag;
    unstructured_mesh->boundary_type[bb0] = IS_BOUNDARY;
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

// Writes out mesh and data
void write_quad_data_to_visit(
    const int nx, const int ny, const int step, double* nodes_x, 
    double* nodes_y, const double* data, const int nodal)
{
  char filename[MAX_STR_LEN];
  sprintf(filename, "output%04d.silo", step);

  DBfile *dbfile = DBCreate(filename, DB_CLOBBER, DB_LOCAL,
      "simulation time step", DB_HDF5);

  int dims[] = {nx+1, ny+1};
  int ndims = 2;
  double *coords[] = {(double*)nodes_x, (double*)nodes_y};
  DBPutQuadmesh(dbfile, "quadmesh", NULL, coords, dims, ndims,
      DB_DOUBLE, DB_NONCOLLINEAR, NULL);

  int local_dims[2];
  if(nodal) {
    local_dims[0] = nx+1;
    local_dims[1] = ny+1;
  }
  else {
    local_dims[0] = nx;
    local_dims[1] = ny;
  }
  DBPutQuadvar1(dbfile, "nodal", "quadmesh", data, local_dims,
      ndims, NULL, 0, DB_DOUBLE, nodal ? DB_NODECENT : DB_ZONECENT, NULL);
  DBClose(dbfile);
}

// Writes out unstructured triangles to visit
void write_unstructured_tris_to_visit(
    const int nnodes, int ncells, const int step, double* nodes_x0, 
    double* nodes_y0, const int* cells_to_nodes)
{
  // Only triangles
  double* coords[] = { (double*)nodes_x0, (double*)nodes_y0 };
  int shapesize[] = { 3 };
  int shapecounts[] = { ncells };
  int shapetype[] = { DB_ZONETYPE_TRIANGLE };
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
  DBClose(dbfile);
}

