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
  allocated += allocate_int_data(&unstructured_mesh->halo_cell, unstructured_mesh->ncells);
  allocated += allocate_int_data(&unstructured_mesh->halo_index, unstructured_mesh->nnodes);
  allocated += allocate_data(&unstructured_mesh->halo_normal_x, nboundary_cells);
  allocated += allocate_data(&unstructured_mesh->halo_normal_y, nboundary_cells);
  allocated += allocate_int_data(&unstructured_mesh->halo_neighbour, nboundary_cells);

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

  // Define the list of nodes surrounding each cell, counter-clockwise
  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      const int cells_nodes_index = ((ii)*nx+(jj))*unstructured_mesh->nnodes_by_cell;
      unstructured_mesh->cells_to_nodes[(cells_nodes_index)+0] = (ii)*(nx+1)+(jj);
      unstructured_mesh->cells_to_nodes[(cells_nodes_index)+1] = (ii)*(nx+1)+(jj+1);
      unstructured_mesh->cells_to_nodes[(cells_nodes_index)+2] = (ii+1)*(nx+1)+(jj+1);
      unstructured_mesh->cells_to_nodes[(cells_nodes_index)+3] = (ii+1)*(nx+1)+(jj);
      unstructured_mesh->cells_to_nodes_off[(ii)*nx+(jj)+1] = 
        unstructured_mesh->cells_to_nodes_off[(ii)*nx+(jj)] +
        unstructured_mesh->nnodes_by_cell;
    }
  }

  // Store the immediate neighbour of a halo cell
  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      int neighbour_index = (ii)*nx+(jj);
      if(ii < mesh->pad) {
        neighbour_index += nx;
      }
      if(ii >= ny-mesh->pad) { 
        neighbour_index -= nx;
      }
      if(jj < mesh->pad) {
        neighbour_index++;
      }
      if(jj >= nx-mesh->pad) {
        neighbour_index--;
      }
      if(neighbour_index != (ii)*nx+(jj)) {
        unstructured_mesh->halo_cell[(ii)*nx+(jj)] = neighbour_index;
      }
    }
  }

  // TODO: Currently serial only, could do some work to parallelise this if
  // needed later on...
  // Store the halo node's neighbour, and normal
  int halo_index = 0;
  for(int ii = 0; ii < (ny+1); ++ii) {
    for(int jj = 0; jj < (nx+1); ++jj) {
      if(ii == 0 || ii == (ny+1)-1 || jj == 0 || jj == (nx+1)-1) {
        unstructured_mesh->halo_index[(ii)*(nx+1)+(jj)] = IS_BOUNDARY;
        continue;
      }
      else if(ii > mesh->pad && ii < (ny+1)-1-mesh->pad && 
          jj > mesh->pad && jj < (nx+1)-1-mesh->pad) {
        unstructured_mesh->halo_index[(ii)*(nx+1)+(jj)] = IS_NOT_HALO;
        continue;
      }

      if(ii == mesh->pad) {
        // Handle corner cases
        if(jj == mesh->pad) {
          unstructured_mesh->halo_neighbour[(halo_index)] = (ii+1)*(nx+1)+(jj+1);
        }
        else if(jj == (nx+1)-1-mesh->pad) {
          unstructured_mesh->halo_neighbour[(halo_index)] = (ii+1)*(nx+1)+(jj-1);
        }
        else {
          unstructured_mesh->halo_normal_x[(halo_index)] = 0.0;
          unstructured_mesh->halo_normal_y[(halo_index)] = 1.0;
          unstructured_mesh->halo_neighbour[(halo_index)] = (ii+1)*(nx+1)+(jj);
        }
      }
      else if(jj == mesh->pad) {
        if(ii == (ny+1)-1-mesh->pad) {
          unstructured_mesh->halo_neighbour[(halo_index)] = (ii-1)*(nx+1)+(jj+1);
        }
        else { 
          unstructured_mesh->halo_normal_x[(halo_index)] = 1.0;
          unstructured_mesh->halo_normal_y[(halo_index)] = 0.0;
          unstructured_mesh->halo_neighbour[(halo_index)] = (ii)*(nx+1)+(jj+1);
        }
      }
      else if(jj == (nx+1)-1-mesh->pad) {
        if(ii == (ny+1)-1-mesh->pad) {
          unstructured_mesh->halo_neighbour[(halo_index)] = (ii-1)*(nx+1)+(jj-1);
        }
        else { 
          unstructured_mesh->halo_normal_x[(halo_index)] = -1.0;
          unstructured_mesh->halo_normal_y[(halo_index)] = 0.0;
          unstructured_mesh->halo_neighbour[(halo_index)] = (ii)*(nx+1)+(jj-1);
        }
      }
      else if(ii == (ny+1)-1-mesh->pad) {
        unstructured_mesh->halo_normal_x[(halo_index)] = 0.0;
        unstructured_mesh->halo_normal_y[(halo_index)] = -1.0;
        unstructured_mesh->halo_neighbour[(halo_index)] = (ii-1)*(nx+1)+(jj);
      }

      unstructured_mesh->halo_index[(ii)*(nx+1)+(jj)] = halo_index++;
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
  allocated += allocate_int_data(&unstructured_mesh->halo_cell, unstructured_mesh->ncells);
  allocated += allocate_int_data(&unstructured_mesh->halo_index, unstructured_mesh->nnodes);

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

    unstructured_mesh->halo_index[(index)] = (is_boundary) ? 1 : 0;
  }

  // Loop through the element file and flatten into data structure
  while(fgets(temp, MAX_STR_LEN, ele_fp)) {
    int index;
    sscanf(temp, "%d", &index); 

    int discard;
    int node1;
    int node2;
    int node3;
    sscanf(temp, "%d%d%d%d", &discard, &node1, &node2, &node3);

    unstructured_mesh->cells_to_nodes[(index*unstructured_mesh->nnodes_by_cell)+0] = node1;
    unstructured_mesh->cells_to_nodes[(index*unstructured_mesh->nnodes_by_cell)+1] = node2;
    unstructured_mesh->cells_to_nodes[(index*unstructured_mesh->nnodes_by_cell)+2] = node3;
    unstructured_mesh->cells_to_nodes_off[(index+1)] = 
      unstructured_mesh->cells_to_nodes_off[(index)] + unstructured_mesh->nnodes_by_cell;

    if(unstructured_mesh->halo_index[(node1)] == IS_BOUNDARY ||
        unstructured_mesh->halo_index[(node2)] == IS_BOUNDARY ||
        unstructured_mesh->halo_index[(node3)] == IS_BOUNDARY) {

      // TODO: need to find neighbour here...
      unstructured_mesh->halo_cell[(index)] = 1;
    }

    const double A = 
      (unstructured_mesh->nodes_x0[node1]*unstructured_mesh->nodes_y0[node2]-
       unstructured_mesh->nodes_x0[node2]*unstructured_mesh->nodes_y0[node1]+
       unstructured_mesh->nodes_x0[node2]*unstructured_mesh->nodes_y0[node3]-
       unstructured_mesh->nodes_x0[node3]*unstructured_mesh->nodes_y0[node2]+
       unstructured_mesh->nodes_x0[node3]*unstructured_mesh->nodes_y0[node1]-
       unstructured_mesh->nodes_x0[node1]*unstructured_mesh->nodes_y0[node3]);
    assert(A > 0.0 && "Nodes are not stored in counter-clockwise order.\n");
  }

  const int nboundary_cells = 0;
  allocated += allocate_data(&unstructured_mesh->halo_normal_x, nboundary_cells);
  allocated += allocate_data(&unstructured_mesh->halo_normal_y, nboundary_cells);
  allocated += allocate_int_data(&unstructured_mesh->halo_neighbour, nboundary_cells);

  return allocated;
}

// Reflect the node centered velocities on the boundary
void handle_unstructured_reflect(
    const int nnodes, const int* halo_index, const int* halo_neighbour, 
    const double* halo_normal_x, const double* halo_normal_y, 
    double* velocity_x, double* velocity_y)
{
  for(int nn = 0; nn < nnodes; ++nn) {
    const int index = halo_index[(nn)];
    if(index == IS_NOT_HALO) {
      continue;
    }

    const int neighbour_index = halo_neighbour[(index)];

    if(index == IS_BOUNDARY) {
      // This is node is an artificial boundary node
      velocity_x[(nn)] = 0.0;
      velocity_y[(nn)] = 0.0;
    }
    else {
      const double dot = (velocity_x[(nn)]*halo_normal_x[(index)]+
          velocity_y[(nn)]*halo_normal_y[(index)]);

      // Calculate the reflected velocity
      velocity_x[(nn)] = velocity_x[(nn)] - halo_normal_x[(index)]*2.0*dot;
      velocity_y[(nn)] = velocity_y[(nn)] - halo_normal_y[(index)]*2.0*dot;

#if 0
      // Project the velocity onto the face direction
      const double halo_parallel_x = halo_normal_y[(index)];
      const double halo_parallel_y = -halo_normal_x[(index)];
      const double vel_dot_parallel = 
        (velocity_x[(nn)]*halo_parallel_x+velocity_y[(nn)]*halo_parallel_y);
      velocity_x[(nn)] = halo_parallel_x*vel_dot_parallel;
      velocity_y[(nn)] = halo_parallel_y*vel_dot_parallel;

      // Calculate the reflected velocity
      const double reflect_x = velocity_x[(nn)] - 
        halo_normal_x[(index)]*2.0*(velocity_x[(nn)]*halo_normal_x[(index)]+
            velocity_y[(nn)]*halo_normal_y[(index)]);
      const double reflect_y = velocity_y[(nn)] - 
        halo_normal_y[(index)]*2.0*(velocity_x[(nn)]*halo_normal_x[(index)]+
            velocity_y[(nn)]*halo_normal_y[(index)]);

      // Project the reflected velocity back to the neighbour
      velocity_x[(neighbour_index)] += halo_normal_x[(index)]*
        (reflect_x*halo_normal_x[(index)]+reflect_y*halo_normal_y[(index)]);
      velocity_y[(neighbour_index)] += halo_normal_y[(index)]*
        (reflect_x*halo_normal_x[(index)]+reflect_y*halo_normal_y[(index)]);
#endif // if 0
    }
  }
}

// Fill boundary cells with interior values
void handle_unstructured_cell_boundary(
    const int ncells, const int* halo_cell, double* arr)
{
  // Perform the local halo update with reflective boundary condition
  for(int cc = 0; cc < ncells; ++cc) {
    if(halo_cell[(cc)]) {
      const int neighbour_index = halo_cell[(cc)];
      arr[(cc)] = arr[(neighbour_index)];
    }
  }
}

// Fill halo nodes with interior values
void handle_unstructured_node_boundary(
    const int nnodes, const int* halo_index, const int* halo_neighbour, double* arr)
{
  for(int nn = 0; nn < nnodes; ++nn) {
    const int index = halo_index[(nn)];

    if(index >= 0) {
      arr[(nn)] = arr[(halo_neighbour[(index)])];
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

