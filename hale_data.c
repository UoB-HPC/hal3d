#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <silo.h>
#include "hale_data.h"
#include "../shared.h"
#include "../mesh.h"

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

// Build a fully unstructured mesh, initially n by n rectilinear layout
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
  unstructured_mesh->ncells = mesh->local_nx*mesh->local_ny;
  unstructured_mesh->nnodes = (nx+1)*(nx+1);

  size_t allocated = allocate_data(&unstructured_mesh->nodes_x0, unstructured_mesh->nnodes);
  allocated += allocate_data(&unstructured_mesh->nodes_y0, unstructured_mesh->nnodes);
  allocated = allocate_data(&unstructured_mesh->nodes_x1, unstructured_mesh->nnodes);
  allocated += allocate_data(&unstructured_mesh->nodes_y1, unstructured_mesh->nnodes);
  allocated += allocate_int_data(&unstructured_mesh->cells_to_nodes, 
      nx*ny*unstructured_mesh->nnodes_by_cell);
  allocated += allocate_int_data(&unstructured_mesh->cells_to_nodes_off, nx*ny+1);
  allocated += allocate_data(&unstructured_mesh->cell_centroids_x, nx*ny);
  allocated += allocate_data(&unstructured_mesh->cell_centroids_y, nx*ny);
  allocated += allocate_int_data(&unstructured_mesh->halo_cell, nx*ny);
  allocated += allocate_int_data(&unstructured_mesh->halo_index, (nx+1)*(ny+1));
  allocated += allocate_data(&unstructured_mesh->halo_normal_x, 2*(nx+ny));
  allocated += allocate_data(&unstructured_mesh->halo_normal_y, 2*(nx+ny));
  allocated += allocate_int_data(&unstructured_mesh->halo_neighbour, 2*(nx+ny));

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

      int neighbour_index = 0;
      double normal_x = 0.0;
      double normal_y = 0.0;
      if(ii == mesh->pad) {
        // Handle corner cases
        if(jj == mesh->pad) {
          normal_x = 1.0;
          normal_y = 1.0;
          neighbour_index = (ii+1)*(nx+1)+(jj+1);
        }
        else if(jj == (nx+1)-1-mesh->pad) {
          normal_x = -1.0;
          normal_y = 1.0;
          neighbour_index = (ii+1)*(nx+1)+(jj-1);
        }
        else {
          normal_x = 0.0;
          normal_y = 1.0;
          neighbour_index = (ii+1)*(nx+1)+(jj);
        }
      }
      else if(jj == mesh->pad) {
        if(ii == (ny+1)-1-mesh->pad) {
          normal_x = 1.0;
          normal_y = -1.0;
          neighbour_index = (ii-1)*(nx+1)+(jj+1);
        }
        else { 
          normal_x = 1.0;
          normal_y = 0.0;
          neighbour_index = (ii)*(nx+1)+(jj+1);
        }
      }
      else if(jj == (nx+1)-1-mesh->pad) {
        if(ii == (ny+1)-1-mesh->pad) {
          normal_x = -1.0;
          normal_y = -1.0;
          neighbour_index = (ii-1)*(nx+1)+(jj-1);
        }
        else { 
          normal_x = -1.0;
          normal_y = 0.0;
          neighbour_index = (ii)*(nx+1)+(jj-1);
        }
      }
      else if(ii == (ny+1)-1-mesh->pad) {
        normal_x = 0.0;
        normal_y = -1.0;
        neighbour_index = (ii-1)*(nx+1)+(jj);
      }

      if(normal_x == 0.0 && normal_y == 0.0) {
        // The node isn't on the halo boundary
        unstructured_mesh->halo_index[(ii)*(nx+1)+(jj)] = IS_NOT_HALO;
      }
      else {
        // To normalise the normal vector
        const double normal_mag = sqrt(normal_x*normal_x+normal_y*normal_y);

        // Store the normal and neighbour
        unstructured_mesh->halo_neighbour[(halo_index)] = neighbour_index;
        unstructured_mesh->halo_normal_x[(halo_index)] = normal_x/normal_mag;
        unstructured_mesh->halo_normal_y[(halo_index)] = normal_y/normal_mag;
        unstructured_mesh->halo_index[(ii)*(nx+1)+(jj)] = halo_index++;
      }
    }
  }

  return allocated;
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

