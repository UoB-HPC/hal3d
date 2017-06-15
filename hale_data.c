#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "hale_data.h"
#include "../shared.h"
#include "../mesh.h"

// Initialises the shared_data variables for two dimensional applications
size_t initialise_hale_data_2d(
    const int local_nx, const int local_ny, HaleData* hale_data)
{
  size_t allocated = allocate_data(&hale_data->rho_u, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->rho_v, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->F_x, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->F_y, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->uF_x, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->uF_y, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->vF_x, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->vF_y, (local_nx+1)*(local_ny+1));
  return allocated;
}

void deallocate_hale_data(
    HaleData* hale_data)
{
  deallocate_data(hale_data->rho_u);
  deallocate_data(hale_data->rho_v);
  deallocate_data(hale_data->F_x);
  deallocate_data(hale_data->F_y);
  deallocate_data(hale_data->uF_x);
  deallocate_data(hale_data->uF_y);
  deallocate_data(hale_data->vF_x);
  deallocate_data(hale_data->vF_y);
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

  size_t allocated = allocate_data(&unstructured_mesh->nodes_x, (nx+1)*(ny+1));
  allocated += allocate_data(&unstructured_mesh->nodes_y, (nx+1)*(ny+1));

  // Construct the list of nodes contiguously, currently Cartesian
  for(int ii = 0; ii < (ny+1); ++ii) {
    for(int jj = 0; jj < (nx+1); ++jj) {
      const int index = (ii)*(nx+1)+(jj);
      const double cell_width = (width/(double)global_nx);
      const double cell_height = (height/(double)global_ny);
      unstructured_mesh->nodes_x[index] = (double)((jj)-PAD)*cell_width;
      unstructured_mesh->nodes_y[index] = (double)((ii)-PAD)*cell_height;
    }
  }

  const int nnodes_by_cell = 4;
  const int ncells_by_node = 4;
  allocated += allocate_data(&unstructured_mesh->nodes_x, (nx+1)*(ny+1));
  allocated += allocate_data(&unstructured_mesh->nodes_y, (nx+1)*(ny+1));
  allocated += allocate_int_data(&unstructured_mesh->nodes_cells_off, nx*ny+1);
  allocated += allocate_int_data(&unstructured_mesh->cells_nodes_off, nx*ny+1);
  allocated += allocate_int_data(&unstructured_mesh->cells_nodes, nx*ny*nnodes_by_cell);
  allocated += 
    allocate_int_data(&unstructured_mesh->nodes_cells, (nx+1)*(ny+1)*ncells_by_node);
  
  // Define the list of nodes surrounding each cell, counter-clockwise
  unstructured_mesh->nodes_cells_off[0] = 0;
  for(int ii = PAD; ii < ny-PAD; ++ii) {
    for(int jj = PAD; jj < nx-PAD; ++jj) {
      const int cells_nodes_index = ((ii)*nx+(jj))*nnodes_by_cell;
      unstructured_mesh->cells_nodes[(cells_nodes_index)+0] = (ii)*(nx+1)+(jj);
      unstructured_mesh->cells_nodes[(cells_nodes_index)+1] = (ii)*(nx+1)+(jj+1);
      unstructured_mesh->cells_nodes[(cells_nodes_index)+2] = (ii+1)*(nx+1)+(jj+1);
      unstructured_mesh->cells_nodes[(cells_nodes_index)+3] = (ii+1)*(nx+1)+(jj);
      unstructured_mesh->cells_nodes_off[(ii)*nx+(jj)+1] += nnodes_by_cell;
    }
  }

  // Define the list of cells connected to each node
  unstructured_mesh->cells_nodes_off[0] = 0;
  for(int ii = PAD; ii < (ny+1)-PAD; ++ii) {
    for(int jj = PAD; jj < (nx+1)-PAD; ++jj) {
      const int nodes_cells_index = ((ii)*(nx+1)+(jj))*ncells_by_node;
      unstructured_mesh->nodes_cells[(nodes_cells_index)+0] = (ii-1)*nx+(jj-1);
      unstructured_mesh->nodes_cells[(nodes_cells_index)+1] = (ii-1)*nx+(jj);
      unstructured_mesh->nodes_cells[(nodes_cells_index)+2] = (ii)*nx+(jj-1);
      unstructured_mesh->nodes_cells[(nodes_cells_index)+3] = (ii)*nx+(jj);
      unstructured_mesh->nodes_cells_off[(ii)*(nx+1)+(jj)+1] += ncells_by_node;
    }
  }

  return allocated;
}

#if 0
  // Ordered by node_vertex_0 is closest to bottom left
  allocate_int_data(&unstructured_mesh->node_vertex0, unstructured_mesh->nnodes);
  allocate_int_data(&unstructured_mesh->node_vertex1, unstructured_mesh->nnodes);
  allocate_int_data(&unstructured_mesh->nodes_cells, unstructured_mesh->nnodes*NCELLS_PER_node);

  // Calculate the nodes connecting each node
  for(int ii = 0; ii < ny+1; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      const int node_index = (ii)*(2*nx+1)+(jj);
      unstructured_mesh->node_vertex0[node_index] = (ii)*(nx+1)+(jj);
      unstructured_mesh->node_vertex1[node_index] = (ii)*(nx+1)+(jj+1);
    }
    if(ii < ny) {
      for(int jj = 0; jj < nx+1; ++jj) {
        const int node_index = (ii)*(2*nx+1)+(jj)+nx;
        unstructured_mesh->node_vertex0[node_index] = (ii)*(nx+1)+(jj);
        unstructured_mesh->node_vertex1[node_index] = (ii+1)*(nx+1)+(jj);
      }
    }
  }

  // Calculate the cells connected to nodes, as a neighbour list.
  for(int ii = 0; ii < ny+1; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      const int node_index = (ii)*(2*nx+1)+(jj);
      unstructured_mesh->nodes_cells[(node_index)*NCELLS_PER_node+0] = (ii)*nx+(jj);
      unstructured_mesh->nodes_cells[(node_index)*NCELLS_PER_node+1] = (ii-1)*nx+(jj);
    }
    if(ii < ny) {
      for(int jj = 0; jj < nx+1; ++jj) {
        const int node_index = (ii)*(2*nx+1)+(jj)+nx;
        unstructured_mesh->nodes_cells[(node_index)*NCELLS_PER_node+0] = (ii)*nx+(jj);
        unstructured_mesh->nodes_cells[(node_index)*NCELLS_PER_node+1] = (ii)*nx+(jj-1);
      }
    }
  }

  // TODO: Make sure that the memory order of the cells_nodes array is
  // optimal for all architectures
  allocate_int_data(&unstructured_mesh->cells_nodes, Nnodes*nx*ny);
  allocate_data(&unstructured_mesh->cell_centroids_x, nx*ny);
  allocate_data(&unstructured_mesh->cell_centroids_y, nx*ny);
  allocate_data(&unstructured_mesh->volume, nx*ny);

  // Initialise cells connecting nodes
  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      unstructured_mesh->cells_nodes[(BOTTOM)*nx*ny+(ii)*nx+(jj)] = (ii)*(2*nx+1)+(jj);
      unstructured_mesh->cells_nodes[(LEFT)*nx*ny+(ii)*nx+(jj)] = 
        unstructured_mesh->cells_nodes[(BOTTOM)*nx*ny+(ii)*nx+(jj)]+nx;
      unstructured_mesh->cells_nodes[(RIGHT)*nx*ny+(ii)*nx+(jj)] =
        unstructured_mesh->cells_nodes[(LEFT)*nx*ny+(ii)*nx+(jj)]+1;
      unstructured_mesh->cells_nodes[(TOP)*nx*ny+(ii)*nx+(jj)] =
        unstructured_mesh->cells_nodes[(RIGHT)*nx*ny+(ii)*nx+(jj)]+nx;
    }
  }

  // Find the (x,y) location of each of the cell centroids, and the cell volume
  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      const int cell_index = (ii)*nx+(jj);

      double A = 0.0;
      double c_x_factor = 0.0;
      double c_y_factor = 0.0;

      for(int kk = 0; kk < Nnodes; ++kk) {
        int node_index = unstructured_mesh->cells_nodes[(kk)*nx*ny+(cell_index)];
        int node_vertex0 = unstructured_mesh->node_vertex0[node_index];
        int node_vertex1 = unstructured_mesh->node_vertex1[node_index];

        // The top and left nodes need to be ordered backwards to ensure
        // correct counter-clockwise access
        double x0 = (kk == TOP || kk == LEFT) 
          ?  unstructured_mesh->nodes_x[node_vertex1] 
          : unstructured_mesh->nodes_x[node_vertex0];
        double y0 = (kk == TOP || kk == LEFT) 
          ? unstructured_mesh->nodes_y[node_vertex1] 
          : unstructured_mesh->nodes_y[node_vertex0];
        double x1 = (kk == TOP || kk == LEFT) 
          ? unstructured_mesh->nodes_x[node_vertex0] 
          : unstructured_mesh->nodes_x[node_vertex1];
        double y1 = (kk == TOP || kk == LEFT) 
          ? unstructured_mesh->nodes_y[node_vertex0] 
          : unstructured_mesh->nodes_y[node_vertex1];

        A += 0.5*(x0*y1-x1*y0);
        c_x_factor += (x0+x1)*(x0*y1-x1*y0);
        c_y_factor += (y0+y1)*(x0*y1-x1*y0);
      }

      // NOTE: This calculation of the volume is actually general to all
      // simple polygons...
      unstructured_mesh->volume[(cell_index)] = A;
      unstructured_mesh->cell_centroids_x[(cell_index)] = (1.0/(6.0*A))*c_x_factor;
      unstructured_mesh->cell_centroids_y[(cell_index)] = (1.0/(6.0*A))*c_y_factor;
    }
  }
#endif // if 0

