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
  size_t allocated = allocate_data(&hale_data->energy0, (local_nx)*(local_ny));
  allocated += allocate_data(&hale_data->density0, (local_nx)*(local_ny));
  allocated += allocate_data(&hale_data->pressure0, (local_nx)*(local_ny));
  allocated += allocate_data(&hale_data->velocity_x0, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->velocity_y0, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->energy1, (local_nx)*(local_ny));
  allocated += allocate_data(&hale_data->density1, (local_nx)*(local_ny));
  allocated += allocate_data(&hale_data->pressure1, (local_nx)*(local_ny));
  allocated += allocate_data(&hale_data->velocity_x1, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->velocity_y1, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->cell_force_x, 
      (local_nx)*(local_ny)*umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->cell_force_y, 
      (local_nx)*(local_ny)*umesh->nnodes_by_cell);
  allocated += allocate_data(&hale_data->node_force_x, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->node_force_y, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->cell_mass, (local_nx)*(local_ny));
  allocated += allocate_data(&hale_data->nodal_mass, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->nodal_volumes, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->nodal_soundspeed, (local_nx+1)*(local_ny+1));
  allocated += allocate_data(&hale_data->limiter, (local_nx+1)*(local_ny+1));

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

// Builds an unstructured mesh with an nx by ny rectilinear layout
size_t initialise_unstructured_mesh(
    Mesh* mesh, UnstructuredMesh* umesh)
{
  // Just setting all cells to have same number of nodes
  const int nx = mesh->local_nx;
  const int ny = mesh->local_nx;
  const int global_nx = mesh->global_nx;
  const int global_ny = mesh->global_nx;
  const double width = mesh->width;
  const double height = mesh->height;
  umesh->nnodes_by_cell = 4;
  umesh->ncells_by_node = 4;
  umesh->ncells = nx*ny;
  umesh->nnodes = (nx+1)*(nx+1);

  const int nboundary_cells = 2*(nx+ny);

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
  allocated += allocate_data(&umesh->boundary_normal_x, nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, nboundary_cells);

  // Construct the list of nodes contiguously, currently Cartesian
  for(int ii = 0; ii < (ny+1); ++ii) {
    for(int jj = 0; jj < (nx+1); ++jj) {
      const int index = (ii)*(nx+1)+(jj);
      const double cell_width = (width/(double)global_nx);
      const double cell_height = (height/(double)global_ny);
      umesh->nodes_x0[index] = (double)((jj)-mesh->pad)*cell_width;
      umesh->nodes_y0[index] = (double)((ii)-mesh->pad)*cell_height;
    }
  }

  for(int ii = 0; ii < ny; ++ii) {
    for(int jj = 0; jj < nx; ++jj) {
      const int offset = umesh->cells_to_nodes_off[(ii)*nx+(jj)];
      umesh->cells_to_nodes[(offset)+0] = (ii)*(nx+1)+(jj);
      umesh->cells_to_nodes[(offset)+1] = (ii)*(nx+1)+(jj+1);
      umesh->cells_to_nodes[(offset)+2] = (ii+1)*(nx+1)+(jj+1);
      umesh->cells_to_nodes[(offset)+3] = (ii+1)*(nx+1)+(jj);
      umesh->cells_to_nodes_off[(ii)*nx+(jj)+1] = 
        umesh->cells_to_nodes_off[(ii)*nx+(jj)] +
        umesh->nnodes_by_cell;
    }
  }

  // TODO: Currently serial only, could do some work to parallelise this if
  // needed later on...
  // Store the halo node's neighbour, and normal
  int index = 0;
  for(int ii = 0; ii < (ny+1); ++ii) {
    for(int jj = 0; jj < (nx+1); ++jj) {
      umesh->boundary_index[(ii)*(nx+1)+(jj)] = IS_INTERIOR_NODE;

      if(ii == 0) { 
        if(jj == 0) { 
          umesh->boundary_type[(index)] = IS_FIXED;
        }
        else if(jj == (nx+1)-1) {
          umesh->boundary_type[(index)] = IS_FIXED;
        }
        else {
          umesh->boundary_type[(index)] = IS_BOUNDARY;
          umesh->boundary_normal_x[(index)] = 0.0;
          umesh->boundary_normal_y[(index)] = 1.0;
        }
        umesh->boundary_index[(ii)*(nx+1)+(jj)] = index++;
      }
      else if(ii == (ny+1)-1) {
        if(jj == 0) { 
          umesh->boundary_type[(index)] = IS_FIXED;
        }
        else if(jj == (nx+1)-1) {
          umesh->boundary_type[(index)] = IS_FIXED;
        }
        else {
          umesh->boundary_type[(index)] = IS_BOUNDARY;
          umesh->boundary_normal_x[(index)] = 0.0;
          umesh->boundary_normal_y[(index)] = -1.0;
        }
        umesh->boundary_index[(ii)*(nx+1)+(jj)] = index++;
      } 
      else if(jj == 0) { 
        umesh->boundary_type[(index)] = IS_BOUNDARY;
        umesh->boundary_normal_x[(index)] = 1.0;
        umesh->boundary_normal_y[(index)] = 0.0;
        umesh->boundary_index[(ii)*(nx+1)+(jj)] = index++;
      }
      else if(jj == (nx+1)-1) {
        umesh->boundary_type[(index)] = IS_BOUNDARY;
        umesh->boundary_normal_x[(index)] = -1.0;
        umesh->boundary_normal_y[(index)] = 0.0;
        umesh->boundary_index[(ii)*(nx+1)+(jj)] = index++;
      }
    }
  }

  return allocated;
}

// Reads an unstructured mesh from an input file
size_t read_unstructured_mesh(
    Mesh* mesh, UnstructuredMesh* umesh)
{
  // Just setting all cells to have same number of nodes
  umesh->nnodes_by_cell = 3;
  umesh->ncells_by_node = 3;

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
  sscanf(temp, "%d", &umesh->ncells);

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
  allocate_int_data(&boundary_edge_list, nboundary_cells*2);

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

    // Only store edges that are on the boundary
    if(nboundary_nodes == 2) {
      for(int nn = 0; nn < umesh->nnodes_by_cell; ++nn) {
        if(umesh->boundary_index[(node[nn])] != IS_INTERIOR_NODE) {
          boundary_edge_list[boundary_edge_index++] = node[nn];
        }
      }
    }

    // TODO: change to loop
    const double A = 
      (umesh->nodes_x0[node[0]]*umesh->nodes_y0[node[1]]-
       umesh->nodes_x0[node[1]]*umesh->nodes_y0[node[0]]+
       umesh->nodes_x0[node[1]]*umesh->nodes_y0[node[2]]-
       umesh->nodes_x0[node[2]]*umesh->nodes_y0[node[1]]+
       umesh->nodes_x0[node[2]]*umesh->nodes_y0[node[0]]-
       umesh->nodes_x0[node[0]]*umesh->nodes_y0[node[2]]);
    assert(A > 0.0 && "Nodes are not stored in counter-clockwise order.\n");
  }

  allocated += allocate_data(&umesh->boundary_normal_x, nboundary_cells);
  allocated += allocate_data(&umesh->boundary_normal_y, nboundary_cells);
  allocated += allocate_int_data(&umesh->boundary_type, nboundary_cells);

  // Loop through all of the boundary cells and find their normals
  for(int nn = 0; nn < umesh->nnodes; ++nn) {
    if(umesh->boundary_index[(nn)] == IS_INTERIOR_NODE) {
      continue;
    }

    const int boundary_index = umesh->boundary_index[(nn)];

    double normal_x = 0.0;
    double normal_y = 0.0;

#if 0
    for(int bb1 = 0; bb1 < nboundary_cells; ++bb1) {
      const int node0 = boundary_edge_list[bb1*2];
      const int node1 = boundary_edge_list[bb1*2+1];

      if(node0 == nn || node1 == nn) {
        const double node0_x = umesh->nodes_x0[(node0)];
        const double node0_y = umesh->nodes_y0[(node0)];
        const double node1_x = umesh->nodes_x0[(node1)];
        const double node1_y = umesh->nodes_y0[(node1)];

        printf("found %.12f %.12f %.12f %.12f\n", node0_x, node0_y, node1_x, node1_y);
        normal_x += node1_y-node0_y;
        normal_y += node1_x+node0_x;
      }
    }
#endif // if 0

    // TODO: REMOVE THIS HACK
    if(umesh->nodes_x0[(nn)] == 0.0) {
      normal_x = 1.0;
    }
    else if(umesh->nodes_x0[(nn)] == 1.0) {
      normal_x = -1.0;
    }

    if(umesh->nodes_y0[(nn)] == 0.0) {
      normal_y = 1.0;
    }
    else if(umesh->nodes_y0[(nn)] == 1.0) {
      normal_y = -1.0;
    }

    const double normal_mag = sqrt(normal_x*normal_x+normal_y*normal_y);
    if(normal_mag > 1.0) {
      umesh->boundary_type[(boundary_index)] = IS_FIXED;
    }
    else {
      umesh->boundary_type[(boundary_index)] = IS_BOUNDARY;
    }

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
    double* nodes_y0, const int* cells_to_nodes, const double* arr, const int nodal)
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
  DBPutUcdvar1(dbfile, "arr", "mesh", arr, (nodal ? nnodes : ncells), NULL, 0,
      DB_DOUBLE, (nodal ? DB_NODECENT : DB_ZONECENT), NULL);
  DBClose(dbfile);
}

