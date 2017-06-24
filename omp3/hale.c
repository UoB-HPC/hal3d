#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include "hale.h"
#include "../hale_data.h"
#include "../hale_interface.h"
#include "../../comms.h"
#include "../../params.h"
#include "../../shared.h"

// Solve a single timestep on the given mesh
void solve_unstructured_hydro_2d(
    Mesh* mesh, const int ncells, const int nnodes, const double visc_coeff1, 
    const double visc_coeff2, double* cell_centroids_x, 
    double* cell_centroids_y, int* cells_to_nodes, int* cells_to_nodes_off, 
    double* nodes_x0, double* nodes_y0, double* nodes_x1, double* nodes_y1, 
    int* boundary_index, int* boundary_type, double* boundary_normal_x, 
    double* boundary_normal_y, double* energy0, double* energy1, double* density0, 
    double* density1, double* pressure0, double* pressure1, double* velocity_x0, 
    double* velocity_y0, double* velocity_x1, double* velocity_y1, 
    double* cell_force_x, double* cell_force_y, double* node_force_x, 
    double* node_force_y, double* cell_mass, double* nodal_mass, 
    double* nodal_volumes, double* nodal_soundspeed, double* limiter)
{
  /*
   *    PREDICTOR
   */

  // Calculate the pressure using the ideal gas equation of state
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    pressure0[(cc)] = (GAM-1.0)*energy0[(cc)]*density0[(cc)];
  }

  // Calculate the cell centroids
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
    const double inv_Np = 1.0/(double)nnodes_around_cell;

    cell_centroids_x[(cc)] = 0.0;
    cell_centroids_y[(cc)] = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = cells_to_nodes[(nodes_off)+(nn)];
      cell_centroids_x[(cc)] += nodes_x0[node_index]*inv_Np;
      cell_centroids_y[(cc)] += nodes_y0[node_index]*inv_Np;
    }
  }

#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    nodal_mass[(nn)] = 0.0;
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }

  double total_mass = 0.0;
  // Calculate the nodal and cell mass
#pragma omp parallel for reduction(+: total_mass)
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
    const double cell_centroid_x = cell_centroids_x[(cc)];
    const double cell_centroid_y = cell_centroids_y[(cc)];

    double cell_volume = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {

      // Determine the three point stencil of nodes around current node
      const int node_l_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];

      const double node_c_x = nodes_x0[(node_c_index)];
      const double node_c_y = nodes_y0[(node_c_index)];

      // Get the midpoints between l and r nodes and current node
      const double node_l_x = 0.5*(nodes_x0[node_l_index]+node_c_x);
      const double node_l_y = 0.5*(nodes_y0[node_l_index]+node_c_y);
      const double node_r_x = 0.5*(node_c_x+nodes_x0[node_r_index]);
      const double node_r_y = 0.5*(node_c_y+nodes_y0[node_r_index]);

      // Use shoelace formula to get the volume between node and cell c
      const double sub_cell_volume =
        0.5*((node_l_x*node_c_y + node_c_x*node_r_y +
              node_r_x*cell_centroid_y + cell_centroid_x*node_l_y) -
            (node_c_x*node_l_y + node_r_x*node_c_y +
             cell_centroid_x*node_r_y + node_l_x*cell_centroid_y));

      // Reduce the total cell volume for later calculation
      cell_volume += sub_cell_volume;

      nodal_mass[(node_c_index)] += density0[(cc)]*sub_cell_volume;

      // Calculate the volume and soundspeed at the node
      nodal_soundspeed[(node_c_index)] += 
        sqrt(GAM*(GAM-1.0)*energy0[(cc)])*sub_cell_volume;
      nodal_volumes[(node_c_index)] += sub_cell_volume;
    }

    // Calculate the mass and store volume for the whole cell
    cell_mass[(cc)] = density0[(cc)]*cell_volume;
    total_mass += cell_mass[(cc)];
  }

  printf("total mass %.12f\n", total_mass);

#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    node_force_x[(nn)] = 0.0;
    node_force_y[(nn)] = 0.0;
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }

  // Calculate the force contributions for pressure gradients
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 

      // Determine the three point stencil of nodes around current node
      const int node_l_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)]
        : cells_to_nodes[(nodes_off)+(nn+1)];

      // Calculate the area vectors away from cell through node, using
      // the half edge vectors adjacent to the node combined
      const double S_x =
        0.5*((nodes_y0[(node_c_index)]-nodes_y0[(node_l_index)]) +
            (nodes_y0[(node_r_index)]-nodes_y0[(node_c_index)]));
      const double S_y =
        -0.5*((nodes_x0[(node_c_index)]-nodes_x0[(node_l_index)]) +
            (nodes_x0[(node_r_index)]-nodes_x0[(node_c_index)]));

      node_force_x[(node_c_index)] += pressure0[(cc)]*S_x;
      node_force_y[(node_c_index)] += pressure0[(cc)]*S_y;
      cell_force_x[(nodes_off)+(nn)] = pressure0[(cc)]*S_x;
      cell_force_y[(nodes_off)+(nn)] = pressure0[(cc)]*S_y;
    }
  }

  calculate_artificial_viscosity(
      ncells, visc_coeff1, visc_coeff2, cells_to_nodes_off, cells_to_nodes, 
      nodes_x0, nodes_y0, cell_centroids_x, cell_centroids_y,
      velocity_x0, velocity_y0, nodal_soundspeed, nodal_mass,
      nodal_volumes, limiter, node_force_x, node_force_y);

  // Calculate the time centered evolved velocities, by first calculating the
  // predicted values at the new timestep and then averaging with current velocity
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    // Determine the predicted velocity
    velocity_x1[(nn)] = velocity_x0[(nn)] + mesh->dt*node_force_x[(nn)]/nodal_mass[(nn)];
    velocity_y1[(nn)] = velocity_y0[(nn)] + mesh->dt*node_force_y[(nn)]/nodal_mass[(nn)];

    // Calculate the time centered velocity
    velocity_x1[(nn)] = 0.5*(velocity_x0[(nn)] + velocity_x1[(nn)]);
    velocity_y1[(nn)] = 0.5*(velocity_y0[(nn)] + velocity_y1[(nn)]);
  }

  handle_unstructured_reflect(
      nnodes, boundary_index, boundary_type, boundary_normal_x, 
      boundary_normal_y, velocity_x1, velocity_y1);

  // Move the nodes by the predicted velocity
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    nodes_x1[(nn)] = nodes_x0[(nn)] + mesh->dt*velocity_x1[(nn)];
    nodes_y1[(nn)] = nodes_y0[(nn)] + mesh->dt*velocity_y1[(nn)];
  }

  set_timestep(
      ncells, cells_to_nodes, cells_to_nodes_off, 
      nodes_x1, nodes_y1, energy0, &mesh->dt);

    // Calculate the predicted energy
#pragma omp parallel for
    for(int cc = 0; cc < ncells; ++cc) {
      const int nodes_off = cells_to_nodes_off[(cc)];
      const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

      // Sum the time centered velocity by the sub-cell forces
      double force = 0.0;
      for(int nn = 0; nn < nnodes_around_cell; ++nn) {
        const int node_index = cells_to_nodes[(nodes_off)+(nn)];
        force += 
          (velocity_x1[(node_index)]*cell_force_x[(nodes_off)+(nn)] +
           velocity_y1[(node_index)]*cell_force_y[(nodes_off)+(nn)]);
      }

      energy1[(cc)] = energy0[(cc)] - mesh->dt*force/cell_mass[(cc)];
    }

  // Using the new volume, calculate the predicted density
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    double cell_volume = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {

      // Determine the three point stencil of nodes around current node
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];

      // Reduce the total cell volume for later calculation
      cell_volume += 0.5*(nodes_x1[node_c_index]+nodes_x1[node_r_index])*
        (nodes_y1[node_r_index]-nodes_y1[node_c_index]);
    }

    density1[(cc)] = cell_mass[(cc)]/cell_volume;
  }

  // Calculate the time centered pressure from mid point between original and 
  // predicted pressures
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    // Calculate the predicted pressure from the equation of state
    pressure1[(cc)] = (GAM-1.0)*energy1[(cc)]*density1[(cc)];

    // Determine the time centered pressure
    pressure1[(cc)] = 0.5*(pressure0[(cc)] + pressure1[(cc)]);
  }

  // Prepare time centered variables for the corrector step
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    nodes_x1[(nn)] = 0.5*(nodes_x1[(nn)] + nodes_x0[(nn)]);
    nodes_y1[(nn)] = 0.5*(nodes_y1[(nn)] + nodes_y0[(nn)]);
    node_force_x[(nn)] = 0.0;
    node_force_y[(nn)] = 0.0;
    nodal_volumes[(nn)] = 0.0;
    nodal_soundspeed[(nn)] = 0.0;
  }

  /*
   *    CORRECTOR
   */

  // Calculate the new cell centroids
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;
    const double inv_Np = 1.0/(double)nnodes_around_cell;

    cell_centroids_x[(cc)] = 0.0;
    cell_centroids_y[(cc)] = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_index = cells_to_nodes[(nodes_off)+(nn)];
      cell_centroids_x[(cc)] += nodes_x1[(node_index)]*inv_Np;
      cell_centroids_y[(cc)] += nodes_y1[(node_index)]*inv_Np;
    }
  }

  // Calculate the new nodal soundspeed and volumes
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    const double cell_centroid_x = cell_centroids_x[(cc)];
    const double cell_centroid_y = cell_centroids_y[(cc)];
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 

      // Determine the three point stencil of nodes around current node
      const int node_l_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];

      const double node_c_x = nodes_x1[(node_c_index)];
      const double node_c_y = nodes_y1[(node_c_index)];

      // Get the midpoints between l and r nodes and current node
      const double node_l_x = 0.5*(nodes_x1[node_l_index]+node_c_x);
      const double node_l_y = 0.5*(nodes_y1[node_l_index]+node_c_y);
      const double node_r_x = 0.5*(node_c_x+nodes_x1[node_r_index]);
      const double node_r_y = 0.5*(node_c_y+nodes_y1[node_r_index]);

      // Use shoelace formula to get the volume between node and cell c
      const double sub_cell_volume =
        0.5*((node_l_x*node_c_y + node_c_x*node_r_y +
              node_r_x*cell_centroid_y + cell_centroid_x*node_l_y) -
            (node_c_x*node_l_y + node_r_x*node_c_y +
             cell_centroid_x*node_r_y + node_l_x*cell_centroid_y));

      // Add contributions to the nodal mass from adjacent sub-cells
      nodal_soundspeed[(node_c_index)] += 
        sqrt(GAM*(GAM-1.0)*energy1[(cc)])*sub_cell_volume;

      // Calculate the volume at the node
      nodal_volumes[(node_c_index)] += sub_cell_volume;
    }
  }

#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    nodal_soundspeed[(nn)] /= nodal_volumes[(nn)];
  }

  // Calculate the force contributions for pressure gradients
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 

      // Determine the three point stencil of nodes around current node
      const int node_l_index = (nn == 0) 
        ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
        : cells_to_nodes[(nodes_off)+(nn-1)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)]
        : cells_to_nodes[(nodes_off)+(nn+1)];

      // Calculate the area vectors away from cell through node, using
      // the half edge vectors adjacent to the node combined
      const double S_x =
        0.5*((nodes_y1[(node_c_index)]-nodes_y1[(node_l_index)]) +
            (nodes_y1[(node_r_index)]-nodes_y1[(node_c_index)]));
      const double S_y =
        -0.5*((nodes_x1[(node_c_index)]-nodes_x1[(node_l_index)]) +
            (nodes_x1[(node_r_index)]-nodes_x1[(node_c_index)]));

      node_force_x[(node_c_index)] += pressure1[(cc)]*S_x;
      node_force_y[(node_c_index)] += pressure1[(cc)]*S_y;
      cell_force_x[(nodes_off)+(nn)] = pressure1[(cc)]*S_x;
      cell_force_y[(nodes_off)+(nn)] = pressure1[(cc)]*S_y;
    }
  }

  calculate_artificial_viscosity(
      ncells, visc_coeff1, visc_coeff2, cells_to_nodes_off, cells_to_nodes, 
      nodes_x1, nodes_y1, cell_centroids_x, cell_centroids_y,
      velocity_x1, velocity_y1, nodal_soundspeed, nodal_mass,
      nodal_volumes, limiter, node_force_x, node_force_y);

  // Calculate the corrected time centered velocities
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    // Calculate the new velocities
    velocity_x1[(nn)] += mesh->dt*node_force_x[(nn)]/nodal_mass[(nn)];
    velocity_y1[(nn)] += mesh->dt*node_force_y[(nn)]/nodal_mass[(nn)];

    // Calculate the corrected time centered velocities
    velocity_x0[(nn)] = 0.5*(velocity_x1[(nn)] + velocity_x0[(nn)]);
    velocity_y0[(nn)] = 0.5*(velocity_y1[(nn)] + velocity_y0[(nn)]);
  }

  handle_unstructured_reflect(
      nnodes, boundary_index, boundary_type, boundary_normal_x, 
      boundary_normal_y, velocity_x0, velocity_y0);

  // Calculate the corrected node movements
#pragma omp parallel for
  for(int nn = 0; nn < nnodes; ++nn) {
    nodes_x0[(nn)] += mesh->dt*velocity_x0[(nn)];
    nodes_y0[(nn)] += mesh->dt*velocity_y0[(nn)];
  }

  set_timestep(
      ncells, cells_to_nodes, cells_to_nodes_off,
      nodes_x0, nodes_y0, energy1, &mesh->dt);

    // Calculate the final energy
#pragma omp parallel for
    for(int cc = 0; cc < ncells; ++cc) {
      const int nodes_off = cells_to_nodes_off[(cc)];
      const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

      // Sum the time centered velocity by the sub-cell forces
      double force = 0.0;
      for(int nn = 0; nn < nnodes_around_cell; ++nn) {
        const int node_index = cells_to_nodes[(nodes_off)+(nn)];
        force += 
          (velocity_x0[(node_index)]*cell_force_x[(nodes_off)+(nn)] +
           velocity_y0[(node_index)]*cell_force_y[(nodes_off)+(nn)]);
      }

      energy0[(cc)] -= mesh->dt*force/cell_mass[(cc)];
    }

  // Using the new corrected volume, calculate the density
#pragma omp parallel for
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    double cell_volume = 0.0;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      // Calculate the new volume of the cell
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];
      cell_volume += 
        0.5*(nodes_x0[node_c_index]+nodes_x0[node_r_index])*
        (nodes_y0[node_r_index]-nodes_y0[node_c_index]);
    }

    // Update the density using the new volume
    density0[(cc)] = cell_mass[(cc)]/cell_volume;
  }
}

// Calculates the artificial viscous forces for momentum acceleration
void calculate_artificial_viscosity(
    const int ncells, const double visc_coeff1, const double visc_coeff2, 
    const int* cells_to_nodes_off, const int* cells_to_nodes, 
    const double* nodes_x, const double* nodes_y, 
    const double* cell_centroids_x, const double* cell_centroids_y,
    const double* velocity_x, const double* velocity_y,
    const double* nodal_soundspeed, const double* nodal_mass,
    const double* nodal_volumes, const double* limiter,
    double* node_force_x, double* node_force_y)
{
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      const int node_c_off = (nodes_off)+(nn);
      const int node_r_off = (nn == nnodes_around_cell-1) 
        ? (nodes_off) : (nodes_off)+(nn+1);
      const int node_c_index = cells_to_nodes[(node_c_off)]; 
      const int node_r_index = cells_to_nodes[(node_r_off)];

      // Get cell center point and edge center point
      const double cell_x = cell_centroids_x[(cc)];
      const double cell_y = cell_centroids_y[(cc)];
      const double edge_mid_x = 
        0.5*(nodes_x[(node_c_index)] + nodes_x[(node_r_index)]);
      const double edge_mid_y = 
        0.5*(nodes_y[(node_c_index)] + nodes_y[(node_r_index)]);

      // Rotate the vector between cell c and edge midpoint to get normal
      const double S_x = (edge_mid_y-cell_y);
      const double S_y = -(edge_mid_x-cell_x);

      // Velocity gradients
      const double grad_velocity_x = 
        velocity_x[(node_r_index)]-velocity_x[(node_c_index)];
      const double grad_velocity_y = 
        velocity_y[(node_r_index)]-velocity_y[(node_c_index)];
      const double grad_velocity_mag =
        sqrt(grad_velocity_x*grad_velocity_x+grad_velocity_y*grad_velocity_y);
      const double grad_velocity_unit_x = 
        (grad_velocity_x != 0.0) ? grad_velocity_x/grad_velocity_mag : 0.0;
      const double grad_velocity_unit_y = 
        (grad_velocity_y != 0.0) ? grad_velocity_y/grad_velocity_mag : 0.0;

      // Calculate the minimum soundspeed
      const double cs = min(
          nodal_soundspeed[(node_c_index)], nodal_soundspeed[(node_r_index)]);

      // Calculate the edge centered density with a harmonic mean
      double nodal_density_l = nodal_mass[(node_c_index)]/nodal_volumes[(node_c_index)];
      double nodal_density_r = nodal_mass[(node_r_index)]/nodal_volumes[(node_r_index)];
      const double density_edge = 
        (2.0*nodal_density_l*nodal_density_r)/(nodal_density_l+nodal_density_r);

      // Calculate the artificial viscous force term for the edge
      const double t = 0.25*(GAM + 1.0);
      double expansion_term = (grad_velocity_x*S_x + grad_velocity_y*S_y);

      // If the cell is compressing, calculate the edge forces and add their
      // contributions to the node forces
      if(expansion_term <= 0.0) {
        const double edge_visc_force_x = 
          density_edge*(visc_coeff2*t*fabs(grad_velocity_x) + 
              sqrt(visc_coeff2*visc_coeff2*t*t*grad_velocity_x*grad_velocity_x +
                visc_coeff1*visc_coeff1*cs*cs))*
          (1.0 - limiter[(node_c_index)])*(grad_velocity_x*S_x)*grad_velocity_unit_x;
        const double edge_visc_force_y = 
          density_edge*(visc_coeff2*t*fabs(grad_velocity_y) +
              sqrt(visc_coeff2*visc_coeff2*t*t*grad_velocity_y*grad_velocity_y +
                visc_coeff1*visc_coeff1*cs*cs))*
          (1.0 - limiter[(node_c_index)])*(grad_velocity_y*S_y)*grad_velocity_unit_y;

        // Add the contributions of the edge based artifical viscous terms
        // to the main force terms
        node_force_x[(node_c_index)] -= edge_visc_force_x;
        node_force_x[(node_r_index)] += edge_visc_force_x;
        node_force_y[(node_c_index)] -= edge_visc_force_y;
        node_force_y[(node_r_index)] += edge_visc_force_y;
      }
    }
  }
}

// Controls the timestep for the simulation
void set_timestep(
    const int ncells, const int* cells_to_nodes, const int* cells_to_nodes_off,
    const double* nodes_x, const double* nodes_y, const double* energy, double* dt)
{
  // Calculate the timestep based on the computational mesh and CFL condition
  double local_dt = DBL_MAX;
#pragma omp parallel for reduction(min: local_dt)
  for(int cc = 0; cc < ncells; ++cc) {
    const int nodes_off = cells_to_nodes_off[(cc)];
    const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

    double shortest_edge = DBL_MAX;
    for(int nn = 0; nn < nnodes_around_cell; ++nn) {
      // Calculate the new volume of the cell
      const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
      const int node_r_index = (nn == nnodes_around_cell-1) 
        ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];
      const double x_component = nodes_x[(node_c_index)]-nodes_x[(node_r_index)];
      const double y_component = nodes_y[(node_c_index)]-nodes_y[(node_r_index)];
      shortest_edge = min(shortest_edge, 
          sqrt(x_component*x_component+y_component*y_component));
    }

    const double soundspeed = sqrt(GAM*(GAM-1.0)*energy[(cc)]);
    local_dt = min(local_dt, shortest_edge/soundspeed);
  }

  *dt = CFL*local_dt;
  printf("Timestep %.8fs\n", *dt);
}

#if 0
// Calculate the limiter
for(int cc = 0; cc < ncells; ++cc) {

  const int nodes_off = cells_to_nodes_off[(cc)];
  const int nnodes_around_cell = cells_to_nodes_off[(cc+1)]-nodes_off;

  for(int nn = 0; nn < nnodes_around_cell; ++nn) {
    const int node_l_index = (nn == 0) 
      ? cells_to_nodes[(nodes_off+nnodes_around_cell-1)] 
      : cells_to_nodes[(nodes_off)+(nn-1)]; 
    const int node_c_index = cells_to_nodes[(nodes_off)+(nn)]; 
    const int node_r_index = (nn == nnodes_around_cell-1) 
      ? cells_to_nodes[(nodes_off)] : cells_to_nodes[(nodes_off)+(nn+1)];

    const double velocity_mag = 
      sqrt((velocity_x1[(node_c_index)]*velocity_x1[(node_c_index)]+
            velocity_y1[(node_c_index)]*velocity_y1[(node_c_index)]);
          const double nodes_mag = 
          sqrt(nodes_x1[(node_c_index)]*nodes_x1[(node_c_index)]+
            nodes_y1[(node_c_index)]*nodes_y1[(node_c_index)]);
          const double velocity_x1_unit = velocity_x1[(node_c_index)]/velocity_mag;
          const double velocity_y1_unit = velocity_y1[(node_c_index)]/velocity_mag;
          const double nodes_x1_unit = nodes_x1[(node_c_index)]/nodes_mag;
          const double nodes_y1_unit = nodes_y1[(node_c_index)]/nodes_mag;

          const double cond = nodes_mag/velocity_mag;
          const double r_l = 
          ((velocity_x1[(node_c_index)]-velocity_x1[(node_l_index)])*velocity_x1_unit+
           (velocity_y1[(node_l_index)]-velocity_y1[(node_l_index)])*velocity_y1_unit)/
          ((nodes_x1[(node_c_index)]-nodes_x1[(node_l_index)])*nodes_x1_unit+
           (nodes_y1[(node_c_index)]-nodes_y1[(node_l_index)])*nodes_y1_unit)*cond;
          const double r_r = 
          ((velocity_x1[(node_r_index)]-velocity_x1[(node_c_index)])*velocity_x1_unit+
           (velocity_y1[(node_r_index)]-velocity_y1[(node_r_index)])*velocity_y1_unit)/
          ((nodes_x1[(node_r_index)]-nodes_x1[(node_c_index)])*nodes_x1_unit+
           (nodes_y1[(node_r_index)]-nodes_y1[(node_c_index)])*nodes_y1_unit)*cond;

          double limiter_i = min(0.5*(r_l+r_r), 2.0*r_l);
          limiter_i = min(limiter_i, 2.0*r_r);
          limiter_i = min(limiter_i, 1.0);
          limiter[(node_c_index)] = max(0.0, limiter_i);
  }
}
#endif // if 0

