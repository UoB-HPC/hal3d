#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <stdio.h>

// Repairs the subcell extrema for mass
void repair_subcell_extrema(const int ncells, const int* cells_offsets,
                            const int* subcells_to_subcells_offsets,
                            const int* subcells_to_subcells,
                            double* subcell_volume, double* subcell_mass);

// Repairs the extrema at the nodal velocities
void repair_velocity_extrema(const int nnodes,
                             const int* nodes_to_nodes_offsets,
                             const int* nodes_to_nodes, double* velocity_x,
                             double* velocity_y, double* velocity_z);

// Repairs the subcell extrema for mass
void repair_energy_extrema(const int ncells, const int* cells_to_faces_offsets,
                           const int* cells_to_faces,
                           const int* faces_to_cells0,
                           const int* faces_to_cells1, double* energy);

// Redistributes the mass according to the determined neighbour availability
void redistribute_subcell_mass(double* mass, const int subcell_index,
                               const int nsubcell_neighbours,
                               const int* subcells_to_subcells,
                               const int subcell_to_subcells_off,
                               const double* dmass_avail_neighbour,
                               const double dmass_avail,
                               const double dmass_need, const double g,
                               const double subcell_vol, const int is_min);

// Performs a conservative repair of the mesh
void mass_repair_phase(UnstructuredMesh* umesh, HaleData* hale_data) {

  // Advects mass and energy through the subcell faces using swept edge approx
  repair_subcell_extrema(umesh->ncells, umesh->cells_offsets,
                         hale_data->subcells_to_subcells_offsets,
                         hale_data->subcells_to_subcells,
                         hale_data->subcell_volume, hale_data->subcell_mass);
}

// Repairs the nodal velocities
void velocity_repair_phase(UnstructuredMesh* umesh, HaleData* hale_data) {

  repair_velocity_extrema(umesh->nnodes, umesh->nodes_to_nodes_offsets,
                          umesh->nodes_to_nodes, hale_data->velocity_x0,
                          hale_data->velocity_y0, hale_data->velocity_z0);
}

// Repairs the energy
void energy_repair_phase(UnstructuredMesh* umesh, HaleData* hale_data) {

  repair_energy_extrema(umesh->ncells, umesh->cells_to_faces_offsets,
                        umesh->cells_to_faces, umesh->faces_to_cells0,
                        umesh->faces_to_cells1, hale_data->energy0);
}

// Repairs the subcell extrema for mass
void repair_velocity_extrema(const int nnodes,
                             const int* nodes_to_nodes_offsets,
                             const int* nodes_to_nodes, double* velocity_x,
                             double* velocity_y, double* velocity_z) {

#if 0
#pragma omp parallel for
#endif // if 0
  for (int nn = 0; nn < nnodes; ++nn) {
    const int node_to_nodes_off = nodes_to_nodes_offsets[(nn)];
    const int nnodes_by_node =
        nodes_to_nodes_offsets[(nn + 1)] - node_to_nodes_off;

    double gmax_vx = -DBL_MAX;
    double gmin_vx = DBL_MAX;
    double gmax_vy = -DBL_MAX;
    double gmin_vy = DBL_MAX;
    double gmax_vz = -DBL_MAX;
    double gmin_vz = DBL_MAX;
    double dvx_total_avail_donate = 0.0;
    double dvx_total_avail_receive = 0.0;
    double dvy_total_avail_donate = 0.0;
    double dvy_total_avail_receive = 0.0;
    double dvz_total_avail_donate = 0.0;
    double dvz_total_avail_receive = 0.0;
    double dvx_avail_donate_neighbour[(nnodes_by_node)];
    double dvx_avail_receive_neighbour[(nnodes_by_node)];
    double dvy_avail_donate_neighbour[(nnodes_by_node)];
    double dvy_avail_receive_neighbour[(nnodes_by_node)];
    double dvz_avail_donate_neighbour[(nnodes_by_node)];
    double dvz_avail_receive_neighbour[(nnodes_by_node)];

    // Loop over the nodes attached to this node
    for (int nn2 = 0; nn2 < nnodes_by_node; ++nn2) {
      const int neighbour_index = nodes_to_nodes[(node_to_nodes_off + nn2)];
      if (neighbour_index == -1) {
        continue;
      }

      const int neighbour_to_nodes_off =
          nodes_to_nodes_offsets[(neighbour_index)];
      const int nnodes_by_neighbour =
          nodes_to_nodes_offsets[(neighbour_index + 1)] -
          neighbour_to_nodes_off;

      vec_t neighbour_v = {velocity_x[(neighbour_index)],
                           velocity_y[(neighbour_index)],
                           velocity_z[(neighbour_index)]};

      double neighbour_gmax_vx = -DBL_MAX;
      double neighbour_gmin_vx = DBL_MAX;
      double neighbour_gmax_vy = -DBL_MAX;
      double neighbour_gmin_vy = DBL_MAX;
      double neighbour_gmax_vz = -DBL_MAX;
      double neighbour_gmin_vz = DBL_MAX;

      for (int nn3 = 0; nn3 < nnodes_by_neighbour; ++nn3) {
        const int neighbour_neighbour_index =
            nodes_to_nodes[(neighbour_to_nodes_off + nn3)];
        if (neighbour_neighbour_index == -1) {
          continue;
        }

        neighbour_gmax_vx =
            max(neighbour_gmax_vx, velocity_x[(neighbour_neighbour_index)]);
        neighbour_gmin_vx =
            min(neighbour_gmin_vx, velocity_x[(neighbour_neighbour_index)]);
        neighbour_gmax_vy =
            max(neighbour_gmax_vy, velocity_y[(neighbour_neighbour_index)]);
        neighbour_gmin_vy =
            min(neighbour_gmin_vy, velocity_y[(neighbour_neighbour_index)]);
        neighbour_gmax_vz =
            max(neighbour_gmax_vz, velocity_z[(neighbour_neighbour_index)]);
        neighbour_gmin_vz =
            min(neighbour_gmin_vz, velocity_z[(neighbour_neighbour_index)]);
      }

      dvx_avail_donate_neighbour[(nn2)] =
          max(neighbour_v.x - neighbour_gmin_vx, 0.0);
      dvx_avail_receive_neighbour[(nn2)] =
          max(neighbour_gmax_vx - neighbour_v.x, 0.0);
      dvy_avail_donate_neighbour[(nn2)] =
          max(neighbour_v.y - neighbour_gmin_vy, 0.0);
      dvy_avail_receive_neighbour[(nn2)] =
          max(neighbour_gmax_vy - neighbour_v.y, 0.0);
      dvz_avail_donate_neighbour[(nn2)] =
          max(neighbour_v.z - neighbour_gmin_vz, 0.0);
      dvz_avail_receive_neighbour[(nn2)] =
          max(neighbour_gmax_vz - neighbour_v.z, 0.0);

      dvx_total_avail_donate += dvx_avail_donate_neighbour[(nn2)];
      dvx_total_avail_receive += dvx_avail_receive_neighbour[(nn2)];
      dvy_total_avail_donate += dvy_avail_donate_neighbour[(nn2)];
      dvy_total_avail_receive += dvy_avail_receive_neighbour[(nn2)];
      dvz_total_avail_donate += dvz_avail_donate_neighbour[(nn2)];
      dvz_total_avail_receive += dvz_avail_receive_neighbour[(nn2)];

      gmax_vx = max(gmax_vx, neighbour_v.x);
      gmin_vx = min(gmin_vx, neighbour_v.x);
      gmax_vy = max(gmax_vy, neighbour_v.y);
      gmin_vy = min(gmin_vy, neighbour_v.y);
      gmax_vz = max(gmax_vz, neighbour_v.z);
      gmin_vz = min(gmin_vz, neighbour_v.z);
    }

    vec_t cell_v = {velocity_x[(nn)], velocity_y[(nn)], velocity_z[(nn)]};
    const double dvx_need_receive = gmin_vx - cell_v.x;
    const double dvx_need_donate = cell_v.x - gmax_vx;
    const double dvy_need_receive = gmin_vy - cell_v.y;
    const double dvy_need_donate = cell_v.y - gmax_vy;
    const double dvz_need_receive = gmin_vz - cell_v.z;
    const double dvz_need_donate = cell_v.z - gmax_vz;

    if (dvx_need_receive > 0.0) {
      velocity_x[(nn)] = gmin_vx;

      // Loop over the nodes attached to this node
      for (int nn2 = 0; nn2 < nnodes_by_node; ++nn2) {
        const int neighbour_index = nodes_to_nodes[(node_to_nodes_off + nn2)];
        if (neighbour_index == -1) {
          continue;
        }
        velocity_x[(neighbour_index)] -=
            (dvx_avail_donate_neighbour[(nn2)] / dvx_total_avail_donate) *
            dvx_need_receive;
      }
    } else if (dvx_need_donate > 0.0) {
      // Loop over the nodes attached to this node
      velocity_x[(nn)] = gmax_vx;
      for (int nn2 = 0; nn2 < nnodes_by_node; ++nn2) {
        const int neighbour_index = nodes_to_nodes[(node_to_nodes_off + nn2)];
        if (neighbour_index == -1) {
          continue;
        }
        velocity_x[(neighbour_index)] +=
            (dvx_avail_receive_neighbour[(nn2)] / dvx_total_avail_receive) *
            dvx_need_donate;
      }
    }

    if (dvy_need_receive > 0.0) {
      velocity_y[(nn)] = gmin_vy;

      // Loop over the nodes attached to this node
      for (int nn2 = 0; nn2 < nnodes_by_node; ++nn2) {
        const int neighbour_index = nodes_to_nodes[(node_to_nodes_off + nn2)];
        if (neighbour_index == -1) {
          continue;
        }
        velocity_y[(neighbour_index)] -=
            (dvy_avail_donate_neighbour[(nn2)] / dvy_total_avail_donate) *
            dvy_need_receive;
      }
    } else if (dvy_need_donate > 0.0) {
      // Loop over the nodes attached to this node
      velocity_y[(nn)] = gmax_vy;
      for (int nn2 = 0; nn2 < nnodes_by_node; ++nn2) {
        const int neighbour_index = nodes_to_nodes[(node_to_nodes_off + nn2)];
        if (neighbour_index == -1) {
          continue;
        }
        velocity_y[(neighbour_index)] +=
            (dvy_avail_receive_neighbour[(nn2)] / dvy_total_avail_receive) *
            dvy_need_donate;
      }
    }

    if (dvz_need_receive > 0.0) {
      velocity_z[(nn)] = gmin_vz;

      // Loop over the nodes attached to this node
      for (int nn2 = 0; nn2 < nnodes_by_node; ++nn2) {
        const int neighbour_index = nodes_to_nodes[(node_to_nodes_off + nn2)];
        if (neighbour_index == -1) {
          continue;
        }
        velocity_z[(neighbour_index)] -=
            (dvz_avail_donate_neighbour[(nn2)] / dvz_total_avail_donate) *
            dvz_need_receive;
      }
    } else if (dvz_need_donate > 0.0) {
      // Loop over the nodes attached to this node
      velocity_z[(nn)] = gmax_vz;
      for (int nn2 = 0; nn2 < nnodes_by_node; ++nn2) {
        const int neighbour_index = nodes_to_nodes[(node_to_nodes_off + nn2)];
        if (neighbour_index == -1) {
          continue;
        }
        velocity_z[(neighbour_index)] +=
            (dvz_avail_receive_neighbour[(nn2)] / dvz_total_avail_receive) *
            dvz_need_donate;
      }
    }

    if (dvx_total_avail_donate < dvx_need_receive ||
        dvx_total_avail_receive < dvx_need_donate ||
        dvy_total_avail_donate < dvy_need_receive ||
        dvy_total_avail_receive < dvy_need_donate ||
        dvz_total_avail_donate < dvz_need_receive ||
        dvz_total_avail_receive < dvz_need_donate) {
      printf("Repair stage needs additional level.\n");
      continue;
    }
  }
}

// Repairs the subcell extrema for mass
void repair_energy_extrema(const int ncells, const int* cells_to_faces_offsets,
                           const int* cells_to_faces,
                           const int* faces_to_cells0,
                           const int* faces_to_cells1, double* energy) {

#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_faces_off = cells_to_faces_offsets[(cc)];
    const int nfaces_by_cell =
        cells_to_faces_offsets[(cc + 1)] - cell_to_faces_off;

    double gmax_ie = -DBL_MAX;
    double gmin_ie = DBL_MAX;
    double die_total_avail_donate = 0.0;
    double die_total_avail_receive = 0.0;
    double die_avail_donate_neighbour[(nfaces_by_cell)];
    double die_avail_receive_neighbour[(nfaces_by_cell)];

    const double cell_ie = energy[(cc)];

    // Loop over the nodes attached to this node
    for (int ff = 0; ff < nfaces_by_cell; ++ff) {
      const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
      const int neighbour_index = (faces_to_cells0[(face_index)] == cc)
                                      ? faces_to_cells1[(face_index)]
                                      : faces_to_cells0[(face_index)];
      if (neighbour_index == -1) {
        continue;
      }

      const double neighbour_ie = energy[(neighbour_index)];

      double neighbour_gmax_ie = -DBL_MAX;
      double neighbour_gmin_ie = DBL_MAX;

      const int neighbour_to_faces_off =
          cells_to_faces_offsets[(neighbour_index)];
      const int nfaces_by_neighbour =
          cells_to_faces_offsets[(neighbour_index + 1)] -
          neighbour_to_faces_off;

      for (int ff2 = 0; ff2 < nfaces_by_neighbour; ++ff2) {
        const int neighbour_face_index =
            cells_to_faces[(neighbour_to_faces_off + ff2)];
        const int neighbour_neighbour_index =
            (faces_to_cells0[(neighbour_face_index)] == neighbour_index)
                ? faces_to_cells1[(neighbour_face_index)]
                : faces_to_cells0[(neighbour_face_index)];

        if (neighbour_neighbour_index == -1) {
          continue;
        }

        neighbour_gmax_ie =
            max(neighbour_gmax_ie, energy[(neighbour_neighbour_index)]);
        neighbour_gmin_ie =
            min(neighbour_gmin_ie, energy[(neighbour_neighbour_index)]);
      }

      die_avail_donate_neighbour[(ff)] =
          max(neighbour_ie - neighbour_gmin_ie, 0.0);
      die_avail_receive_neighbour[(ff)] =
          max(neighbour_gmax_ie - neighbour_ie, 0.0);

      die_total_avail_donate += die_avail_donate_neighbour[(ff)];
      die_total_avail_receive += die_avail_receive_neighbour[(ff)];

      gmax_ie = max(gmax_ie, neighbour_ie);
      gmin_ie = min(gmin_ie, neighbour_ie);
    }

    const double die_need_receive = gmin_ie - cell_ie;
    const double die_need_donate = cell_ie - gmax_ie;

    if (die_need_receive > 0.0) {
      energy[(cc)] = gmin_ie;

      for (int ff = 0; ff < nfaces_by_cell; ++ff) {
        const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
        const int neighbour_index = (faces_to_cells0[(face_index)] == cc)
                                        ? faces_to_cells1[(face_index)]
                                        : faces_to_cells0[(face_index)];
        if (neighbour_index == -1) {
          continue;
        }

        energy[(neighbour_index)] -=
            (die_avail_donate_neighbour[(ff)] / die_total_avail_donate) *
            die_need_receive;
      }
    } else if (die_need_donate > 0.0) {
      // Loop over the nodes attached to this node
      energy[(cc)] = gmax_ie;
      for (int ff = 0; ff < nfaces_by_cell; ++ff) {
        const int face_index = cells_to_faces[(cell_to_faces_off + ff)];
        const int neighbour_index = (faces_to_cells0[(face_index)] == cc)
                                        ? faces_to_cells1[(face_index)]
                                        : faces_to_cells0[(face_index)];
        if (neighbour_index == -1) {
          continue;
        }
        energy[(neighbour_index)] +=
            (die_avail_receive_neighbour[(ff)] / die_total_avail_receive) *
            die_need_donate;
      }
    }

    if (die_total_avail_donate < die_need_receive ||
        die_total_avail_receive < die_need_donate) {
      printf("Repair stage needs additional level.\n");
      continue;
    }
  }
}

// Repairs the subcell extrema for mass
void repair_subcell_extrema(const int ncells, const int* cells_offsets,
                            const int* subcells_to_subcells_offsets,
                            const int* subcells_to_subcells,
                            double* subcell_volume, double* subcell_mass) {

#if 0
#pragma omp parallel for
#endif // if 0
  for (int cc = 0; cc < ncells; ++cc) {
    const int cell_to_nodes_off = cells_offsets[(cc)];
    const int nnodes_by_cell = cells_offsets[(cc + 1)] - cell_to_nodes_off;

    // Looping over corner subcells here
    for (int nn = 0; nn < nnodes_by_cell; ++nn) {
      const int subcell_index = cell_to_nodes_off + nn;
      const int subcell_to_subcells_off =
          subcells_to_subcells_offsets[(subcell_index)];
      const int nsubcell_neighbours =
          subcells_to_subcells_offsets[(subcell_index + 1)] -
          subcell_to_subcells_off;

      const double subcell_vol = subcell_volume[(subcell_index)];
      const double subcell_m_density =
          subcell_mass[(subcell_index)] / subcell_vol;

      double gmax_m = -DBL_MAX;
      double gmin_m = DBL_MAX;
      double dm_avail_donate = 0.0;
      double dm_avail_receive = 0.0;
      double dm_avail_donate_neighbour[(nsubcell_neighbours)];
      double dm_avail_receive_neighbour[(nsubcell_neighbours)];

      // Loop over neighbours
      for (int ss = 0; ss < nsubcell_neighbours; ++ss) {
        const int neighbour_index =
            subcells_to_subcells[(subcell_to_subcells_off + ss)];

        // Ignore boundary neighbours
        if (neighbour_index == -1) {
          continue;
        }

        const int neighbour_to_subcells_off =
            subcells_to_subcells_offsets[(neighbour_index)];
        const int nneighbour_neighbours =
            subcells_to_subcells_offsets[(neighbour_index + 1)] -
            neighbour_to_subcells_off;

        const double neighbour_vol = subcell_volume[(neighbour_index)];
        const double neighbour_m_density =
            subcell_mass[(neighbour_index)] / neighbour_vol;

        double neighbour_gmax_m = -DBL_MAX;
        double neighbour_gmin_m = DBL_MAX;

        // Loop over neighbour's neighbours
        for (int ss2 = 0; ss2 < nneighbour_neighbours; ++ss2) {
          const int neighbour_neighbour_index =
              subcells_to_subcells[(neighbour_to_subcells_off + ss2)];

          // Ignore boundary neighbours
          if (neighbour_neighbour_index == -1) {
            continue;
          }

          const double neighbour_neighbour_vol =
              subcell_volume[(neighbour_neighbour_index)];
          const double neighbour_neighbour_m_density =
              subcell_mass[(neighbour_neighbour_index)] /
              neighbour_neighbour_vol;

          // Store the maximum / minimum values for rho in the neighbourhood
          neighbour_gmax_m =
              max(neighbour_gmax_m, neighbour_neighbour_m_density);
          neighbour_gmin_m =
              min(neighbour_gmin_m, neighbour_neighbour_m_density);
        }

        dm_avail_donate_neighbour[(ss)] =
            max((neighbour_m_density - neighbour_gmin_m) * subcell_vol, 0.0);
        dm_avail_receive_neighbour[(ss)] =
            max((neighbour_gmax_m - neighbour_m_density) * subcell_vol, 0.0);

        dm_avail_donate += dm_avail_donate_neighbour[(ss)];
        dm_avail_receive += dm_avail_receive_neighbour[(ss)];

        gmax_m = max(gmax_m, neighbour_m_density);
        gmin_m = min(gmin_m, neighbour_m_density);
      }

      const double dm_need_receive = (gmin_m - subcell_m_density) * subcell_vol;
      const double dm_need_donate = (subcell_m_density - gmax_m) * subcell_vol;

      if (dm_need_receive > 0.0) {
        redistribute_subcell_mass(subcell_mass, subcell_index,
                                  nsubcell_neighbours, subcells_to_subcells,
                                  subcell_to_subcells_off,
                                  dm_avail_donate_neighbour, dm_avail_donate,
                                  dm_need_receive, gmin_m, subcell_vol, 1);

      } else if (dm_need_donate > 0.0) {
        redistribute_subcell_mass(subcell_mass, subcell_index,
                                  nsubcell_neighbours, subcells_to_subcells,
                                  subcell_to_subcells_off,
                                  dm_avail_receive_neighbour, dm_avail_receive,
                                  dm_need_donate, gmax_m, subcell_vol, 0);
      }

      if (dm_avail_donate < dm_need_receive ||
          dm_avail_receive < dm_need_donate) {
        printf("dm_avail_donate %.12e dm_need_receive %.12e dm_avail_receive "
               "%.12e dm_need_donate %.12e\n",
               dm_avail_donate, dm_need_receive, dm_avail_receive,
               dm_need_donate);
        printf("Repair stage needs additional level.\n");
        continue;
      }
    }
  }
}

// Redistributes the mass according to the determined neighbour availability
void redistribute_subcell_mass(double* mass, const int subcell_index,
                               const int nsubcell_neighbours,
                               const int* subcells_to_subcells,
                               const int subcell_to_subcells_off,
                               const double* dmass_avail_neighbour,
                               const double dmass_avail,
                               const double dmass_need, const double g,
                               const double subcell_vol, const int is_min) {

  mass[(subcell_index)] = g * subcell_vol;

  // Loop over neighbours
  for (int ss = 0; ss < nsubcell_neighbours; ++ss) {
    const int neighbour_index =
        subcells_to_subcells[(subcell_to_subcells_off + ss)];
    mass[(neighbour_index)] += (is_min ? -1.0 : 1.0) *
                               (dmass_avail_neighbour[(ss)] / dmass_avail) *
                               dmass_need;
  }
}
