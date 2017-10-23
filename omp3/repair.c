#include "../../shared.h"
#include "hale.h"
#include <float.h>
#include <stdio.h>

// Advects mass and energy through the subcell faces using swept edge approx
void repair_extrema(const int ncells, const int* cells_offsets,
                    const int* subcells_to_subcells_offsets,
                    const int* subcells_to_subcells, double* subcell_volume,
                    double* subcell_momentum_x, double* subcell_momentum_y,
                    double* subcell_momentum_z, double* subcell_mass,
                    double* subcell_ie_mass);

// Redistributes the mass according to the determined neighbour availability
void redistribute_mass(double* mass, const int subcell_index,
                       const int nsubcell_neighbours,
                       const int* subcells_to_subcells,
                       const int subcell_to_subcells_off,
                       const double* dmass_avail_local,
                       const double dmass_avail, const double dmass_need,
                       const double g, const double vol, const int is_min);

// Performs a conservative repair of the mesh
void repair_phase(UnstructuredMesh* umesh, HaleData* hale_data) {

  // Advects mass and energy through the subcell faces using swept edge approx
  repair_extrema(umesh->ncells, umesh->cells_offsets,
                 hale_data->subcells_to_subcells_offsets,
                 hale_data->subcells_to_subcells, hale_data->subcell_volume,
                 hale_data->subcell_momentum_x, hale_data->subcell_momentum_y,
                 hale_data->subcell_momentum_z, hale_data->subcell_mass,
                 hale_data->subcell_ie_mass);
}

// Advects mass and energy through the subcell faces using swept edge approx
void repair_extrema(const int ncells, const int* cells_offsets,
                    const int* subcells_to_subcells_offsets,
                    const int* subcells_to_subcells, double* subcell_volume,
                    double* subcell_momentum_x, double* subcell_momentum_y,
                    double* subcell_momentum_z, double* subcell_mass,
                    double* subcell_ie_mass) {

#pragma omp parallel for
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

      const double vol = subcell_volume[(subcell_index)];
      const double subcell_m_density = subcell_mass[(subcell_index)] / vol;
      const double subcell_ie_density = subcell_ie_mass[(subcell_index)] / vol;
      const double subcell_vx_density =
          subcell_momentum_x[(subcell_index)] / vol;
      const double subcell_vy_density =
          subcell_momentum_y[(subcell_index)] / vol;
      const double subcell_vz_density =
          subcell_momentum_z[(subcell_index)] / vol;

      double gmax_m = -DBL_MAX;
      double gmin_m = DBL_MAX;
      double gmax_ie = -DBL_MAX;
      double gmin_ie = DBL_MAX;
      double gmin_vx = DBL_MAX;
      double gmax_vx = -DBL_MAX;
      double gmin_vy = DBL_MAX;
      double gmax_vy = -DBL_MAX;
      double gmin_vz = DBL_MAX;
      double gmax_vz = -DBL_MAX;
      double dm_avail_donate = 0.0;
      double dm_avail_receive = 0.0;
      double die_avail_donate = 0.0;
      double die_avail_receive = 0.0;
      double dvx_avail_receive = 0.0;
      double dvx_avail_donate = 0.0;
      double dvy_avail_receive = 0.0;
      double dvy_avail_donate = 0.0;
      double dvz_avail_receive = 0.0;
      double dvz_avail_donate = 0.0;
      double dm_avail_donate_local[(nsubcell_neighbours)];
      double dm_avail_receive_local[(nsubcell_neighbours)];
      double die_avail_donate_local[(nsubcell_neighbours)];
      double die_avail_receive_local[(nsubcell_neighbours)];
      double dvx_avail_receive_local[(nsubcell_neighbours)];
      double dvx_avail_donate_local[(nsubcell_neighbours)];
      double dvy_avail_receive_local[(nsubcell_neighbours)];
      double dvy_avail_donate_local[(nsubcell_neighbours)];
      double dvz_avail_receive_local[(nsubcell_neighbours)];
      double dvz_avail_donate_local[(nsubcell_neighbours)];

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
        const double neighbour_ie_density =
            subcell_ie_mass[(neighbour_index)] / neighbour_vol;
        const double neighbour_vx_density =
            subcell_momentum_x[(neighbour_index)] / neighbour_vol;
        const double neighbour_vy_density =
            subcell_momentum_y[(neighbour_index)] / neighbour_vol;
        const double neighbour_vz_density =
            subcell_momentum_z[(neighbour_index)] / neighbour_vol;

        double neighbour_gmax_m = -DBL_MAX;
        double neighbour_gmin_m = DBL_MAX;
        double neighbour_gmax_ie = -DBL_MAX;
        double neighbour_gmin_ie = DBL_MAX;
        double neighbour_gmin_vx = DBL_MAX;
        double neighbour_gmax_vx = -DBL_MAX;
        double neighbour_gmin_vy = DBL_MAX;
        double neighbour_gmax_vy = -DBL_MAX;
        double neighbour_gmin_vz = DBL_MAX;
        double neighbour_gmax_vz = -DBL_MAX;

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
          const double neighbour_neighbour_ie_density =
              subcell_ie_mass[(neighbour_neighbour_index)] /
              neighbour_neighbour_vol;
          const double neighbour_neighbour_vx_density =
              subcell_momentum_x[(neighbour_neighbour_index)] /
              neighbour_neighbour_vol;
          const double neighbour_neighbour_vy_density =
              subcell_momentum_y[(neighbour_neighbour_index)] /
              neighbour_neighbour_vol;
          const double neighbour_neighbour_vz_density =
              subcell_momentum_z[(neighbour_neighbour_index)] /
              neighbour_neighbour_vol;

          // Store the maximum / minimum values for rho in the neighbourhood
          neighbour_gmax_m =
              max(neighbour_gmax_m, neighbour_neighbour_m_density);
          neighbour_gmin_m =
              min(neighbour_gmin_m, neighbour_neighbour_m_density);
          neighbour_gmax_ie =
              max(neighbour_gmax_ie, neighbour_neighbour_ie_density);
          neighbour_gmin_ie =
              min(neighbour_gmin_ie, neighbour_neighbour_ie_density);
          neighbour_gmax_vx =
              max(neighbour_gmax_vx, neighbour_neighbour_vx_density);
          neighbour_gmin_vx =
              min(neighbour_gmin_vx, neighbour_neighbour_vx_density);
          neighbour_gmax_vy =
              max(neighbour_gmax_vy, neighbour_neighbour_vy_density);
          neighbour_gmin_vy =
              min(neighbour_gmin_vy, neighbour_neighbour_vy_density);
          neighbour_gmax_vz =
              max(neighbour_gmax_vz, neighbour_neighbour_vz_density);
          neighbour_gmin_vz =
              min(neighbour_gmin_vz, neighbour_neighbour_vz_density);
        }

        dm_avail_donate_local[(ss)] =
            max((neighbour_m_density - neighbour_gmin_m) * vol, 0.0);
        dm_avail_receive_local[(ss)] =
            max((neighbour_gmax_m - neighbour_m_density) * vol, 0.0);
        die_avail_donate_local[(ss)] =
            max((neighbour_ie_density - neighbour_gmin_ie) * vol, 0.0);
        die_avail_receive_local[(ss)] =
            max((neighbour_gmax_ie - neighbour_ie_density) * vol, 0.0);
        dvx_avail_donate_local[(ss)] =
            max((neighbour_vx_density - neighbour_gmin_vx) * vol, 0.0);
        dvx_avail_receive_local[(ss)] =
            max((neighbour_gmax_vx - neighbour_vx_density) * vol, 0.0);
        dvy_avail_donate_local[(ss)] =
            max((neighbour_vy_density - neighbour_gmin_vy) * vol, 0.0);
        dvy_avail_receive_local[(ss)] =
            max((neighbour_gmax_vy - neighbour_vy_density) * vol, 0.0);
        dvz_avail_donate_local[(ss)] =
            max((neighbour_vz_density - neighbour_gmin_vz) * vol, 0.0);
        dvz_avail_receive_local[(ss)] =
            max((neighbour_gmax_vz - neighbour_vz_density) * vol, 0.0);

        dm_avail_donate += dm_avail_donate_local[(ss)];
        dm_avail_receive += dm_avail_receive_local[(ss)];
        die_avail_donate += die_avail_donate_local[(ss)];
        die_avail_receive += die_avail_receive_local[(ss)];

        dvx_avail_donate += dvx_avail_donate_local[(ss)];
        dvx_avail_receive += dvx_avail_receive_local[(ss)];
        dvy_avail_donate += dvy_avail_donate_local[(ss)];
        dvy_avail_receive += dvy_avail_receive_local[(ss)];
        dvz_avail_donate += dvz_avail_donate_local[(ss)];
        dvz_avail_receive += dvz_avail_receive_local[(ss)];

        gmax_m = max(gmax_m, neighbour_m_density);
        gmin_m = min(gmin_m, neighbour_m_density);
        gmax_ie = max(gmax_ie, neighbour_ie_density);
        gmin_ie = min(gmin_ie, neighbour_ie_density);
        gmax_vx = max(gmax_vx, neighbour_vx_density);
        gmin_vx = min(gmin_vx, neighbour_vx_density);
        gmax_vy = max(gmax_vy, neighbour_vy_density);
        gmin_vy = min(gmin_vy, neighbour_vy_density);
        gmax_vz = max(gmax_vz, neighbour_vz_density);
        gmin_vz = min(gmin_vz, neighbour_vz_density);
      }

      const double dm_need_receive = (gmin_m - subcell_m_density) * vol;
      const double dm_need_donate = (subcell_m_density - gmax_m) * vol;
      const double die_need_receive = (gmin_ie - subcell_ie_density) * vol;
      const double die_need_donate = (subcell_ie_density - gmax_ie) * vol;
      const double dvx_need_receive = (gmin_vx - subcell_vx_density) * vol;
      const double dvx_need_donate = (subcell_vx_density - gmax_vx) * vol;
      const double dvy_need_receive = (gmin_vy - subcell_vy_density) * vol;
      const double dvy_need_donate = (subcell_vy_density - gmax_vy) * vol;
      const double dvz_need_receive = (gmin_vz - subcell_vz_density) * vol;
      const double dvz_need_donate = (subcell_vz_density - gmax_vz) * vol;

      if (dm_need_receive > 0.0) {
        redistribute_mass(subcell_mass, subcell_index, nsubcell_neighbours,
                          subcells_to_subcells, subcell_to_subcells_off,
                          dm_avail_donate_local, dm_avail_donate,
                          dm_need_receive, gmin_m, vol, 1);

      } else if (dm_need_donate > 0.0) {
        redistribute_mass(subcell_mass, subcell_index, nsubcell_neighbours,
                          subcells_to_subcells, subcell_to_subcells_off,
                          dm_avail_receive_local, dm_avail_receive,
                          dm_need_donate, gmax_m, vol, 0);
      }

      if (die_need_receive > 0.0) {
        redistribute_mass(subcell_ie_mass, subcell_index, nsubcell_neighbours,
                          subcells_to_subcells, subcell_to_subcells_off,
                          die_avail_donate_local, die_avail_donate,
                          die_need_receive, gmin_ie, vol, 1);

      } else if (die_need_donate > 0.0) {
        redistribute_mass(subcell_ie_mass, subcell_index, nsubcell_neighbours,
                          subcells_to_subcells, subcell_to_subcells_off,
                          die_avail_receive_local, die_avail_receive,
                          die_need_donate, gmax_ie, vol, 0);
      }

      if (dvx_need_receive > 0.0) {
        redistribute_mass(subcell_momentum_x, subcell_index,
                          nsubcell_neighbours, subcells_to_subcells,
                          subcell_to_subcells_off, dvx_avail_donate_local,
                          dvx_avail_donate, dvx_need_receive, gmin_vx, vol, 1);

      } else if (dvx_need_donate > 0.0) {
        redistribute_mass(subcell_momentum_x, subcell_index,
                          nsubcell_neighbours, subcells_to_subcells,
                          subcell_to_subcells_off, dvx_avail_receive_local,
                          dvx_avail_receive, dvx_need_donate, gmax_vx, vol, 0);
      }

      if (dvy_need_receive > 0.0) {
        redistribute_mass(subcell_momentum_y, subcell_index,
                          nsubcell_neighbours, subcells_to_subcells,
                          subcell_to_subcells_off, dvy_avail_donate_local,
                          dvy_avail_donate, dvy_need_receive, gmin_vy, vol, 1);

      } else if (dvy_need_donate > 0.0) {
        redistribute_mass(subcell_momentum_y, subcell_index,
                          nsubcell_neighbours, subcells_to_subcells,
                          subcell_to_subcells_off, dvy_avail_receive_local,
                          dvy_avail_receive, dvy_need_donate, gmax_vy, vol, 0);
      }

      if (dvz_need_receive > 0.0) {
        redistribute_mass(subcell_momentum_z, subcell_index,
                          nsubcell_neighbours, subcells_to_subcells,
                          subcell_to_subcells_off, dvz_avail_donate_local,
                          dvz_avail_donate, dvz_need_receive, gmin_vz, vol, 1);

      } else if (dvz_need_donate > 0.0) {
        redistribute_mass(subcell_momentum_z, subcell_index,
                          nsubcell_neighbours, subcells_to_subcells,
                          subcell_to_subcells_off, dvz_avail_receive_local,
                          dvz_avail_receive, dvz_need_donate, gmax_vz, vol, 0);
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
void redistribute_mass(double* mass, const int subcell_index,
                       const int nsubcell_neighbours,
                       const int* subcells_to_subcells,
                       const int subcell_to_subcells_off,
                       const double* dmass_avail_local,
                       const double dmass_avail, const double dmass_need,
                       const double g, const double vol, const int is_min) {

  mass[(subcell_index)] = g * vol;

// Loop over neighbours
#pragma omp parallel for
  for (int ss = 0; ss < nsubcell_neighbours; ++ss) {
    const int neighbour_index =
        subcells_to_subcells[(subcell_to_subcells_off + ss)];
    mass[(neighbour_index)] += (is_min ? -1.0 : 1.0) *
                               (dmass_avail_local[(ss)] / dmass_avail) *
                               dmass_need;
  }
}
