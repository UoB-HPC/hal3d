#include "hale_data.h"
#include "../shared.h"

// Initialises the shared_data variables for two dimensional applications
void initialise_hale_data_2d(
    const int local_nx, const int local_ny, HaleData* hale_data)
{
  allocate_data(&hale_data->rho_u, (local_nx+1)*(local_ny+1));
  allocate_data(&hale_data->rho_v, (local_nx+1)*(local_ny+1));
  allocate_data(&hale_data->F_x, (local_nx+1)*(local_ny+1));
  allocate_data(&hale_data->F_y, (local_nx+1)*(local_ny+1));
  allocate_data(&hale_data->uF_x, (local_nx+1)*(local_ny+1));
  allocate_data(&hale_data->uF_y, (local_nx+1)*(local_ny+1));
  allocate_data(&hale_data->vF_x, (local_nx+1)*(local_ny+1));
  allocate_data(&hale_data->vF_y, (local_nx+1)*(local_ny+1));
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



