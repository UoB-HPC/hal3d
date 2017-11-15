#include "../hale_data.h"

// Performs the predictor step of the Lagrangian phase
void predictor(Mesh* mesh, UnstructuredMesh* umesh, HaleData* hale_data);

// Performs the corrector step of the Lagrangian phase
void corrector(Mesh* mesh, UnstructuredMesh* umesh, HaleData* hale_data);

