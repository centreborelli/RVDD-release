// Compile with cmake (CMakeLists.txt is provided) or with the following lines in bash:
// g++ -c -fPIC libautosim.cpp -o libautosim.o
// g++ -shared -Wl,-soname,libautosim.so -o libautosim.so libautosim.o


#include <string>
#include <sstream>
#include <iostream>
#include <list>
#include <vector>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <float.h>
#include <limits>
#include <cmath>
#include <limits>


#ifndef DISABLE_OMP
#include <omp.h>
#endif //DISABLE_OMP

#define PAR_DEFAULT_NPROC 0
#define PAR_DEFAULT_TAU 0.25
#define PAR_DEFAULT_LAMBDA 0.15
#define PAR_DEFAULT_THETA 0.3
#define PAR_DEFAULT_NSCALES 100
#define PAR_DEFAULT_FSCALE 0
#define PAR_DEFAULT_ZFACTOR 0.5
#define PAR_DEFAULT_NWARPS 5
#define PAR_DEFAULT_EPSILON 0.01
#define PAR_DEFAULT_VERBOSE 0

// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{

#include "tvl1flow_lib.h"

  void tvl1flow(float *I0, float *I1, float *u, int nx, int ny)
  {

    int fauxval = -1;
    int nproc = (fauxval > 0) ? fauxval : PAR_DEFAULT_NPROC;
    float tau = (fauxval > 0) ? fauxval : PAR_DEFAULT_TAU;
    float lambda = (fauxval > 0) ? fauxval : PAR_DEFAULT_LAMBDA;
    float theta = (fauxval > 0) ? fauxval : PAR_DEFAULT_THETA;
    int nscales = (fauxval > 0) ? fauxval : PAR_DEFAULT_NSCALES;
    int fscale = (fauxval > 0) ? fauxval : PAR_DEFAULT_FSCALE;
    float zfactor = (fauxval > 0) ? fauxval : PAR_DEFAULT_ZFACTOR;
    int nwarps = (fauxval > 0) ? fauxval : PAR_DEFAULT_NWARPS;
    float epsilon = (fauxval > 0) ? fauxval : PAR_DEFAULT_EPSILON;
    int verbose = (fauxval > 0) ? fauxval : PAR_DEFAULT_VERBOSE;
    
    //check parameters
    if (nproc < 0)
    {
      nproc = PAR_DEFAULT_NPROC;
      if (verbose)
        fprintf(stderr, "warning: "
                        "nproc changed to %d\n",
                nproc);
    }
    if (tau <= 0 || tau > 0.25)
    {
      tau = PAR_DEFAULT_TAU;
      if (verbose)
        fprintf(stderr, "warning: "
                        "tau changed to %g\n",
                tau);
    }
    if (lambda <= 0)
    {
      lambda = PAR_DEFAULT_LAMBDA;
      if (verbose)
        fprintf(stderr, "warning: "
                        "lambda changed to %g\n",
                lambda);
    }
    if (theta <= 0)
    {
      theta = PAR_DEFAULT_THETA;
      if (verbose)
        fprintf(stderr, "warning: "
                        "theta changed to %g\n",
                theta);
    }
    if (nscales <= 0)
    {
      nscales = PAR_DEFAULT_NSCALES;
      if (verbose)
        fprintf(stderr, "warning: "
                        "nscales changed to %d\n",
                nscales);
    }
    if (zfactor <= 0 || zfactor >= 1)
    {
      zfactor = PAR_DEFAULT_ZFACTOR;
      if (verbose)
        fprintf(stderr, "warning: "
                        "zfactor changed to %g\n",
                zfactor);
    }
    if (nwarps <= 0)
    {
      nwarps = PAR_DEFAULT_NWARPS;
      if (verbose)
        fprintf(stderr, "warning: "
                        "nwarps changed to %d\n",
                nwarps);
    }
    if (epsilon <= 0)
    {
      epsilon = PAR_DEFAULT_EPSILON;
      if (verbose)
        fprintf(stderr, "warning: "
                        "epsilon changed to %f\n",
                epsilon);
    }

#ifndef DISABLE_OMP
    if (nproc > 0)
      omp_set_num_threads(nproc);
#endif //DISABLE_OMP


    //Set the number of scales according to the size of the
    //images.  The value N is computed to assure that the smaller
    //images of the pyramid don't have a size smaller than 16x16
    const float N = 1 + log(hypot(nx, ny) / 16.0) / log(1 / zfactor);
    if (N < nscales)
      nscales = N;
    if (nscales < fscale)
      fscale = nscales;

    if (verbose)
      fprintf(stderr,
              "nproc=%d tau=%f lambda=%f theta=%f nscales=%d "
              "zfactor=%f nwarps=%d epsilon=%g\n",
              nproc, tau, lambda, theta, nscales,
              zfactor, nwarps, epsilon);

    // ----- u is allocated by python
    // //allocate memory for the flow
    // u = xmalloc((size_t)(2 * nx * ny * sizeof *u));
    float *v = u + nx * ny;

    //compute the optical flow
    Dual_TVL1_optic_flow_multiscale(
        I0, I1, u, v, nx, ny, tau, lambda, theta,
        nscales, fscale, zfactor, nwarps, epsilon, verbose);

    
    // ---- The memory will be controlled by python ----
    // //delete allocated memory
    // free(I0);
    // free(I1);
    // free(u);
  }

}



