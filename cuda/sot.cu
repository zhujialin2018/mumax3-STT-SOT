// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013, 2016

#include <stdint.h>
#include "float3.h"
#include "constants.h"
#include "amul.h"

extern "C" __global__ void
addsotorque(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                      float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                      float* __restrict__ Ms_,        float  Ms_mul,
                      float* __restrict__ jz_,        float  jz_mul,
                      float* __restrict__ px_,        float  px_mul,
                      float* __restrict__ py_,        float  py_mul,
                      float* __restrict__ pz_,        float  pz_mul,
                      float* __restrict__ alpha_,     float  alpha_mul,
                      float* __restrict__ spinhall_,  float  spinhall_mul,
                      float* __restrict__ hfloverhdl_,float  hfloverhdl_mul,
                      float* __restrict__ thickness_, float  thickness_mul,
                      int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m = make_float3(mx[i], my[i], mz[i]);
        float  J = amul(jz_, jz_mul, i);
        float3 p = normalized(vmul(px_, py_, pz_, px_mul, py_mul, pz_mul, i));
        float  Ms           = amul(Ms_, Ms_mul, i);
        float  alpha        = amul(alpha_, alpha_mul, i);
        float  spinhall     = amul(spinhall_, spinhall_mul, i);
        float  hfloverhdl   = amul(hfloverhdl_, hfloverhdl_mul, i);

        float thickness = amul(thickness_, thickness_mul, i);

        if (J == 0.0f || Ms == 0.0f) {
            return;
        }

        float beta    = (HBAR / QE) * (spinhall * J / (2.0 * thickness * Ms) );

        float B = beta * hfloverhdl;

        float gilb     = 1.0f / (1.0f + alpha * alpha);
        float mxpxmFac = gilb * (beta + alpha * B);
        float pxmFac   = gilb * (B - alpha * beta);

        float3 pxm      = cross(p, m);
        float3 mxpxm    = cross(m, pxm);

        tx[i] += mxpxmFac * mxpxm.x + pxmFac * pxm.x;
        ty[i] += mxpxmFac * mxpxm.y + pxmFac * pxm.y;
        tz[i] += mxpxmFac * mxpxm.z + pxmFac * pxm.z;
    }
}

