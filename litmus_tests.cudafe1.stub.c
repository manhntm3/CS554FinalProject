#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "litmus_tests.fatbin.c"
extern void __device_stub__Z9sb_kernelPiS_S_S_iib(int *, int *, int *, int *, int, int, bool);
extern void __device_stub__Z9mp_kernelPiS_S_S_iib(int *, int *, int *, int *, int, int, bool);
extern void __device_stub__Z9lb_kernelPiS_S_S_iib(int *, int *, int *, int *, int, int, bool);
extern void __device_stub__Z11iriw_kernelPiS_S_S_S_S_ii(int *, int *, int *, int *, int *, int *, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z9sb_kernelPiS_S_S_iib(int *__par0, int *__par1, int *__par2, int *__par3, int __par4, int __par5, bool __par6){__cudaLaunchPrologue(7);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 32UL);__cudaSetupArgSimple(__par5, 36UL);__cudaSetupArgSimple(__par6, 40UL);__cudaLaunch(((char *)((void ( *)(int *, int *, int *, int *, int, int, bool))sb_kernel)));}
# 238 "litmus_tests.cu"
void sb_kernel( int *__cuda_0,int *__cuda_1,int *__cuda_2,int *__cuda_3,int __cuda_4,int __cuda_5,bool __cuda_6)
# 240 "litmus_tests.cu"
{__device_stub__Z9sb_kernelPiS_S_S_iib( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 313 "litmus_tests.cu"
}
# 1 "litmus_tests.cudafe1.stub.c"
void __device_stub__Z9mp_kernelPiS_S_S_iib( int *__par0,  int *__par1,  int *__par2,  int *__par3,  int __par4,  int __par5,  bool __par6) {  __cudaLaunchPrologue(7); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(int *, int *, int *, int *, int, int, bool))mp_kernel))); }
# 329 "litmus_tests.cu"
void mp_kernel( int *__cuda_0,int *__cuda_1,int *__cuda_2,int *__cuda_3,int __cuda_4,int __cuda_5,bool __cuda_6)
# 331 "litmus_tests.cu"
{__device_stub__Z9mp_kernelPiS_S_S_iib( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 418 "litmus_tests.cu"
}
# 1 "litmus_tests.cudafe1.stub.c"
void __device_stub__Z9lb_kernelPiS_S_S_iib( int *__par0,  int *__par1,  int *__par2,  int *__par3,  int __par4,  int __par5,  bool __par6) {  __cudaLaunchPrologue(7); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 36UL); __cudaSetupArgSimple(__par6, 40UL); __cudaLaunch(((char *)((void ( *)(int *, int *, int *, int *, int, int, bool))lb_kernel))); }
# 431 "litmus_tests.cu"
void lb_kernel( int *__cuda_0,int *__cuda_1,int *__cuda_2,int *__cuda_3,int __cuda_4,int __cuda_5,bool __cuda_6)
# 433 "litmus_tests.cu"
{__device_stub__Z9lb_kernelPiS_S_S_iib( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6);
# 497 "litmus_tests.cu"
}
# 1 "litmus_tests.cudafe1.stub.c"
void __device_stub__Z11iriw_kernelPiS_S_S_S_S_ii( int *__par0,  int *__par1,  int *__par2,  int *__par3,  int *__par4,  int *__par5,  int __par6,  int __par7) {  __cudaLaunchPrologue(8); __cudaSetupArgSimple(__par0, 0UL); __cudaSetupArgSimple(__par1, 8UL); __cudaSetupArgSimple(__par2, 16UL); __cudaSetupArgSimple(__par3, 24UL); __cudaSetupArgSimple(__par4, 32UL); __cudaSetupArgSimple(__par5, 40UL); __cudaSetupArgSimple(__par6, 48UL); __cudaSetupArgSimple(__par7, 52UL); __cudaLaunch(((char *)((void ( *)(int *, int *, int *, int *, int *, int *, int, int))iriw_kernel))); }
# 515 "litmus_tests.cu"
void iriw_kernel( int *__cuda_0,int *__cuda_1,int *__cuda_2,int *__cuda_3,int *__cuda_4,int *__cuda_5,int __cuda_6,int __cuda_7)
# 518 "litmus_tests.cu"
{__device_stub__Z11iriw_kernelPiS_S_S_S_S_ii( __cuda_0,__cuda_1,__cuda_2,__cuda_3,__cuda_4,__cuda_5,__cuda_6,__cuda_7);
# 588 "litmus_tests.cu"
}
# 1 "litmus_tests.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T23) {  __nv_dummy_param_ref(__T23); __nv_save_fatbinhandle_for_managed_rt(__T23); __cudaRegisterEntry(__T23, ((void ( *)(int *, int *, int *, int *, int *, int *, int, int))iriw_kernel), _Z11iriw_kernelPiS_S_S_S_S_ii, (-1)); __cudaRegisterEntry(__T23, ((void ( *)(int *, int *, int *, int *, int, int, bool))lb_kernel), _Z9lb_kernelPiS_S_S_iib, (-1)); __cudaRegisterEntry(__T23, ((void ( *)(int *, int *, int *, int *, int, int, bool))mp_kernel), _Z9mp_kernelPiS_S_S_iib, (-1)); __cudaRegisterEntry(__T23, ((void ( *)(int *, int *, int *, int *, int, int, bool))sb_kernel), _Z9sb_kernelPiS_S_S_iib, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
