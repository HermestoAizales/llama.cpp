#include "sycl_hw.hpp"

using namespace std;

/*defined in
* /opt/intel/oneapi/compiler/latest/include/sycl/ext/oneapi/experimental/device_architecture.def
*/
static map<gpu_arch, const char*> arch2name = {
    {gpu_arch::intel_gpu_bdw,     "intel_gpu_bdw"},
    {gpu_arch::intel_gpu_skl,     "intel_gpu_skl"},
    {gpu_arch::intel_gpu_kbl,     "intel_gpu_kbl"},
    {gpu_arch::intel_gpu_cfl,     "intel_gpu_cfl"},
    {gpu_arch::intel_gpu_apl,     "intel_gpu_apl"},
    {gpu_arch::intel_gpu_glk,     "intel_gpu_glk"},
    {gpu_arch::intel_gpu_whl,     "intel_gpu_whl"},
    {gpu_arch::intel_gpu_aml,     "intel_gpu_aml"},
    {gpu_arch::intel_gpu_cml,     "intel_gpu_cml"},
    {gpu_arch::intel_gpu_icllp,   "intel_gpu_icllp"},
    {gpu_arch::intel_gpu_ehl,     "intel_gpu_ehl"},
    {gpu_arch::intel_gpu_tgllp,   "intel_gpu_tgllp"},
    {gpu_arch::intel_gpu_rkl,     "intel_gpu_rkl"},
    {gpu_arch::intel_gpu_adl_s,   "intel_gpu_adl_s"},
    {gpu_arch::intel_gpu_adl_p,   "intel_gpu_adl_p"},
    {gpu_arch::intel_gpu_adl_n,   "intel_gpu_adl_n"},
    {gpu_arch::intel_gpu_dg1,     "intel_gpu_dg1"},
    {gpu_arch::intel_gpu_acm_g10, "intel_gpu_acm_g10"},
    {gpu_arch::intel_gpu_acm_g11, "intel_gpu_acm_g11"},
    {gpu_arch::intel_gpu_acm_g12, "intel_gpu_acm_g12"},
    {gpu_arch::intel_gpu_pvc,     "intel_gpu_pvc"},
    {gpu_arch::intel_gpu_pvc_vg,  "intel_gpu_pvc_vg"},
    {gpu_arch::intel_gpu_mtl_u,   "intel_gpu_mtl_u"},
    {gpu_arch::intel_gpu_mtl_h,   "intel_gpu_mtl_h"},
    {gpu_arch::intel_gpu_arl_h,   "intel_gpu_arl_h"},
    {gpu_arch::intel_gpu_bmg_g21, "intel_gpu_bmg_g21"},
    {gpu_arch::intel_gpu_bmg_g31, "intel_gpu_bmg_g31"},
    {gpu_arch::intel_gpu_lnl_m,   "intel_gpu_lnl_m"},
    {gpu_arch::intel_gpu_ptl_h,   "intel_gpu_ptl_h"},
    {gpu_arch::intel_gpu_ptl_u,   "intel_gpu_ptl_u"},
    {gpu_arch::intel_gpu_wcl,     "intel_gpu_wcl"}
};

sycl_hw_info get_device_hw_info(sycl::device* device_ptr) {
  sycl_hw_info res;
  int32_t id =
      device_ptr->get_info<sycl::ext::intel::info::device::device_id>();
  res.device_id = id;

  res.name = device_ptr->get_info<sycl::info::device::name>();

  syclex::architecture arch =
      device_ptr->get_info<syclex::info::device::architecture>();
  res.arch = arch;

  map<syclex::architecture, const char*>::iterator it = arch2name.find(res.arch);
  if (it != arch2name.end()) {
    res.arch_name = it->second;
  } else {
    res.arch_name = "unknown";
  }

  return res;
}
