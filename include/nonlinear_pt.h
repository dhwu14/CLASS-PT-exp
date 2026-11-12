/** @file nonlinear.h Documented includes for trg module */
#include <unistd.h>
#include <errno.h>
#include "primordial.h"
/* DW: make the Fourier spectrum interface available to the exact-time PT module. */
#include "fourier.h"

#ifndef __cplusplus
#include <complex.h>
#endif

#ifndef __NONLINEAR_PT__
#define __NONLINEAR_PT__

#ifdef __cplusplus
extern "C" {
/* In C++ context, use void* for double complex pointers (same size) */
#define _dcomplex_ptr_ void *
#define _dcomplex_ void
#else
#define _dcomplex_ptr_ double complex *
#define _dcomplex_ double complex
#endif

/**
 * Maximum number of values of redshift at which the spectra will be
 * written in output files
 */

#define _Z_PK_NUM_MAX_ 100


enum non_linear_method_pt {nlpt_none,nlpt_spt};
enum non_linear_irresumm_pt {irres_yes,irres_no};
enum non_linear_bias_pt {bias_yes,bias_no};
enum non_linear_rsd_pt {rsd_yes,rsd_no};
// enum non_linear_cb_pt {cb_yes,cb_no};
// enum non_linear_fftlogn {fftlog_fast,fftlog_norm};
enum non_linear_AP_effect_pt {AP_effect_yes,AP_effect_no};
enum non_linear_fNL_equil_ortho {fNL_equil_ortho_yes,fNL_equil_ortho_no};
enum non_linear_qcdm_qscf_kernels {qcdm_qscf_kernels_auto,qcdm_qscf_kernels_no,qcdm_qscf_kernels_exact_qs};

//enum halofit_integral_type {halofit_integral_one, halofit_integral_two, halofit_integral_three};
//enum halofit_statement {ok, too_small};

/**
 * Structure containing all information on non-linear spectra.
 *
 * Once initialized by nonlinear_init(), contains a table for all two points correlation functions
 * and for all the ai,bj functions (containing the three points correlation functions), for each
 * time and wave-number.
 */

struct nonlinear_pt {

  /** @name - input parameters initialized by user in input module
      (all other quantities are computed in this module, given these
      parameters and the content of the 'precision', 'background',
      'thermo', 'primordial' and 'spectra' structures) */

  //@{

enum non_linear_method_pt method; /**< method for computing non-linear corrections (none, Halogit, etc.) */
enum non_linear_irresumm_pt irres;
  /* DW: require an explicit opt-in before using the legacy isotropic BAO
     damping as an approximation in the exact-time two-fluid RSD branch. */
  short qcdm_allow_isotropic_rsd_ir_approximation;
enum non_linear_bias_pt bias;
enum non_linear_rsd_pt rsd;
// enum non_linear_cb_pt cb;
// enum non_linear_fftlogn norm;
enum non_linear_AP_effect_pt AP_effect;
enum non_linear_fNL_equil_ortho fNL_equil_ortho_switch;
enum non_linear_qcdm_qscf_kernels qcdm_qscf_kernels;

  /* DW: add explicit amplitudes for the exact-time coupled-fluid primordial
     non-Gaussianity templates. PNG=yes controls whether the unit-amplitude
     equilateral and orthogonal P12 integrals are evaluated; these two
     coefficients control whether they are added to the reported spectra. */
  double qcdm_fNL_equilateral;
  double qcdm_fNL_orthogonal;

  /* DW: QCDM two-fluid tracer coefficients for exact-QS direct RSD branch.
     These are only used when qcdm_qscf_kernels != no. The expansion is
       delta_g = b_m delta_m + b_r delta_r + b_theta_r Theta_r
               + 1/2 b_mm delta_m^2 + b_mr delta_m delta_r
               + 1/2 b_rr delta_r^2
               + b_Gmm G2[mm] + b_Gmr G2[mr] + b_Grr G2[rr].
     The observed tracer velocity is Theta_g = Theta_m + c_vr Theta_r. */
  double qcdm_tf_b_m;
  double qcdm_tf_b_r;
  double qcdm_tf_b_theta_r;
  double qcdm_tf_c_vr;
/* DW: add nonlinear relative-velocity bias coefficients to the two-fluid tracer model. */
  double qcdm_tf_c_mr;
  double qcdm_tf_c_K;
  double qcdm_tf_b_mm;
  double qcdm_tf_b_mr;
  double qcdm_tf_b_rr;
  double qcdm_tf_b_Gmm;
  double qcdm_tf_b_Gmr;
  double qcdm_tf_b_Grr;
/* DW: add the third-order nonlocal Gamma3 tracer-bias coefficient. */
  double qcdm_tf_b_Gamma3;
  double qcdm_tf_ct0;
  double qcdm_tf_ct2;
  double qcdm_tf_ct4;
/* DW: add independent relative-mode redshift-space EFT counterterms. */
  double qcdm_tf_ct_r0;
  double qcdm_tf_ct_r2;
  double qcdm_tf_ct_r4;
  double qcdm_tf_stoch0;
  double qcdm_tf_stoch2;
  double qcdm_tf_stoch2_mu2;

  /* DW: add matter-only EFT coefficients for the explicitly decomposed
     matter spectrum. qcdm_matter_ct_k2_h is in (Mpc/h)^2 and multiplies
     -k_h^2 P_L; qcdm_matter_stoch_k4_h is in (Mpc/h)^7 and multiplies k_h^4. */
  double qcdm_matter_ct_k2_h;
  /* DW: add the independent two-fluid matter counterterm multiplying k_h^2 P_mr. */
  double qcdm_matter_ct_r_k2_h;
  double qcdm_matter_stoch_k4_h;
  int qcdm_tf_Nq;
  int qcdm_tf_Nx;
  int qcdm_tf_Nphi;
  int qcdm_tf_Nmu;

/* DW: expose and validate exact-time kernel ODE resolution controls. */
  /* Exact-time kernel and direct-loop controls. Wavenumbers carrying the
     suffix _h are specified in h/Mpc in the input file. */
  double qcdm_kernel_z_ini;
  int qcdm_kernel_steps_f1;
  int qcdm_kernel_steps_f2;
  int qcdm_kernel_steps_f3;
  double qcdm_loop_q_min_h;
  double qcdm_loop_q_max_h;
  double qcdm_loop_k_max_h;

  /* Optional disk output switches for CLASS-PT/QCDM files. */
  short qcdm_write_pk_nl;
  short qcdm_write_twofluid_templates;
  short qcdm_write_multipoles;
  short qcdm_write_tracer_real;
  /* DW: control the detailed cumulative matter EFTofLSS breakdown file. */
  short qcdm_write_matter_breakdown;

  /* DW: exact-QS fast tracer/nuisance layer.
     The expensive cosmology-dependent kernel solutions are generated once
     per nonlinear_pt_init call and retained for all requested z_pk values.
     Galaxy-bias/EFT/stochastic parameters can then be changed through the
     public fast-pkpoles interface without rerunning background,
     perturbations, Fourier, or the exact kernel ODEs. */
  short qcdm_kernel_cache_started;
  short qcdm_fast_tracer_ready;
  int qcdm_fast_nmax;
  double *qcdm_fast_kdisc;
  double *qcdm_fast_Pbin;
  double *qcdm_fast_Pphi;
  double *qcdm_fast_Mphi;
  short *qcdm_fast_has_png;
  short *qcdm_fast_z_ready;

  /* Root prefix used for writing CLASS-PT/QCDM output files. */
  FileName qcdm_pt_output_root;

  //@}

  /** @name - table non-linear corrections for matter density, sqrt(P_NL(k,z)/P_NL(k,z)) */

  //@{

    // M.I. Global collable variables have to be defined here (they are called with prefix pnlpt->)

    int z_pk_num;
    double z_pk[_Z_PK_NUM_MAX_];

  int k_size;      /**< k_size = total number of k values */
  int ln_k_size;   /**< k_size = total number of k values */
  double * k;      /**< k[index_k] = list of k values */
  double * ln_k;      /**< k[index_k] = list of k values */
  int tau_size;    /**< tau_size = number of values */
  double * tau;    /**< tau[index_tau] = list of time values */
  double * ln_tau;    /**< ln_tau is absolutely different from psp->ln_tau!! It is a simple log of tau with size 609*/

  int index_md_scalars,ic_size,tp_size;
  double * dd_sources_tp_delta_m;
  double * dd_sources_tp_delta_cb;
  double * sources_tp_delta_m;
  double * sources_tp_delta_cb;

short fast_output;
short cb;

//  double * pk_nl_out;

  //@{

  int k_size_cmb;  /**< k_size_cmb[index_md] number of k values used
                        for CMB calculations, requiring a fine
                        sampling in k-space */

  int k_size_cl;  /**< k_size_cl[index_md] number of k values used
                       for non-CMB \f$ C_l \f$ calculations, requiring a coarse
                       sampling in k-space. */

  double k_min;     /**< minimum value (over all modes) */
  double k_max;     /**< maximum value (over all modes) */

  //@}

double OmfidAP;

double * M13_oneline;
double * M22_oneline;
double * M22basic_oneline;
double * IFG2_oneline;
//double * M12_oneline;    //GC!


double * M12_oneline;

    //GC: ORTHOGONAL -- start

double * M12_oneline_ortho;
    
    //GC: ORTHOGONAL -- finish
    
/**/
double * M12_oneline_matter_multipoles_vv0_f2;
double * M12_oneline_matter_multipoles_vv0_f3;
double * M12_oneline_matter_multipoles_vd0_f1;
double * M12_oneline_matter_multipoles_vd0_f2;
double * M12_oneline_matter_multipoles_dd0_f0;
double * M12_oneline_matter_multipoles_dd0_f1;
double * M12_oneline_matter_multipoles_vv2_f3;
double * M12_oneline_matter_multipoles_vd2_f2;
double * M12_oneline_matter_multipoles_vv4_f3;
double * M12_oneline_matter_multipoles_vd4_f2;
/**/
    
    //GC: ORTHOGONAL -- start

    double * M12_oneline_matter_multipoles_vv0_f2_ortho;
    double * M12_oneline_matter_multipoles_vv0_f3_ortho;
    double * M12_oneline_matter_multipoles_vd0_f1_ortho;
    double * M12_oneline_matter_multipoles_vd0_f2_ortho;
    double * M12_oneline_matter_multipoles_dd0_f0_ortho;
    double * M12_oneline_matter_multipoles_dd0_f1_ortho;
    double * M12_oneline_matter_multipoles_vv2_f3_ortho;
    double * M12_oneline_matter_multipoles_vd2_f2_ortho;
    double * M12_oneline_matter_multipoles_vv4_f3_ortho;
    double * M12_oneline_matter_multipoles_vd4_f2_ortho;
    
    //GC: ORTHOGONAL -- finish

 
double * M12_oneline_matter_mu_powers_vd2_f1;
double * M12_oneline_matter_mu_powers_vd2_f2;
double * M12_oneline_matter_mu_powers_dd2_f1;
double * M12_oneline_matter_mu_powers_vv4_f2;
double * M12_oneline_matter_mu_powers_vd4_f2;
double * M12_oneline_matter_mu_powers_vv6_f3;

    
    //GC: ORTHOGONAL -- start

double * M12_oneline_matter_mu_powers_vd2_f1_ortho;
double * M12_oneline_matter_mu_powers_vd2_f2_ortho;
double * M12_oneline_matter_mu_powers_dd2_f1_ortho;
double * M12_oneline_matter_mu_powers_vv4_f2_ortho;
double * M12_oneline_matter_mu_powers_vd4_f2_ortho;
double * M12_oneline_matter_mu_powers_vv6_f3_ortho;

    
    //GC: ORTHOGONAL -- finish

    
/**/
double * M12_oneline_bias_real_space_b2;
double * M12_oneline_bias_real_space_bG2;
/**/

    
    //GC: ORTHOGONAL -- start

    
    double * M12_oneline_bias_real_space_b2_ortho;
    double * M12_oneline_bias_real_space_bG2_ortho;

    
    //GC: ORTHOGONAL -- finish

 
double * M12_oneline_bias_multipoles_b2_vv0_f1;
double * M12_oneline_bias_multipoles_bG2_vv0_f1;
    
    //GC: ORTHOGONAL -- finish

    double * M12_oneline_bias_multipoles_b2_vv0_f1_ortho;
    double * M12_oneline_bias_multipoles_bG2_vv0_f1_ortho;
        
    //GC: ORTHOGONAL -- finish

    
    //GC --> go now to the complexified ones...
    
    //GC --> THESE ALL REMAIN...

_dcomplex_ptr_ M13_oneline_complex;
_dcomplex_ptr_ M22_oneline_complex;
_dcomplex_ptr_ M22basic_oneline_complex;
_dcomplex_ptr_ IFG2_oneline_complex;

_dcomplex_ptr_ M22_oneline_0_vv_complex;
_dcomplex_ptr_ M13_0_vv_oneline_complex;

_dcomplex_ptr_ M22_oneline_0_vd_complex;
_dcomplex_ptr_ M13_0_vd_oneline_complex;

_dcomplex_ptr_ M22_oneline_0_dd_complex;
_dcomplex_ptr_ M13_0_dd_oneline_complex;

    _dcomplex_ptr_ M22_oneline_2_vv_complex;
    _dcomplex_ptr_ M13_2_vv_oneline_complex;

    _dcomplex_ptr_ M22_oneline_4_vv_complex;
    _dcomplex_ptr_ M13_4_vv_oneline_complex;

    _dcomplex_ptr_ M22_oneline_2_vd_complex;
    _dcomplex_ptr_ M13_2_vd_oneline_complex;

    _dcomplex_ptr_ M22_oneline_4_vd_complex;
    _dcomplex_ptr_ M13_4_vd_oneline_complex;
    _dcomplex_ptr_ M22_oneline_4_dd_complex;

    _dcomplex_ptr_ M22_oneline_2_dd_complex;
    _dcomplex_ptr_ M13_2_dd_oneline_complex;

_dcomplex_ptr_ M22_0_b1b2_oneline_complex;
_dcomplex_ptr_ M22_0_b2_oneline_complex;
_dcomplex_ptr_ M22_0_b1bG2_oneline_complex;
    _dcomplex_ptr_ M22_0_bG2_oneline_complex;


    _dcomplex_ptr_ M22_2_b1b2_oneline_complex;
    _dcomplex_ptr_ M22_2_b2_oneline_complex;
    _dcomplex_ptr_ M22_2_b1bG2_oneline_complex;
    _dcomplex_ptr_ M22_2_bG2_oneline_complex;

    _dcomplex_ptr_ M22_4_b2_oneline_complex;
    _dcomplex_ptr_ M22_4_bG2_oneline_complex;

_dcomplex_ptr_ M_Id2;
_dcomplex_ptr_ M_IG2;
_dcomplex_ptr_ M_Id2G2;
_dcomplex_ptr_ M_IG2G2;

_dcomplex_ptr_ M22_oneline_mu2_vd_complex;
_dcomplex_ptr_ M22_oneline_mu2_dd_complex;
    _dcomplex_ptr_ M22_oneline_mu4_vv_complex;
    _dcomplex_ptr_ M22_oneline_mu4_vd_complex;

    _dcomplex_ptr_ M22_oneline_mu4_dd_complex;
    _dcomplex_ptr_ M22_oneline_mu6_vv_complex;
    _dcomplex_ptr_ M22_oneline_mu6_vd_complex;
    _dcomplex_ptr_ M22_oneline_mu8_complex;

_dcomplex_ptr_ M13_mu2_dd_oneline_complex;
    _dcomplex_ptr_ M13_mu2_vd_oneline_complex;
    _dcomplex_ptr_ M13_mu4_vv_oneline_complex;
    _dcomplex_ptr_ M13_mu4_vd_oneline_complex;
    _dcomplex_ptr_ M13_mu6_oneline_complex;
    
    
    //GC!
    
    
    _dcomplex_ptr_ M12_oneline_complex;
    
    //GC: ORTHOGONAL -- start

    _dcomplex_ptr_ M12_oneline_complex_ortho;

    //GC: ORTHOGONAL -- finish



    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vv0_f2;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vv0_f3;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vd0_f1;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vd0_f2;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_dd0_f0;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_dd0_f1;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vv2_f3;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vd2_f2;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vv4_f3;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vd4_f2;
    
    
    //GC: ORTHOGONAL -- start
    
    
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vv0_f2_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vv0_f3_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vd0_f1_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vd0_f2_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_dd0_f0_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_dd0_f1_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vv2_f3_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vd2_f2_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vv4_f3_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_multipoles_vd4_f2_ortho;


    //GC: ORTHOGONAL -- finish



    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vd2_f1;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vd2_f2;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_dd2_f1;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vv4_f2;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vd4_f2;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vv6_f3;
    
    
    //GC: ORTHOGONAL -- start
    
    
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vd2_f1_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vd2_f2_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_dd2_f1_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vv4_f2_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vd4_f2_ortho;
    _dcomplex_ptr_ M12_oneline_complex_matter_mu_powers_vv6_f3_ortho;


    //GC: ORTHOGONAL -- finish



    _dcomplex_ptr_ M12_oneline_complex_bias_real_space_b2;
    _dcomplex_ptr_ M12_oneline_complex_bias_real_space_bG2;
    
    
    //GC: ORTHOGONAL -- start

    _dcomplex_ptr_ M12_oneline_complex_bias_real_space_b2_ortho;
    _dcomplex_ptr_ M12_oneline_complex_bias_real_space_bG2_ortho;

    //GC: ORTHOGONAL -- finish



    _dcomplex_ptr_ M12_oneline_complex_bias_multipoles_b2_vv0_f1;
    _dcomplex_ptr_ M12_oneline_complex_bias_multipoles_bG2_vv0_f1;
    
    
    //GC: ORTHOGONAL -- start

    _dcomplex_ptr_ M12_oneline_complex_bias_multipoles_b2_vv0_f1_ortho;
    _dcomplex_ptr_ M12_oneline_complex_bias_multipoles_bG2_vv0_f1_ortho;
    

    //GC: ORTHOGONAL -- finish



    _dcomplex_ptr_ M12_oneline_0_vv_complex;
    _dcomplex_ptr_ M12_oneline_0_vd_complex;
    _dcomplex_ptr_ M12_oneline_0_dd_complex;
    _dcomplex_ptr_ M12_oneline_2_vv_complex;
    _dcomplex_ptr_ M12_oneline_2_vd_complex;
    _dcomplex_ptr_ M12_oneline_2_dd_complex;
    _dcomplex_ptr_ M12_oneline_4_vv_complex;
    _dcomplex_ptr_ M12_oneline_4_vd_complex;
    _dcomplex_ptr_ M12_oneline_mu2_vd_complex;
    _dcomplex_ptr_ M12_oneline_mu2_dd_complex;
    _dcomplex_ptr_ M12_oneline_mu4_vv_complex;
    _dcomplex_ptr_ M12_oneline_mu4_vd_complex;
    _dcomplex_ptr_ M12_oneline_mu6_vv_complex;
    
    
    //GC: ORTHOGONAL -- start

    _dcomplex_ptr_ M12_oneline_0_vv_complex_ortho;
    _dcomplex_ptr_ M12_oneline_0_vd_complex_ortho;
    _dcomplex_ptr_ M12_oneline_0_dd_complex_ortho;
    _dcomplex_ptr_ M12_oneline_2_vv_complex_ortho;
    _dcomplex_ptr_ M12_oneline_2_vd_complex_ortho;
    _dcomplex_ptr_ M12_oneline_2_dd_complex_ortho;
    _dcomplex_ptr_ M12_oneline_4_vv_complex_ortho;
    _dcomplex_ptr_ M12_oneline_4_vd_complex_ortho;
    _dcomplex_ptr_ M12_oneline_mu2_vd_complex_ortho;
    _dcomplex_ptr_ M12_oneline_mu2_dd_complex_ortho;
    _dcomplex_ptr_ M12_oneline_mu4_vv_complex_ortho;
    _dcomplex_ptr_ M12_oneline_mu4_vd_complex_ortho;
    _dcomplex_ptr_ M12_oneline_mu6_vv_complex_ortho;

    //GC: ORTHOGONAL -- finish

    
    
    _dcomplex_ptr_ M12_2_bG2_oneline_complex;
    _dcomplex_ptr_ M12_2_b2_oneline_complex;
    _dcomplex_ptr_ M12_0_bG2_oneline_complex;
    _dcomplex_ptr_ M12_0_b1bG2_oneline_complex;
    _dcomplex_ptr_ M12_0_b2_oneline_complex;
    _dcomplex_ptr_ M12_0_b1b2_oneline_complex;
    
    
    //GC: ORTHOGONAL -- start

    _dcomplex_ptr_ M12_2_bG2_oneline_complex_ortho;
    _dcomplex_ptr_ M12_2_b2_oneline_complex_ortho;
    _dcomplex_ptr_ M12_0_bG2_oneline_complex_ortho;
    _dcomplex_ptr_ M12_0_b1bG2_oneline_complex_ortho;
    _dcomplex_ptr_ M12_0_b2_oneline_complex_ortho;
    _dcomplex_ptr_ M12_0_b1b2_oneline_complex_ortho;
    

    //GC: ORTHOGONAL -- finish


    _dcomplex_ptr_ M_fNLd2;
    _dcomplex_ptr_ M_fNLG2;
    
    
    //GC: ORTHOGONAL -- start

    _dcomplex_ptr_ M_fNLd2_ortho;
    _dcomplex_ptr_ M_fNLG2_ortho;

    //GC: ORTHOGONAL -- finish
    
    
    //GC!


double * ln_pk_nl;     /*For classy[i_z*pnlpt->k_size+index_k]*/
double * ln_pk_Id2d2;
    double * ln_pk_Id2d2_2;
    double * ln_pk_Id2d2_4;
double * ln_pk_Id2;
double * ln_pk_IG2;
double * ln_pk_Id2G2;
    double * ln_pk_Id2G2_2;
    double * ln_pk_Id2G2_4;
double * ln_pk_IG2G2;
    double * ln_pk_IG2G2_2;
    double * ln_pk_IG2G2_4;
double * ln_pk_IFG2;
    double * ln_pk_IFG2_0b1;
    double * ln_pk_IFG2_0;
    double * ln_pk_IFG2_2;
double * ln_pk_CTR;
    double * ln_pk_CTR_0;
    double * ln_pk_CTR_2;
    double * ln_pk_CTR_4;
double * ln_pk_Tree;
    double * ln_pk_Tree_0_vv;
    double * ln_pk_Tree_0_vd;
    double * ln_pk_Tree_0_dd;
    double * ln_pk_Tree_2_vv;
    double * ln_pk_Tree_2_vd;
    double * ln_pk_Tree_4_vv;

double * ln_pk_0_vv;
double * ln_pk_0_vd;
double * ln_pk_0_dd;

    double * ln_pk_2_vv;
    double * ln_pk_2_vd;
    double * ln_pk_2_dd;

    double * ln_pk_4_vv;
    double * ln_pk_4_vd;

    double * ln_pk_4_dd;

double * ln_pk_0_b1b2;
double * ln_pk_0_b1bG2;
double * ln_pk_0_b2;
    double * ln_pk_0_bG2;

    double * ln_pk_2_b1b2;
    double * ln_pk_2_b1bG2;
    double * ln_pk_2_b2;
    double * ln_pk_2_bG2;

    double * ln_pk_4_b2;
    double * ln_pk_4_bG2;
    double * ln_pk_4_b1b2;
    double * ln_pk_4_b1bG2;
    
    //GC!
        
    double * ln_pk_fNL_0_vv;
    double * ln_pk_fNL_0_vd;
    double * ln_pk_fNL_0_dd;

    double * ln_pk_fNL_2_vv;
    double * ln_pk_fNL_2_vd;
    double * ln_pk_fNL_2_dd;

    double * ln_pk_fNL_4_vv;
    double * ln_pk_fNL_4_vd;
    double * ln_pk_fNL_4_dd;
    
    
    //GC: ORTHOGONAL -- start


    double * ln_pk_fNL_0_vv_ortho;
    double * ln_pk_fNL_0_vd_ortho;
    double * ln_pk_fNL_0_dd_ortho;

    double * ln_pk_fNL_2_vv_ortho;
    double * ln_pk_fNL_2_vd_ortho;
    double * ln_pk_fNL_2_dd_ortho;

    double * ln_pk_fNL_4_vv_ortho;
    double * ln_pk_fNL_4_vd_ortho;
    double * ln_pk_fNL_4_dd_ortho;

    
    //GC: ORTHOGONAL -- finish


    
    double * ln_pk12_0_b1b2;
    double * ln_pk12_0_b2;
    double * ln_pk12_0_b1bG2;
    double * ln_pk12_0_bG2;
    double * ln_pk12_2_b1b2;
    double * ln_pk12_2_b2;
    double * ln_pk12_2_b1bG2;
    double * ln_pk12_2_bG2;
    double * ln_pk12_4_b1b2;
    double * ln_pk12_4_b2;
    double * ln_pk12_4_b1bG2;
    double * ln_pk12_4_bG2;
    
    
    
    //GC: ORTHOGONAL -- start
    
    
    double * ln_pk12_0_b1b2_ortho;
    double * ln_pk12_0_b2_ortho;
    double * ln_pk12_0_b1bG2_ortho;
    double * ln_pk12_0_bG2_ortho;
    double * ln_pk12_2_b1b2_ortho;
    double * ln_pk12_2_b2_ortho;
    double * ln_pk12_2_b1bG2_ortho;
    double * ln_pk12_2_bG2_ortho;
    double * ln_pk12_4_b1b2_ortho;
    double * ln_pk12_4_b2_ortho;
    double * ln_pk12_4_b1bG2_ortho;
    double * ln_pk12_4_bG2_ortho;


    //GC: ORTHOGONAL -- finish


    
    double * ln_pk_nl_fNL;
    double * ln_pk_fNLd2;
    double * ln_pk_fNLG2;
    
    
    //GC: ORTHOGONAL -- start
    
    
    double * ln_pk_nl_fNL_ortho;
    double * ln_pk_fNLd2_ortho;
    double * ln_pk_fNLG2_ortho;


    //GC: ORTHOGONAL -- finish


    //GC!

    double * growthf;   /*For RSD effect[i_z]*/
    double * hratio_array;   /*For RSD effect[i_z]*/
    double * Dratio_array;   /*For RSD effect[i_z]*/

    double * gauss;
    double * gauss_x;
    double * gauss_w;

    // double *koff;
    // double *Poff;
    // double *offtab;

/*< nl_corr_density[index_tau * ppt->k_size + index_k] */
    double * nl_corr_density;
    /*
    double * nl_corr_density;
    double * nl_corr_Id2d2;
    double * nl_corr_IG2;
    double * nl_corr_Id2;
    double * nl_corr_Id2G2;
    double * nl_corr_IG2G2;
    double * nl_corr_IFG2;
    double * nl_corr_CTR;
    double * nl_corr_Tree;
    */


    char input_pk[500];
    int replace_pk;
    int replace_background;
    int no_wiggle;
    int wiggle_only;
    double alpha_rs;
    double replace_Hz_value;
    double replace_DAz_value;
    double replace_Dz_value;
    double replace_fz_value;

//  double * k_nl;  /**< wavenumber at which non-linear corrections become important, defined differently by different non_linear_method's */
  int index_tau_min_nl; /**< index of smallest value of tau at which nonlinear corrections have been computed (so, for tau<tau_min_nl, the array nl_corr_density only contains some factors 1 */

  //@}

  /** @name - technical parameters */

  //@{

  short nonlinear_pt_verbose;  	/**< amount of information written in standard output */
  double pt_k_max_h;  /**< max k in h/Mpc for perturbations grid (default 100; lower = faster but less accurate) */

  ErrorMsg error_message; 	/**< zone for writing error messages */

  //@}
};

/********************************************************************************/


extern void zspmv_(char*, int*, _dcomplex_ptr_, _dcomplex_ptr_, _dcomplex_ptr_, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*);
extern _dcomplex_ zdotu_(int*, _dcomplex_ptr_, int*, _dcomplex_ptr_, int*);
extern void zsymv_(char*, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*, _dcomplex_ptr_, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*);
extern void zsymm_(char*, char*, int*, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*, _dcomplex_ptr_, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*);
extern void zgemv_(char*, int*, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*, _dcomplex_ptr_, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*);
extern void zgemm_(char*, char*, int*, int*, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*, _dcomplex_ptr_, int*, _dcomplex_ptr_, _dcomplex_ptr_, int*);

/* Real BLAS for optimized real-valued PT loop integrals */
extern void dsymm_(char*, char*, int*, int*, double*, double*, int*,
                   double*, int*, double*, double*, int*);
extern void dgemv_(char*, int*, int*, double*, double*, int*,
                   double*, int*, double*, double*, int*);
extern double ddot_(int*, double*, int*, double*, int*);

#ifdef __cplusplus
} /* close first extern "C" block (struct + BLAS declarations) */
#endif

/* @cond INCLUDE_WITH_DOXYGEN */
/*
 * Boilerplate for C++
 */
#ifdef __cplusplus
extern "C" {
#endif


  int nonlinear_pt_init(
                     struct precision *ppr,
                     struct background *pba,
                     struct thermodynamics *pth,
                     struct perturbations *ppt,
                     struct primordial *ppm,
/* DW: pass the initialized Fourier module into nonlinear_pt so the physical CLASS linear matter spectrum is used. */
                     struct fourier *pfo,
                     struct nonlinear_pt *pnlpt
                     );

  int perturb_get_k_list_nl(
                         struct precision * ppr,
                         struct background * pba,
                         struct thermodynamics * pth,
                         struct perturbations * ppt,
                         struct nonlinear_pt *pnlpt
                         );

  int nonlinear_pt_free(
                     struct nonlinear_pt *pnlpt
                     );

  /* DW: exact-QS fast tracer interface for likelihood sampling.
     k_h is in h/Mpc and the returned poles are in (Mpc/h)^3.
     tracer_params must contain QCDM_TF_FAST_NPARAM values in the order
       b_m, b_r, b_theta_r, c_vr, c_mr, c_K,
       b_mm, b_mr, b_rr, b_Gmm, b_Gmr, b_Grr, b_Gamma3,
       ct0, ct2, ct4, ct_r0, ct_r2, ct_r4,
       stoch0, stoch2, stoch2_mu2, fNL_equilateral, fNL_orthogonal.
     This changes only tracer/nuisance contractions; the exact c-c' kernel
     physics and cosmology are held fixed. */
#define QCDM_TF_FAST_NPARAM 24
#define QCDM_TF_BASIS_NCOEFF 18
  int nonlinear_pt_qcdm_fast_pkpoles(
                     struct background *pba,
                     struct nonlinear_pt *pnlpt,
                     double z,
                     const double *k_h,
                     int k_size,
                     const double *tracer_params,
                     int n_tracer_params,
                     double *pk0_h,
                     double *pk2_h,
                     double *pk4_h,
                     int use_ap_override,
                     double hratio_override,
                     double Dratio_override
                     );


  /* DW: exact-time QCDM/QSCF deterministic tree-level redshift-space
     bispectrum monopole. Triangle sides are in h/Mpc and b0_h is returned
     in (Mpc/h)^6. The same QCDM_TF_FAST_NPARAM tracer vector is used as by
     nonlinear_pt_qcdm_fast_pkpoles(). The routine includes exact Z1/Z2
     dynamics, LOS averaging, AP remapping and leading IR resummation. */
  int nonlinear_pt_qcdm_fast_b0_tree(
                     struct background *pba,
                     struct nonlinear_pt *pnlpt,
                     double z,
                     const double *k1_h,
                     const double *k2_h,
                     const double *k3_h,
                     int ntri,
                     const double *tracer_params,
                     int n_tracer_params,
                     int nmu,
                     int nphi,
                     double *b0_h,
                     int use_ap_override,
                     double hratio_override,
                     double Dratio_override
                     );

  /* DW: DESI-convention exact-QS bispectrum monopole. This extends the
     validated tree B0 with the DESI DR1 bispectrum FoG damping parameter
     sigma_B_h [Mpc/h], applied inside the LOS integral after AP remapping.
     sigma_B_h=0 exactly recovers nonlinear_pt_qcdm_fast_b0_tree().
     Survey/bin calibration and residual Poisson shot noise remain likelihood
     products and are intentionally not hard-coded in CLASS-PT. */
  int nonlinear_pt_qcdm_fast_b0_desi(
                     struct background *pba,
                     struct nonlinear_pt *pnlpt,
                     double z,
                     const double *k1_h,
                     const double *k2_h,
                     const double *k3_h,
                     int ntri,
                     const double *tracer_params,
                     int n_tracer_params,
                     int nmu,
                     int nphi,
                     double sigma_B_h,
                     double *b0_h,
                     int use_ap_override,
                     double hratio_override,
                     double Dratio_override
                     );

  /* DW: exact native polynomial basis for the DESI M0 nuisance model.
     basis{0,2,4}_h are coefficient-major arrays of length
     QCDM_TF_BASIS_NCOEFF*k_size. The coefficient ordering is
       1, b_m, b_mm, b_Gmm, b_m^2, b_mm^2, b_Gmm^2,
       b_m*b_mm, b_m*b_Gmm, b_mm*b_Gmm,
       ct0, ct2, ct4, stoch0, stoch2, stoch2_mu2,
      b_Gamma3, b_m*b_Gamma3.
     Columns 0--15 retain their validated legacy definitions; the
     final two columns are the exact one-loop P13 Gamma3 responses.
     All other tracer coefficients are held fixed at tracer_params. */
  int nonlinear_pt_qcdm_fast_pkpole_basis(
                     struct background *pba,
                     struct nonlinear_pt *pnlpt,
                     double z,
                     const double *k_h,
                     int k_size,
                     const double *tracer_params,
                     int n_tracer_params,
                     double *basis0_h,
                     double *basis2_h,
                     double *basis4_h,
                     int use_ap_override,
                     double hratio_override,
                     double Dratio_override
                     );

  int nonlinear_pt_pk_l(
                     struct background *pba,
                     struct perturbations *ppt,
                     struct primordial *ppm,
                     struct nonlinear_pt *pnlpt,
                     int index_tau,
                     double *pk_l,
                     double *lnk,
                     double *lnpk,
                     double *ddlnpk); //,
                     //double *tk_l,
                     //double *lntk,
                     //double *ddlntk); //GC

//GC!

//GC: the function that extracts the primordial PS to get the transfer function {\cal M}(k)...

    int nonlinear_pt_pPRIMk_l(
                              //struct background *pba,
                              struct perturbations *ppt,
                              struct primordial *ppm,
                              struct nonlinear_pt *pnlpt,
                              double *lnk,
                              double *pPRIMk_l,
                              double *lnpPRIMk,
                              double *ddlnpPRIMk);

//GC!


    int nonlinear_pt_loop(
                          struct precision *ppr,
                          struct background *pba,
                          struct primordial *ppm,
                          struct thermodynamics *pth,
                          struct nonlinear_pt *pnlpt,
                          double tau,
                          double f,
                          double hratio,
                          double Dratio,
                          double *pk_l,
                          /* DW: pass the total-matter linear spectrum separately
                             when cb=yes so the exact two-fluid loop can use the
                             clustering b+QCDM spectrum while diagnostics can
                             reconstruct the hybrid total-matter spectrum. */
                          double *pk_l_total,
                          double *pPRIMk_l, //GC! Never used...
                          double *pk_l_0_vv,
                          double *pk_l_0_vd,
                          double *pk_l_0_dd,
                          double *pk_l_2_vv,
                          double *pk_l_2_vd,
                          double *pk_l_2_dd,
                          double *pk_l_4_vv,
                          double *pk_l_4_vd,
                          double *pk_l_4_dd,
                          double *pk_nl,
                          double *pk_nl_fNL, //GC!
                          double *pk_fNLd2, //GC!
                          double *pk_fNLG2, //GC!
                          //GC: ORTHOGONAL -- start
                          double *pk_nl_fNL_ortho, //GC!
                          double *pk_fNLd2_ortho, //GC!
                          double *pk_fNLG2_ortho, //GC!
                          //GC: ORTHOGONAL -- finish
                          double *pk_Id2d2,
                          double *pk_Id2d2_2,
                          double *pk_Id2d2_4,
                          double *pk_l_0_b1b2,
                          double *pk_l_0_b2,
                          double *pk_l_0_b1bG2,
                          double *pk_l_0_bG2,
                          double *pk_l_2_b1b2,
                          double *pk_l_2_b2,
                          double *pk_l_2_b1bG2,
                          double *pk_l_2_bG2,
                          double *pk_l_4_b2,
                          double *pk_l_4_bG2,
                          double *pk_l_4_b1b2,
                          double *pk_l_4_b1bG2,
                          double *pk_Id2,
                          double *pk_IG2,
                          double *pk_Id2G2,
                          double *pk_Id2G2_2,
                          double *pk_Id2G2_4,
                          double *pk_IG2G2,
                          double *pk_IG2G2_2,
                          double *pk_IG2G2_4,
                          double *pk_IFG2,
                          double *pk_IFG2_0b1,
                          double *pk_IFG2_0,
                          double *pk_IFG2_2,
                          double *pk_CTR,
                          double *pk_CTR_0,
                          double *pk_CTR_2,
                          double *pk_CTR_4,
                          double *pk_Tree,
                          double *pk_Tree_0_vv,
                          double *pk_Tree_0_vd,
                          double *pk_Tree_0_dd,
                          double *pk_Tree_2_vv,
                          double *pk_Tree_2_vd,
                          double *pk_Tree_4_vv,
                          //GC!
                          double *pk_l_fNL_0_vv,
                          double *pk_l_fNL_0_vd,
                          double *pk_l_fNL_0_dd,
                          double *pk_l_fNL_2_vv,
                          double *pk_l_fNL_2_vd,
                          double *pk_l_fNL_2_dd,
                          double *pk_l_fNL_4_vv,
                          double *pk_l_fNL_4_vd,
                          double *pk_l_fNL_4_dd,
                          double *pk12_l_0_b1b2,
                          double *pk12_l_0_b2,
                          double *pk12_l_0_b1bG2,
                          double *pk12_l_0_bG2,
                          double *pk12_l_2_b1b2,
                          double *pk12_l_2_b2,
                          double *pk12_l_2_b1bG2,
                          double *pk12_l_2_bG2,
                          double *pk12_l_4_b1b2,
                          double *pk12_l_4_b2,
                          double *pk12_l_4_b1bG2,
                          double *pk12_l_4_bG2,
                          //GC: ORTHOGONAL -- start
                          double *pk_l_fNL_0_vv_ortho,
                          double *pk_l_fNL_0_vd_ortho,
                          double *pk_l_fNL_0_dd_ortho,
                          double *pk_l_fNL_2_vv_ortho,
                          double *pk_l_fNL_2_vd_ortho,
                          double *pk_l_fNL_2_dd_ortho,
                          double *pk_l_fNL_4_vv_ortho,
                          double *pk_l_fNL_4_vd_ortho,
                          double *pk_l_fNL_4_dd_ortho,
                          double *pk12_l_0_b1b2_ortho,
                          double *pk12_l_0_b2_ortho,
                          double *pk12_l_0_b1bG2_ortho,
                          double *pk12_l_0_bG2_ortho,
                          double *pk12_l_2_b1b2_ortho,
                          double *pk12_l_2_b2_ortho,
                          double *pk12_l_2_b1bG2_ortho,
                          double *pk12_l_2_bG2_ortho,
                          double *pk12_l_4_b1b2_ortho,
                          double *pk12_l_4_b2_ortho,
                          double *pk12_l_4_b1bG2_ortho,
                          double *pk12_l_4_bG2_ortho,
                          //GC: ORTHOGONAL -- finish
                          //GC!
                          double *lnk_l,
                          double *lnpk_l,
                          double *lnpPRIMk_l //GC!
                          );


    /**
     * Function definitions of the FFT tool, used by transfer.c and spectra.c
     * For more information, see fft.c
     */

#ifndef FFT_DEFINED
#define FFT_DEFINED
    /**
     * Compute FFT of two real inputs and store into two complex outputs of sizes N
     * Assumes arrays are allocated and of size N
     */
    void FFT_real(double* input_real_1, double* input_real_2, double* output_real_1, double* output_imag_1, double* output_real_2, double* output_imag_2, int N);
    /**
     * Compute FFT of two real inputs and store into two complex outputs of sizes N
     * Assumes arrays are allocated and of size N
     * Only gives up nonredundant part for real FFT which have c_(N-n)=c_n
     */
    void FFT_real_short(double* input_real_1, double* input_real_2, double* output_real_1, double* output_imag_1, double* output_real_2, double* output_imag_2, int N);
    /**
     * Compute FFT of single complex input and stores into single complex output of size N
     * Assumes arrays are allocated and of size N
     */
    void FFT(double* input_real, double* input_imag, double* output_real, double* output_imag, int N, int stepsize);
    void DCT_real(double* input_real,double* input_imag,double* output_real,double* output_imag,int N);
#endif


  /* Output/interpolation functions for PT spectra */

  int nonlinear_pt_bias_at_z_i(
                       struct nonlinear_pt *pnlpt,
                       enum linear_or_logarithmic mode,
                       int i_z,
                       double * output_tot,
                       double * output_tot_Id2d2,
                       double * output_tot_Id2,
                       double * output_tot_IG2,
                       double * output_tot_Id2G2,
                       double * output_tot_IG2G2,
                       double * output_tot_IFG2,
                       double * output_tot_IFG2_0b1,
                       double * output_tot_IFG2_0,
                       double * output_tot_IFG2_2,
                       double * output_tot_CTR,
                       double * output_tot_CTR_0,
                       double * output_tot_CTR_2,
                       double * output_tot_CTR_4,
                       double * output_tot_Tree,
                       double * output_tot_Tree_0_vv,
                       double * output_tot_Tree_0_vd,
                       double * output_tot_Tree_0_dd,
                       double * output_tot_Tree_2_vv,
                       double * output_tot_Tree_2_vd,
                       double * output_tot_Tree_4_vv,
                       double * output_tot_0_vv,
                       double * output_tot_0_vd,
                       double * output_tot_0_dd,
                       double * output_tot_2_vv,
                       double * output_tot_2_vd,
                       double * output_tot_2_dd,
                       double * output_tot_4_vv,
                       double * output_tot_4_vd,
                       double * output_tot_4_dd,
                       double * output_tot_0_b1b2,
                       double * output_tot_0_b2,
                       double * output_tot_0_b1bG2,
                       double * output_tot_0_bG2,
                       double * output_tot_2_b1b2,
                       double * output_tot_2_b2,
                       double * output_tot_2_b1bG2,
                       double * output_tot_2_bG2,
                       double * output_tot_4_b2,
                       double * output_tot_4_bG2,
                       double * output_tot_4_b1b2,
                       double * output_tot_4_b1bG2,
                       double * output_tot_2_b2b2,
                       double * output_tot_2_b2bG2,
                       double * output_tot_2_bG2bG2,
                       double * output_tot_4_b2b2,
                       double * output_tot_4_b2bG2,
                       double * output_tot_4_bG2bG2,
                       double * output_tot_pk_nl_fNL,
                       double * output_tot_pk_fNLd2,
                       double * output_tot_pk_fNLG2,
                       double * output_tot_pk_l_fNL_0_vv,
                       double * output_tot_pk_l_fNL_0_vd,
                       double * output_tot_pk_l_fNL_0_dd,
                       double * output_tot_pk_l_fNL_2_vv,
                       double * output_tot_pk_l_fNL_2_vd,
                       double * output_tot_pk_l_fNL_2_dd,
                       double * output_tot_pk_l_fNL_4_vv,
                       double * output_tot_pk_l_fNL_4_vd,
                       double * output_tot_pk_l_fNL_4_dd,
                       double * output_tot_pk12_l_0_b1b2,
                       double * output_tot_pk12_l_0_b2,
                       double * output_tot_pk12_l_0_b1bG2,
                       double * output_tot_pk12_l_0_bG2,
                       double * output_tot_pk12_l_2_b1b2,
                       double * output_tot_pk12_l_2_b2,
                       double * output_tot_pk12_l_2_b1bG2,
                       double * output_tot_pk12_l_2_bG2,
                       double * output_tot_pk12_l_4_b1b2,
                       double * output_tot_pk12_l_4_b2,
                       double * output_tot_pk12_l_4_b1bG2,
                       double * output_tot_pk12_l_4_bG2,
                       double * output_tot_pk_nl_fNL_ortho,
                       double * output_tot_pk_fNLd2_ortho,
                       double * output_tot_pk_fNLG2_ortho,
                       double * output_tot_pk_l_fNL_0_vv_ortho,
                       double * output_tot_pk_l_fNL_0_vd_ortho,
                       double * output_tot_pk_l_fNL_0_dd_ortho,
                       double * output_tot_pk_l_fNL_2_vv_ortho,
                       double * output_tot_pk_l_fNL_2_vd_ortho,
                       double * output_tot_pk_l_fNL_2_dd_ortho,
                       double * output_tot_pk_l_fNL_4_vv_ortho,
                       double * output_tot_pk_l_fNL_4_vd_ortho,
                       double * output_tot_pk_l_fNL_4_dd_ortho,
                       double * output_tot_pk12_l_0_b1b2_ortho,
                       double * output_tot_pk12_l_0_b2_ortho,
                       double * output_tot_pk12_l_0_b1bG2_ortho,
                       double * output_tot_pk12_l_0_bG2_ortho,
                       double * output_tot_pk12_l_2_b1b2_ortho,
                       double * output_tot_pk12_l_2_b2_ortho,
                       double * output_tot_pk12_l_2_b1bG2_ortho,
                       double * output_tot_pk12_l_2_bG2_ortho,
                       double * output_tot_pk12_l_4_b1b2_ortho,
                       double * output_tot_pk12_l_4_b2_ortho,
                       double * output_tot_pk12_l_4_b1bG2_ortho,
                       double * output_tot_pk12_l_4_bG2_ortho
                       );

  int nonlinear_pt_pk_at_k_and_z(
                             struct nonlinear_pt * pnlpt,
                             double k,
                             double z,
                             double * pk_tot,
                             double * pk_tot_Id2d2,
                             double * pk_tot_Id2,
                             double * pk_tot_IG2,
                             double * pk_tot_Id2G2,
                             double * pk_tot_IG2G2,
                             double * pk_tot_IFG2,
                             double * pk_tot_IFG2_0b1,
                             double * pk_tot_IFG2_0,
                             double * pk_tot_IFG2_2,
                             double * pk_tot_CTR,
                             double * pk_tot_CTR_0,
                             double * pk_tot_CTR_2,
                             double * pk_tot_CTR_4,
                             double * pk_tot_Tree,
                             double * pk_tot_Tree_0_vv,
                             double * pk_tot_Tree_0_vd,
                             double * pk_tot_Tree_0_dd,
                             double * pk_tot_Tree_2_vv,
                             double * pk_tot_Tree_2_vd,
                             double * pk_tot_Tree_4_vv,
                             double * pk_tot_0_vv,
                             double * pk_tot_0_vd,
                             double * pk_tot_0_dd,
                             double * pk_tot_2_vv,
                             double * pk_tot_2_vd,
                             double * pk_tot_2_dd,
                             double * pk_tot_4_vv,
                             double * pk_tot_4_vd,
                             double * pk_tot_4_dd,
                             double * pk_tot_0_b1b2,
                             double * pk_tot_0_b2,
                             double * pk_tot_0_b1bG2,
                             double * pk_tot_0_bG2,
                             double * pk_tot_2_b1b2,
                             double * pk_tot_2_b2,
                             double * pk_tot_2_b1bG2,
                             double * pk_tot_2_bG2,
                             double * pk_tot_4_b2,
                             double * pk_tot_4_bG2,
                             double * pk_tot_4_b1b2,
                             double * pk_tot_4_b1bG2,
                             double * pk_tot_2_b2b2,
                             double * pk_tot_2_b2bG2,
                             double * pk_tot_2_bG2bG2,
                             double * pk_tot_4_b2b2,
                             double * pk_tot_4_b2bG2,
                             double * pk_tot_4_bG2bG2,
                             double * pk_tot_fNL,
                             double * pk_tot_fNLd2,
                             double * pk_tot_fNLG2,
                             double * pk_tot_fNL_0_vv,
                             double * pk_tot_fNL_0_vd,
                             double * pk_tot_fNL_0_dd,
                             double * pk_tot_fNL_2_vv,
                             double * pk_tot_fNL_2_vd,
                             double * pk_tot_fNL_2_dd,
                             double * pk_tot_fNL_4_vv,
                             double * pk_tot_fNL_4_vd,
                             double * pk_tot_fNL_4_dd,
                             double * pk_tot_fNL_0_b1b2,
                             double * pk_tot_fNL_0_b2,
                             double * pk_tot_fNL_0_b1bG2,
                             double * pk_tot_fNL_0_bG2,
                             double * pk_tot_fNL_2_b1b2,
                             double * pk_tot_fNL_2_b2,
                             double * pk_tot_fNL_2_b1bG2,
                             double * pk_tot_fNL_2_bG2,
                             double * pk_tot_fNL_4_b1b2,
                             double * pk_tot_fNL_4_b2,
                             double * pk_tot_fNL_4_b1bG2,
                             double * pk_tot_fNL_4_bG2,
                             double * pk_tot_fNL_ortho,
                             double * pk_tot_fNLd2_ortho,
                             double * pk_tot_fNLG2_ortho,
                             double * pk_tot_fNL_0_vv_ortho,
                             double * pk_tot_fNL_0_vd_ortho,
                             double * pk_tot_fNL_0_dd_ortho,
                             double * pk_tot_fNL_2_vv_ortho,
                             double * pk_tot_fNL_2_vd_ortho,
                             double * pk_tot_fNL_2_dd_ortho,
                             double * pk_tot_fNL_4_vv_ortho,
                             double * pk_tot_fNL_4_vd_ortho,
                             double * pk_tot_fNL_4_dd_ortho,
                             double * pk_tot_fNL_0_b1b2_ortho,
                             double * pk_tot_fNL_0_b2_ortho,
                             double * pk_tot_fNL_0_b1bG2_ortho,
                             double * pk_tot_fNL_0_bG2_ortho,
                             double * pk_tot_fNL_2_b1b2_ortho,
                             double * pk_tot_fNL_2_b2_ortho,
                             double * pk_tot_fNL_2_b1bG2_ortho,
                             double * pk_tot_fNL_2_bG2_ortho,
                             double * pk_tot_fNL_4_b1b2_ortho,
                             double * pk_tot_fNL_4_b2_ortho,
                             double * pk_tot_fNL_4_b1bG2_ortho,
                             double * pk_tot_fNL_4_bG2_ortho
                             );

  int nonlinear_pt_pk_mult_at_kvec_and_z(
                             struct nonlinear_pt * pnlpt,
                             double * kvec,
                             int n_k,
                             double z,
                             double * pk_mult
                             );


#ifdef __cplusplus
}
#endif

/**************************************************************/

#endif
/* @endcond */
