/**
 * @file nonlinear_pt.c
 * @brief One-loop perturbation theory engine for CLASS-PT.
 *
 * @author Mikhail M. Ivanov (original), with extensions by
 *         Chudaykin, Philcox, Simonovic, Cabass et al.
 * @date 17.06.2018
 *
 * Computes one-loop SPT power spectra for dark matter and biased tracers
 * in real space and redshift space, with:
 *  - FFTLog decomposition of loop integrals (EdS approximation)
 *  - IR resummation of BAO (Senatore & Zaldarriaga method)
 *  - Alcock-Paczynski projection
 *  - Non-Gaussian initial conditions (local/equilateral/orthogonal fNL)
 *
 * Pipeline: input -> background -> thermodynamics -> perturbations ->
 *           primordial -> nonlinear -> [this module] -> transfer -> spectra
 *
 * Reference: Chudaykin, Ivanov, Philcox, Simonovic (arXiv:2004.10607)
 *
 * CLASS-PT-exp extension: Exact time kernels implemented
 * @author: Obied, Vafa, Wu
 * @date: 11.11.2026
 * 
 * Primarily used for the FDS model (arXiv:2507.03090); benchmarked with LCDM (c,c'->0)
 * Additions can be found from comments starting with DW
 * Reference:	Obied, Vafa, Wu (arXiv:2611.XXXXX "The Full-Shape Analysis of the Fading Dark Sector")
 */

#define _GNU_SOURCE
#include "nonlinear_pt.h"
#include "fft.h"
#include <time.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>
/* DW: include floating-point limits used by endpoint-safe exact-time background interpolation. */
#include <float.h>

/* DW: forward declaration; implementation lives beside the exact-QS IR helpers. */
static void qcdm_tf_ir_fast_cache_reset(void);

/* Opt-in linear-only F1 diagnostic; implemented beside the exact-QS F1 solver.
 * Called from nonlinear_pt_init even when method == nlpt_none. */
static int qcdm_qscf_dump_relvel_f1(struct background *pba,
                                     struct nonlinear_pt *pnlpt);

/* DW: QCDM: robust finite check. Many CLASS/CLASS-PT builds use aggressive
 * floating-point flags. Compiler builtins like qcdm_qscf_finite_double() can then be optimized
 * under the assumption that NaNs/Infs do not occur. This IEEE-754 bit test is
 * used only in the QCDM/QSCF exact-QS branch to guarantee that science runs
 * stop loudly instead of writing NaNs. */
static int qcdm_qscf_finite_double(double x) {
    uint64_t u;
    memcpy(&u, &x, sizeof(double));
    return ((u & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL);
}


/* OpenBLAS thread control — allows setting BLAS to single-threaded
 * so we can run many independent BLAS calls in parallel via OpenMP */
/* DW: make OpenBLAS thread-control calls optional when only a generic BLAS library is linked. */
extern void openblas_set_num_threads(int num_threads) __attribute__((weak));
extern int openblas_get_num_threads(void) __attribute__((weak));

static int classpt_blas_get_num_threads(void) {
    return (openblas_get_num_threads != NULL) ? openblas_get_num_threads() : 1;
}

static void classpt_blas_set_num_threads(int n) {
    if (openblas_set_num_threads != NULL) openblas_set_num_threads(n);
}

#define TIMER_DECL(name) double _t_##name = 0.; struct timespec _ts0_##name, _ts1_##name
#define TIMER_START(name) clock_gettime(CLOCK_MONOTONIC, &_ts0_##name)
#define TIMER_ADD(name) do { \
    clock_gettime(CLOCK_MONOTONIC, &_ts1_##name); \
    _t_##name += (_ts1_##name.tv_sec - _ts0_##name.tv_sec) \
               + 1e-9 * (_ts1_##name.tv_nsec - _ts0_##name.tv_nsec); \
} while (0)

/* === K-GRID CONSTRUCTION === */

/**
 * Build the wavenumber grid for the PT module.
 *
 * Constructs a non-uniform k-grid with dense sampling around the BAO
 * scale (k_rec = 2pi/r_s) and logarithmic spacing at high k.
 * Three regimes: k < k_max_cmb (linear steps), k_max_cmb..k_max_cl
 * (log steps with BAO refinement), k_max_cl..k_max (log steps).
 */
int perturb_get_k_list_nl(
    struct precision *ppr,
    struct background *pba,
    struct thermodynamics *pth,
    struct perturbations *ppt,
    struct nonlinear_pt *pnlpt) {
    int index_k, index_k_output, index_mode;
    double k, k_min = 0., k_rec, step, tau1;
    double k_max_cmb;
    double k_max_cl;
    double k_max = 0.;
    double scale2;
    double *tmp_k_list;
    int newk_size, index_newk, add_k_output_value;

    /** Summary: */

    class_test(ppr->k_step_transition == 0.,
               pnlpt->error_message,
               "stop to avoid division by zero");

    class_test(pth->rs_rec == 0.,
               pnlpt->error_message,
               "stop to avoid division by zero");

    /** - scalar modes */

    if (ppt->has_scalars == _TRUE_) {
        k_min = ppr->k_min_tau0 / pba->conformal_age;

        /* first value */
        if (pba->sgnK == 0) {
            /* K<0 (flat)  : start close to zero */
            k_min = ppr->k_min_tau0 / pba->conformal_age;
        }
        else if (pba->sgnK == -1) {
            /* K<0 (open): start close to sqrt(-K)
             * (in transfer modules, for scalars, this will correspond to q close to zero;
             * for vectors and tensors, this value is even smaller than the minimum necessary value) */
            k_min = sqrt(-pba->K + pow(ppr->k_min_tau0 / pba->conformal_age / pth->angular_rescaling, 2));
        }
        else if (pba->sgnK == 1) {
            /* K>0 (closed): start from q=sqrt(k2+(1+m)K) equal to 3sqrt(K), i.e. k=sqrt((8-m)K) */
            k_min = sqrt((8. - 1.e-4) * pba->K);
        }

        /** - --> find k_max (as well as k_max_cmb[ppt->index_md_scalars], k_max_cl[ppt->index_md_scalars]) */

        k_rec = 2. * _PI_ / pth->rs_rec; /* comoving scale corresponding to sound horizon at recombination */

        k_max_cmb = k_min;
        k_max_cl = k_min;
        k_max = k_min;

        /* find k_max: */

        if ((ppt->has_pk_matter == _TRUE_) || (ppt->has_density_transfers == _TRUE_) || (ppt->has_velocity_transfers == _TRUE_))
            k_max = MAX(k_max, ppt->k_max_for_pk);

        if (ppt->has_nl_corrections_based_on_delta_m == _TRUE_)
            k_max = MAX(k_max, ppr->nonlinear_min_k_max);

        /** - --> test that result for k_min, k_max make sense */

        class_test(k_min < 0.,
                   pnlpt->error_message,
                   "buggy definition of k_min");

        class_test(k_max < 0.,
                   pnlpt->error_message,
                   "buggy definition of k_max");

        class_test(k_max < k_min,
                   pnlpt->error_message,
                   "buggy definition of k_min and/or k_max");

        /* if K>0, the transfer function will be calculated for discrete
       integer values of nu=3,4,5,... where nu=sqrt(k2+(1+m)K) and
       m=0,1,2 for scalars/vectors/tensors. However we are free to
       define in the perturbation module some arbitrary values of k:
       later on, the transfer module will interpolate at values of k
       corresponding exactly to integer values of nu. Hence, apart
       from the value of k_min and the step size in the vicinity of
       k_min, we define exactly the same sampling in the three cases
       K=0, K<0, K>0 */

        /* allocate array with, for the moment, the largest possible size */
        class_alloc(pnlpt->k,
                    ((int)((k_max_cmb - k_min) / k_rec / MIN(ppr->k_step_super, ppr->k_step_sub)) +
                     (int)(MAX(ppr->k_per_decade_for_pk, ppr->k_per_decade_for_bao) * log(k_max / k_min) / log(10.)) + 3) *
                        sizeof(double),
                    pnlpt->error_message);

        /* first value */

        index_k = 0;
        k = k_min;
        pnlpt->k[index_k] = k;
        index_k++;

        /* values until k_max_cmb[ppt->index_md_scalars] */

        while (k < k_max_cmb) {
            /* the linear step is not constant, it has a step-like shape,
         centered around the characteristic scale set by the sound
         horizon at recombination (associated to the comoving wavenumber
         k_rec) */

            step = (ppr->k_step_super + 0.5 * (tanh((k - k_rec) / k_rec / ppr->k_step_transition) + 1.) * (ppr->k_step_sub - ppr->k_step_super)) * k_rec;

            /* there is one other thing to take into account in the step
         size. There are two other characteristic scales that matter for
         the sampling: the Hubble scale today, k0=a0H0, and eventually
         curvature scale sqrt(|K|). We define "scale2" as the sum of the
         squared Hubble radius and squared curvature radius. We need to
         increase the sampling for k<sqrt(scale2), in order to get the
         first mutipoles accurate enough. The formula below reduces it
         gradually in the k-->0 limit, by up to a factor 10. The actual
         stepsize is still fixed by k_step_super, this is just a
         reduction factor. */

            scale2 = pow(pba->H0, 2) + fabs(pba->K); /* a_today=1 in v3 */

            step *= (k * k / scale2 + 1.) / (k * k / scale2 + 1. / ppr->k_step_super_reduction);

            class_test(step / k < ppr->smallest_allowed_variation,
                       pnlpt->error_message,
                       "k step =%e < machine precision : leads either to numerical error or infinite loop",
                       step * k_rec);

            k += step;

            class_test(k <= pnlpt->k[index_k - 1],
                       pnlpt->error_message,
                       "consecutive values of k should differ and should be in growing order");

            pnlpt->k[index_k] = k;

            index_k++;
        }

        pnlpt->k_size_cmb = index_k;

        /* values until k_max_cl[ppt->index_md_scalars] */

        while (k < k_max_cl) {
            k *= pow(10., 1. / (ppr->k_per_decade_for_pk + (ppr->k_per_decade_for_bao - ppr->k_per_decade_for_pk) * (1. - tanh(pow((log(k) - log(ppr->k_bao_center * k_rec)) / log(ppr->k_bao_width), 4)))));

            pnlpt->k[index_k] = k;
            index_k++;
        }

        pnlpt->k_size_cl = index_k;

        /* values until k_max */

        while (k < k_max) {
            k *= pow(10., 1. / (ppr->k_per_decade_for_pk + (ppr->k_per_decade_for_bao - ppr->k_per_decade_for_pk) * (1. - tanh(pow((log(k) - log(ppr->k_bao_center * k_rec)) / log(ppr->k_bao_width), 4)))));

            pnlpt->k[index_k] = k;
            index_k++;
        }

        pnlpt->k_size = index_k;

        class_realloc(pnlpt->k,
                      pnlpt->k_size * sizeof(double),
                      pnlpt->error_message);
    }

    /** - vector modes skipped */

    /** - tensor modes skipped */

    /** - If user asked for k_output_values, add those to all k lists: */
    if (ppt->k_output_values_num > 0) {
        /* Allocate storage */
        class_alloc(ppt->index_k_output_values, sizeof(double) * 1 * ppt->k_output_values_num, pnlpt->error_message);

        /** - --> Find indices in ppt->k[index_md] corresponding to 'k_output_values'.
        We are assuming that ppt->k is sorted and growing, and we have made sure
        that ppt->k_output_values is also sorted and growing.*/

        newk_size = pnlpt->k_size + ppt->k_output_values_num;

        class_alloc(tmp_k_list, sizeof(double) * newk_size, pnlpt->error_message);

        index_k = 0;
        index_k_output = 0;
        for (index_newk = 0; index_newk < newk_size; index_newk++) {
            /** - --> Decide if we should add k_output_value now. This has to be this complicated, since we
            can only compare the k-values when both indices are in range.*/
            if (index_k >= pnlpt->k_size)
                add_k_output_value = _TRUE_;
            else if (index_k_output >= ppt->k_output_values_num)
                add_k_output_value = _FALSE_;
            else if (ppt->k_output_values[index_k_output] < pnlpt->k[index_k])
                add_k_output_value = _TRUE_;
            else
                add_k_output_value = _FALSE_;

            if (add_k_output_value == _TRUE_) {
                tmp_k_list[index_newk] = ppt->k_output_values[index_k_output];
                ppt->index_k_output_values[0 * ppt->k_output_values_num + index_k_output] = index_newk;
                index_k_output++;
            } else {
                tmp_k_list[index_newk] = pnlpt->k[index_k];
                index_k++;
            }
        }

        free(pnlpt->k);
        pnlpt->k = tmp_k_list;
        pnlpt->k_size = newk_size;

        index_k = newk_size - 1;
        while (pnlpt->k[index_k] > k_max_cl)
            index_k--;
        pnlpt->k_size_cl = MIN(index_k + 2, pnlpt->k_size);

        index_k = newk_size - 1;
        while (pnlpt->k[index_k] > k_max_cmb)
            index_k--;
        pnlpt->k_size_cmb = MIN(index_k + 2, pnlpt->k_size);

        /** - --> The two MIN statements are here because in a normal run, the cl and cmb
          arrays contain a single k value larger than their respective k_max.
          We are mimicking this behavior. */
    }

    /** - finally, find the global k_min and k_max for the ensemble of all modes scalars, vectors, tensors) */

    pnlpt->k_min = _HUGE_;
    pnlpt->k_max = 0.;
    if (ppt->has_scalars == _TRUE_) {
        pnlpt->k_min = MIN(pnlpt->k_min, pnlpt->k[0]);                 /* first value, inferred from perturbations structure */
        pnlpt->k_max = MAX(pnlpt->k_max, pnlpt->k[pnlpt->k_size - 1]); /* last value, inferred from perturbations structure */
    }

    return _SUCCESS_;
}

/* === MODULE INITIALIZATION === */

/**
 * Initialize the non-linear PT module.
 *
 * Reads precomputed kernel matrices (M22, M13, IFG2, fNL),
 * allocates output arrays, extracts the linear P(k) at each
 * requested redshift, and calls nonlinear_pt_loop() to compute
 * the one-loop spectra.
 */
/* Offset tables for data-driven allocation and storage of P(k) arrays
 * and M22/M13 complex kernel matrices. These use offsetof() to enable
 * a single loop to allocate/initialize all struct members, replacing
 * ~100 individual allocation calls. */
#include <stddef.h>

/** Offsets of all ln_pk_* output arrays in struct nonlinear_pt (87 total) */
static const size_t ln_pk_offsets[] = {
    offsetof(struct nonlinear_pt, ln_pk_nl),
    offsetof(struct nonlinear_pt, ln_pk_Id2d2),
    offsetof(struct nonlinear_pt, ln_pk_Id2d2_2),
    offsetof(struct nonlinear_pt, ln_pk_Id2d2_4),
    offsetof(struct nonlinear_pt, ln_pk_Id2),
    offsetof(struct nonlinear_pt, ln_pk_IG2),
    offsetof(struct nonlinear_pt, ln_pk_Id2G2),
    offsetof(struct nonlinear_pt, ln_pk_Id2G2_2),
    offsetof(struct nonlinear_pt, ln_pk_Id2G2_4),
    offsetof(struct nonlinear_pt, ln_pk_IG2G2),
    offsetof(struct nonlinear_pt, ln_pk_IG2G2_2),
    offsetof(struct nonlinear_pt, ln_pk_IG2G2_4),
    offsetof(struct nonlinear_pt, ln_pk_IFG2),
    offsetof(struct nonlinear_pt, ln_pk_IFG2_0b1),
    offsetof(struct nonlinear_pt, ln_pk_IFG2_0),
    offsetof(struct nonlinear_pt, ln_pk_IFG2_2),
    offsetof(struct nonlinear_pt, ln_pk_CTR),
    offsetof(struct nonlinear_pt, ln_pk_CTR_0),
    offsetof(struct nonlinear_pt, ln_pk_CTR_2),
    offsetof(struct nonlinear_pt, ln_pk_CTR_4),
    offsetof(struct nonlinear_pt, ln_pk_Tree),
    offsetof(struct nonlinear_pt, ln_pk_Tree_0_vv),
    offsetof(struct nonlinear_pt, ln_pk_Tree_0_vd),
    offsetof(struct nonlinear_pt, ln_pk_Tree_0_dd),
    offsetof(struct nonlinear_pt, ln_pk_Tree_2_vv),
    offsetof(struct nonlinear_pt, ln_pk_Tree_2_vd),
    offsetof(struct nonlinear_pt, ln_pk_Tree_4_vv),
    offsetof(struct nonlinear_pt, ln_pk_0_vv),
    offsetof(struct nonlinear_pt, ln_pk_0_vd),
    offsetof(struct nonlinear_pt, ln_pk_0_dd),
    offsetof(struct nonlinear_pt, ln_pk_2_vv),
    offsetof(struct nonlinear_pt, ln_pk_2_vd),
    offsetof(struct nonlinear_pt, ln_pk_2_dd),
    offsetof(struct nonlinear_pt, ln_pk_4_vv),
    offsetof(struct nonlinear_pt, ln_pk_4_vd),
    offsetof(struct nonlinear_pt, ln_pk_4_dd),
    offsetof(struct nonlinear_pt, ln_pk_0_b1b2),
    offsetof(struct nonlinear_pt, ln_pk_0_b2),
    offsetof(struct nonlinear_pt, ln_pk_0_b1bG2),
    offsetof(struct nonlinear_pt, ln_pk_0_bG2),
    offsetof(struct nonlinear_pt, ln_pk_2_b1b2),
    offsetof(struct nonlinear_pt, ln_pk_2_b2),
    offsetof(struct nonlinear_pt, ln_pk_2_b1bG2),
    offsetof(struct nonlinear_pt, ln_pk_2_bG2),
    offsetof(struct nonlinear_pt, ln_pk_4_b2),
    offsetof(struct nonlinear_pt, ln_pk_4_bG2),
    offsetof(struct nonlinear_pt, ln_pk_4_b1b2),
    offsetof(struct nonlinear_pt, ln_pk_4_b1bG2),
    offsetof(struct nonlinear_pt, ln_pk_fNL_0_vv),
    offsetof(struct nonlinear_pt, ln_pk_fNL_0_vd),
    offsetof(struct nonlinear_pt, ln_pk_fNL_0_dd),
    offsetof(struct nonlinear_pt, ln_pk_fNL_2_vv),
    offsetof(struct nonlinear_pt, ln_pk_fNL_2_vd),
    offsetof(struct nonlinear_pt, ln_pk_fNL_2_dd),
    offsetof(struct nonlinear_pt, ln_pk_fNL_4_vv),
    offsetof(struct nonlinear_pt, ln_pk_fNL_4_vd),
    offsetof(struct nonlinear_pt, ln_pk_fNL_4_dd),
    offsetof(struct nonlinear_pt, ln_pk_fNL_0_vv_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNL_0_vd_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNL_0_dd_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNL_2_vv_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNL_2_vd_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNL_2_dd_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNL_4_vv_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNL_4_vd_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNL_4_dd_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_0_b1b2),
    offsetof(struct nonlinear_pt, ln_pk12_0_b2),
    offsetof(struct nonlinear_pt, ln_pk12_0_b1bG2),
    offsetof(struct nonlinear_pt, ln_pk12_0_bG2),
    offsetof(struct nonlinear_pt, ln_pk12_2_b1b2),
    offsetof(struct nonlinear_pt, ln_pk12_2_b2),
    offsetof(struct nonlinear_pt, ln_pk12_2_b1bG2),
    offsetof(struct nonlinear_pt, ln_pk12_2_bG2),
    offsetof(struct nonlinear_pt, ln_pk12_4_b1b2),
    offsetof(struct nonlinear_pt, ln_pk12_4_b2),
    offsetof(struct nonlinear_pt, ln_pk12_4_b1bG2),
    offsetof(struct nonlinear_pt, ln_pk12_4_bG2),
    offsetof(struct nonlinear_pt, ln_pk12_0_b1b2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_0_b2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_0_b1bG2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_0_bG2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_2_b1b2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_2_b2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_2_b1bG2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_2_bG2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_4_b1b2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_4_b2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_4_b1bG2_ortho),
    offsetof(struct nonlinear_pt, ln_pk12_4_bG2_ortho),
    offsetof(struct nonlinear_pt, ln_pk_nl_fNL),
    offsetof(struct nonlinear_pt, ln_pk_fNLd2),
    offsetof(struct nonlinear_pt, ln_pk_fNLG2),
    offsetof(struct nonlinear_pt, ln_pk_nl_fNL_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNLd2_ortho),
    offsetof(struct nonlinear_pt, ln_pk_fNLG2_ortho),
};
#define N_LN_PK (sizeof(ln_pk_offsets) / sizeof(ln_pk_offsets[0]))


static const size_t m22_complex_offsets[] = {
    offsetof(struct nonlinear_pt, M22_oneline_complex),
    offsetof(struct nonlinear_pt, M22basic_oneline_complex),
    offsetof(struct nonlinear_pt, M22_0_b1b2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_0_b2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_0_b1bG2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_0_bG2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_2_b1b2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_2_b2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_2_b1bG2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_2_bG2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_4_b2_oneline_complex),
    offsetof(struct nonlinear_pt, M22_4_bG2_oneline_complex),
    offsetof(struct nonlinear_pt, M_IG2G2),
    offsetof(struct nonlinear_pt, M_Id2),
    offsetof(struct nonlinear_pt, M_IG2),
    offsetof(struct nonlinear_pt, M_Id2G2),
    offsetof(struct nonlinear_pt, M22_oneline_0_vv_complex),
    offsetof(struct nonlinear_pt, M22_oneline_0_vd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_0_dd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_2_vv_complex),
    offsetof(struct nonlinear_pt, M22_oneline_2_vd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_2_dd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_4_vv_complex),
    offsetof(struct nonlinear_pt, M22_oneline_4_vd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_4_dd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_mu2_vd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_mu2_dd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_mu4_vd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_mu4_vv_complex),
    offsetof(struct nonlinear_pt, M22_oneline_mu4_dd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_mu6_vv_complex),
    offsetof(struct nonlinear_pt, M22_oneline_mu6_vd_complex),
    offsetof(struct nonlinear_pt, M22_oneline_mu8_complex),
    offsetof(struct nonlinear_pt, M12_oneline_complex),
    offsetof(struct nonlinear_pt, M12_oneline_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vv0_f2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vv0_f3),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vd0_f1),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vd0_f2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_dd0_f0),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_dd0_f1),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vv2_f3),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vd2_f2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vv4_f3),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vd4_f2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vv0_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vv0_f3_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vd0_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vd0_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_dd0_f0_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_dd0_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vv2_f3_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vd2_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vv4_f3_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_multipoles_vd4_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vd2_f1),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vd2_f2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_dd2_f1),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vv4_f2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vd4_f2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vv6_f3),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vd2_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vd2_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_dd2_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vv4_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vd4_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_matter_mu_powers_vv6_f3_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_bias_real_space_b2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_bias_real_space_bG2),
    offsetof(struct nonlinear_pt, M12_oneline_complex_bias_real_space_b2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_bias_real_space_bG2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_bias_multipoles_b2_vv0_f1),
    offsetof(struct nonlinear_pt, M12_oneline_complex_bias_multipoles_bG2_vv0_f1),
    offsetof(struct nonlinear_pt, M12_oneline_complex_bias_multipoles_b2_vv0_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_complex_bias_multipoles_bG2_vv0_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_0_vv_complex),
    offsetof(struct nonlinear_pt, M12_oneline_0_vd_complex),
    offsetof(struct nonlinear_pt, M12_oneline_0_dd_complex),
    offsetof(struct nonlinear_pt, M12_oneline_2_vv_complex),
    offsetof(struct nonlinear_pt, M12_oneline_2_vd_complex),
    offsetof(struct nonlinear_pt, M12_oneline_2_dd_complex),
    offsetof(struct nonlinear_pt, M12_oneline_4_vv_complex),
    offsetof(struct nonlinear_pt, M12_oneline_4_vd_complex),
    offsetof(struct nonlinear_pt, M12_oneline_mu2_vd_complex),
    offsetof(struct nonlinear_pt, M12_oneline_mu2_dd_complex),
    offsetof(struct nonlinear_pt, M12_oneline_mu4_vv_complex),
    offsetof(struct nonlinear_pt, M12_oneline_mu4_vd_complex),
    offsetof(struct nonlinear_pt, M12_oneline_mu6_vv_complex),
    offsetof(struct nonlinear_pt, M12_oneline_0_vv_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_0_vd_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_0_dd_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_2_vv_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_2_vd_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_2_dd_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_4_vv_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_4_vd_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_mu2_vd_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_mu2_dd_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_mu4_vv_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_mu4_vd_complex_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_mu6_vv_complex_ortho),
    offsetof(struct nonlinear_pt, M12_2_bG2_oneline_complex),
    offsetof(struct nonlinear_pt, M12_2_b2_oneline_complex),
    offsetof(struct nonlinear_pt, M12_0_bG2_oneline_complex),
    offsetof(struct nonlinear_pt, M12_0_b1bG2_oneline_complex),
    offsetof(struct nonlinear_pt, M12_0_b2_oneline_complex),
    offsetof(struct nonlinear_pt, M12_0_b1b2_oneline_complex),
    offsetof(struct nonlinear_pt, M12_2_bG2_oneline_complex_ortho),
    offsetof(struct nonlinear_pt, M12_2_b2_oneline_complex_ortho),
    offsetof(struct nonlinear_pt, M12_0_bG2_oneline_complex_ortho),
    offsetof(struct nonlinear_pt, M12_0_b1bG2_oneline_complex_ortho),
    offsetof(struct nonlinear_pt, M12_0_b2_oneline_complex_ortho),
    offsetof(struct nonlinear_pt, M12_0_b1b2_oneline_complex_ortho),
    offsetof(struct nonlinear_pt, M_fNLd2),
    offsetof(struct nonlinear_pt, M_fNLG2),
    offsetof(struct nonlinear_pt, M_fNLd2_ortho),
    offsetof(struct nonlinear_pt, M_fNLG2_ortho),
};
#define N_M22_COMPLEX (sizeof(m22_complex_offsets) / sizeof(m22_complex_offsets[0]))

static const size_t m13_complex_offsets[] = {
    offsetof(struct nonlinear_pt, M13_oneline_complex),
    offsetof(struct nonlinear_pt, IFG2_oneline_complex),
    offsetof(struct nonlinear_pt, M13_0_vv_oneline_complex),
    offsetof(struct nonlinear_pt, M13_0_vd_oneline_complex),
    offsetof(struct nonlinear_pt, M13_0_dd_oneline_complex),
    offsetof(struct nonlinear_pt, M13_2_vv_oneline_complex),
    offsetof(struct nonlinear_pt, M13_2_vd_oneline_complex),
    offsetof(struct nonlinear_pt, M13_2_dd_oneline_complex),
    offsetof(struct nonlinear_pt, M13_4_vv_oneline_complex),
    offsetof(struct nonlinear_pt, M13_4_vd_oneline_complex),
    offsetof(struct nonlinear_pt, M13_mu2_dd_oneline_complex),
    offsetof(struct nonlinear_pt, M13_mu2_vd_oneline_complex),
    offsetof(struct nonlinear_pt, M13_mu4_vv_oneline_complex),
    offsetof(struct nonlinear_pt, M13_mu4_vd_oneline_complex),
    offsetof(struct nonlinear_pt, M13_mu6_oneline_complex),
};
#define N_M13_COMPLEX (sizeof(m13_complex_offsets) / sizeof(m13_complex_offsets[0]))

/* Offsets of real-valued M12 kernel matrices (21 regular + 21 ortho = 42 total) */
static const size_t m12_real_offsets[] = {
    offsetof(struct nonlinear_pt, M12_oneline),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vv0_f2),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vv0_f3),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vd0_f1),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vd0_f2),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_dd0_f0),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_dd0_f1),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vv2_f3),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vd2_f2),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vv4_f3),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vd4_f2),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vd2_f1),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vd2_f2),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_dd2_f1),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vv4_f2),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vd4_f2),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vv6_f3),
    offsetof(struct nonlinear_pt, M12_oneline_bias_real_space_b2),
    offsetof(struct nonlinear_pt, M12_oneline_bias_real_space_bG2),
    offsetof(struct nonlinear_pt, M12_oneline_bias_multipoles_b2_vv0_f1),
    offsetof(struct nonlinear_pt, M12_oneline_bias_multipoles_bG2_vv0_f1),
    offsetof(struct nonlinear_pt, M12_oneline_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vv0_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vv0_f3_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vd0_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vd0_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_dd0_f0_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_dd0_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vv2_f3_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vd2_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vv4_f3_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_multipoles_vd4_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vd2_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vd2_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_dd2_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vv4_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vd4_f2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_matter_mu_powers_vv6_f3_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_bias_real_space_b2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_bias_real_space_bG2_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_bias_multipoles_b2_vv0_f1_ortho),
    offsetof(struct nonlinear_pt, M12_oneline_bias_multipoles_bG2_vv0_f1_ortho),
};
#define N_M12_REAL (sizeof(m12_real_offsets) / sizeof(m12_real_offsets[0]))

/* Compact spline setup: compute spline derivative table */
#define SPLINE_SETUP(x, n, data, dd, mode)                                    \
    class_call(array_spline_table_columns(x, n, data, 1, dd, mode,            \
                                          pnlpt->error_message),              \
               pnlpt->error_message, pnlpt->error_message)

/* Compact spline interpolation: evaluate spline at a single point */
#define SPLINE_EVAL(x, n, data, dd, xval, last_idx, out)                      \
    class_call(array_interpolate_spline(x, n, data, dd, 1, xval,              \
                                        last_idx, out, 1,                     \
                                        pnlpt->error_message),               \
               pnlpt->error_message, pnlpt->error_message)



/* DW: QCDM: write the CLASS-PT exact-QS outputs to plain files.
 * Standard CLASS output.c writes *_pk_nl.dat only for the Fourier/Halofit
 * nonlinear module. CLASS-PT keeps its spectra in struct nonlinear_pt, so
 * we write the QCDM/QSCF PT spectra directly here. */
static int qcdm_qscf_write_one_z_output_files(
    struct background *pba,
    struct nonlinear_pt *pnlpt,
    int i_z,
    double *pk_Tree,
    double *pk_nl,
    /* DW: exact-time unit-f_NL matter P12 templates for output auditing. */
    double *pk_nl_fNL,
    double *pk_nl_fNL_ortho,
    double *pk_CTR,
    double *pk_Id2,
    double *pk_Id2d2,
    double *pk_IG2,
    double *pk_IFG2,
    double *pk_l_0_dd,
    double *pk_l_2_dd,
    double *pk_l_4_dd,
    /* DW: exact-time unit-f_NL redshift-space multipole templates. */
    double *pk_l_fNL_0_dd,
    double *pk_l_fNL_2_dd,
    double *pk_l_fNL_4_dd,
    double *pk_l_fNL_0_dd_ortho,
    double *pk_l_fNL_2_dd_ortho,
    double *pk_l_fNL_4_dd_ortho,
    double *pk_Tree_0_dd,
    double *pk_Tree_2_vd,
    double *pk_Tree_4_vv)
{
    FileName fname;
    char zsuffix[16];
    FILE *fp;
    int index_k;
    const double large_matter = 5.e4;
    const double large_fNL = 5.e4;
    const double large_big = 1.e6;
    double h3 = pba->h*pba->h*pba->h;

    if (pnlpt->z_pk_num == 1) zsuffix[0] = '\0';
    else class_sprintf(zsuffix,"z%d_",i_z+1);

    if (pnlpt->qcdm_write_pk_nl == _TRUE_) {
        class_sprintf(fname,"%s%sqcdm_pt_pk_nl.dat",pnlpt->qcdm_pt_output_root,zsuffix);
        fp = fopen(fname,"w");
        class_test(fp == NULL, pnlpt->error_message, "Could not open %s for writing", fname);
        fprintf(fp,"# QCDM/QSCF exact-QS CLASS-PT real-space matter spectrum at z = %.12g\n",pnlpt->z_pk[i_z]);
        fprintf(fp,"# DW: cb=%s; P12 columns are unit-f_NL equilateral/orthogonal templates; PPNG uses amplitudes %.12g and %.12g.\n",
                pnlpt->cb==_TRUE_?"yes":"no",pnlpt->qcdm_fNL_equilateral,pnlpt->qcdm_fNL_orthogonal);
        fprintf(fp,"# k[h/Mpc] P_tree P_1loop_Gaussian P12eq_unit P12ortho_unit PPNG P_tree_plus_loop_PNG P_ct_basis [all P in (Mpc/h)^3]\n");
        for (index_k=0; index_k<pnlpt->k_size; index_k++) {
            double k_h = pnlpt->k[index_k]/pba->h;
            double Ptree = pk_Tree[index_k];
            double Ploop = pk_nl[index_k] - large_matter;
            double Peq = pk_nl_fNL[index_k]-large_fNL;
            double Portho = pk_nl_fNL_ortho[index_k]-large_fNL;
            double Ppng=pnlpt->qcdm_fNL_equilateral*Peq+pnlpt->qcdm_fNL_orthogonal*Portho;
            double Pnl = Ptree + Ploop + Ppng;
            double Pct = pk_CTR[index_k];
            class_test((qcdm_qscf_finite_double(k_h) == 0) || (qcdm_qscf_finite_double(Ptree) == 0) ||
                       (qcdm_qscf_finite_double(Ploop) == 0) || (qcdm_qscf_finite_double(Peq) == 0) ||
                       (qcdm_qscf_finite_double(Portho) == 0) || (qcdm_qscf_finite_double(Pnl) == 0) ||
                       (qcdm_qscf_finite_double(Pct) == 0),
                       pnlpt->error_message,
                       "Non-finite QCDM exact-QS pk_nl/PNG output at z index %d, k index %d",
                       i_z,index_k);
            fprintf(fp,"%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n",
                    k_h,Ptree*h3,Ploop*h3,Peq*h3,Portho*h3,Ppng*h3,Pnl*h3,Pct*h3);
        }
        fclose(fp);
    }

    if (pnlpt->qcdm_write_twofluid_templates == _TRUE_) {
        class_sprintf(fname,"%s%sqcdm_pt_twofluid_templates.dat",pnlpt->qcdm_pt_output_root,zsuffix);
        fp = fopen(fname,"w");
        class_test(fp == NULL, pnlpt->error_message, "Could not open %s for writing", fname);
        fprintf(fp,"# QCDM/QSCF exact-QS two-fluid template building blocks at z = %.12g\n",pnlpt->z_pk[i_z]);
        fprintf(fp,"# k [h/Mpc]    P_mr_tree(Id2)    P_rr_tree(Id2d2)    P_mtheta_r_tree(IG2)    P_thetar_thetar_tree(IFG2)\n");
        for (index_k=0; index_k<pnlpt->k_size; index_k++) {
            double k_h = pnlpt->k[index_k]/pba->h;
            double Pmr = pk_Id2[index_k];
            double Prr = pk_Id2d2[index_k] - large_big;
            double Pmtr = pk_IG2[index_k];
            double Ptrtr = pk_IFG2[index_k];
            class_test((qcdm_qscf_finite_double(k_h) == 0) || (qcdm_qscf_finite_double(Pmr) == 0) ||
                       (qcdm_qscf_finite_double(Prr) == 0) || (qcdm_qscf_finite_double(Pmtr) == 0) ||
                       (qcdm_qscf_finite_double(Ptrtr) == 0),
                       pnlpt->error_message,
                       "Non-finite QCDM exact-QS two-fluid template output at z index %d, k index %d: k_h=%e Pmr=%e Prr=%e Pmtr=%e Ptrtr=%e",
                       i_z,index_k,k_h,Pmr,Prr,Pmtr,Ptrtr);
            fprintf(fp,"%.12e %.12e %.12e %.12e %.12e\n",k_h,Pmr*h3,Prr*h3,Pmtr*h3,Ptrtr*h3);
        }
        fclose(fp);
    }

    if ((pnlpt->rsd == rsd_yes) && (pnlpt->qcdm_write_multipoles == _TRUE_)) {
        class_sprintf(fname,"%s%sqcdm_pt_multipoles.dat",pnlpt->qcdm_pt_output_root,zsuffix);
        fp = fopen(fname,"w");
        class_test(fp == NULL, pnlpt->error_message, "Could not open %s for writing", fname);
        fprintf(fp,"# QCDM/QSCF exact-QS two-fluid tracer multipoles at z = %.12g\n",pnlpt->z_pk[i_z]);
        /* DW: label total, AP-remapped tree, and exact-time PNG unit
         * templates separately for full-shape validation. */
        fprintf(fp,"# DW: AP=%s, cb=%s, H_true/H_fid=%.12g, D_M,true/D_M,fid=%.12g, Omfid=%.12g, fNL_equilateral=%.12g, fNL_orthogonal=%.12g.\n",
                pnlpt->AP_effect==AP_effect_yes?"yes":"no",pnlpt->cb==_TRUE_?"yes":"no",
                pnlpt->hratio_array[i_z],pnlpt->Dratio_array[i_z],pnlpt->OmfidAP,
                pnlpt->qcdm_fNL_equilateral,pnlpt->qcdm_fNL_orthogonal);
        fprintf(fp,"# k[h/Mpc] P0_total P2_total P4_total P0_tree P2_tree P4_tree P0eq_unit P2eq_unit P4eq_unit P0ortho_unit P2ortho_unit P4ortho_unit P0PNG P2PNG P4PNG [all P in (Mpc/h)^3]\n");
        for (index_k=0; index_k<pnlpt->k_size; index_k++) {
            double k_h = pnlpt->k[index_k]/pba->h;
            double P0 = pk_l_0_dd[index_k] - large_big;
            double P2 = pk_l_2_dd[index_k] - large_big;
            double P4 = pk_l_4_dd[index_k] - large_big;
            double T0 = pk_Tree_0_dd[index_k] - large_big;
            double T2 = pk_Tree_2_vd[index_k] - large_big;
            double T4 = pk_Tree_4_vv[index_k] - large_big;
            double E0=pk_l_fNL_0_dd[index_k]-large_fNL;
            double E2=pk_l_fNL_2_dd[index_k]-large_fNL;
            double E4=pk_l_fNL_4_dd[index_k]-large_fNL;
            double O0=pk_l_fNL_0_dd_ortho[index_k]-large_fNL;
            double O2=pk_l_fNL_2_dd_ortho[index_k]-large_fNL;
            double O4=pk_l_fNL_4_dd_ortho[index_k]-large_fNL;
            double N0=pnlpt->qcdm_fNL_equilateral*E0+pnlpt->qcdm_fNL_orthogonal*O0;
            double N2=pnlpt->qcdm_fNL_equilateral*E2+pnlpt->qcdm_fNL_orthogonal*O2;
            double N4=pnlpt->qcdm_fNL_equilateral*E4+pnlpt->qcdm_fNL_orthogonal*O4;
            class_test((qcdm_qscf_finite_double(k_h) == 0) || (qcdm_qscf_finite_double(P0) == 0) ||
                       (qcdm_qscf_finite_double(P2) == 0) || (qcdm_qscf_finite_double(P4) == 0) ||
                       (qcdm_qscf_finite_double(T0) == 0) || (qcdm_qscf_finite_double(T2) == 0) ||
                       (qcdm_qscf_finite_double(T4) == 0) || (qcdm_qscf_finite_double(E0) == 0) ||
                       (qcdm_qscf_finite_double(O0) == 0),
                       pnlpt->error_message,
                       "Non-finite QCDM exact-QS AP/PNG multipole output at z index %d, k index %d",
                       i_z,index_k);
            fprintf(fp,"%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n",
                    k_h,P0*h3,P2*h3,P4*h3,T0*h3,T2*h3,T4*h3,
                    E0*h3,E2*h3,E4*h3,O0*h3,O2*h3,O4*h3,N0*h3,N2*h3,N4*h3);
        }
        fclose(fp);
    }

    if (pnlpt->nonlinear_pt_verbose > 0) {
/* DW: add the explicitly decomposed real-space two-fluid tracer output switch. */
        printf("DW/QCDM-QSCF: exact-QS PT output switches at root %s%s: pk_nl=%s, tracer_real=%s, templates=%s, multipoles=%s\n",
               pnlpt->qcdm_pt_output_root, zsuffix,
               pnlpt->qcdm_write_pk_nl == _TRUE_ ? "yes" : "no",
/* DW: add the explicitly decomposed real-space two-fluid tracer output switch. */
               ((pnlpt->bias == bias_yes) && (pnlpt->qcdm_write_tracer_real == _TRUE_)) ? "yes" : "no",
               pnlpt->qcdm_write_twofluid_templates == _TRUE_ ? "yes" : "no",
               ((pnlpt->rsd == rsd_yes) && (pnlpt->qcdm_write_multipoles == _TRUE_)) ? "yes" : "no");
    }

    return _SUCCESS_;
}

int nonlinear_pt_init(
    struct precision *ppr,
    struct background *pba,
    struct thermodynamics *pth,
    struct perturbations *ppt,
    struct primordial *ppm,
/* DW: pass the initialized Fourier module into nonlinear_pt so the physical CLASS linear matter spectrum is used. */
    struct fourier *pfo,
    struct nonlinear_pt *pnlpt) {
    int index_ncdm;
    int index_k;
    int index_tau;
    double *pk_l;
    double *pk_l_0_vv;
    double *pk_l_0_vd;
    double *pk_l_0_dd;
    double *pk_l_2_vv;
    double *pk_l_2_vd;
    double *pk_l_2_dd;
    double *pk_l_4_vv;
    double *pk_l_4_vd;
    double *pk_l_4_dd;
    double *pk_nl;
    double *pk_Id2d2;
    double *pk_Id2d2_2;
    double *pk_Id2d2_4;
    double *pk_l_0_b1b2;
    double *pk_l_0_b2;
    double *pk_l_0_b1bG2;
    double *pk_l_0_bG2;
    double *pk_l_2_b1b2;
    double *pk_l_2_b2;
    double *pk_l_2_b1bG2;
    double *pk_l_2_bG2;
    double *pk_l_4_b2;
    double *pk_l_4_bG2;
    double *pk_l_4_b1b2;
    double *pk_l_4_b1bG2;
    double *pk_Id2;
    double *pk_IG2;
    double *pk_Id2G2;
    double *pk_Id2G2_2;
    double *pk_Id2G2_4;
    double *pk_IG2G2;
    double *pk_IG2G2_2;
    double *pk_IG2G2_4;
    double *pk_IFG2;
    double *pk_IFG2_0b1;
    double *pk_IFG2_0;
    double *pk_IFG2_2;
    double *pk_CTR;
    double *pk_CTR_0;
    double *pk_CTR_2;
    double *pk_CTR_4;
    double *pk_Tree;
    double *pk_Tree_0_vv;
    double *pk_Tree_0_vd;
    double *pk_Tree_0_dd;
    double *pk_Tree_2_vv;
    double *pk_Tree_2_vd;
    double *pk_Tree_4_vv;
    double *lnk_l;
    double *lnpk_l;
    double *ddlnpk_l;
    double *tau_req;

    /* Primordial P(k) arrays, used to extract the transfer function T(k) for fNL */
    double *pPRIMk_l;
    double *lnpPRIMk_l;
    double *ddlnpPRIMk_l;
    double *pk_l_fNL_0_vv;
    double *pk_l_fNL_0_vd;
    double *pk_l_fNL_0_dd;
    double *pk_l_fNL_2_vv;
    double *pk_l_fNL_2_vd;
    double *pk_l_fNL_2_dd;
    double *pk_l_fNL_4_vv;
    double *pk_l_fNL_4_vd;
    double *pk_l_fNL_4_dd;
    double *pk_l_fNL_0_vv_ortho;
    double *pk_l_fNL_0_vd_ortho;
    double *pk_l_fNL_0_dd_ortho;
    double *pk_l_fNL_2_vv_ortho;
    double *pk_l_fNL_2_vd_ortho;
    double *pk_l_fNL_2_dd_ortho;
    double *pk_l_fNL_4_vv_ortho;
    double *pk_l_fNL_4_vd_ortho;
    double *pk_l_fNL_4_dd_ortho;
    double *pk12_l_0_b1b2;
    double *pk12_l_0_b2;
    double *pk12_l_0_b1bG2;
    double *pk12_l_0_bG2;
    double *pk12_l_2_b1b2;
    double *pk12_l_2_b2;
    double *pk12_l_2_b1bG2;
    double *pk12_l_2_bG2;
    double *pk12_l_4_b1b2;
    double *pk12_l_4_b2;
    double *pk12_l_4_b1bG2;
    double *pk12_l_4_bG2;
    double *pk12_l_0_b1b2_ortho;
    double *pk12_l_0_b2_ortho;
    double *pk12_l_0_b1bG2_ortho;
    double *pk12_l_0_bG2_ortho;
    double *pk12_l_2_b1b2_ortho;
    double *pk12_l_2_b2_ortho;
    double *pk12_l_2_b1bG2_ortho;
    double *pk12_l_2_bG2_ortho;
    double *pk12_l_4_b1b2_ortho;
    double *pk12_l_4_b2_ortho;
    double *pk12_l_4_b1bG2_ortho;
    double *pk12_l_4_bG2_ortho;
    double *pk_nl_fNL;
    double *pk_fNLd2;
    double *pk_fNLG2;
    double *pk_nl_fNL_ortho;
    double *pk_fNLd2_ortho;
    double *pk_fNLG2_ortho;

    short print_warning = _FALSE_;
    int last_index;
    double a, z;

    last_index = 0;

    /* DW: each nonlinear_pt_init call defines a new cosmology.  Keep the
     * exact kernel cache across the several requested z_pk values within
     * this call, but force a reset before the first redshift so kernels can
     * never leak between different cosmological parameter points. */
    pnlpt->qcdm_kernel_cache_started = _FALSE_;

    /* DW: discard any previous fast-tracer snapshot before rebuilding the
     * PT tables.  This also makes recompute_nonlinear_pt() safe. */
    free(pnlpt->qcdm_fast_kdisc);   pnlpt->qcdm_fast_kdisc = NULL;
    free(pnlpt->qcdm_fast_Pbin);    pnlpt->qcdm_fast_Pbin = NULL;
    free(pnlpt->qcdm_fast_Pphi);    pnlpt->qcdm_fast_Pphi = NULL;
    free(pnlpt->qcdm_fast_Mphi);    pnlpt->qcdm_fast_Mphi = NULL;
    free(pnlpt->qcdm_fast_has_png); pnlpt->qcdm_fast_has_png = NULL;
    free(pnlpt->qcdm_fast_z_ready); pnlpt->qcdm_fast_z_ready = NULL;
    pnlpt->qcdm_fast_nmax = 0;
    pnlpt->qcdm_fast_tracer_ready = _FALSE_;
    qcdm_tf_ir_fast_cache_reset();

    /* Opt-in exact-QS F1-only diagnostic.  With non_linear=none this avoids
     * F2/F3, all PT loops, AP, IR, and galaxy multipoles.  Normal runs are
     * unaffected unless the environment variable is explicitly supplied. */
    {
        const char *relvel_out = getenv("QCDM_RELVEL_OUT");
        if (relvel_out != NULL && relvel_out[0] != '\0') {
            class_test(pnlpt->method != nlpt_none,pnlpt->error_message,
                       "QCDM_RELVEL_OUT requires non_linear=none (F1-only mode)");
            class_call(qcdm_qscf_dump_relvel_f1(pba,pnlpt),
                       pnlpt->error_message,pnlpt->error_message);
        }
    }

    /** Summary
     *
     * (a) First deal with the case where non non-linear corrections requested */

    if (pnlpt->method == nlpt_none) {
        if (pnlpt->nonlinear_pt_verbose > 0)
            printf("No non-linear spectra requested. Nonlinear PT module skipped.\n");
    }

    /** (b) Compute the non-linear spectrum */

    else if (pnlpt->method == nlpt_spt) {
        if (pnlpt->nonlinear_pt_verbose > 0)
            printf("Computing non-linear matter and bias power spectra in perturbation theory.\n");

        /** - copy list of (k,tau) from perturbation module */

        pnlpt->index_md_scalars = ppt->index_md_scalars;
        pnlpt->ic_size = ppt->ic_size[ppt->index_md_scalars];
        pnlpt->tp_size = ppt->tp_size[ppt->index_md_scalars];

        if (ppt->has_cls == _TRUE_ && pnlpt->fast_output == _TRUE_) {
            class_call(perturb_get_k_list_nl(ppr,
                                             pba,
                                             pth,
                                             ppt,
                                             pnlpt),
                       pnlpt->error_message,
                       pnlpt->error_message);
        } else {
            pnlpt->k_size = ppt->k_size[ppt->index_md_scalars];
            class_alloc(pnlpt->k, pnlpt->k_size * sizeof(double), pnlpt->error_message);
            for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
                pnlpt->k[index_k] = ppt->k[ppt->index_md_scalars][index_k];
            }
        }

        pnlpt->ln_k_size = pnlpt->k_size;
        class_alloc(pnlpt->ln_k, pnlpt->ln_k_size * sizeof(double), pnlpt->error_message);
        for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
            pnlpt->ln_k[index_k] = log(pnlpt->k[index_k]);
        }

        pnlpt->tau_size = ppt->tau_size;
        class_alloc(pnlpt->tau, pnlpt->tau_size * sizeof(double), pnlpt->error_message);
        class_alloc(pnlpt->ln_tau, pnlpt->tau_size * sizeof(double), pnlpt->error_message);
        for (index_tau = 0; index_tau < pnlpt->tau_size; index_tau++) {
            pnlpt->tau[index_tau] = ppt->tau_sampling[index_tau];
            pnlpt->ln_tau[index_tau] = log(ppt->tau_sampling[index_tau]);
        }

        /* --- Allocate output arrays for all one-loop spectra --- */
        class_alloc(pnlpt->nl_corr_density, pnlpt->tau_size * ppt->k_size[pnlpt->index_md_scalars] * sizeof(double), pnlpt->error_message);
        {
            double **_pk_ptrs[] = {
                &pk_l, &pk_l_0_vv, &pk_l_0_vd, &pk_l_0_dd,
                &pk_l_2_vv, &pk_l_2_vd, &pk_l_2_dd, &pk_l_4_vv, &pk_l_4_vd, &pk_l_4_dd,
                &pk_l_0_b1b2, &pk_l_0_b2, &pk_l_0_b1bG2, &pk_l_0_bG2,
                &pk_l_2_b1b2, &pk_l_2_b2, &pk_l_2_b1bG2, &pk_l_2_bG2,
                &pk_l_4_b2, &pk_l_4_bG2, &pk_l_4_b1b2, &pk_l_4_b1bG2,
                &pk_nl, &pk_Id2d2, &pk_Id2d2_2, &pk_Id2d2_4,
                &pk_Id2, &pk_IG2, &pk_Id2G2, &pk_Id2G2_2, &pk_Id2G2_4,
                &pk_IG2G2, &pk_IG2G2_2, &pk_IG2G2_4,
                &pk_IFG2, &pk_IFG2_0b1, &pk_IFG2_0, &pk_IFG2_2,
                &pk_CTR, &pk_CTR_0, &pk_CTR_2, &pk_CTR_4,
                &pk_Tree, &pk_Tree_0_vv, &pk_Tree_0_vd, &pk_Tree_0_dd,
                &pk_Tree_2_vv, &pk_Tree_2_vd, &pk_Tree_4_vv,
                &lnk_l, &lnpk_l, &ddlnpk_l, &pPRIMk_l, &lnpPRIMk_l, &ddlnpPRIMk_l,
                &pk_l_fNL_0_vv, &pk_l_fNL_0_vd, &pk_l_fNL_0_dd,
                &pk_l_fNL_2_vv, &pk_l_fNL_2_vd, &pk_l_fNL_2_dd,
                &pk_l_fNL_4_vv, &pk_l_fNL_4_vd, &pk_l_fNL_4_dd,
                &pk_l_fNL_0_vv_ortho, &pk_l_fNL_0_vd_ortho, &pk_l_fNL_0_dd_ortho,
                &pk_l_fNL_2_vv_ortho, &pk_l_fNL_2_vd_ortho, &pk_l_fNL_2_dd_ortho,
                &pk_l_fNL_4_vv_ortho, &pk_l_fNL_4_vd_ortho, &pk_l_fNL_4_dd_ortho,
                &pk12_l_0_b1b2, &pk12_l_0_b2, &pk12_l_0_b1bG2, &pk12_l_0_bG2,
                &pk12_l_2_b1b2, &pk12_l_2_b2, &pk12_l_2_b1bG2, &pk12_l_2_bG2,
                &pk12_l_4_b1b2, &pk12_l_4_b2, &pk12_l_4_b1bG2, &pk12_l_4_bG2,
                &pk12_l_0_b1b2_ortho, &pk12_l_0_b2_ortho, &pk12_l_0_b1bG2_ortho, &pk12_l_0_bG2_ortho,
                &pk12_l_2_b1b2_ortho, &pk12_l_2_b2_ortho, &pk12_l_2_b1bG2_ortho, &pk12_l_2_bG2_ortho,
                &pk12_l_4_b1b2_ortho, &pk12_l_4_b2_ortho, &pk12_l_4_b1bG2_ortho, &pk12_l_4_bG2_ortho,
                &pk_nl_fNL, &pk_fNLd2, &pk_fNLG2,
                &pk_nl_fNL_ortho, &pk_fNLd2_ortho, &pk_fNLG2_ortho
            };
            const size_t _pk_sz = pnlpt->k_size * sizeof(double);
            for (int ai_ = 0; ai_ < 103; ai_++)
                class_alloc(*_pk_ptrs[ai_], _pk_sz, pnlpt->error_message);
        }
        class_alloc(tau_req, pnlpt->z_pk_num * sizeof(double), pnlpt->error_message);

        int i_z = 0;
        for (i_z = 0; i_z < pnlpt->z_pk_num; i_z++) {
            class_call(background_tau_of_z(pba, pnlpt->z_pk[i_z], &tau_req[i_z]),
                       pba->error_message,
                       pnlpt->error_message);
        }

        /* --- Read precomputed PT kernel matrices from disk --- */
        /* M22 (P22 kernels), M13 (P13 kernels), IFG2 (bias),
         * M12 (fNL transfer), and associated RSD/bias/multipole matrices.
         * Matrix size is set by ppr->nmax_nlpt (128, 256, 512, or 1024). */

        char file2openM22[256];
        char file2openM13[256];
        char file2openM22basic[256];
        char file2openM13basic[256];

        int has_fNL = 0;

        if (pnlpt->fNL_equil_ortho_switch == fNL_equil_ortho_yes) {
            has_fNL = 1;
        }

        /* Determine resolution string for matrix file paths */
        const char *Nstr;
        const char *Nstr_M22;
        if (ppr->nmax_nlpt == 128) {
            Nstr = "N128";
            Nstr_M22 = "N128";
        }
        else if (ppr->nmax_nlpt == 512) {
            Nstr = "N512";
            Nstr_M22 = "N512";
        } else {
            Nstr = "N256";
            Nstr_M22 = "N256_packed";
        }

        sprintf(file2openM22, "%s/pt_matrices/M22oneline_%s.dat", __CLASSDIR__, Nstr_M22);
        sprintf(file2openM13, "%s/pt_matrices/M13oneline_%s.dat", __CLASSDIR__, Nstr);
        sprintf(file2openM22basic, "%s/pt_matrices/M22basiconeline_%s.dat", __CLASSDIR__, Nstr_M22);
        sprintf(file2openM13basic, "%s/pt_matrices/IFG2oneline_%s.dat", __CLASSDIR__, Nstr);


        /* --- Macros for matrix I/O and real-to-complex conversion --- */

#define LOAD_MATRIX(array, filepath, size)                                   \
    do {                                                                     \
        class_alloc(array, (size) * sizeof(double), pnlpt->error_message);   \
        char binpath_[512];                                                  \
        snprintf(binpath_, sizeof(binpath_), "%s.bin", filepath);            \
        FILE *fptr_lm_ = fopen(binpath_, "rb");                              \
        if (fptr_lm_ != NULL) {                                              \
            size_t nread_ = fread((array), sizeof(double), (size), fptr_lm_);\
            fclose(fptr_lm_);                                                \
            if ((int)nread_ != (size)) {                                     \
                printf("Error: binary read %s: got %zu of %d\n",             \
                       binpath_, nread_, (size));                             \
                exit(1);                                                     \
            }                                                                \
        } else {                                                             \
            fptr_lm_ = fopen(filepath, "r");                                 \
            if (fptr_lm_ == NULL) {                                          \
                printf("Error: cannot open %s\n", filepath);                 \
                exit(1);                                                     \
            }                                                                \
            for (int idx_lm_ = 0; idx_lm_ < (size); idx_lm_++)              \
                fscanf(fptr_lm_, "%lf", &(array)[idx_lm_]);                  \
            fclose(fptr_lm_);                                                \
        }                                                                    \
    } while (0)

#define CONVERT_REAL_TO_COMPLEX_M22(dest, src, Np1)                                         \
    for (count = 0; count < (Np1) * ((Np1) + 1) / 2; count++)                               \
    {                                                                                       \
        (dest)[count] = (src)[count] + _Complex_I * (src)[count + (Np1) * ((Np1) + 1) / 2]; \
    }

#define CONVERT_REAL_TO_COMPLEX_M13(dest, src, Np1)                       \
    for (count = 0; count < (Np1); count++)                               \
    {                                                                     \
        (dest)[count] = (src)[count] + _Complex_I * (src)[count + (Np1)]; \
    }

/* Store log(pk) into pnlpt struct array at redshift slice i_z */
#define STORE_LN_PK(member, pk_arr) \
    pnlpt->member[i_z * pnlpt->k_size + index_k] = log(pk_arr[index_k])

        LOAD_MATRIX(pnlpt->M22_oneline, file2openM22, ((ppr->nmax_nlpt + 1) * (ppr->nmax_nlpt + 2)));

        LOAD_MATRIX(pnlpt->M13_oneline, file2openM13, ((ppr->nmax_nlpt + 1) * 2));

        LOAD_MATRIX(pnlpt->M22basic_oneline, file2openM22basic, ((ppr->nmax_nlpt + 1) * (ppr->nmax_nlpt + 2)));

        LOAD_MATRIX(pnlpt->IFG2_oneline, file2openM13basic, ((ppr->nmax_nlpt + 1) * 2));

        /* M12 matrix loading — bundled binary for fNL, zero-alloc otherwise */
        {
            const int m12_size = (ppr->nmax_nlpt + 1) * (ppr->nmax_nlpt + 2);
            const int n_m12 = 21;  /* must match M12_PAIRS in bundle_m12.py */
            double **m12_regular[21] = {
                &pnlpt->M12_oneline, &pnlpt->M12_oneline_matter_multipoles_vv0_f2,
                &pnlpt->M12_oneline_matter_multipoles_vv0_f3, &pnlpt->M12_oneline_matter_multipoles_vd0_f1,
                &pnlpt->M12_oneline_matter_multipoles_vd0_f2, &pnlpt->M12_oneline_matter_multipoles_dd0_f0,
                &pnlpt->M12_oneline_matter_multipoles_dd0_f1, &pnlpt->M12_oneline_matter_multipoles_vv2_f3,
                &pnlpt->M12_oneline_matter_multipoles_vd2_f2, &pnlpt->M12_oneline_matter_multipoles_vv4_f3,
                &pnlpt->M12_oneline_matter_multipoles_vd4_f2, &pnlpt->M12_oneline_matter_mu_powers_vd2_f1,
                &pnlpt->M12_oneline_matter_mu_powers_vd2_f2, &pnlpt->M12_oneline_matter_mu_powers_dd2_f1,
                &pnlpt->M12_oneline_matter_mu_powers_vv4_f2, &pnlpt->M12_oneline_matter_mu_powers_vd4_f2,
                &pnlpt->M12_oneline_matter_mu_powers_vv6_f3, &pnlpt->M12_oneline_bias_real_space_b2,
                &pnlpt->M12_oneline_bias_real_space_bG2, &pnlpt->M12_oneline_bias_multipoles_b2_vv0_f1,
                &pnlpt->M12_oneline_bias_multipoles_bG2_vv0_f1,
            };
            double **m12_ortho[21] = {
                &pnlpt->M12_oneline_ortho, &pnlpt->M12_oneline_matter_multipoles_vv0_f2_ortho,
                &pnlpt->M12_oneline_matter_multipoles_vv0_f3_ortho, &pnlpt->M12_oneline_matter_multipoles_vd0_f1_ortho,
                &pnlpt->M12_oneline_matter_multipoles_vd0_f2_ortho, &pnlpt->M12_oneline_matter_multipoles_dd0_f0_ortho,
                &pnlpt->M12_oneline_matter_multipoles_dd0_f1_ortho, &pnlpt->M12_oneline_matter_multipoles_vv2_f3_ortho,
                &pnlpt->M12_oneline_matter_multipoles_vd2_f2_ortho, &pnlpt->M12_oneline_matter_multipoles_vv4_f3_ortho,
                &pnlpt->M12_oneline_matter_multipoles_vd4_f2_ortho, &pnlpt->M12_oneline_matter_mu_powers_vd2_f1_ortho,
                &pnlpt->M12_oneline_matter_mu_powers_vd2_f2_ortho, &pnlpt->M12_oneline_matter_mu_powers_dd2_f1_ortho,
                &pnlpt->M12_oneline_matter_mu_powers_vv4_f2_ortho, &pnlpt->M12_oneline_matter_mu_powers_vd4_f2_ortho,
                &pnlpt->M12_oneline_matter_mu_powers_vv6_f3_ortho, &pnlpt->M12_oneline_bias_real_space_b2_ortho,
                &pnlpt->M12_oneline_bias_real_space_bG2_ortho, &pnlpt->M12_oneline_bias_multipoles_b2_vv0_f1_ortho,
                &pnlpt->M12_oneline_bias_multipoles_bG2_vv0_f1_ortho,
            };

            if (has_fNL) {
                /* Load from bundled binary files (2 fopen instead of 42) */
                char bpath_[256];
                sprintf(bpath_, "%s/pt_matrices/M12_bundle_%s.bin", __CLASSDIR__, Nstr);
                FILE *fb_ = fopen(bpath_, "rb");
                if (fb_ == NULL) { printf("Error: cannot open %s\n", bpath_); exit(1); }
                for (int m12i_ = 0; m12i_ < n_m12; m12i_++) {
                    class_alloc(*m12_regular[m12i_], m12_size * sizeof(double), pnlpt->error_message);
                    if (fread(*m12_regular[m12i_], sizeof(double), m12_size, fb_) != (size_t)m12_size) {
                        printf("Error: short read from %s at matrix %d\n", bpath_, m12i_); exit(1);
                    }
                }
                fclose(fb_);

                sprintf(bpath_, "%s/pt_matrices/M12_bundle_%s_ortho.bin", __CLASSDIR__, Nstr);
                fb_ = fopen(bpath_, "rb");
                if (fb_ == NULL) { printf("Error: cannot open %s\n", bpath_); exit(1); }
                for (int m12i_ = 0; m12i_ < n_m12; m12i_++) {
                    class_alloc(*m12_ortho[m12i_], m12_size * sizeof(double), pnlpt->error_message);
                    if (fread(*m12_ortho[m12i_], sizeof(double), m12_size, fb_) != (size_t)m12_size) {
                        printf("Error: short read from %s at matrix %d\n", bpath_, m12i_); exit(1);
                    }
                }
                fclose(fb_);
            } else {
                for (int m12i_ = 0; m12i_ < n_m12; m12i_++) {
                    class_alloc(*m12_regular[m12i_], m12_size * sizeof(double), pnlpt->error_message);
                    class_alloc(*m12_ortho[m12i_], m12_size * sizeof(double), pnlpt->error_message);
                }
            }
        }

        char file2openGauss[256];
        sprintf(file2openGauss, "%s/pt_matrices/gauss_tab.dat", __CLASSDIR__);
        int index_gauss = 0;

        class_alloc(pnlpt->gauss_w, 40 * sizeof(double), pnlpt->error_message);
        class_alloc(pnlpt->gauss_x, 40 * sizeof(double), pnlpt->error_message);
        LOAD_MATRIX(pnlpt->gauss, file2openGauss, 80);

        for (index_gauss = 0; index_gauss < 40; index_gauss++) {
            pnlpt->gauss_x[index_gauss] = pnlpt->gauss[index_gauss];
            pnlpt->gauss_w[index_gauss] = pnlpt->gauss[40 + index_gauss];
        }

        /* --- Convert real-valued kernel matrices to packed complex (Hermitian) form --- */
        /* The complex matrices are stored in LAPACK packed format for use
         * with zspmv (symmetric packed matrix-vector multiply). */

        /* Allocate all M22-type complex matrices */
        {
            const int m22_csize = ((ppr->nmax_nlpt + 1) * (ppr->nmax_nlpt + 2) / 2) * sizeof(complex double);
            for (int ci_ = 0; ci_ < N_M22_COMPLEX; ci_++) {
                complex double **ptr = (complex double **)((char *)pnlpt + m22_complex_offsets[ci_]);
                class_alloc(*ptr, m22_csize, pnlpt->error_message);
            }
            const int m13_csize = (ppr->nmax_nlpt + 1) * sizeof(complex double);
            for (int ci_ = 0; ci_ < N_M13_COMPLEX; ci_++) {
                complex double **ptr = (complex double **)((char *)pnlpt + m13_complex_offsets[ci_]);
                class_alloc(*ptr, m13_csize, pnlpt->error_message);
            }
        }

        int count = 0;

        CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M22_oneline_complex, pnlpt->M22_oneline, ppr->nmax_nlpt + 1);
        CONVERT_REAL_TO_COMPLEX_M13(pnlpt->M13_oneline_complex, pnlpt->M13_oneline, ppr->nmax_nlpt + 1);
        CONVERT_REAL_TO_COMPLEX_M13(pnlpt->IFG2_oneline_complex, pnlpt->IFG2_oneline, ppr->nmax_nlpt + 1);
        CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M22basic_oneline_complex, pnlpt->M22basic_oneline, ppr->nmax_nlpt + 1);
        if (has_fNL) {
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex, pnlpt->M12_oneline, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_ortho, pnlpt->M12_oneline_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vv0_f2, pnlpt->M12_oneline_matter_multipoles_vv0_f2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vv0_f3, pnlpt->M12_oneline_matter_multipoles_vv0_f3, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vd0_f1, pnlpt->M12_oneline_matter_multipoles_vd0_f1, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vd0_f2, pnlpt->M12_oneline_matter_multipoles_vd0_f2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_dd0_f0, pnlpt->M12_oneline_matter_multipoles_dd0_f0, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_dd0_f1, pnlpt->M12_oneline_matter_multipoles_dd0_f1, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vv2_f3, pnlpt->M12_oneline_matter_multipoles_vv2_f3, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vd2_f2, pnlpt->M12_oneline_matter_multipoles_vd2_f2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vv4_f3, pnlpt->M12_oneline_matter_multipoles_vv4_f3, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vd4_f2, pnlpt->M12_oneline_matter_multipoles_vd4_f2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vv0_f2_ortho, pnlpt->M12_oneline_matter_multipoles_vv0_f2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vv0_f3_ortho, pnlpt->M12_oneline_matter_multipoles_vv0_f3_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vd0_f1_ortho, pnlpt->M12_oneline_matter_multipoles_vd0_f1_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vd0_f2_ortho, pnlpt->M12_oneline_matter_multipoles_vd0_f2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_dd0_f0_ortho, pnlpt->M12_oneline_matter_multipoles_dd0_f0_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_dd0_f1_ortho, pnlpt->M12_oneline_matter_multipoles_dd0_f1_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vv2_f3_ortho, pnlpt->M12_oneline_matter_multipoles_vv2_f3_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vd2_f2_ortho, pnlpt->M12_oneline_matter_multipoles_vd2_f2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vv4_f3_ortho, pnlpt->M12_oneline_matter_multipoles_vv4_f3_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_multipoles_vd4_f2_ortho, pnlpt->M12_oneline_matter_multipoles_vd4_f2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f1, pnlpt->M12_oneline_matter_mu_powers_vd2_f1, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f2, pnlpt->M12_oneline_matter_mu_powers_vd2_f2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_dd2_f1, pnlpt->M12_oneline_matter_mu_powers_dd2_f1, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vv4_f2, pnlpt->M12_oneline_matter_mu_powers_vv4_f2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vd4_f2, pnlpt->M12_oneline_matter_mu_powers_vd4_f2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vv6_f3, pnlpt->M12_oneline_matter_mu_powers_vv6_f3, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f1_ortho, pnlpt->M12_oneline_matter_mu_powers_vd2_f1_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f2_ortho, pnlpt->M12_oneline_matter_mu_powers_vd2_f2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_dd2_f1_ortho, pnlpt->M12_oneline_matter_mu_powers_dd2_f1_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vv4_f2_ortho, pnlpt->M12_oneline_matter_mu_powers_vv4_f2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vd4_f2_ortho, pnlpt->M12_oneline_matter_mu_powers_vd4_f2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_matter_mu_powers_vv6_f3_ortho, pnlpt->M12_oneline_matter_mu_powers_vv6_f3_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_bias_real_space_b2, pnlpt->M12_oneline_bias_real_space_b2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_bias_real_space_bG2, pnlpt->M12_oneline_bias_real_space_bG2, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_bias_real_space_b2_ortho, pnlpt->M12_oneline_bias_real_space_b2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_bias_real_space_bG2_ortho, pnlpt->M12_oneline_bias_real_space_bG2_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_bias_multipoles_b2_vv0_f1, pnlpt->M12_oneline_bias_multipoles_b2_vv0_f1, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_bias_multipoles_bG2_vv0_f1, pnlpt->M12_oneline_bias_multipoles_bG2_vv0_f1, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_bias_multipoles_b2_vv0_f1_ortho, pnlpt->M12_oneline_bias_multipoles_b2_vv0_f1_ortho, ppr->nmax_nlpt + 1);
            CONVERT_REAL_TO_COMPLEX_M22(pnlpt->M12_oneline_complex_bias_multipoles_bG2_vv0_f1_ortho, pnlpt->M12_oneline_bias_multipoles_bG2_vv0_f1_ortho, ppr->nmax_nlpt + 1);
        }

        /* --- All PT kernel matrices loaded; allocate output storage arrays --- */
        /* Output arrays for spectra at each redshift (accessed by classy via spectra_pk_nl_bias_at_z) */
        /* Allocate all ln_pk output arrays */
        {
            const size_t ln_pk_size = sizeof(double) * pnlpt->z_pk_num * pnlpt->k_size;
            for (int lpi_ = 0; lpi_ < N_LN_PK; lpi_++) {
                double **ptr = (double **)((char *)pnlpt + ln_pk_offsets[lpi_]);
                class_alloc(*ptr, ln_pk_size, pnlpt->error_message);
            }
        }

        int index_md;
        index_md = pnlpt->index_md_scalars;
        int index_ic = 0;

        /* --- Fast output: spline sources for interpolation to arbitrary tau --- */
        if (ppt->has_cls == _TRUE_ && pnlpt->fast_output == _TRUE_) {
            if (pnlpt->cb == _TRUE_) {
                class_alloc(pnlpt->dd_sources_tp_delta_cb,
                            ppt->k_size[index_md] * ppt->tau_size * sizeof(double),
                            pnlpt->error_message);
                class_alloc(pnlpt->sources_tp_delta_cb,
                            ppt->k_size[index_md] * ppt->tau_size * sizeof(double),
                            pnlpt->error_message);

                class_call(array_spline_table_columns(ppt->k[index_md],
                                                      ppt->k_size[index_md],
                                                      ppt->sources[index_md]
                                                                  [index_ic * ppt->tp_size[index_md] + ppt->index_tp_delta_cb],
                                                      ppt->tau_size,
                                                      pnlpt->dd_sources_tp_delta_cb,
                                                      _SPLINE_EST_DERIV_,
                                                      pnlpt->error_message),
                           pnlpt->error_message,
                           pnlpt->error_message);

                for (index_tau = pnlpt->tau_size - 1; index_tau >= 0; index_tau--) {
                    for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
                        if (pnlpt->k[index_k] <= ppt->k[index_md][ppt->k_size[index_md] - 1]) {
                            class_call(array_interpolate_spline_one_column(ppt->k[index_md],
                                                                           ppt->k_size[index_md],
                                                                           ppt->sources[index_md]
                                                                                       [index_ic * ppt->tp_size[index_md] + ppt->index_tp_delta_cb],
                                                                           pnlpt->tau_size,
                                                                           index_tau,
                                                                           pnlpt->dd_sources_tp_delta_cb,
                                                                           pnlpt->k[index_k],
                                                                           &pnlpt->sources_tp_delta_cb[index_tau * pnlpt->k_size + index_k],
                                                                           pnlpt->error_message),
                                       pnlpt->error_message,
                                       pnlpt->error_message);

                        } /* end k <= k_max condition */
                        /* Crude padding for k beyond perturbation module range (only relevant for non-zero curvature) */
                        /* This is done only for the non-zero spatial curvature, for which the kmax from perturbations does not coincude with the kmax from the spectra */
                        else
                        {
                            pnlpt->sources_tp_delta_cb[index_tau * pnlpt->k_size + index_k] = pnlpt->sources_tp_delta_cb[index_tau * pnlpt->k_size + index_k - 1];
                        }
                    }
                }
            } else {
                class_alloc(pnlpt->dd_sources_tp_delta_m,
                            ppt->k_size[index_md] * ppt->tau_size * sizeof(double),
                            pnlpt->error_message);
                class_alloc(pnlpt->sources_tp_delta_m,
                            ppt->k_size[index_md] * ppt->tau_size * sizeof(double),
                            pnlpt->error_message);

                class_call(array_spline_table_columns(ppt->k[index_md],
                                                      ppt->k_size[index_md],
                                                      ppt->sources[index_md]
                                                                  [index_ic * ppt->tp_size[index_md] + ppt->index_tp_delta_m],
                                                      ppt->tau_size,
                                                      pnlpt->dd_sources_tp_delta_m,
                                                      _SPLINE_EST_DERIV_,
                                                      pnlpt->error_message),
                           pnlpt->error_message,
                           pnlpt->error_message);
                for (index_tau = pnlpt->tau_size - 1; index_tau >= 0; index_tau--) {
                    for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
                        if (pnlpt->k[index_k] <= ppt->k[index_md][ppt->k_size[index_md] - 1]) {
                            class_call(array_interpolate_spline_one_column(ppt->k[index_md],
                                                                           ppt->k_size[index_md],
                                                                           ppt->sources[index_md]
                                                                                       [index_ic * ppt->tp_size[index_md] + ppt->index_tp_delta_m],
                                                                           pnlpt->tau_size,
                                                                           index_tau,
                                                                           pnlpt->dd_sources_tp_delta_m,
                                                                           pnlpt->k[index_k],
                                                                           &pnlpt->sources_tp_delta_m[index_tau * pnlpt->k_size + index_k],
                                                                           pnlpt->error_message),
                                       pnlpt->error_message,
                                       pnlpt->error_message);
                        } /* end k <= k_max condition */
                        /* Crude padding for k beyond perturbation module range (only relevant for non-zero curvature) */
                        /* This is done only for curvature, for which the kmax from perturbations does not coincide with the kmax from the spectra */
                        else
                        {
                            pnlpt->sources_tp_delta_m[index_tau * pnlpt->k_size + index_k] = pnlpt->sources_tp_delta_m[index_tau * pnlpt->k_size + index_k - 1];
                        }
                    }
                }
            }
            /* End cb (cold baryon) transfer function interpolation */
        }

        /* End fast_output source interpolation */

        /* Build full P_L(k, tau), ln P_L, and spline tables over all tau */
        double *lnpk_l_full;
        double *pk_l_full;
        double *ddlnpk_l_full;

        class_alloc(lnpk_l_full, sizeof(double) * pnlpt->tau_size * pnlpt->k_size, pnlpt->error_message);
        class_alloc(pk_l_full, sizeof(double) * pnlpt->tau_size * pnlpt->k_size, pnlpt->error_message);
        class_alloc(ddlnpk_l_full, sizeof(double) * pnlpt->tau_size * pnlpt->k_size, pnlpt->error_message);

        /* Loop over conformal time to fill P_L(k,tau) arrays */
        for (index_tau = pnlpt->tau_size - 1; index_tau >= 0; index_tau--) {
            /* get P_L(k) at this time */

            class_call(nonlinear_pt_pk_l(pba, ppt, ppm, pnlpt, index_tau, pk_l, lnk_l, lnpk_l, ddlnpk_l),
                       pnlpt->error_message,
                       pnlpt->error_message);

            /* get P_L(k,tau) lnP_L(k,tau) and ddP_L(k,tau) */

            for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
                lnpk_l_full[index_tau * pnlpt->k_size + index_k] = lnpk_l[index_k];
                pk_l_full[index_tau * pnlpt->k_size + index_k] = pk_l[index_k];
                ddlnpk_l_full[index_tau * pnlpt->k_size + index_k] = ddlnpk_l[index_k];
            }
        }
        /*end cycle over tau*/

        class_call(nonlinear_pt_pPRIMk_l(
                       ppt, ppm, pnlpt, lnk_l,
                       pPRIMk_l, lnpPRIMk_l, ddlnpPRIMk_l),
                   pnlpt->error_message,
                   pnlpt->error_message);

        /* --- Interpolate P_L(k) and P_PRIM(k) to each requested redshift --- */
        double *ln_pk_l_at_z_req;
        double *pk_l_at_z_req;
        /* DW: keep a separate full total-matter linear spectrum when cb=yes.
         * The exact kernels evolve the clustering baryon+QCDM sector, while
         * lensing/matter diagnostics require the total including linear ncdm. */
        double *pk_l_total_at_z_req;
        double *ln_pPRIMk_l_req;
        double *pPRIMk_l_req;

        class_alloc(ln_pPRIMk_l_req, sizeof(double) * pnlpt->k_size, pnlpt->error_message);
        class_alloc(pPRIMk_l_req, sizeof(double) * pnlpt->k_size, pnlpt->error_message);

        double Dref;

        class_alloc(ln_pk_l_at_z_req, sizeof(double) * pnlpt->k_size, pnlpt->error_message);
        class_alloc(pk_l_at_z_req, sizeof(double) * pnlpt->k_size, pnlpt->error_message);
        class_alloc(pk_l_total_at_z_req, sizeof(double) * pnlpt->k_size, pnlpt->error_message);

        /* Retrieve the logarithmic growth rate f for RSD */

        class_alloc(pnlpt->growthf, sizeof(double) * pnlpt->z_pk_num, pnlpt->error_message);
        class_alloc(pnlpt->hratio_array, sizeof(double) * pnlpt->z_pk_num, pnlpt->error_message);
        class_alloc(pnlpt->Dratio_array, sizeof(double) * pnlpt->z_pk_num, pnlpt->error_message);

        if (pnlpt->rsd == rsd_yes) {
            double *pvecbackf;
            int last_indexf;
            class_alloc(pvecbackf, pba->bg_size * sizeof(double), pnlpt->error_message);

            /* Variables for Alcock-Paczynski projection */
            int j;
            int Nz = 2000;
            double Omfid = pnlpt->OmfidAP;
            double dz;
            double kmsMpc = 3.33564095198145e-6;

            if (pnlpt->nonlinear_pt_verbose > 0)
                if (pnlpt->AP_effect == AP_effect_yes) {
                    printf("Computing the Alcock-Paczynski effect for fiducial cosmology with Om=%lf\n", Omfid);
                } else {
                    printf("No Alcock-Paczynski effect.\n");
                }

            /* --- Main redshift loop: compute 1-loop spectra at each z_pk --- */
            for (i_z = 0; i_z < pnlpt->z_pk_num; i_z++) {
                /* Extract background quantities (D, f, H, D_A) at this redshift */
                double Da = 0;
                double Dfid = 0;
                double hnew = 0;
                double hfid = 0;

                class_call(background_at_tau(pba, tau_req[i_z], long_info, inter_normal, &last_indexf, pvecbackf),
                           pnlpt->error_message,
                           pnlpt->error_message);

                /* Override f(z) and D(z) with user-supplied values if replace_background is set */
                if (pnlpt->replace_background) {
                    pnlpt->growthf[i_z] = pnlpt->replace_fz_value;
                    Dref = pnlpt->replace_Dz_value;
                } else {
                    pnlpt->growthf[i_z] = pvecbackf[pba->index_bg_f];
                    Dref = pvecbackf[pba->index_bg_D];
                }

                if (pnlpt->AP_effect == AP_effect_yes) {
                    if (pnlpt->z_pk[i_z] == 0.) {
                        pnlpt->hratio_array[i_z] = 1.;
                        pnlpt->Dratio_array[i_z] = 1.;
                        hnew = 1.;
                    } else {
                        hfid = pow((Omfid * pow((1. + pnlpt->z_pk[i_z]), 3.) + (1. - Omfid) + (pba->Omega0_g) * pow((1. + pnlpt->z_pk[i_z]), 4.)), 0.5);
                        if (pnlpt->replace_background) {
                            hnew = pnlpt->replace_Hz_value / kmsMpc / 100.0 / pba->h;
                        } else {
                            hnew = pvecbackf[pba->index_bg_H] / kmsMpc / 100.0 / pba->h;
                        }
                        pnlpt->hratio_array[i_z] = hnew / hfid;

                        dz = pnlpt->z_pk[i_z] / (double)(Nz - 1);

                        /* DW: integrate the fiducial comoving distance with the
                         * radiation term evaluated at each quadrature redshift.
                         * The previous code froze (1+z)^4 at the endpoint z_pk,
                         * which was dimensionally consistent but not the AP
                         * distance integral actually intended. */
                        for (j = 1; j < Nz; j++) {
                            double z_hi=dz*j;
                            double z_lo=dz*(j-1);
                            double E_hi=sqrt(Omfid*pow(1.0+z_hi,3.0)
                                             +(1.0-Omfid)
                                             +pba->Omega0_g*pow(1.0+z_hi,4.0));
                            double E_lo=sqrt(Omfid*pow(1.0+z_lo,3.0)
                                             +(1.0-Omfid)
                                             +pba->Omega0_g*pow(1.0+z_lo,4.0));
                            Dfid += 0.5*dz*(1.0/E_hi+1.0/E_lo);
                        }
                        if (pnlpt->replace_background) {
                            Da = pnlpt->replace_DAz_value * kmsMpc * 100 * pba->h * (1. + pnlpt->z_pk[i_z]);
                        } else {
                            Da = pvecbackf[pba->index_bg_ang_distance] * kmsMpc * 100 * pba->h * (1. + pnlpt->z_pk[i_z]);
                        }

                        pnlpt->Dratio_array[i_z] = Da / Dfid;
                    }
                } else {
                    pnlpt->hratio_array[i_z] = 1.;
                    pnlpt->Dratio_array[i_z] = 1.;
                }
            }

            free(pvecbackf);
        } else {
            /* No RSD: set growth/AP ratios to unity, retrieve D(z) from background */
            double *pvecbackf;
            int last_indexf = 0;
            class_alloc(pvecbackf, pba->bg_size * sizeof(double), pnlpt->error_message);

            for (i_z = 0; i_z < pnlpt->z_pk_num; i_z++) {
                pnlpt->growthf[i_z] = 1.;
                pnlpt->hratio_array[i_z] = 1.;
                pnlpt->Dratio_array[i_z] = 1.;
                class_call(background_at_tau(pba, tau_req[i_z], long_info, inter_normal, &last_indexf, pvecbackf),
                           pnlpt->error_message,
                           pnlpt->error_message);
                Dref = pvecbackf[pba->index_bg_D];
            }
            free(pvecbackf);
        }

        /* End RSD/AP parameter setup */

        /* Interpolate P_L(k) at each requested redshift.
         *
         * IMPORTANT: lnpk_l_full is stored as [tau][k]. The spline
         * second-derivative table ddlnpk_l_full was constructed column-by-column
         * in the k direction earlier, so it must NOT be reused as a tau-spline
         * table. Doing so can return NaNs for most k bins. For the requested
         * redshift output we only need a stable interpolation at fixed k, so use
         * a robust linear interpolation in ln(tau). */

        for (i_z = 0; i_z < pnlpt->z_pk_num; i_z++) {
            double lt_req = log(tau_req[i_z]);
            int itau_lo = 0;
            int itau_hi = 0;
            double wt_tau = 0.0;

            if (lt_req <= pnlpt->ln_tau[0]) {
                itau_lo = 0;
                itau_hi = 0;
                wt_tau = 0.0;
            }
            else if (lt_req >= pnlpt->ln_tau[pnlpt->tau_size-1]) {
                itau_lo = pnlpt->tau_size-1;
                itau_hi = pnlpt->tau_size-1;
                wt_tau = 0.0;
            }
            else {
                int ilo_tau = 0;
                int ihi_tau = pnlpt->tau_size-1;
                while (ihi_tau - ilo_tau > 1) {
                    int imid_tau = (ilo_tau + ihi_tau)/2;
                    if (pnlpt->ln_tau[imid_tau] > lt_req)
                        ihi_tau = imid_tau;
                    else
                        ilo_tau = imid_tau;
                }
                itau_lo = ilo_tau;
                itau_hi = ihi_tau;
                wt_tau = (lt_req - pnlpt->ln_tau[itau_lo])/(pnlpt->ln_tau[itau_hi] - pnlpt->ln_tau[itau_lo]);
            }

            for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
                /* Interpolate the linear power itself, not log(P).
                 * In this model/gauge the total-matter transfer can pass through
                 * zero at isolated grid points. Then P_L=0 is valid, but log(P_L)
                 * is -inf and must not be used as the primary interpolation
                 * variable. */
                double P_lo = pk_l_full[itau_lo*pnlpt->k_size + index_k];
                double P_hi = pk_l_full[itau_hi*pnlpt->k_size + index_k];

                class_test((qcdm_qscf_finite_double(P_lo) == 0) ||
                           (qcdm_qscf_finite_double(P_hi) == 0) ||
                           (P_lo < 0.0) || (P_hi < 0.0),
                           pnlpt->error_message,
                           "Invalid linear P(k,tau) before QCDM PT interpolation at z index %d, k index %d: P_lo=%e P_hi=%e",
                           i_z,index_k,P_lo,P_hi);

                pk_l_at_z_req[index_k] = (1.0-wt_tau)*P_lo + wt_tau*P_hi;

                class_test((qcdm_qscf_finite_double(pk_l_at_z_req[index_k]) == 0) ||
                           (pk_l_at_z_req[index_k] < 0.0),
                           pnlpt->error_message,
                           "Invalid interpolated linear P(k) before QCDM PT at z index %d, k index %d: P=%e",
                           i_z,index_k,pk_l_at_z_req[index_k]);

                /* Keep a finite logarithmic companion only for legacy FFTLog/
                 * bookkeeping paths. The exact-QS branch uses pk_l_at_z_req. */
                ln_pk_l_at_z_req[index_k] = log(MAX(pk_l_at_z_req[index_k],1.e-300));

                ln_pPRIMk_l_req[index_k] = lnpPRIMk_l[index_k];
                pPRIMk_l_req[index_k] = exp(ln_pPRIMk_l_req[index_k]);
            }

/* DW: replace reconstructed source power by the fully assembled CLASS Fourier linear matter spectrum. */
            /* The exact coupled-fluid branch must use the same physical
             * linear matter spectrum as CLASS's Fourier module.  Rebuilding
             * P_m from a single perturbation source is not reliable here:
             * the source can be gauge converted and can cross zero, while
             * Fourier already performs the complete primordial-IC sum and
             * time interpolation. */
            if ((pba->has_qcdm == _TRUE_) &&
                (pnlpt->qcdm_qscf_kernels != qcdm_qscf_kernels_no)) {
                class_test(pfo->has_pk_m == _FALSE_,pnlpt->error_message,
                           "QCDM exact-QS PT requires the CLASS linear matter spectrum (mPk)");
                for (index_k=0; index_k<pnlpt->k_size; index_k++) {
                    double Pclass_total=0.0;
                    double Pclass_cluster=0.0;
                    class_call(fourier_pk_at_k_and_z(pba,ppm,pfo,pk_linear,
                                                     pnlpt->k[index_k],pnlpt->z_pk[i_z],
                                                     pfo->index_pk_m,&Pclass_total,NULL),
                               pfo->error_message,pnlpt->error_message);
                    /* DW: use CLASS P_cb when requested and available.  CLASS
                     * defines delta_cb before adding ncdm, and this branch's
                     * modified perturbations include QCDM in that source. */
                    if ((pnlpt->cb == _TRUE_) && (pfo->has_pk_cb == _TRUE_)) {
                        class_call(fourier_pk_at_k_and_z(pba,ppm,pfo,pk_linear,
                                                         pnlpt->k[index_k],pnlpt->z_pk[i_z],
                                                         pfo->index_pk_cb,&Pclass_cluster,NULL),
                                   pfo->error_message,pnlpt->error_message);
                    }
                    else {
                        Pclass_cluster=Pclass_total;
                    }
                    class_test((qcdm_qscf_finite_double(Pclass_total)==0)||(Pclass_total<0.0) ||
                               (qcdm_qscf_finite_double(Pclass_cluster)==0)||(Pclass_cluster<0.0),
                               pnlpt->error_message,
                               "Invalid Fourier linear spectrum for QCDM PT: z=%e k=%e Pm=%e Pcluster=%e",
                               pnlpt->z_pk[i_z],pnlpt->k[index_k],Pclass_total,Pclass_cluster);
                    pk_l_total_at_z_req[index_k]=Pclass_total;
                    pk_l_at_z_req[index_k]=Pclass_cluster;
                    ln_pk_l_at_z_req[index_k]=log(MAX(Pclass_cluster,1.e-300));
                }
            }
            else {
                /* DW: legacy single-fluid branch has no separate clustering
                 * reconstruction requirement, so both pointers carry P_L. */
                for (index_k=0; index_k<pnlpt->k_size; index_k++)
                    pk_l_total_at_z_req[index_k]=pk_l_at_z_req[index_k];
            }

            /* get P_NL(k) at tau_req */

            if (print_warning == _FALSE_) {
                /* Call the main 1-loop computation engine */
                class_call(nonlinear_pt_loop(ppr,
                                             pba,
                                             ppm,
                                             pth,
                                             pnlpt,
                                             tau_req[i_z],
                                             pnlpt->growthf[i_z],
                                             pnlpt->hratio_array[i_z],
                                             pnlpt->Dratio_array[i_z],
                                             pk_l_at_z_req,
                                             /* DW: pass total matter beside cb for the linear-ncdm hybrid reconstruction. */
                                             pk_l_total_at_z_req,
                                             pPRIMk_l_req,
                                             pk_l_0_vv,
                                             pk_l_0_vd,
                                             pk_l_0_dd,
                                             pk_l_2_vv,
                                             pk_l_2_vd,
                                             pk_l_2_dd,
                                             pk_l_4_vv,
                                             pk_l_4_vd,
                                             pk_l_4_dd,
                                             pk_nl,
                                             pk_nl_fNL,
                                             pk_fNLd2,
                                             pk_fNLG2,
                                             pk_nl_fNL_ortho,
                                             pk_fNLd2_ortho,
                                             pk_fNLG2_ortho,
                                             pk_Id2d2,
                                             pk_Id2d2_2,
                                             pk_Id2d2_4,
                                             pk_l_0_b1b2,
                                             pk_l_0_b2,
                                             pk_l_0_b1bG2,
                                             pk_l_0_bG2,
                                             pk_l_2_b1b2,
                                             pk_l_2_b2,
                                             pk_l_2_b1bG2,
                                             pk_l_2_bG2,
                                             pk_l_4_b2,
                                             pk_l_4_bG2,
                                             pk_l_4_b1b2,
                                             pk_l_4_b1bG2,
                                             pk_Id2,
                                             pk_IG2,
                                             pk_Id2G2,
                                             pk_Id2G2_2,
                                             pk_Id2G2_4,
                                             pk_IG2G2,
                                             pk_IG2G2_2,
                                             pk_IG2G2_4,
                                             pk_IFG2,
                                             pk_IFG2_0b1,
                                             pk_IFG2_0,
                                             pk_IFG2_2,
                                             pk_CTR,
                                             pk_CTR_0,
                                             pk_CTR_2,
                                             pk_CTR_4,
                                             pk_Tree,
                                             pk_Tree_0_vv,
                                             pk_Tree_0_vd,
                                             pk_Tree_0_dd,
                                             pk_Tree_2_vv,
                                             pk_Tree_2_vd,
                                             pk_Tree_4_vv,
                                             pk_l_fNL_0_vv,
                                             pk_l_fNL_0_vd,
                                             pk_l_fNL_0_dd,
                                             pk_l_fNL_2_vv,
                                             pk_l_fNL_2_vd,
                                             pk_l_fNL_2_dd,
                                             pk_l_fNL_4_vv,
                                             pk_l_fNL_4_vd,
                                             pk_l_fNL_4_dd,
                                             pk12_l_0_b1b2,
                                             pk12_l_0_b2,
                                             pk12_l_0_b1bG2,
                                             pk12_l_0_bG2,
                                             pk12_l_2_b1b2,
                                             pk12_l_2_b2,
                                             pk12_l_2_b1bG2,
                                             pk12_l_2_bG2,
                                             pk12_l_4_b1b2,
                                             pk12_l_4_b2,
                                             pk12_l_4_b1bG2,
                                             pk12_l_4_bG2,
                                             pk_l_fNL_0_vv_ortho,
                                             pk_l_fNL_0_vd_ortho,
                                             pk_l_fNL_0_dd_ortho,
                                             pk_l_fNL_2_vv_ortho,
                                             pk_l_fNL_2_vd_ortho,
                                             pk_l_fNL_2_dd_ortho,
                                             pk_l_fNL_4_vv_ortho,
                                             pk_l_fNL_4_vd_ortho,
                                             pk_l_fNL_4_dd_ortho,
                                             pk12_l_0_b1b2_ortho,
                                             pk12_l_0_b2_ortho,
                                             pk12_l_0_b1bG2_ortho,
                                             pk12_l_0_bG2_ortho,
                                             pk12_l_2_b1b2_ortho,
                                             pk12_l_2_b2_ortho,
                                             pk12_l_2_b1bG2_ortho,
                                             pk12_l_2_bG2_ortho,
                                             pk12_l_4_b1b2_ortho,
                                             pk12_l_4_b2_ortho,
                                             pk12_l_4_b1bG2_ortho,
                                             pk12_l_4_bG2_ortho,
                                             lnk_l,
                                             ln_pk_l_at_z_req,
                                             ln_pPRIMk_l_req),
                           pnlpt->error_message,
                           pnlpt->error_message);
                /* --- Store results: copy 1-loop spectra into pnlpt arrays --- */
                {
                    double *store_pk_arrs_[N_LN_PK] = {
                        pk_nl, pk_Id2d2, pk_Id2d2_2, pk_Id2d2_4,
                        pk_Id2, pk_IG2, pk_Id2G2, pk_Id2G2_2, pk_Id2G2_4,
                        pk_IG2G2, pk_IG2G2_2, pk_IG2G2_4,
                        pk_IFG2, pk_IFG2_0b1, pk_IFG2_0, pk_IFG2_2,
                        pk_CTR, pk_CTR_0, pk_CTR_2, pk_CTR_4,
                        pk_Tree, pk_Tree_0_vv, pk_Tree_0_vd, pk_Tree_0_dd,
                        pk_Tree_2_vv, pk_Tree_2_vd, pk_Tree_4_vv,
                        pk_l_0_vv, pk_l_0_vd, pk_l_0_dd,
                        pk_l_2_vv, pk_l_2_vd, pk_l_2_dd,
                        pk_l_4_vv, pk_l_4_vd, pk_l_4_dd,
                        pk_l_0_b1b2, pk_l_0_b2, pk_l_0_b1bG2, pk_l_0_bG2,
                        pk_l_2_b1b2, pk_l_2_b2, pk_l_2_b1bG2, pk_l_2_bG2,
                        pk_l_4_b2, pk_l_4_bG2, pk_l_4_b1b2, pk_l_4_b1bG2,
                        pk_l_fNL_0_vv, pk_l_fNL_0_vd, pk_l_fNL_0_dd,
                        pk_l_fNL_2_vv, pk_l_fNL_2_vd, pk_l_fNL_2_dd,
                        pk_l_fNL_4_vv, pk_l_fNL_4_vd, pk_l_fNL_4_dd,
                        pk_l_fNL_0_vv_ortho, pk_l_fNL_0_vd_ortho, pk_l_fNL_0_dd_ortho,
                        pk_l_fNL_2_vv_ortho, pk_l_fNL_2_vd_ortho, pk_l_fNL_2_dd_ortho,
                        pk_l_fNL_4_vv_ortho, pk_l_fNL_4_vd_ortho, pk_l_fNL_4_dd_ortho,
                        pk12_l_0_b1b2, pk12_l_0_b2, pk12_l_0_b1bG2, pk12_l_0_bG2,
                        pk12_l_2_b1b2, pk12_l_2_b2, pk12_l_2_b1bG2, pk12_l_2_bG2,
                        pk12_l_4_b1b2, pk12_l_4_b2, pk12_l_4_b1bG2, pk12_l_4_bG2,
                        pk12_l_0_b1b2_ortho, pk12_l_0_b2_ortho, pk12_l_0_b1bG2_ortho, pk12_l_0_bG2_ortho,
                        pk12_l_2_b1b2_ortho, pk12_l_2_b2_ortho, pk12_l_2_b1bG2_ortho, pk12_l_2_bG2_ortho,
                        pk12_l_4_b1b2_ortho, pk12_l_4_b2_ortho, pk12_l_4_b1bG2_ortho, pk12_l_4_bG2_ortho,
                        pk_nl_fNL, pk_fNLd2, pk_fNLG2,
                        pk_nl_fNL_ortho, pk_fNLd2_ortho, pk_fNLG2_ortho
                    };
                    for (index_k = 0; index_k < pnlpt->k_size; index_k++)
                        for (int si_ = 0; si_ < N_LN_PK; si_++) {
                            double *dest_ = *(double **)((char *)pnlpt + ln_pk_offsets[si_]);
                            dest_[i_z * pnlpt->k_size + index_k] = log(store_pk_arrs_[si_][index_k]);
                        }
                }

                /* DW: QCDM: write CLASS-PT/QCDM outputs to disk. The standard
                 * CLASS output module does not write these when non linear=PT. */
                class_call(qcdm_qscf_write_one_z_output_files(pba,
                                                              pnlpt,
                                                              i_z,
                                                              pk_Tree,
                                                              pk_nl,
                                                              pk_nl_fNL,
                                                              pk_nl_fNL_ortho,
                                                              pk_CTR,
                                                              pk_Id2,
                                                              pk_Id2d2,
                                                              pk_IG2,
                                                              pk_IFG2,
                                                              pk_l_0_dd,
                                                              pk_l_2_dd,
                                                              pk_l_4_dd,
                                                              pk_l_fNL_0_dd,
                                                              pk_l_fNL_2_dd,
                                                              pk_l_fNL_4_dd,
                                                              pk_l_fNL_0_dd_ortho,
                                                              pk_l_fNL_2_dd_ortho,
                                                              pk_l_fNL_4_dd_ortho,
                                                              pk_Tree_0_dd,
                                                              pk_Tree_2_vd,
                                                              pk_Tree_4_vv),
                           pnlpt->error_message,
                           pnlpt->error_message);
            }
        }

        /* This is required for lensing with one-loop ! */

        /* --- Lensing module: compute nl_corr_density ratio for CMB lensing --- */
        if (ppt->has_cl_cmb_lensing_potential == _TRUE_) {
            /* Lensing corrections active */
            double Dplus = 0;
            double *pvecbackD;
            int last_indexD = 0;
            class_alloc(pvecbackD, pba->bg_size * sizeof(double), pnlpt->error_message);

            double *dd_pk_Tree;
            double *dd_pk_nl;
            double *dd_pk_l_at_z_req;
            double *pk_Tree_int;
            double *pk_nl_int;
            double *pk_l_at_z_req_int;

            last_index = 0;

            double large_for_logs_matter = 5.e4;

            if (ppt->has_cls == _TRUE_ && pnlpt->fast_output == _TRUE_) {
                class_alloc(dd_pk_Tree, pnlpt->k_size * sizeof(double), pnlpt->error_message);
                class_alloc(dd_pk_nl, pnlpt->k_size * sizeof(double), pnlpt->error_message);
                class_alloc(dd_pk_l_at_z_req, pnlpt->k_size * sizeof(double), pnlpt->error_message);
                class_alloc(pk_Tree_int, ppt->k_size[pnlpt->index_md_scalars] * sizeof(double), pnlpt->error_message);
                class_alloc(pk_nl_int, ppt->k_size[pnlpt->index_md_scalars] * sizeof(double), pnlpt->error_message);
                class_alloc(pk_l_at_z_req_int, ppt->k_size[pnlpt->index_md_scalars] * sizeof(double), pnlpt->error_message);

                SPLINE_SETUP(pnlpt->k, pnlpt->k_size, pk_Tree, dd_pk_Tree, _SPLINE_EST_DERIV_);
                SPLINE_SETUP(pnlpt->k, pnlpt->k_size, pk_nl, dd_pk_nl, _SPLINE_EST_DERIV_);
                SPLINE_SETUP(pnlpt->k, pnlpt->k_size, pk_l_at_z_req, dd_pk_l_at_z_req, _SPLINE_EST_DERIV_);
                for (index_k = 0; index_k < ppt->k_size[pnlpt->index_md_scalars]; index_k++) {
                    if (ppt->k[pnlpt->index_md_scalars][index_k] < pnlpt->k[pnlpt->k_size - 1]) {
                        SPLINE_EVAL(pnlpt->k, pnlpt->k_size, pk_Tree, dd_pk_Tree, ppt->k[pnlpt->index_md_scalars][index_k], &last_index, &pk_Tree_int[index_k]);
                        SPLINE_EVAL(pnlpt->k, pnlpt->k_size, pk_nl, dd_pk_nl, ppt->k[pnlpt->index_md_scalars][index_k], &last_index, &pk_nl_int[index_k]);
                        SPLINE_EVAL(pnlpt->k, pnlpt->k_size, pk_l_at_z_req, dd_pk_l_at_z_req, ppt->k[pnlpt->index_md_scalars][index_k], &last_index, &pk_l_at_z_req_int[index_k]);
                    } else {
                        pk_Tree_int[index_k] = pk_Tree_int[index_k - 1];
                        pk_nl_int[index_k] = pk_nl_int[index_k - 1];
                        pk_l_at_z_req_int[index_k] = pk_l_at_z_req_int[index_k - 1];
                    }
                }
            }

            for (index_tau = pnlpt->tau_size - 1; index_tau >= 0; index_tau--) {
                class_call(background_at_tau(pba, pnlpt->tau[index_tau], long_info, inter_normal, &last_indexD, pvecbackD), pnlpt->error_message, pnlpt->error_message);

                Dplus = pvecbackD[pba->index_bg_D];
                if (ppt->has_cls == _TRUE_ && pnlpt->fast_output == _TRUE_) {
                    for (index_k = 0; index_k < ppt->k_size[pnlpt->index_md_scalars]; index_k++)
                        pnlpt->nl_corr_density[index_tau * ppt->k_size[pnlpt->index_md_scalars] + index_k] = sqrt(fabs((pk_Tree_int[index_k] + Dplus * Dplus * (pk_nl_int[index_k] - large_for_logs_matter) /
                                                                                                                                                   Dref / Dref)) /
                                                                                                                  pk_l_at_z_req_int[index_k]);
                } else {
                    for (index_k = 0; index_k < ppt->k_size[pnlpt->index_md_scalars]; index_k++)
                        pnlpt->nl_corr_density[index_tau * ppt->k_size[pnlpt->index_md_scalars] + index_k] = sqrt(fabs((pk_Tree[index_k] + Dplus * Dplus * (pk_nl[index_k] - large_for_logs_matter) /
                                                                                                                                               Dref / Dref)) /
                                                                                                                  pk_l_at_z_req[index_k]);
                }
            }
            free(pvecbackD);

            if (ppt->has_cls == _TRUE_ && pnlpt->fast_output == _TRUE_) {
                free(dd_pk_Tree);
                free(dd_pk_nl);
                free(dd_pk_l_at_z_req);
                free(pk_Tree_int);
                free(pk_nl_int);
                free(pk_l_at_z_req_int);
            }
        } else {
            /* No CMB lensing: set nl_corr_density = 1 */
            for (index_tau = pnlpt->tau_size - 1; index_tau >= 0; index_tau--) {
                for (index_k = 0; index_k < ppt->k_size[pnlpt->index_md_scalars]; index_k++) {
                    pnlpt->nl_corr_density[index_tau * ppt->k_size[pnlpt->index_md_scalars] + index_k] = 1.;
                }
            }
        }

        /* --- End lensing module --- */

        { void *_f[] = {
            pk_l, pk_l_0_vv, pk_l_0_vd, pk_l_0_dd, pk_l_2_vv, pk_l_2_vd, pk_l_2_dd,
            pk_l_4_vv, pk_l_4_vd, pk_l_4_dd,
            pk_l_0_b1b2, pk_l_0_b2, pk_l_0_b1bG2, pk_l_0_bG2,
            pk_l_2_b1b2, pk_l_2_b2, pk_l_2_b1bG2, pk_l_2_bG2,
            pk_l_4_b2, pk_l_4_bG2, pk_l_4_b1b2, pk_l_4_b1bG2,
            pk_nl, pk_Id2d2, pk_Id2d2_2, pk_Id2d2_4,
            pk_IG2, pk_Id2G2, pk_IG2G2, pk_Id2G2_2, pk_IG2G2_2, pk_Id2G2_4, pk_IG2G2_4,
            pk_IFG2, pk_IFG2_0, pk_IFG2_0b1, pk_IFG2_2, pk_Id2,
            pk_CTR, pk_CTR_0, pk_CTR_2, pk_CTR_4,
            pk_Tree, pk_Tree_0_vv, pk_Tree_0_vd, pk_Tree_0_dd, pk_Tree_2_vv, pk_Tree_2_vd, pk_Tree_4_vv,
            lnk_l, lnpk_l, ddlnpk_l, tau_req,
            lnpk_l_full, pk_l_full, ddlnpk_l_full,
            /* DW: release the separate total-matter spectrum allocated for
             * cb-aware exact-QS reconstruction. */
            ln_pk_l_at_z_req, pk_l_at_z_req, pk_l_total_at_z_req, ln_pPRIMk_l_req, pPRIMk_l_req,
            pk_l_fNL_0_vv, pk_l_fNL_0_vd, pk_l_fNL_0_dd,
            pk_l_fNL_2_vv, pk_l_fNL_2_vd, pk_l_fNL_2_dd,
            pk_l_fNL_4_vv, pk_l_fNL_4_vd, pk_l_fNL_4_dd,
            pk12_l_0_b1b2, pk12_l_0_b2, pk12_l_0_b1bG2, pk12_l_0_bG2,
            pk12_l_2_b1b2, pk12_l_2_b2, pk12_l_2_b1bG2, pk12_l_2_bG2,
            pk12_l_4_b1b2, pk12_l_4_b2, pk12_l_4_b1bG2, pk12_l_4_bG2,
            pk_nl_fNL, pk_fNLd2, pk_fNLG2,
            pk_l_fNL_0_vv_ortho, pk_l_fNL_0_vd_ortho, pk_l_fNL_0_dd_ortho,
            pk_l_fNL_2_vv_ortho, pk_l_fNL_2_vd_ortho, pk_l_fNL_2_dd_ortho,
            pk_l_fNL_4_vv_ortho, pk_l_fNL_4_vd_ortho, pk_l_fNL_4_dd_ortho,
            pk12_l_0_b1b2_ortho, pk12_l_0_b2_ortho, pk12_l_0_b1bG2_ortho, pk12_l_0_bG2_ortho,
            pk12_l_2_b1b2_ortho, pk12_l_2_b2_ortho, pk12_l_2_b1bG2_ortho, pk12_l_2_bG2_ortho,
            pk12_l_4_b1b2_ortho, pk12_l_4_b2_ortho, pk12_l_4_b1bG2_ortho, pk12_l_4_bG2_ortho,
            pk_nl_fNL_ortho, pk_fNLd2_ortho, pk_fNLG2_ortho,
            pPRIMk_l, lnpPRIMk_l, ddlnpPRIMk_l
        };
        for (int fi_ = 0; fi_ < 111; fi_++) free(_f[fi_]); }

        if (ppt->has_cls == _TRUE_ && pnlpt->fast_output == _TRUE_) {
            if (pnlpt->cb == _TRUE_) {
                free(pnlpt->dd_sources_tp_delta_cb);
                free(pnlpt->sources_tp_delta_cb);
            } else {
                free(pnlpt->dd_sources_tp_delta_m);
                free(pnlpt->sources_tp_delta_m);
            }
        }

        if (pnlpt->nonlinear_pt_verbose > 0)
            printf(" 'nonlinear_pt_init' module executed successfully\n");
    } else {
        class_stop(pnlpt->error_message,
                   "Your non-linear method variable is set to %d, out of the range defined in nonlinear_pt.h", pnlpt->method);
    }

    return _SUCCESS_;
}

/* === MODULE CLEANUP === */

/**
 * Free all memory allocated by the PT module.
 */
int nonlinear_pt_free(
    struct nonlinear_pt *pnlpt) {
    if (pnlpt->method > nlpt_none) {
        if (pnlpt->method == nlpt_spt) {
            /* Free kernel matrices */

            free(pnlpt->M13_oneline);
            free(pnlpt->M22_oneline);
            free(pnlpt->M22basic_oneline);
            free(pnlpt->IFG2_oneline);
            free(pnlpt->k);
            free(pnlpt->ln_k);
            free(pnlpt->tau);
            free(pnlpt->ln_tau);
            /* Free all complex matrices */
            for (int ci_ = 0; ci_ < N_M22_COMPLEX; ci_++) {
                complex double **ptr = (complex double **)((char *)pnlpt + m22_complex_offsets[ci_]);
                free(*ptr);
            }
            for (int ci_ = 0; ci_ < N_M13_COMPLEX; ci_++) {
                complex double **ptr = (complex double **)((char *)pnlpt + m13_complex_offsets[ci_]);
                free(*ptr);
            }
            /* Free all real M12 kernel matrices */
            for (int ri_ = 0; ri_ < N_M12_REAL; ri_++) {
                double **ptr = (double **)((char *)pnlpt + m12_real_offsets[ri_]);
                free(*ptr);
            }
            free(pnlpt->growthf);
            free(pnlpt->hratio_array);
            free(pnlpt->Dratio_array);
            free(pnlpt->nl_corr_density);
            /* Free all ln_pk output arrays */
            for (int lpi_ = 0; lpi_ < N_LN_PK; lpi_++) {
                double **ptr = (double **)((char *)pnlpt + ln_pk_offsets[lpi_]);
                free(*ptr);
            }
            free(pnlpt->gauss_x);
            free(pnlpt->gauss);
            free(pnlpt->gauss_w);

            /* DW: release the exact-QS fast tracer snapshots.  The global
             * kernel-cache storage keeps only reusable capacity; its logical
             * contents are reset at the next nonlinear_pt_init cosmology. */
            free(pnlpt->qcdm_fast_kdisc);   pnlpt->qcdm_fast_kdisc=NULL;
            free(pnlpt->qcdm_fast_Pbin);    pnlpt->qcdm_fast_Pbin=NULL;
            free(pnlpt->qcdm_fast_Pphi);    pnlpt->qcdm_fast_Pphi=NULL;
            free(pnlpt->qcdm_fast_Mphi);    pnlpt->qcdm_fast_Mphi=NULL;
            free(pnlpt->qcdm_fast_has_png); pnlpt->qcdm_fast_has_png=NULL;
            free(pnlpt->qcdm_fast_z_ready); pnlpt->qcdm_fast_z_ready=NULL;
            pnlpt->qcdm_fast_nmax=0;
            pnlpt->qcdm_fast_tracer_ready=_FALSE_;
            pnlpt->qcdm_kernel_cache_started=_FALSE_;
            qcdm_tf_ir_fast_cache_reset();

        }
    }
    return _SUCCESS_;
}

/* === LINEAR POWER SPECTRUM EXTRACTION === */

/**
 * Extract the linear matter power spectrum P_L(k) at a given time.
 *
 * Combines perturbation source functions and primordial spectrum
 * to produce P_L(k), stored as ln(P) with spline derivatives
 * for interpolation.
 */
int nonlinear_pt_pk_l(
    struct background *pba,
    struct perturbations *ppt,
    struct primordial *ppm,
    struct nonlinear_pt *pnlpt,
    int index_tau,
    double *pk_l,
    double *lnk,
    double *lnpk,
    double *ddlnpk) {
    int index_md;
    int index_ic = 0;
    int index_type;
    int index_k;
    int index_ic1, index_ic2, index_ic1_ic2;
    double *primordial_pk;
    double source_ic1, source_ic2;

    /* Replace linear spectrum with a given input file */
    if (pnlpt->replace_pk) {
        /* open input interpolation file */
        char line[100000];
        FILE *fp;
        errno = 0;
        fp = fopen(pnlpt->input_pk, "r");

        if (fp == NULL) {
            fprintf(stderr, "Interpolation file %s not found\n", pnlpt->input_pk);
            fprintf(stderr, "Trying again in 3sec\n");
            sleep(3);
            fp = fopen(pnlpt->input_pk, "r");
            if (fp == NULL) {
                fprintf(stderr, "Interpolation file %s still not found\n Aborting!\n", pnlpt->input_pk);
                fprintf(stderr, "The error code is %d.\n", errno);
                fprintf(stderr, "Error opening file: %s\n", strerror(errno));
                exit(0);
            }
        }
        /* Count lines to construct the correct size */
        int nline = 0;
        while (fgets(line, 10000, fp) != NULL) {
            if (line[0] == '#')
                continue; /* skip comment lines starting with '#' */
            if (line[0] == '\n')
                continue;
            nline++;
        }
        rewind(fp);

        /* Allocate memory for k and P(k) arrays */
        double kint[nline], pkint[nline], kmin, kmax;

        int line_count = 0;
        int counter = 0;

        /* Read in values from file */
        while (fgets(line, 100000, fp) != NULL) {
            /* Skip comment and blank lines */
            if (line[0] == '#')
                continue;
            if (line[0] == '\n')
                continue;

            /* Split into tab-separated variables */
            char *split_string;
            split_string = strtok(line, "\t");
            counter = 0;

            /* Iterate over columns in line */
            while (split_string != NULL) {
                if (counter == 0) {
                    kint[line_count] = atof(split_string);
                }
                if (counter == 1) {
                    pkint[line_count] = atof(split_string);
                }
                if (counter > 1) {
                    fprintf(stderr, "Incorrect file format");
                    abort();
                }
                split_string = strtok(NULL, "\t");
                counter++;
            }
            line_count++;
        }
        kmin = kint[0];
        kmax = kint[nline - 1];
        double this_pk[1], this_k;

        for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
            lnk[index_k] = log(pnlpt->k[index_k]);

            this_k = pnlpt->k[index_k];
            this_pk[0] = -1;
            if (this_k < kmin)
                this_pk[0] = pkint[0];
            else if (this_k > kmax)
                this_pk[0] = pkint[nline - 1];
            else
            {
                int segment;
                for (segment = 0; segment < nline - 1; segment++) {
                    if ((kint[segment + 1] >= this_k) && (kint[segment] <= this_k)) {
                        this_pk[0] = pkint[segment] + (pkint[segment + 1] - pkint[segment]) * (this_k - kint[segment]) / (kint[segment + 1] - kint[segment]);
                        break;
                    }
                }
                if (this_pk[0] == -1)
                    abort();
            }
            pk_l[index_k] = this_pk[0];
            class_test((qcdm_qscf_finite_double(pk_l[index_k]) == 0) || (pk_l[index_k] < 0.0),
                       pnlpt->error_message,
                       "Invalid replacement linear P(k): index_k=%d k=%e P=%e",
                       index_k,pnlpt->k[index_k],pk_l[index_k]);
            lnpk[index_k] = log(MAX(pk_l[index_k],1.e-300));
        }
        fclose(fp);
    }
    /* Compute linear power from CLASS as usual */
    else
    {
        index_md = ppt->index_md_scalars;

        class_alloc(primordial_pk, ppm->ic_ic_size[index_md] * sizeof(double), pnlpt->error_message);

        for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
            class_call(primordial_spectrum_at_k(ppm,
                                                index_md,
                                                linear,
                                                pnlpt->k[index_k],
                                                primordial_pk),
                       ppm->error_message,
                       pnlpt->error_message);

            pk_l[index_k] = 0;

            /* part diagonal in initial conditions */
            for (index_ic1 = 0; index_ic1 < ppm->ic_size[index_md]; index_ic1++) {
                index_ic1_ic2 = index_symmetric_matrix(index_ic1, index_ic1, ppm->ic_size[index_md]);

                if (ppt->has_cls == _TRUE_ && pnlpt->fast_output == _TRUE_) {
                    if (pnlpt->cb == _TRUE_) {
                        source_ic1 = pnlpt->sources_tp_delta_cb[index_tau * pnlpt->k_size + index_k];
                    } else {
                        source_ic1 = pnlpt->sources_tp_delta_m[index_tau * pnlpt->k_size + index_k];
                    }
                } else {
                    if (pnlpt->cb == _TRUE_) {
                        source_ic1 = ppt->sources[index_md][index_ic1 * ppt->tp_size[index_md] + ppt->index_tp_delta_cb][index_tau * ppt->k_size[index_md] + index_k];
                    } else {
                        source_ic1 = ppt->sources[index_md][index_ic1 * ppt->tp_size[index_md] + ppt->index_tp_delta_m][index_tau * ppt->k_size[index_md] + index_k];
                    }
                }

                /* source_ic1 and source_ic2 are the transfer functions */

                pk_l[index_k] += 2. * _PI_ * _PI_ / pow(pnlpt->k[index_k], 3) * source_ic1 * source_ic1 * primordial_pk[index_ic1_ic2];
            }

            /* part non-diagonal in initial conditions */
            for (index_ic1 = 0; index_ic1 < ppm->ic_size[index_md]; index_ic1++) {
                for (index_ic2 = index_ic1 + 1; index_ic2 < ppm->ic_size[index_md]; index_ic2++) {
                    index_ic1_ic2 = index_symmetric_matrix(index_ic1, index_ic2, ppm->ic_size[index_md]);
                    if (ppm->is_non_zero[index_md][index_ic1_ic2] == _TRUE_) {
                        source_ic1 = ppt->sources[index_md]
                                                 [index_ic1 * ppt->tp_size[index_md] + ppt->index_tp_delta_m]
                                                 [index_tau * ppt->k_size[index_md] + index_k];
                        source_ic2 = ppt->sources[index_md]
                                                 [index_ic2 * ppt->tp_size[index_md] + ppt->index_tp_delta_m]
                                                 [index_tau * ppt->k_size[index_md] + index_k];
                        pk_l[index_k] += 2. * 2. * _PI_ * _PI_ / pow(pnlpt->k[index_k], 3) * source_ic1 * source_ic2 * primordial_pk[index_ic1_ic2]; /* extra factor of 2 for the symmetric term (ic2,ic1) */
                    }
                }
            }

            class_test((qcdm_qscf_finite_double(pk_l[index_k]) == 0) || (pk_l[index_k] < 0.0),
                       pnlpt->error_message,
                       "Invalid CLASS linear P(k) in nonlinear_pt_pk_l: index_tau=%d index_k=%d k=%e P=%e",
                       index_tau,index_k,pnlpt->k[index_k],pk_l[index_k]);
            lnk[index_k] = log(pnlpt->k[index_k]);
            lnpk[index_k] = log(MAX(pk_l[index_k],1.e-300));
        }
        free(primordial_pk);
    }

    SPLINE_SETUP(lnk, pnlpt->k_size, lnpk, ddlnpk, _SPLINE_NATURAL_);

    return _SUCCESS_;

} /* end nonlinear_pt_pk_l */

/* === PRIMORDIAL POWER SPECTRUM EXTRACTION === */

/**
 * Extract the primordial power spectrum P_prim(k).
 *
 * Used for fNL calculations: the transfer function T(k) is obtained
 * by dividing P_L(k) by P_prim(k), since P_L = T^2 * P_prim.
 */
int nonlinear_pt_pPRIMk_l(
    struct perturbations *ppt,
    struct primordial *ppm,
    struct nonlinear_pt *pnlpt,
    double *lnk,
    double *pPRIMk_l,
    double *lnpPRIMk,
    double *ddlnpPRIMk) {
    int index_md;
    int index_ic = 0;
    int index_type;
    int index_k;
    int index_ic1, index_ic2, index_ic1_ic2;
    double *primordial_pk;

    index_md = ppt->index_md_scalars;

    class_alloc(primordial_pk, ppm->ic_ic_size[index_md] * sizeof(double), pnlpt->error_message);

    for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
        class_call(primordial_spectrum_at_k(ppm,
                                            index_md,
                                            linear,
                                            pnlpt->k[index_k],
                                            primordial_pk),
                   ppm->error_message,
                   pnlpt->error_message);

        pPRIMk_l[index_k] = 0;

        /* part diagonal in initial conditions */
        for (index_ic1 = 0; index_ic1 < ppm->ic_size[index_md]; index_ic1++) {
            index_ic1_ic2 = index_symmetric_matrix(index_ic1, index_ic1, ppm->ic_size[index_md]);

            /* source_ic1 and source_ic2 are the transfer functions */

            pPRIMk_l[index_k] += 2. * _PI_ * _PI_ * (pnlpt->k[index_k]) * primordial_pk[index_ic1_ic2] / ppm->A_s;
        }

        /* part non-diagonal in initial conditions */
        for (index_ic1 = 0; index_ic1 < ppm->ic_size[index_md]; index_ic1++) {
            for (index_ic2 = index_ic1 + 1; index_ic2 < ppm->ic_size[index_md]; index_ic2++) {
                index_ic1_ic2 = index_symmetric_matrix(index_ic1, index_ic2, ppm->ic_size[index_md]);
                if (ppm->is_non_zero[index_md][index_ic1_ic2] == _TRUE_) {
                    pPRIMk_l[index_k] += 2. * 2. * _PI_ * _PI_ * (pnlpt->k[index_k]) * primordial_pk[index_ic1_ic2] / ppm->A_s;
                }
            }
        }

        lnk[index_k] = log(pnlpt->k[index_k]);

        lnpPRIMk[index_k] = log(pPRIMk_l[index_k]);
    }

    free(primordial_pk);

    SPLINE_SETUP(lnk, pnlpt->k_size, lnpPRIMk, ddlnpPRIMk, _SPLINE_NATURAL_);

    return _SUCCESS_;

} /* end nonlinear_pt_pk_l */

/* ================================================================== */
/* === MAIN ONE-LOOP COMPUTATION ENGINE === */
/* ================================================================== */

/**
 * Compute all one-loop power spectra at a single redshift.
 *
 * This is the core computational routine. Given the linear P(k) and
 * growth rate f at one redshift, it computes:
 *   1. FFTLog decomposition of P_L(k) into power-law basis
 *   2. P22 and P13 one-loop integrals via precomputed kernel matrices
 *   3. IR resummation of BAO wiggles (Senatore & Zaldarriaga)
 *   4. RSD multipoles (ell=0,2,4) for velocity-velocity, velocity-density,
 *      and density-density components
 *   5. Alcock-Paczynski projection via Gauss-Legendre quadrature
 *   6. Bias operator integrals (Id2d2, Id2G2, IG2G2, IFG2)
 *   7. fNL contributions (local, equilateral, orthogonal)
 */

/* ==================================================================
 * BATCH BLAS-3 HELPERS FOR REAL-VALUED PT COMPUTATIONS
 *
 * The FFTLog decomposition represents P(k) as a sum of power laws:
 *   P(k) = sum_m c_m * k^{eta_m}
 * where c_m are complex coefficients and eta_m = b + 2*pi*i*m/(N*Delta).
 *
 * The one-loop integrals become matrix operations on X[c,j]:
 *   X[c,j] = c_m * k_j^{eta_m}   (the "X matrix")
 *   P13[j] = X[j,:]^T * M13      (dot product, dgemv)
 *   P22[j] = X[j,:]^T * M22 * X[j,:]  (quadratic form, dsymm+ddot)
 *   P12[j] = Xl[j,:]^T * M12 * Xr[j,:]  (bilinear form)
 *
 * All operations use real BLAS (dgemv, dsymm, ddot) by separating
 * real and imaginary parts, avoiding complex BLAS dependencies.
 * ================================================================== */

/** Build X matrix: X[c,j] = coeffs[c] * k[j]^exponents[c] for all j.
 *  Layout: column-major, X[c + j*ncoeff] for coefficient c at wavenumber j.
 *
 *  Optimization: exponents have the form b + i*omega[c] where b is a real
 *  constant (the FFTLog bias). So k^(b+i*omega) = k^b * exp(i*omega*ln(k)).
 *  kb_cache and lnk_cache are pre-computed arrays of k^b and ln(k);
 *  if NULL, they are computed internally (fallback). */
static void build_X_matrix(
    int nk, int ncoeff,
    const double complex *coeffs,
    const double complex *exponents,
    const double *k,
    double *Xr, double *Xi) {
    double b_real = creal(exponents[0]);

    for (int j = 0; j < nk; j++) {
        double lnk = log(k[j]);
        double kb = pow(k[j], b_real);
        int offset = j * ncoeff;
        for (int c = 0; c < ncoeff; c++) {
            double omega_c = cimag(exponents[c]);
            double s, co;
            sincos(omega_c * lnk, &s, &co);
            double cr_c = creal(coeffs[c]);
            double ci_c = cimag(coeffs[c]);
            Xr[c + offset] = kb * (cr_c * co - ci_c * s);
            Xi[c + offset] = kb * (cr_c * s + ci_c * co);
        }
    }
}

/** Build X matrix with cached k^b and ln(k) arrays to avoid recomputing pow/log. */
static void build_X_matrix_cached(
    int nk, int ncoeff,
    const double complex *coeffs,
    const double complex *exponents,
    const double *kb_cache,
    const double *lnk_cache,
    double *Xr, double *Xi) {
    for (int j = 0; j < nk; j++) {
        double lnk = lnk_cache[j];
        double kb = kb_cache[j];
        int offset = j * ncoeff;
        for (int c = 0; c < ncoeff; c++) {
            double omega_c = cimag(exponents[c]);
            double s, co;
            sincos(omega_c * lnk, &s, &co);
            double cr_c = creal(coeffs[c]);
            double ci_c = cimag(coeffs[c]);
            Xr[c + offset] = kb * (cr_c * co - ci_c * s);
            Xi[c + offset] = kb * (cr_c * s + ci_c * co);
        }
    }
}

/** Precompute sincos table for given exponents and ln(k) values.
 *  Layout: sc_cos[c + j*ncoeff], sc_sin[c + j*ncoeff].
 *  Caller must allocate sc_cos, sc_sin of size nk*ncoeff. */
static void precompute_sincos_table(
    int nk, int ncoeff,
    const double complex *exponents,
    const double *lnk_cache,
    double *sc_cos, double *sc_sin) {
    for (int j = 0; j < nk; j++) {
        double lnk = lnk_cache[j];
        int offset = j * ncoeff;
        for (int c = 0; c < ncoeff; c++) {
            double omega_c = cimag(exponents[c]);
            sincos(omega_c * lnk, &sc_sin[c + offset], &sc_cos[c + offset]);
        }
    }
}

/** Build X matrix from precomputed sincos table (no trig calls). */
static void build_X_from_tables(
    int nk, int ncoeff,
    const double complex *coeffs,
    const double *kb_cache,
    const double *sc_cos, const double *sc_sin,
    double *Xr, double *Xi) {
    /* Only parallelize if we have enough work per thread (nk=128, reasonable up to 8 threads) */
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < nk; j++) {
        double kb = kb_cache[j];
        int offset = j * ncoeff;
        for (int c = 0; c < ncoeff; c++) {
            double co = sc_cos[c + offset];
            double s = sc_sin[c + offset];
            double cr_c = creal(coeffs[c]);
            double ci_c = cimag(coeffs[c]);
            Xr[c + offset] = kb * (cr_c * co - ci_c * s);
            Xi[c + offset] = kb * (cr_c * s + ci_c * co);
        }
    }
}

/** Batch dot product: f[j] = X[j,:]^T · m  for all j (using real BLAS)
 *  mr_ws, mi_ws: workspace arrays of size ncoeff (caller-allocated) */
static void batch_dot_product(
    int ncoeff, int nk,
    const double *Xr, const double *Xi,
    const double complex *m_complex,
    double *f,
    double *mr_ws, double *mi_ws) {
    for (int c = 0; c < ncoeff; c++) {
        mr_ws[c] = creal(m_complex[c]);
        mi_ws[c] = cimag(m_complex[c]);
    }

    double one = 1., zero = 0., neg = -1.;
    int inc = 1;
    /* Critical section: OpenBLAS memory allocator is not thread-safe
     * when called concurrently from multiple OMP threads. */
    #pragma omp critical(blas_alloc)
    {
    dgemv_("T", &ncoeff, &nk, &one, (double *)Xr, &ncoeff, mr_ws, &inc, &zero, f, &inc);
    dgemv_("T", &ncoeff, &nk, &neg, (double *)Xi, &ncoeff, mi_ws, &inc, &one, f, &inc);
    }
}

/** Batch quadratic form: f[j] = X[j,:]^T · M · X[j,:]  for all j */
static void batch_quadratic_form(
    int ncoeff, int nk,
    const double complex *M_packed,
    const double *Xr, const double *Xi,
    double *f,
    double *Mr_ws, double *Mi_ws,
    double *Y1_ws, double *Y2_ws) {
    /* Unpack complex symmetric M (upper-triangular packed order) into real Mr and Mi */
    int k = 0;
    for (int j = 0; j < ncoeff; j++) {
        for (int i = j; i < ncoeff; i++) {
            Mr_ws[j + i * ncoeff] = Mr_ws[i + j * ncoeff] = creal(M_packed[k]);
            Mi_ws[j + i * ncoeff] = Mi_ws[i + j * ncoeff] = cimag(M_packed[k]);
            k++;
        }
    }

    /* f = diag(Xr^T Mr Xr - Xi^T Mr Xi - 2 Xr^T Mi Xi) via dsymm.
     * Critical section: OpenBLAS memory allocator is not thread-safe
     * when called concurrently from multiple OMP threads. */
    double one = 1., zero = 0.;

    /* Y1 = Mr * Xr, Y2 = Mr * Xi */
    #pragma omp critical(blas_alloc)
    {
    dsymm_("L", "U", &ncoeff, &nk, &one, Mr_ws, &ncoeff, (double *)Xr, &ncoeff, &zero, Y1_ws, &ncoeff);
    dsymm_("L", "U", &ncoeff, &nk, &one, Mr_ws, &ncoeff, (double *)Xi, &ncoeff, &zero, Y2_ws, &ncoeff);
    }

    /* Extract diagonal: f[j] = Xr[:,j].Y1[:,j] - Xi[:,j].Y2[:,j] */
    for (int j = 0; j < nk; j++) {
        const double *xr_j = &Xr[j * ncoeff];
        const double *xi_j = &Xi[j * ncoeff];
        const double *y1_j = &Y1_ws[j * ncoeff];
        const double *y2_j = &Y2_ws[j * ncoeff];
        double s = 0.;
        for (int c = 0; c < ncoeff; c++)
            s += xr_j[c] * y1_j[c] - xi_j[c] * y2_j[c];
        f[j] = s;
    }

    /* Y1 = Mi * Xi */
    #pragma omp critical(blas_alloc)
    dsymm_("L", "U", &ncoeff, &nk, &one, Mi_ws, &ncoeff, (double *)Xi, &ncoeff, &zero, Y1_ws, &ncoeff);

    /* f[j] -= 2 * Xr[:,j].Y1[:,j] */
    for (int j = 0; j < nk; j++) {
        const double *xr_j = &Xr[j * ncoeff];
        const double *y1_j = &Y1_ws[j * ncoeff];
        double s = 0.;
        for (int c = 0; c < ncoeff; c++)
            s += xr_j[c] * y1_j[c];
        f[j] -= 2. * s;
    }
}

/** Batch bilinear form: f[j] = Re[ Xl[j,:]^H · M · Xr[j,:] ] for all j
 *  where Xl = Xlr + i*Xli, Xr = Xrr + i*Xri, M = Mr + i*Mi (symmetric).
 *  Uses 4 dsymm calls: Mr*Xrr, Mi*Xrr, Mr*Xri, Mi*Xri. */
static void batch_bilinear_form(
    int ncoeff, int nk,
    const double complex *M_packed,
    const double *Xlr, const double *Xli,
    const double *Xrr, const double *Xri,
    double *f,
    double *Mr_ws, double *Mi_ws, double *Y_ws) {
    /* Unpack M (upper-triangular packed order) */
    int k = 0;
    for (int j = 0; j < ncoeff; j++) {
        for (int i = j; i < ncoeff; i++) {
            Mr_ws[j + i * ncoeff] = Mr_ws[i + j * ncoeff] = creal(M_packed[k]);
            Mi_ws[j + i * ncoeff] = Mi_ws[i + j * ncoeff] = cimag(M_packed[k]);
            k++;
        }
    }

    /* Critical section: OpenBLAS memory allocator is not thread-safe
     * when called concurrently from multiple OMP threads. */
    double one = 1., zero = 0.;

    /* Y = Mr * Xrr;  f[j] = Xlr[:,j] . Y[:,j] */
    #pragma omp critical(blas_alloc)
    dsymm_("L", "U", &ncoeff, &nk, &one, Mr_ws, &ncoeff, (double *)Xrr, &ncoeff, &zero, Y_ws, &ncoeff);
    for (int j = 0; j < nk; j++) {
        double s = 0.;
        for (int c = 0; c < ncoeff; c++)
            s += Xlr[j * ncoeff + c] * Y_ws[j * ncoeff + c];
        f[j] = s;
    }

    /* Y = Mi * Xrr;  f[j] -= Xli[:,j] . Y[:,j] */
    #pragma omp critical(blas_alloc)
    dsymm_("L", "U", &ncoeff, &nk, &one, Mi_ws, &ncoeff, (double *)Xrr, &ncoeff, &zero, Y_ws, &ncoeff);
    for (int j = 0; j < nk; j++) {
        double s = 0.;
        for (int c = 0; c < ncoeff; c++)
            s += Xli[j * ncoeff + c] * Y_ws[j * ncoeff + c];
        f[j] -= s;
    }

    /* Y = Mi * Xri;  f[j] -= Xlr[:,j] . Y[:,j] */
    #pragma omp critical(blas_alloc)
    dsymm_("L", "U", &ncoeff, &nk, &one, Mi_ws, &ncoeff, (double *)Xri, &ncoeff, &zero, Y_ws, &ncoeff);
    for (int j = 0; j < nk; j++) {
        double s = 0.;
        for (int c = 0; c < ncoeff; c++)
            s += Xlr[j * ncoeff + c] * Y_ws[j * ncoeff + c];
        f[j] -= s;
    }

    /* Y = Mr * Xri;  f[j] -= Xli[:,j] . Y[:,j] */
    #pragma omp critical(blas_alloc)
    dsymm_("L", "U", &ncoeff, &nk, &one, Mr_ws, &ncoeff, (double *)Xri, &ncoeff, &zero, Y_ws, &ncoeff);
    for (int j = 0; j < nk; j++) {
        double s = 0.;
        for (int c = 0; c < ncoeff; c++)
            s += Xli[j * ncoeff + c] * Y_ws[j * ncoeff + c];
        f[j] -= s;
    }
}

/* Assembly helpers: combine FFTLog coefficients with UV counterterms and
 * apply high-k exponential cutoff exp(-(k/cutoff)^6) to regularize. */

/** Assemble P13: P13[j] = (k^3 * f * P_L + P13UV) * exp(-cutoff) */
static void assemble_P13(int nk, const double *k, const double *f,
                         const double *Pbin, const double *P13UV,
                         double cutoff, double *P13) {
    double inv_cutoff = 1.0 / cutoff;
    for (int j = 0; j < nk; j++) {
        double kc = k[j] * inv_cutoff; double kc2 = kc * kc; double kc6 = kc2 * kc2 * kc2;
        P13[j] = (k[j] * k[j] * k[j] * f[j] * Pbin[j] + P13UV[j]) * exp(-kc6);
    }
}

/** Assemble P22: P22[j] = k^3 * f * exp(-cutoff) */
static void assemble_P22(int nk, const double *k, const double *f,
                         double cutoff, double *P22) {
    double inv_cutoff = 1.0 / cutoff;
    for (int j = 0; j < nk; j++) {
        double kc = k[j] * inv_cutoff; double kc2 = kc * kc; double kc6 = kc2 * kc2 * kc2;
        P22[j] = k[j] * k[j] * k[j] * f[j] * exp(-kc6);
    }
}


/* ========================================================================== */
/* === DW: QCDM-QSCF quasi-static exact real-space kernels =================== */
/* ========================================================================== */
/* This branch bypasses the precomputed EdS FFTLog M22/M13 matrices for the
 * real-space matter P22/P13 pieces whenever qcdm is present. QCDM replaces CDM;
 * the loop field is baryons + qcdm. The scalar perturbation is integrated out in
 * the quasi-static approximation and appears as mu_q(k,a) in the qcdm Euler
 * equation. Returned F2/F3 kernels are reduced with respect to the final-time
 * total-matter linear field, so the direct loop integrals can use P_L^mm(tau). */

typedef struct qcdm_qscf_qs_coefficients {
    /* DW: CLASS index_bg_H is the physical Hubble rate H.  Store both H and
     * the conformal rate Hconf=aH explicitly so time-variable conversions
     * cannot silently lose a factor of the scale factor. */
    double a, Hphys, Hconf, Hconf_prime_over_Hconf2, Omega_b, Omega_q;
    /* DW: scalar velocity with respect to N=ln(a), phi_N=phi'/Hconf. */
    double dphi_dN;
    double ealpha, dalpha, ddalpha;
    double meff2;
    double A_b, A_q;
} qcdm_qscf_qs_coefficients;

typedef struct qcdm_qscf_kvec3 { double x, y, z; } qcdm_qscf_kvec3;

typedef struct qcdm_qscf_kernel2_result {
    double F2_b, G2_b, F2_q, G2_q;
    double F2_m_red, G2_m_red;
    double F2_r_red, G2_r_red;
    double F1m_k1, F1m_k2;
    double F1r_k1, F1r_k2;
    double G1m_k1, G1m_k2;
    double G1r_k1, G1r_k2;
} qcdm_qscf_kernel2_result;

typedef struct qcdm_qscf_kernel1_result {
    double F1_m;
    double G1_m_red;
    double F1_r_red;
    double G1_r_red;
} qcdm_qscf_kernel1_result;

typedef struct qcdm_qscf_kernel3_result {
    double F3_m_red, G3_m_red;
    double F3_r_red, G3_r_red;
} qcdm_qscf_kernel3_result;


typedef struct qcdm_qscf_f1_cache_entry {
    double tau, k;
    qcdm_qscf_kernel1_result res;
} qcdm_qscf_f1_cache_entry;

typedef struct qcdm_qscf_f1m_cache_entry {
    double tau, k;
    double F1m;
} qcdm_qscf_f1m_cache_entry;

typedef struct qcdm_qscf_f2_cache_entry {
    double tau, k1, k2, mu12;
    qcdm_qscf_kernel2_result res;
} qcdm_qscf_f2_cache_entry;

typedef struct qcdm_qscf_f3_cache_entry {
    double tau, k, q, mu;
    qcdm_qscf_kernel3_result res;
} qcdm_qscf_f3_cache_entry;

typedef struct qcdm_qscf_f3m_cache_entry {
    double tau, k, q, mu;
    double F3m;
} qcdm_qscf_f3m_cache_entry;

typedef struct qcdm_qscf_qs_cache {
    qcdm_qscf_f1_cache_entry *f1;
    int f1_size, f1_cap;
    int *f1_slot; int f1_slot_cap;
    qcdm_qscf_f1m_cache_entry *f1m;
    int f1m_size, f1m_cap;
    int *f1m_slot; int f1m_slot_cap;
    qcdm_qscf_f2_cache_entry *f2;
    int f2_size, f2_cap;
    int *f2_slot; int f2_slot_cap;
    qcdm_qscf_f3_cache_entry *f3;
    int f3_size, f3_cap;
    int *f3_slot; int f3_slot_cap;
    qcdm_qscf_f3m_cache_entry *f3m;
    int f3m_size, f3m_cap;
    int *f3m_slot; int f3m_slot_cap;
    long f1_hits, f1_misses;
    long f1m_hits, f1m_misses;
    long f2_hits, f2_misses;
    long f3_hits, f3_misses;
    long f3m_hits, f3m_misses;
} qcdm_qscf_qs_cache;

static qcdm_qscf_qs_cache qcdm_qscf_cache = {0};

/* DW: during the native-basis OpenMP contraction the exact kernel cache is
 * intentionally read-only.  nonlinear_pt_init() has already populated the
 * required F1/F2/F3 entries.  If a supposedly warm lookup misses while this
 * flag is set, the solver returns an error instead of mutating the global
 * hash tables from multiple threads. */
static int qcdm_qscf_cache_readonly_parallel = _FALSE_;
/* DW speed stage 2: initialization RSD also parallelizes over external k,
 * but unlike the native basis it is responsible for creating many kernel
 * entries.  To avoid races in the process-global hash tables, a parallel
 * no-store mode allows cache hits but computes misses privately without
 * inserting them.  The RK background-stage caches are prewarmed serially. */
static int qcdm_qscf_cache_parallel_nostore = _FALSE_;

/* DW: expose and validate exact-time kernel ODE resolution controls. */
/* Runtime controls copied from the nonlinear_pt input structure before each
 * exact-QS redshift calculation.  This module already uses a process-global
 * kernel cache, so keeping the matching integration controls beside it does
 * not introduce an additional re-entrancy limitation. */
static double qcdm_qscf_kernel_z_ini_runtime = 99.0;
static int qcdm_qscf_kernel_steps_f1_runtime = 128;
static int qcdm_qscf_kernel_steps_f2_runtime = 128;
static int qcdm_qscf_kernel_steps_f3_runtime = 128;

/* DW speed: the matching conformal time and the RK4 background coefficients
 * depend only on the cosmological background, target redshift, and RK4
 * resolution, not on the Fourier triangle. Cache them once per exact-QS
 * redshift calculation. This changes no kernel equations or quadrature. */
static double qcdm_qscf_kernel_tau_zini_cached = -1.0;

typedef struct qcdm_qscf_rk_stage {
    double tau;
    double fac;
    qcdm_qscf_qs_coefficients qc;
} qcdm_qscf_rk_stage;

typedef struct qcdm_qscf_rk_stage_cache {
    int valid;
    double tau_final;
    int nstep;
    qcdm_qscf_rk_stage *stage; /* three unique RK nodes per step: x, x+h/2, x+h */
    int stage_cap;
} qcdm_qscf_rk_stage_cache;

/* F1/F2/F3 can use different user-selected RK resolutions, hence three slots. */
static qcdm_qscf_rk_stage_cache qcdm_qscf_rk_cache[3] = {{0}};

static void qcdm_qscf_qs_configure_runtime(const struct nonlinear_pt *pnlpt) {
    qcdm_qscf_kernel_z_ini_runtime = pnlpt->qcdm_kernel_z_ini;
    qcdm_qscf_kernel_steps_f1_runtime = pnlpt->qcdm_kernel_steps_f1;
    qcdm_qscf_kernel_steps_f2_runtime = pnlpt->qcdm_kernel_steps_f2;
    qcdm_qscf_kernel_steps_f3_runtime = pnlpt->qcdm_kernel_steps_f3;
}

static int qcdm_qscf_cache_close(double a,double b) {
    double scale = 1.0 + fabs(a) + fabs(b);
    return (fabs(a-b) <= 1.e-12*scale) ? _TRUE_ : _FALSE_;
}

static void qcdm_qscf_qs_cache_reset(void) {
    qcdm_qscf_kernel_tau_zini_cached = -1.0;
    for (int irk=0; irk<3; irk++) qcdm_qscf_rk_cache[irk].valid = _FALSE_;
    qcdm_qscf_cache.f1_size = 0;
    qcdm_qscf_cache.f1m_size = 0;
    qcdm_qscf_cache.f2_size = 0;
    qcdm_qscf_cache.f3_size = 0;
    qcdm_qscf_cache.f3m_size = 0;
    if (qcdm_qscf_cache.f1_slot) memset(qcdm_qscf_cache.f1_slot,0,(size_t)qcdm_qscf_cache.f1_slot_cap*sizeof(int));
    if (qcdm_qscf_cache.f1m_slot) memset(qcdm_qscf_cache.f1m_slot,0,(size_t)qcdm_qscf_cache.f1m_slot_cap*sizeof(int));
    if (qcdm_qscf_cache.f2_slot) memset(qcdm_qscf_cache.f2_slot,0,(size_t)qcdm_qscf_cache.f2_slot_cap*sizeof(int));
    if (qcdm_qscf_cache.f3_slot) memset(qcdm_qscf_cache.f3_slot,0,(size_t)qcdm_qscf_cache.f3_slot_cap*sizeof(int));
    if (qcdm_qscf_cache.f3m_slot) memset(qcdm_qscf_cache.f3m_slot,0,(size_t)qcdm_qscf_cache.f3m_slot_cap*sizeof(int));
    qcdm_qscf_cache.f1_hits = qcdm_qscf_cache.f1_misses = 0;
    qcdm_qscf_cache.f1m_hits = qcdm_qscf_cache.f1m_misses = 0;
    qcdm_qscf_cache.f2_hits = qcdm_qscf_cache.f2_misses = 0;
    qcdm_qscf_cache.f3_hits = qcdm_qscf_cache.f3_misses = 0;
    qcdm_qscf_cache.f3m_hits = qcdm_qscf_cache.f3m_misses = 0;
}

static int qcdm_qscf_cache_grow(void **ptr,int *cap,size_t item_size,ErrorMsg errmsg) {
    int new_cap = (*cap == 0) ? 1024 : 2*(*cap);
    void *new_ptr = realloc(*ptr,(size_t)new_cap*item_size);
    class_test(new_ptr == NULL,errmsg,"Could not allocate QCDM/QSCF kernel cache");
    *ptr = new_ptr;
    *cap = new_cap;
    return _SUCCESS_;
}

/* DW speed: O(1)-average open-addressing indices for the kernel caches.
 * The old implementation linearly scanned all previous entries, so a run with
 * O(10^4-10^5) unique triangles paid O(N^2) key-comparison cost even when
 * every lookup was a miss. The hash index changes only cache lookup mechanics;
 * a miss still launches exactly the same kernel ODE solve. */
static uint64_t qcdm_qscf_hash_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static uint64_t qcdm_qscf_hash_double(double x) {
    uint64_t u;
    if (x==0.0) x=0.0; /* canonicalize -0 */
    memcpy(&u,&x,sizeof(double));
    return qcdm_qscf_hash_mix64(u);
}

static uint64_t qcdm_qscf_hash2(double a,double b) {
    uint64_t h=qcdm_qscf_hash_double(a);
    h ^= qcdm_qscf_hash_mix64(qcdm_qscf_hash_double(b)+0x9e3779b97f4a7c15ULL);
    return qcdm_qscf_hash_mix64(h);
}

static uint64_t qcdm_qscf_hash4(double a,double b,double c,double d) {
    uint64_t h=qcdm_qscf_hash2(a,b);
    h ^= qcdm_qscf_hash_mix64(qcdm_qscf_hash_double(c)+0x517cc1b727220a95ULL);
    h ^= qcdm_qscf_hash_mix64(qcdm_qscf_hash_double(d)+0x6eed0e9da4d94a4fULL);
    return qcdm_qscf_hash_mix64(h);
}

static int qcdm_qscf_rehash_f1(int newcap,ErrorMsg errmsg) {
    int *slot=calloc((size_t)newcap,sizeof(int));
    class_test(slot==NULL,errmsg,"Could not allocate QCDM F1 hash index");
    for (int i=0;i<qcdm_qscf_cache.f1_size;i++) {
        qcdm_qscf_f1_cache_entry *e=&qcdm_qscf_cache.f1[i];
        int pos=(int)(qcdm_qscf_hash2(e->tau,e->k)&(uint64_t)(newcap-1));
        while(slot[pos]!=0) pos=(pos+1)&(newcap-1);
        slot[pos]=i+1;
    }
    free(qcdm_qscf_cache.f1_slot); qcdm_qscf_cache.f1_slot=slot; qcdm_qscf_cache.f1_slot_cap=newcap;
    return _SUCCESS_;
}
static int qcdm_qscf_rehash_f1m(int newcap,ErrorMsg errmsg) {
    int *slot=calloc((size_t)newcap,sizeof(int));
    class_test(slot==NULL,errmsg,"Could not allocate QCDM F1m hash index");
    for (int i=0;i<qcdm_qscf_cache.f1m_size;i++) {
        qcdm_qscf_f1m_cache_entry *e=&qcdm_qscf_cache.f1m[i];
        int pos=(int)(qcdm_qscf_hash2(e->tau,e->k)&(uint64_t)(newcap-1));
        while(slot[pos]!=0) pos=(pos+1)&(newcap-1);
        slot[pos]=i+1;
    }
    free(qcdm_qscf_cache.f1m_slot); qcdm_qscf_cache.f1m_slot=slot; qcdm_qscf_cache.f1m_slot_cap=newcap;
    return _SUCCESS_;
}
static int qcdm_qscf_rehash_f2(int newcap,ErrorMsg errmsg) {
    int *slot=calloc((size_t)newcap,sizeof(int));
    class_test(slot==NULL,errmsg,"Could not allocate QCDM F2 hash index");
    for (int i=0;i<qcdm_qscf_cache.f2_size;i++) {
        qcdm_qscf_f2_cache_entry *e=&qcdm_qscf_cache.f2[i];
        int pos=(int)(qcdm_qscf_hash4(e->tau,e->k1,e->k2,e->mu12)&(uint64_t)(newcap-1));
        while(slot[pos]!=0) pos=(pos+1)&(newcap-1);
        slot[pos]=i+1;
    }
    free(qcdm_qscf_cache.f2_slot); qcdm_qscf_cache.f2_slot=slot; qcdm_qscf_cache.f2_slot_cap=newcap;
    return _SUCCESS_;
}
static int qcdm_qscf_rehash_f3(int newcap,ErrorMsg errmsg) {
    int *slot=calloc((size_t)newcap,sizeof(int));
    class_test(slot==NULL,errmsg,"Could not allocate QCDM F3 hash index");
    for (int i=0;i<qcdm_qscf_cache.f3_size;i++) {
        qcdm_qscf_f3_cache_entry *e=&qcdm_qscf_cache.f3[i];
        int pos=(int)(qcdm_qscf_hash4(e->tau,e->k,e->q,e->mu)&(uint64_t)(newcap-1));
        while(slot[pos]!=0) pos=(pos+1)&(newcap-1);
        slot[pos]=i+1;
    }
    free(qcdm_qscf_cache.f3_slot); qcdm_qscf_cache.f3_slot=slot; qcdm_qscf_cache.f3_slot_cap=newcap;
    return _SUCCESS_;
}
static int qcdm_qscf_rehash_f3m(int newcap,ErrorMsg errmsg) {
    int *slot=calloc((size_t)newcap,sizeof(int));
    class_test(slot==NULL,errmsg,"Could not allocate QCDM F3m hash index");
    for (int i=0;i<qcdm_qscf_cache.f3m_size;i++) {
        qcdm_qscf_f3m_cache_entry *e=&qcdm_qscf_cache.f3m[i];
        int pos=(int)(qcdm_qscf_hash4(e->tau,e->k,e->q,e->mu)&(uint64_t)(newcap-1));
        while(slot[pos]!=0) pos=(pos+1)&(newcap-1);
        slot[pos]=i+1;
    }
    free(qcdm_qscf_cache.f3m_slot); qcdm_qscf_cache.f3m_slot=slot; qcdm_qscf_cache.f3m_slot_cap=newcap;
    return _SUCCESS_;
}

#define QCDM_HASH_ENSURE(kind,errmsg) do { \
    int *_cap=&qcdm_qscf_cache.kind##_slot_cap; \
    int _size=qcdm_qscf_cache.kind##_size; \
    if ((*_cap==0) || ((_size+1)*10 > (*_cap)*7)) { \
        int _newcap=(*_cap==0)?2048:2*(*_cap); \
        class_call(qcdm_qscf_rehash_##kind(_newcap,errmsg),errmsg,errmsg); \
    } \
} while(0)

static int qcdm_qscf_cache_get_f1(double tau,double k,qcdm_qscf_kernel1_result *res) {
    if (qcdm_qscf_cache.f1_slot_cap>0) {
        int mask=qcdm_qscf_cache.f1_slot_cap-1;
        int pos=(int)(qcdm_qscf_hash2(tau,k)&(uint64_t)mask);
        while(qcdm_qscf_cache.f1_slot[pos]!=0) {
            qcdm_qscf_f1_cache_entry *e=&qcdm_qscf_cache.f1[qcdm_qscf_cache.f1_slot[pos]-1];
            if (qcdm_qscf_cache_close(e->tau,tau) && qcdm_qscf_cache_close(e->k,k)) {
                *res=e->res; if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f1_hits++; return _TRUE_;
            }
            pos=(pos+1)&mask;
        }
    }
    if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f1_misses++; return _FALSE_;
}
static int qcdm_qscf_cache_put_f1(double tau,double k,const qcdm_qscf_kernel1_result *res,ErrorMsg errmsg) {
    if (qcdm_qscf_cache.f1_size >= qcdm_qscf_cache.f1_cap)
        class_call(qcdm_qscf_cache_grow((void**)&qcdm_qscf_cache.f1,&qcdm_qscf_cache.f1_cap,sizeof(qcdm_qscf_f1_cache_entry),errmsg),errmsg,errmsg);
    QCDM_HASH_ENSURE(f1,errmsg);
    int idx=qcdm_qscf_cache.f1_size++;
    qcdm_qscf_f1_cache_entry *e=&qcdm_qscf_cache.f1[idx]; e->tau=tau; e->k=k; e->res=*res;
    int mask=qcdm_qscf_cache.f1_slot_cap-1,pos=(int)(qcdm_qscf_hash2(tau,k)&(uint64_t)mask);
    while(qcdm_qscf_cache.f1_slot[pos]!=0) pos=(pos+1)&mask; qcdm_qscf_cache.f1_slot[pos]=idx+1;
    return _SUCCESS_;
}

static int qcdm_qscf_cache_get_f1m(double tau,double k,double *F1m) {
    if (qcdm_qscf_cache.f1m_slot_cap>0) {
        int mask=qcdm_qscf_cache.f1m_slot_cap-1,pos=(int)(qcdm_qscf_hash2(tau,k)&(uint64_t)mask);
        while(qcdm_qscf_cache.f1m_slot[pos]!=0) {
            qcdm_qscf_f1m_cache_entry *e=&qcdm_qscf_cache.f1m[qcdm_qscf_cache.f1m_slot[pos]-1];
            if (qcdm_qscf_cache_close(e->tau,tau) && qcdm_qscf_cache_close(e->k,k)) {
                *F1m=e->F1m; if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f1m_hits++; return _TRUE_;
            }
            pos=(pos+1)&mask;
        }
    }
    if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f1m_misses++; return _FALSE_;
}
static int qcdm_qscf_cache_put_f1m(double tau,double k,double F1m,ErrorMsg errmsg) {
    if (qcdm_qscf_cache.f1m_size >= qcdm_qscf_cache.f1m_cap)
        class_call(qcdm_qscf_cache_grow((void**)&qcdm_qscf_cache.f1m,&qcdm_qscf_cache.f1m_cap,sizeof(qcdm_qscf_f1m_cache_entry),errmsg),errmsg,errmsg);
    QCDM_HASH_ENSURE(f1m,errmsg);
    int idx=qcdm_qscf_cache.f1m_size++;
    qcdm_qscf_f1m_cache_entry *e=&qcdm_qscf_cache.f1m[idx]; e->tau=tau; e->k=k; e->F1m=F1m;
    int mask=qcdm_qscf_cache.f1m_slot_cap-1,pos=(int)(qcdm_qscf_hash2(tau,k)&(uint64_t)mask);
    while(qcdm_qscf_cache.f1m_slot[pos]!=0) pos=(pos+1)&mask; qcdm_qscf_cache.f1m_slot[pos]=idx+1;
    return _SUCCESS_;
}

static int qcdm_qscf_cache_get_f2(double tau,double k1,double k2,double mu12,qcdm_qscf_kernel2_result *res) {
    if (qcdm_qscf_cache.f2_slot_cap>0) {
        int mask=qcdm_qscf_cache.f2_slot_cap-1,pos=(int)(qcdm_qscf_hash4(tau,k1,k2,mu12)&(uint64_t)mask);
        while(qcdm_qscf_cache.f2_slot[pos]!=0) {
            qcdm_qscf_f2_cache_entry *e=&qcdm_qscf_cache.f2[qcdm_qscf_cache.f2_slot[pos]-1];
            if (qcdm_qscf_cache_close(e->tau,tau) && qcdm_qscf_cache_close(e->k1,k1) &&
                qcdm_qscf_cache_close(e->k2,k2) && qcdm_qscf_cache_close(e->mu12,mu12)) {
                *res=e->res; if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f2_hits++; return _TRUE_;
            }
            pos=(pos+1)&mask;
        }
    }
    if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f2_misses++; return _FALSE_;
}
static int qcdm_qscf_cache_put_f2(double tau,double k1,double k2,double mu12,const qcdm_qscf_kernel2_result *res,ErrorMsg errmsg) {
    if (qcdm_qscf_cache.f2_size >= qcdm_qscf_cache.f2_cap)
        class_call(qcdm_qscf_cache_grow((void**)&qcdm_qscf_cache.f2,&qcdm_qscf_cache.f2_cap,sizeof(qcdm_qscf_f2_cache_entry),errmsg),errmsg,errmsg);
    QCDM_HASH_ENSURE(f2,errmsg);
    int idx=qcdm_qscf_cache.f2_size++;
    qcdm_qscf_f2_cache_entry *e=&qcdm_qscf_cache.f2[idx]; e->tau=tau; e->k1=k1; e->k2=k2; e->mu12=mu12; e->res=*res;
    int mask=qcdm_qscf_cache.f2_slot_cap-1,pos=(int)(qcdm_qscf_hash4(tau,k1,k2,mu12)&(uint64_t)mask);
    while(qcdm_qscf_cache.f2_slot[pos]!=0) pos=(pos+1)&mask; qcdm_qscf_cache.f2_slot[pos]=idx+1;
    return _SUCCESS_;
}

static int qcdm_qscf_cache_get_f3(double tau,double k,double q,double mu,qcdm_qscf_kernel3_result *res) {
    if (qcdm_qscf_cache.f3_slot_cap>0) {
        int mask=qcdm_qscf_cache.f3_slot_cap-1,pos=(int)(qcdm_qscf_hash4(tau,k,q,mu)&(uint64_t)mask);
        while(qcdm_qscf_cache.f3_slot[pos]!=0) {
            qcdm_qscf_f3_cache_entry *e=&qcdm_qscf_cache.f3[qcdm_qscf_cache.f3_slot[pos]-1];
            if (qcdm_qscf_cache_close(e->tau,tau) && qcdm_qscf_cache_close(e->k,k) &&
                qcdm_qscf_cache_close(e->q,q) && qcdm_qscf_cache_close(e->mu,mu)) {
                *res=e->res; if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f3_hits++; return _TRUE_;
            }
            pos=(pos+1)&mask;
        }
    }
    if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f3_misses++; return _FALSE_;
}
static int qcdm_qscf_cache_put_f3(double tau,double k,double q,double mu,const qcdm_qscf_kernel3_result *res,ErrorMsg errmsg) {
    if (qcdm_qscf_cache.f3_size >= qcdm_qscf_cache.f3_cap)
        class_call(qcdm_qscf_cache_grow((void**)&qcdm_qscf_cache.f3,&qcdm_qscf_cache.f3_cap,sizeof(qcdm_qscf_f3_cache_entry),errmsg),errmsg,errmsg);
    QCDM_HASH_ENSURE(f3,errmsg);
    int idx=qcdm_qscf_cache.f3_size++;
    qcdm_qscf_f3_cache_entry *e=&qcdm_qscf_cache.f3[idx]; e->tau=tau; e->k=k; e->q=q; e->mu=mu; e->res=*res;
    int mask=qcdm_qscf_cache.f3_slot_cap-1,pos=(int)(qcdm_qscf_hash4(tau,k,q,mu)&(uint64_t)mask);
    while(qcdm_qscf_cache.f3_slot[pos]!=0) pos=(pos+1)&mask; qcdm_qscf_cache.f3_slot[pos]=idx+1;
    return _SUCCESS_;
}

static int qcdm_qscf_cache_get_f3m(double tau,double k,double q,double mu,double *F3m) {
    if (qcdm_qscf_cache.f3m_slot_cap>0) {
        int mask=qcdm_qscf_cache.f3m_slot_cap-1,pos=(int)(qcdm_qscf_hash4(tau,k,q,mu)&(uint64_t)mask);
        while(qcdm_qscf_cache.f3m_slot[pos]!=0) {
            qcdm_qscf_f3m_cache_entry *e=&qcdm_qscf_cache.f3m[qcdm_qscf_cache.f3m_slot[pos]-1];
            if (qcdm_qscf_cache_close(e->tau,tau) && qcdm_qscf_cache_close(e->k,k) &&
                qcdm_qscf_cache_close(e->q,q) && qcdm_qscf_cache_close(e->mu,mu)) {
                *F3m=e->F3m; if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f3m_hits++; return _TRUE_;
            }
            pos=(pos+1)&mask;
        }
    }
    if ((qcdm_qscf_cache_readonly_parallel!=_TRUE_) && (qcdm_qscf_cache_parallel_nostore!=_TRUE_)) qcdm_qscf_cache.f3m_misses++; return _FALSE_;
}
static int qcdm_qscf_cache_put_f3m(double tau,double k,double q,double mu,double F3m,ErrorMsg errmsg) {
    if (qcdm_qscf_cache.f3m_size >= qcdm_qscf_cache.f3m_cap)
        class_call(qcdm_qscf_cache_grow((void**)&qcdm_qscf_cache.f3m,&qcdm_qscf_cache.f3m_cap,sizeof(qcdm_qscf_f3m_cache_entry),errmsg),errmsg,errmsg);
    QCDM_HASH_ENSURE(f3m,errmsg);
    int idx=qcdm_qscf_cache.f3m_size++;
    qcdm_qscf_f3m_cache_entry *e=&qcdm_qscf_cache.f3m[idx]; e->tau=tau; e->k=k; e->q=q; e->mu=mu; e->F3m=F3m;
    int mask=qcdm_qscf_cache.f3m_slot_cap-1,pos=(int)(qcdm_qscf_hash4(tau,k,q,mu)&(uint64_t)mask);
    while(qcdm_qscf_cache.f3m_slot[pos]!=0) pos=(pos+1)&mask; qcdm_qscf_cache.f3m_slot[pos]=idx+1;
    return _SUCCESS_;
}

#undef QCDM_HASH_ENSURE

static void qcdm_qscf_qs_cache_print_stats(void) {
    printf("DW/QCDM-QSCF kernel cache: F1 %ld/%ld, F1m %ld/%ld, F2 %ld/%ld, F3 %ld/%ld, F3m %ld/%ld hits/misses\n",
           qcdm_qscf_cache.f1_hits,qcdm_qscf_cache.f1_misses,
           qcdm_qscf_cache.f1m_hits,qcdm_qscf_cache.f1m_misses,
           qcdm_qscf_cache.f2_hits,qcdm_qscf_cache.f2_misses,
           qcdm_qscf_cache.f3_hits,qcdm_qscf_cache.f3_misses,
           qcdm_qscf_cache.f3m_hits,qcdm_qscf_cache.f3m_misses);
}

static double qcdm_qscf_dot(qcdm_qscf_kvec3 a, qcdm_qscf_kvec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static qcdm_qscf_kvec3 qcdm_qscf_add(qcdm_qscf_kvec3 a, qcdm_qscf_kvec3 b) { qcdm_qscf_kvec3 c={a.x+b.x,a.y+b.y,a.z+b.z}; return c; }
static double qcdm_qscf_norm(qcdm_qscf_kvec3 a) { return sqrt(MAX(qcdm_qscf_dot(a,a),0.0)); }
static double qcdm_qscf_alpha_vec(qcdm_qscf_kvec3 k1, qcdm_qscf_kvec3 k2) {
    double k1sq=qcdm_qscf_dot(k1,k1); if (k1sq<=0.0) return 0.0; return 1.0+qcdm_qscf_dot(k1,k2)/k1sq;
}
static double qcdm_qscf_beta_vec(qcdm_qscf_kvec3 k1, qcdm_qscf_kvec3 k2) {
    double k1sq=qcdm_qscf_dot(k1,k1), k2sq=qcdm_qscf_dot(k2,k2); if ((k1sq<=0.0)||(k2sq<=0.0)) return 0.0;
    qcdm_qscf_kvec3 ks=qcdm_qscf_add(k1,k2); return qcdm_qscf_dot(ks,ks)*qcdm_qscf_dot(k1,k2)/(2.0*k1sq*k2sq);
}

static int qcdm_qscf_qs_get_coefficients(struct background *pba, double tau, qcdm_qscf_qs_coefficients *qc, ErrorMsg errmsg) {
/* DW: clamp only roundoff-sized RK endpoint excursions while retaining hard out-of-range errors. */
    int last_index=0;
    double tau_eval=tau;
    double tau_min=pba->tau_table[0];
    double tau_max=pba->tau_table[pba->bt_size-1];
    double tau_tol=64.0*DBL_EPSILON*(1.0+fabs(tau_min)+fabs(tau_max)+fabs(tau));
    /* DW speed: this routine sits inside every RK stage. Avoid millions of
     * heap allocations by using a stack workspace for the CLASS background
     * vector. bg_size is modest and fixed for a given CLASS run. */
    double pvecback_storage[pba->bg_size];
    double *pvecback=pvecback_storage;

    /* Fixed-step RK stages can land a few ulps outside the tabulated CLASS
     * background interval at the last step.  Clamp only this round-off-sized
     * overshoot; a genuinely out-of-range time remains a hard error. */
    class_test(tau < tau_min-tau_tol || tau > tau_max+tau_tol,
               errmsg,
               "QCDM kernel requested background outside table: tau=%e, range=[%e,%e]",
               tau,tau_min,tau_max);
    if (tau_eval < tau_min) tau_eval=tau_min;
    if (tau_eval > tau_max) tau_eval=tau_max;

    class_call(background_at_tau(pba,tau_eval,long_info,inter_normal,&last_index,pvecback),pba->error_message,errmsg);

    qc->a=pvecback[pba->index_bg_a];
    /* DW: background.c defines index_bg_H as physical H and
     * index_bg_H_prime=dH/dtau.  The kernel hierarchy is written in
     * N=ln(a), hence Hconf=aH and
     * Hconf'/Hconf^2 = 1 + H'/(a H^2). */
    qc->Hphys=pvecback[pba->index_bg_H];
    qc->Hconf=qc->a*qc->Hphys;
    qc->Hconf_prime_over_Hconf2=1.0
      +pvecback[pba->index_bg_H_prime]/(qc->a*qc->Hphys*qc->Hphys);
    qc->Omega_b=pvecback[pba->index_bg_rho_b]/pvecback[pba->index_bg_rho_crit];
    qc->Omega_q=(pba->has_qcdm==_TRUE_)?pvecback[pba->index_bg_rho_qcdm]/pvecback[pba->index_bg_rho_crit]:0.0;

    qc->dphi_dN=0.0;
    qc->ealpha=1.0;
    qc->dalpha=0.0;
    qc->ddalpha=0.0;
    qc->meff2=0.0;

    if (pba->has_qscf==_TRUE_) {
        double step=pvecback[pba->index_bg_step_qscf];
        double dstep=pvecback[pba->index_bg_dstep_qscf];
        double ddstep=pvecback[pba->index_bg_ddstep_qscf];
        double mass=pvecback[pba->index_bg_mass_qscf];
        double dmass=pvecback[pba->index_bg_dmass_qscf];
        double ddmass=pvecback[pba->index_bg_ddmass_qscf];
        double numer=pba->qscf_f*(dstep*mass+step*dmass);
        double dnumer=pba->qscf_f*(ddstep*mass+2.0*dstep*dmass+step*ddmass);

        /* DW: phi_prime_qscf is dphi/dtau.  Divide by aH, not H, to
         * obtain dphi/d ln(a), the quantity entering the dimensionless
         * QCDM Euler friction coefficient. */
        qc->dphi_dN=pvecback[pba->index_bg_phi_prime_qscf]/qc->Hconf;
        qc->ealpha=1.0+pba->qscf_f*step*mass;
        if (fabs(qc->ealpha)>0.0) {
            qc->dalpha=numer/qc->ealpha;
            qc->ddalpha=dnumer/qc->ealpha-(numer*numer)/(qc->ealpha*qc->ealpha);
        }

        /* Match perturbations.c scalar equation:
         * k^2 + a^2 ddV + 3 a^2 rho_qcdm ealpha (dalpha^2 + ddalpha).
         * The quantity stored here is the mass^2 part before the a^2 factor. */
        qc->meff2=pvecback[pba->index_bg_ddV_qscf];
        if (pba->has_qcdm==_TRUE_) {
            qc->meff2 += 3.0*pvecback[pba->index_bg_rho_qcdm]*qc->ealpha*(qc->dalpha*qc->dalpha+qc->ddalpha);
        }
    }

    if (qc->meff2<0.0) qc->meff2=0.0;
    /* DW: with Theta=-theta/Hconf and N=ln(a),
     * dTheta/dN = -[1+Hconf'/Hconf^2]Theta + sources.  Therefore
     * A_b=1+Hconf'/Hconf^2=2+d ln(H)/d ln(a). */
    qc->A_b=1.0+qc->Hconf_prime_over_Hconf2;

    /* DW: match perturbations.c QCDM Euler equation
     * theta_q' = ... - (Hconf + phi' dalpha) theta_q.  The extra
     * dimensionless friction is dalpha*phi'/Hconf=dalpha*dphi/dN. */
    qc->A_q=qc->A_b+qc->dalpha*qc->dphi_dN;

    return _SUCCESS_;
}

static double qcdm_qscf_qs_mu_q(const qcdm_qscf_qs_coefficients *qc, double k) {
    double k2=k*k, a2m2=qc->a*qc->a*qc->meff2;
    if (k2+a2m2<=0.0) return 1.0;

    /* QS scalar force matched to perturbations.c conventions.
     * The coupling is dalpha = d ln(ealpha)/d phi, not bare qscf_beta.
     * The ealpha factor comes from the scalar source term in perturbations.c. */
    return 1.0+2.0*qc->ealpha*qc->dalpha*qc->dalpha*k2/(k2+a2m2);
}

static void qcdm_qscf_qs_linear_rhs(const qcdm_qscf_qs_coefficients *qc, double k,
    double Fb,double Gb,double Fq,double Gq,double *dFb,double *dGb,double *dFq,double *dGq);

static int qcdm_qscf_qs_f2_rhs_stage(const qcdm_qscf_rk_stage *st,
                                      double k1,double k2,double K,
                                      double alpha12,double alpha21,double beta12,
                                      const double y[12],double dy[12],ErrorMsg errmsg) {
    const qcdm_qscf_qs_coefficients *qc=&st->qc;
    double dF,dG,dFq,dGq;
    double fac=st->fac;
    qcdm_qscf_qs_linear_rhs(qc,k1,y[0],y[1],y[2],y[3],&dF,&dG,&dFq,&dGq);
    dy[0]=fac*dF; dy[1]=fac*dG; dy[2]=fac*dFq; dy[3]=fac*dGq;
    qcdm_qscf_qs_linear_rhs(qc,k2,y[4],y[5],y[6],y[7],&dF,&dG,&dFq,&dGq);
    dy[4]=fac*dF; dy[5]=fac*dG; dy[6]=fac*dFq; dy[7]=fac*dGq;
    double muqK=qcdm_qscf_qs_mu_q(qc,K);
    double Sd_b=0.5*(alpha12*y[1]*y[4]+alpha21*y[5]*y[0]);
    double St_b=beta12*y[1]*y[5];
    double Sd_q=0.5*(alpha12*y[3]*y[6]+alpha21*y[7]*y[2]);
    double St_q=beta12*y[3]*y[7];
    double dF2b=y[9]+Sd_b;
    double dG2b=-qc->A_b*y[9]+1.5*(qc->Omega_b*y[8]+qc->Omega_q*y[10])+St_b;
    double dF2q=y[11]+Sd_q;
    double dG2q=-qc->A_q*y[11]+1.5*(qc->Omega_b*y[8]+qc->Omega_q*muqK*y[10])+St_q;
    dy[8]=fac*dF2b; dy[9]=fac*dG2b; dy[10]=fac*dF2q; dy[11]=fac*dG2q;
    for(int i=0;i<12;i++){
        class_test(qcdm_qscf_finite_double(dy[i])==0,errmsg,
                   "Non-finite QCDM F2 RHS: tau=%e k1=%e k2=%e K=%e component=%d y=%e dy=%e alpha12=%e alpha21=%e beta12=%e A_b=%e A_q=%e muqK=%e",
                   st->tau,k1,k2,K,i,y[i],dy[i],alpha12,alpha21,beta12,qc->A_b,qc->A_q,muqK);
    }
    return _SUCCESS_;
}

/* DW: initialize second- and third-order kernels in the growing mode at the matching redshift. */
/* EdS growing-mode kernels used only to match the pressureless kernel system
 * at its high-redshift starting surface.  The third-order kernel must not be
 * initialized to zero: doing so excites a transient and removes the part of
 * P13 generated before the matching redshift. */
static void qcdm_qscf_eds_F2G2_vec(qcdm_qscf_kvec3 k1,
                                   qcdm_qscf_kvec3 k2,
                                   double *F2,
                                   double *G2) {
    double n1=qcdm_qscf_norm(k1), n2=qcdm_qscf_norm(k2);
    if (n1<=0.0 || n2<=0.0) {
        *F2=0.0;
        *G2=0.0;
        return;
    }
    double mu=qcdm_qscf_dot(k1,k2)/(n1*n2);
    if (mu>1.0) mu=1.0;
    if (mu<-1.0) mu=-1.0;
    double r=n1/n2;
    *F2=5.0/7.0+0.5*mu*(r+1.0/r)+2.0/7.0*mu*mu;
    *G2=3.0/7.0+0.5*mu*(r+1.0/r)+4.0/7.0*mu*mu;
}

static void qcdm_qscf_eds_F3G3_sym(qcdm_qscf_kvec3 p[3],
                                    double *F3,
                                    double *G3) {
    int perms[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    double Sd=0.0, St=0.0;
    for (int ip=0; ip<6; ip++) {
        int i=perms[ip][0], j=perms[ip][1], l=perms[ip][2];
        qcdm_qscf_kvec3 pjl=qcdm_qscf_add(p[j],p[l]);
        qcdm_qscf_kvec3 pij=qcdm_qscf_add(p[i],p[j]);
        double F2jl,G2jl,F2ij,G2ij;
        qcdm_qscf_eds_F2G2_vec(p[j],p[l],&F2jl,&G2jl);
        qcdm_qscf_eds_F2G2_vec(p[i],p[j],&F2ij,&G2ij);
        (void)F2ij;
        Sd += qcdm_qscf_alpha_vec(p[i],pjl)*F2jl
            + qcdm_qscf_alpha_vec(pij,p[l])*G2ij;
        St += qcdm_qscf_beta_vec(p[i],pjl)*G2jl
            + qcdm_qscf_beta_vec(pij,p[l])*G2ij;
    }
    Sd/=6.0;
    St/=6.0;

    /* In EdS, dF3/d ln a=3F3 and dG3/d ln a=3G3.  Combining
     * 3F3=G3+Sd and (3+1/2)G3=(3/2)F3+St gives: */
    *F3=(3.5*Sd+St)/9.0;
    *G3=3.0*(*F3)-Sd;
}

static void qcdm_qscf_qs_linear_rhs(const qcdm_qscf_qs_coefficients *qc, double k,
    double Fb,double Gb,double Fq,double Gq,double *dFb,double *dGb,double *dFq,double *dGq) {
    double muq=qcdm_qscf_qs_mu_q(qc,k);
    *dFb=Gb; *dGb=-qc->A_b*Gb+1.5*(qc->Omega_b*Fb+qc->Omega_q*Fq);
    *dFq=Gq; *dGq=-qc->A_q*Gq+1.5*(qc->Omega_b*Fb+qc->Omega_q*muq*Fq);
}




static int qcdm_qscf_qs_F3_sources(struct background *pba,double tau,qcdm_qscf_kvec3 p[3],double *Sd_b,double *St_b,double *Sd_q,double *St_q,ErrorMsg errmsg);

/* DW: QCDM science-run kernel ODE integrators.
 *
 * The exact-QS kernels are evolved in x=ln(tau), not with an explicit
 * Euler step in tau.  The right-hand sides below return
 * dY/dx = tau Hconf dY/dN, with Hconf=aH and x=ln(tau).
 * We use fixed-step RK4 because these kernel equations are smooth but can be
 * numerically unstable under first-order Euler stepping.  This is a numerical
 * accuracy/stability fix, not a NaN guard: all non-finite intermediate values
 * still trigger class_stop through class_test. */
static int qcdm_qscf_qs_f1_rhs_x(struct background *pba,double x,double k,const double y[4],double dy[4],ErrorMsg errmsg) {
    double tau=exp(x);
    qcdm_qscf_qs_coefficients qc;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc,errmsg),errmsg,errmsg);
    double dFb,dGb,dFq,dGq;
    qcdm_qscf_qs_linear_rhs(&qc,k,y[0],y[1],y[2],y[3],&dFb,&dGb,&dFq,&dGq);
    double fac=tau*qc.Hconf;
    dy[0]=fac*dFb; dy[1]=fac*dGb; dy[2]=fac*dFq; dy[3]=fac*dGq;
    class_test((qcdm_qscf_finite_double(dy[0])==0) || (qcdm_qscf_finite_double(dy[1])==0) ||
               (qcdm_qscf_finite_double(dy[2])==0) || (qcdm_qscf_finite_double(dy[3])==0),errmsg,
               "Non-finite QCDM F1 RHS: tau=%e k=%e y=(%e,%e,%e,%e) dy=(%e,%e,%e,%e)",
               tau,k,y[0],y[1],y[2],y[3],dy[0],dy[1],dy[2],dy[3]);
    return _SUCCESS_;
}


/* Choose a physically sensible initial time for the pressureless exact-QS
 * kernel ODEs.  Starting at tau_final*1e-3 for z=0 puts the solver in
 * radiation domination, where the late-time pressureless two-fluid SPT
 * kernel equations and EdS initial conditions are not appropriate and can
 * numerically excite unstable modes.  We instead start after recombination
 * and well inside matter domination, at z=99 by default. */
static int qcdm_qscf_qs_get_tau_zini(struct background *pba,
                                      double *tau_zini,
                                      ErrorMsg errmsg) {
    if (qcdm_qscf_kernel_tau_zini_cached <= 0.0) {
        class_call(background_tau_of_z(pba, qcdm_qscf_kernel_z_ini_runtime,
                                       &qcdm_qscf_kernel_tau_zini_cached),
                   pba->error_message, errmsg);
    }
    *tau_zini = qcdm_qscf_kernel_tau_zini_cached;
    return _SUCCESS_;
}

static int qcdm_qscf_qs_kernel_tau_ini(struct background *pba,
                                       double tau_final,
                                       double *tau_ini,
                                       ErrorMsg errmsg) {
    double tau_zini = 0.0;
/* DW speed: the z_ini -> tau conversion is identical for every Fourier kernel. */
    class_call(qcdm_qscf_qs_get_tau_zini(pba,&tau_zini,errmsg),errmsg,errmsg);

    *tau_ini = tau_zini;

    if ((qcdm_qscf_finite_double(*tau_ini) == 0) ||
        (*tau_ini <= 0.0) ||
        (*tau_ini >= tau_final)) {
        *tau_ini = 1.e-2 * tau_final;
    }

    if (*tau_ini >= tau_final) {
        *tau_ini = 0.5 * tau_final;
    }

    class_test((qcdm_qscf_finite_double(*tau_ini) == 0) ||
               (*tau_ini <= 0.0) ||
               (*tau_ini >= tau_final),
               errmsg,
               "Invalid QCDM kernel initial time: tau_ini=%e tau_final=%e",
               *tau_ini, tau_final);

    return _SUCCESS_;
}



static int qcdm_qscf_qs_kernel_is_initial_time(struct background *pba,
                                               double tau_final,
                                               int *is_initial,
                                               ErrorMsg errmsg) {
    double tau_zini = 0.0;
    class_call(qcdm_qscf_qs_get_tau_zini(pba,&tau_zini,errmsg),errmsg,errmsg);

    *is_initial = _FALSE_;
    if ((qcdm_qscf_finite_double(tau_zini) == 1) &&
        (tau_zini > 0.0) &&
        (tau_final <= tau_zini * (1.0 + 1.e-10))) {
        *is_initial = _TRUE_;
    }
    return _SUCCESS_;
}

/* DW speed: precompute the background/QSCF coefficients on the exact RK4
 * stage times. Every Fourier kernel at one redshift traverses the same time
 * grid, so repeating background_at_tau() inside each ODE solve is redundant. */
static int qcdm_qscf_qs_get_rk_stage_cache(struct background *pba,
                                           double tau_final,
                                           int nstep,
                                           qcdm_qscf_rk_stage_cache **out,
                                           ErrorMsg errmsg) {
    class_test(nstep<1,errmsg,"Invalid QCDM RK4 step count %d",nstep);
    for (int i=0;i<3;i++) {
        if (qcdm_qscf_rk_cache[i].valid &&
            qcdm_qscf_rk_cache[i].nstep==nstep &&
            qcdm_qscf_cache_close(qcdm_qscf_rk_cache[i].tau_final,tau_final)) {
            *out=&qcdm_qscf_rk_cache[i];
            return _SUCCESS_;
        }
    }

    int slot=-1;
    for (int i=0;i<3;i++) if (!qcdm_qscf_rk_cache[i].valid && qcdm_qscf_rk_cache[i].nstep==nstep) { slot=i; break; }
    if (slot<0) for (int i=0;i<3;i++) if (!qcdm_qscf_rk_cache[i].valid) { slot=i; break; }
    if (slot<0) slot=0;
    qcdm_qscf_rk_stage_cache *cache=&qcdm_qscf_rk_cache[slot];

    int need=3*nstep;
    if (cache->stage_cap<need) {
        qcdm_qscf_rk_stage *tmp=realloc(cache->stage,(size_t)need*sizeof(qcdm_qscf_rk_stage));
        class_test(tmp==NULL,errmsg,"Could not allocate QCDM RK4 coefficient cache");
        cache->stage=tmp;
        cache->stage_cap=need;
    }

    double tau_ini;
    class_call(qcdm_qscf_qs_kernel_tau_ini(pba,tau_final,&tau_ini,errmsg),errmsg,errmsg);
    class_test((tau_ini<=0.0)||(tau_final<=tau_ini),errmsg,
               "Invalid QCDM RK4 coefficient interval: tau_ini=%e tau_final=%e",tau_ini,tau_final);
    double x0=log(tau_ini), h=(log(tau_final)-x0)/((double)nstep);
    for (int n=0;n<nstep;n++) {
        double x=x0+n*h;
        double xs[3]={x,x+0.5*h,x+h};
        for (int is=0;is<3;is++) {
            qcdm_qscf_rk_stage *st=&cache->stage[3*n+is];
            st->tau=exp(xs[is]);
            class_call(qcdm_qscf_qs_get_coefficients(pba,st->tau,&st->qc,errmsg),errmsg,errmsg);
            st->fac=st->tau*st->qc.Hconf;
        }
    }
    cache->tau_final=tau_final;
    cache->nstep=nstep;
    cache->valid=_TRUE_;
    *out=cache;
    return _SUCCESS_;
}

/* DW speed stage 2b: before any parallel kernel solve at one redshift,
 * invalidate only the RK-stage-cache validity flags and rebuild the stage
 * grids needed by F1/F2/F3 serially for this exact tau.  The stage buffers
 * themselves are retained, so this is cheap.  This guarantees that worker
 * threads only read qcdm_qscf_rk_cache and never enter its replacement/
 * realloc path concurrently.  There are at most three distinct step counts
 * and exactly three cache slots. */
static int qcdm_qscf_qs_prewarm_rk_for_tau(struct background *pba,
                                            double tau,
                                            ErrorMsg errmsg) {
    for (int i=0;i<3;i++) qcdm_qscf_rk_cache[i].valid=_FALSE_;

    qcdm_qscf_rk_stage_cache *tmp=NULL;
    class_call(qcdm_qscf_qs_get_rk_stage_cache(
                   pba,tau,qcdm_qscf_kernel_steps_f1_runtime,&tmp,errmsg),
               errmsg,errmsg);
    class_call(qcdm_qscf_qs_get_rk_stage_cache(
                   pba,tau,qcdm_qscf_kernel_steps_f2_runtime,&tmp,errmsg),
               errmsg,errmsg);
    class_call(qcdm_qscf_qs_get_rk_stage_cache(
                   pba,tau,qcdm_qscf_kernel_steps_f3_runtime,&tmp,errmsg),
               errmsg,errmsg);
    return _SUCCESS_;
}

static int qcdm_qscf_qs_f1_rhs_stage(const qcdm_qscf_rk_stage *st,
                                      double k,const double y[4],double dy[4],
                                      ErrorMsg errmsg) {
    double dFb,dGb,dFq,dGq;
    qcdm_qscf_qs_linear_rhs(&st->qc,k,y[0],y[1],y[2],y[3],&dFb,&dGb,&dFq,&dGq);
    dy[0]=st->fac*dFb; dy[1]=st->fac*dGb; dy[2]=st->fac*dFq; dy[3]=st->fac*dGq;
    class_test((qcdm_qscf_finite_double(dy[0])==0) || (qcdm_qscf_finite_double(dy[1])==0) ||
               (qcdm_qscf_finite_double(dy[2])==0) || (qcdm_qscf_finite_double(dy[3])==0),errmsg,
               "Non-finite QCDM F1 RHS: tau=%e k=%e y=(%e,%e,%e,%e) dy=(%e,%e,%e,%e)",
               st->tau,k,y[0],y[1],y[2],y[3],dy[0],dy[1],dy[2],dy[3]);
    return _SUCCESS_;
}

static int qcdm_qscf_qs_rk4_f1(struct background *pba,double tau_final,double k,double y[4],ErrorMsg errmsg) {
/* DW: use the user-selected RK4 resolution for exact-time kernel evolution. */
    int Nstep=qcdm_qscf_kernel_steps_f1_runtime;
    double tau_ini;
    class_call(qcdm_qscf_qs_kernel_tau_ini(pba,tau_final,&tau_ini,errmsg),errmsg,errmsg);
    class_test((tau_ini<=0.0) || (tau_final<=tau_ini),errmsg,
               "Invalid QCDM F1 RK4 interval: tau_ini=%e tau_final=%e",tau_ini,tau_final);
    double x0=log(tau_ini), h=(log(tau_final)-x0)/((double)Nstep);
    qcdm_qscf_rk_stage_cache *rkcache=NULL;
    class_call(qcdm_qscf_qs_get_rk_stage_cache(pba,tau_final,Nstep,&rkcache,errmsg),errmsg,errmsg);
    for(int n=0;n<Nstep;n++){
        double k1v[4],k2v[4],k3v[4],k4v[4],yt[4];
        const qcdm_qscf_rk_stage *s0=&rkcache->stage[3*n];
        const qcdm_qscf_rk_stage *sm=&rkcache->stage[3*n+1];
        const qcdm_qscf_rk_stage *s1=&rkcache->stage[3*n+2];
        class_call(qcdm_qscf_qs_f1_rhs_stage(s0,k,y,k1v,errmsg),errmsg,errmsg);
        for(int i=0;i<4;i++) yt[i]=y[i]+0.5*h*k1v[i];
        class_call(qcdm_qscf_qs_f1_rhs_stage(sm,k,yt,k2v,errmsg),errmsg,errmsg);
        for(int i=0;i<4;i++) yt[i]=y[i]+0.5*h*k2v[i];
        class_call(qcdm_qscf_qs_f1_rhs_stage(sm,k,yt,k3v,errmsg),errmsg,errmsg);
        for(int i=0;i<4;i++) yt[i]=y[i]+h*k3v[i];
        class_call(qcdm_qscf_qs_f1_rhs_stage(s1,k,yt,k4v,errmsg),errmsg,errmsg);
        for(int i=0;i<4;i++) y[i]+=h*(k1v[i]+2.0*k2v[i]+2.0*k3v[i]+k4v[i])/6.0;
        class_test((qcdm_qscf_finite_double(y[0])==0) || (qcdm_qscf_finite_double(y[1])==0) ||
                   (qcdm_qscf_finite_double(y[2])==0) || (qcdm_qscf_finite_double(y[3])==0),errmsg,
                   "Non-finite QCDM F1 RK4 state: n=%d k=%e tau=%e y=(%e,%e,%e,%e)",
                   n,k,s1->tau,y[0],y[1],y[2],y[3]);
    }
    return _SUCCESS_;
}

static int qcdm_qscf_qs_f2_rhs_x(struct background *pba,double x,double k1,double k2,double K,double alpha12,double alpha21,double beta12,const double y[12],double dy[12],ErrorMsg errmsg) {
    double tau=exp(x);
    qcdm_qscf_qs_coefficients qc;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc,errmsg),errmsg,errmsg);
    double dF,dG,dFq,dGq;
    double fac=tau*qc.Hconf;
    qcdm_qscf_qs_linear_rhs(&qc,k1,y[0],y[1],y[2],y[3],&dF,&dG,&dFq,&dGq);
    dy[0]=fac*dF; dy[1]=fac*dG; dy[2]=fac*dFq; dy[3]=fac*dGq;
    qcdm_qscf_qs_linear_rhs(&qc,k2,y[4],y[5],y[6],y[7],&dF,&dG,&dFq,&dGq);
    dy[4]=fac*dF; dy[5]=fac*dG; dy[6]=fac*dFq; dy[7]=fac*dGq;
    double muqK=qcdm_qscf_qs_mu_q(&qc,K);
    double Sd_b=0.5*(alpha12*y[1]*y[4]+alpha21*y[5]*y[0]);
    double St_b=beta12*y[1]*y[5];
    double Sd_q=0.5*(alpha12*y[3]*y[6]+alpha21*y[7]*y[2]);
    double St_q=beta12*y[3]*y[7];
    double dF2b=y[9]+Sd_b;
    double dG2b=-qc.A_b*y[9]+1.5*(qc.Omega_b*y[8]+qc.Omega_q*y[10])+St_b;
    double dF2q=y[11]+Sd_q;
    double dG2q=-qc.A_q*y[11]+1.5*(qc.Omega_b*y[8]+qc.Omega_q*muqK*y[10])+St_q;
    dy[8]=fac*dF2b; dy[9]=fac*dG2b; dy[10]=fac*dF2q; dy[11]=fac*dG2q;
    for(int i=0;i<12;i++){
        class_test(qcdm_qscf_finite_double(dy[i])==0,errmsg,
                   "Non-finite QCDM F2 RHS: tau=%e k1=%e k2=%e K=%e component=%d y=%e dy=%e alpha12=%e alpha21=%e beta12=%e A_b=%e A_q=%e muqK=%e",
                   tau,k1,k2,K,i,y[i],dy[i],alpha12,alpha21,beta12,qc.A_b,qc.A_q,muqK);
    }
    return _SUCCESS_;
}

/* DW: initialize second- and third-order kernels in the growing mode at the matching redshift. */
/* Coupled collapsed F3 system ------------------------------------------------
 *
 * A one-loop P13 evaluation only needs F3(k,q,-q).  The previous
 * implementation evaluated the F3 source by launching fresh F2 and F1
 * integrations at every RK4 substage.  Besides being prohibitively slow,
 * that nested procedure sampled slightly different histories at the RK
 * stages.  We instead evolve one closed 24-component system:
 *
 *   0..3   : F1/G1 for |k|, baryons and qcdm
 *   4..7   : F1/G1 for |q|, baryons and qcdm
 *   8..11  : F2/G2 for ( q,-q)
 *   12..15 : F2/G2 for ( k,-q)
 *   16..19 : F2/G2 for ( k, q)
 *   20..23 : F3/G3 for ( k, q,-q)
 *
 * This is the exact-time pressureless two-fluid hierarchy in the same QS
 * approximation as the linear CLASS implementation. */

static int qcdm_qscf_pair_id(int i,int j) {
    if (i>j) { int t=i; i=j; j=t; }
    if (i==1 && j==2) return 0;
    if (i==0 && j==2) return 1;
    return 2; /* (0,1) */
}

static const double *qcdm_qscf_f1_block(const double y[24],int label) {
    return (label==0) ? y : y+4;
}

static void qcdm_qscf_qs_f2_block_rhs_N(
    const qcdm_qscf_qs_coefficients *qc,
    qcdm_qscf_kvec3 ka,
    qcdm_qscf_kvec3 kb,
    const double f1a[4],
    const double f1b[4],
    const double f2[4],
    double df2[4]) {

    double K=qcdm_qscf_norm(qcdm_qscf_add(ka,kb));
    double alpha_ab=qcdm_qscf_alpha_vec(ka,kb);
    double alpha_ba=qcdm_qscf_alpha_vec(kb,ka);
    double beta_ab=qcdm_qscf_beta_vec(ka,kb);
    double muqK=qcdm_qscf_qs_mu_q(qc,K);

    double Sd_b=0.5*(alpha_ab*f1a[1]*f1b[0]+alpha_ba*f1b[1]*f1a[0]);
    double St_b=beta_ab*f1a[1]*f1b[1];
    double Sd_q=0.5*(alpha_ab*f1a[3]*f1b[2]+alpha_ba*f1b[3]*f1a[2]);
    double St_q=beta_ab*f1a[3]*f1b[3];

    df2[0]=f2[1]+Sd_b;
    df2[1]=-qc->A_b*f2[1]+1.5*(qc->Omega_b*f2[0]+qc->Omega_q*f2[2])+St_b;
    df2[2]=f2[3]+Sd_q;
    df2[3]=-qc->A_q*f2[3]+1.5*(qc->Omega_b*f2[0]+qc->Omega_q*muqK*f2[2])+St_q;
}

static int qcdm_qscf_qs_f3_coupled_rhs_x(
    struct background *pba,
    double x,
    qcdm_qscf_kvec3 p[3],
    const double y[24],
    double dy[24],
    ErrorMsg errmsg) {

    double tau=exp(x);
    qcdm_qscf_qs_coefficients qc;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc,errmsg),errmsg,errmsg);
    double fac=tau*qc.Hconf;
    double dFb,dGb,dFq,dGq;

    qcdm_qscf_qs_linear_rhs(&qc,qcdm_qscf_norm(p[0]),y[0],y[1],y[2],y[3],
                             &dFb,&dGb,&dFq,&dGq);
    dy[0]=fac*dFb; dy[1]=fac*dGb; dy[2]=fac*dFq; dy[3]=fac*dGq;
    qcdm_qscf_qs_linear_rhs(&qc,qcdm_qscf_norm(p[1]),y[4],y[5],y[6],y[7],
                             &dFb,&dGb,&dFq,&dGq);
    dy[4]=fac*dFb; dy[5]=fac*dGb; dy[6]=fac*dFq; dy[7]=fac*dGq;

    const int pair_labels[3][2]={{1,2},{0,2},{0,1}};
    for (int ipair=0; ipair<3; ipair++) {
        int ia=pair_labels[ipair][0], ib=pair_labels[ipair][1];
        const double *f1a=qcdm_qscf_f1_block(y,ia);
        const double *f1b=qcdm_qscf_f1_block(y,ib);
        const double *f2=y+8+4*ipair;
        double df2[4];
        qcdm_qscf_qs_f2_block_rhs_N(&qc,p[ia],p[ib],f1a,f1b,f2,df2);
        for (int j=0;j<4;j++) dy[8+4*ipair+j]=fac*df2[j];
    }

    int perms[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    double Sd_b=0.0,St_b=0.0,Sd_q=0.0,St_q=0.0;
    for (int iperm=0; iperm<6; iperm++) {
        int i=perms[iperm][0],j=perms[iperm][1],l=perms[iperm][2];
        const double *F1i=qcdm_qscf_f1_block(y,i);
        const double *F1l=qcdm_qscf_f1_block(y,l);
        const double *F2jl=y+8+4*qcdm_qscf_pair_id(j,l);
        const double *F2ij=y+8+4*qcdm_qscf_pair_id(i,j);
        qcdm_qscf_kvec3 pjl=qcdm_qscf_add(p[j],p[l]);
        qcdm_qscf_kvec3 pij=qcdm_qscf_add(p[i],p[j]);
        double ai_jl=qcdm_qscf_alpha_vec(p[i],pjl);
        double bi_jl=qcdm_qscf_beta_vec(p[i],pjl);
        double aij_l=qcdm_qscf_alpha_vec(pij,p[l]);
        double bij_l=qcdm_qscf_beta_vec(pij,p[l]);

        Sd_b += ai_jl*F1i[1]*F2jl[0] + aij_l*F2ij[1]*F1l[0];
        St_b += bi_jl*F1i[1]*F2jl[1] + bij_l*F2ij[1]*F1l[1];
        Sd_q += ai_jl*F1i[3]*F2jl[2] + aij_l*F2ij[3]*F1l[2];
        St_q += bi_jl*F1i[3]*F2jl[3] + bij_l*F2ij[3]*F1l[3];
    }
    Sd_b/=6.0; St_b/=6.0; Sd_q/=6.0; St_q/=6.0;

    const double *f3=y+20;
    double muqK=qcdm_qscf_qs_mu_q(&qc,qcdm_qscf_norm(qcdm_qscf_add(qcdm_qscf_add(p[0],p[1]),p[2])));
    double df3[4];
    df3[0]=f3[1]+Sd_b;
    df3[1]=-qc.A_b*f3[1]+1.5*(qc.Omega_b*f3[0]+qc.Omega_q*f3[2])+St_b;
    df3[2]=f3[3]+Sd_q;
    df3[3]=-qc.A_q*f3[3]+1.5*(qc.Omega_b*f3[0]+qc.Omega_q*muqK*f3[2])+St_q;
    for (int j=0;j<4;j++) dy[20+j]=fac*df3[j];

    for (int j=0;j<24;j++) {
        class_test(qcdm_qscf_finite_double(dy[j])==0,errmsg,
                   "Non-finite coupled QCDM F3 RHS: tau=%e component=%d y=%e dy=%e",
                   tau,j,y[j],dy[j]);
    }
    return _SUCCESS_;
}

static int qcdm_qscf_qs_f3_coupled_rhs_stage(
    const qcdm_qscf_rk_stage *st,
    qcdm_qscf_kvec3 p[3],
    const double y[24],
    double dy[24],
    ErrorMsg errmsg) {

    const qcdm_qscf_qs_coefficients *qc=&st->qc;
    double fac=st->fac;
    double dFb,dGb,dFq,dGq;

    qcdm_qscf_qs_linear_rhs(qc,qcdm_qscf_norm(p[0]),y[0],y[1],y[2],y[3],
                             &dFb,&dGb,&dFq,&dGq);
    dy[0]=fac*dFb; dy[1]=fac*dGb; dy[2]=fac*dFq; dy[3]=fac*dGq;
    qcdm_qscf_qs_linear_rhs(qc,qcdm_qscf_norm(p[1]),y[4],y[5],y[6],y[7],
                             &dFb,&dGb,&dFq,&dGq);
    dy[4]=fac*dFb; dy[5]=fac*dGb; dy[6]=fac*dFq; dy[7]=fac*dGq;

    const int pair_labels[3][2]={{1,2},{0,2},{0,1}};
    for (int ipair=0; ipair<3; ipair++) {
        int ia=pair_labels[ipair][0], ib=pair_labels[ipair][1];
        const double *f1a=qcdm_qscf_f1_block(y,ia);
        const double *f1b=qcdm_qscf_f1_block(y,ib);
        const double *f2=y+8+4*ipair;
        double df2[4];
        qcdm_qscf_qs_f2_block_rhs_N(qc,p[ia],p[ib],f1a,f1b,f2,df2);
        for (int j=0;j<4;j++) dy[8+4*ipair+j]=fac*df2[j];
    }

    int perms[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    double Sd_b=0.0,St_b=0.0,Sd_q=0.0,St_q=0.0;
    for (int iperm=0; iperm<6; iperm++) {
        int i=perms[iperm][0],j=perms[iperm][1],l=perms[iperm][2];
        const double *F1i=qcdm_qscf_f1_block(y,i);
        const double *F1l=qcdm_qscf_f1_block(y,l);
        const double *F2jl=y+8+4*qcdm_qscf_pair_id(j,l);
        const double *F2ij=y+8+4*qcdm_qscf_pair_id(i,j);
        qcdm_qscf_kvec3 pjl=qcdm_qscf_add(p[j],p[l]);
        qcdm_qscf_kvec3 pij=qcdm_qscf_add(p[i],p[j]);
        double ai_jl=qcdm_qscf_alpha_vec(p[i],pjl);
        double bi_jl=qcdm_qscf_beta_vec(p[i],pjl);
        double aij_l=qcdm_qscf_alpha_vec(pij,p[l]);
        double bij_l=qcdm_qscf_beta_vec(pij,p[l]);

        Sd_b += ai_jl*F1i[1]*F2jl[0] + aij_l*F2ij[1]*F1l[0];
        St_b += bi_jl*F1i[1]*F2jl[1] + bij_l*F2ij[1]*F1l[1];
        Sd_q += ai_jl*F1i[3]*F2jl[2] + aij_l*F2ij[3]*F1l[2];
        St_q += bi_jl*F1i[3]*F2jl[3] + bij_l*F2ij[3]*F1l[3];
    }
    Sd_b/=6.0; St_b/=6.0; Sd_q/=6.0; St_q/=6.0;

    const double *f3=y+20;
    double muqK=qcdm_qscf_qs_mu_q(qc,qcdm_qscf_norm(qcdm_qscf_add(qcdm_qscf_add(p[0],p[1]),p[2])));
    double df3[4];
    df3[0]=f3[1]+Sd_b;
    df3[1]=-qc->A_b*f3[1]+1.5*(qc->Omega_b*f3[0]+qc->Omega_q*f3[2])+St_b;
    df3[2]=f3[3]+Sd_q;
    df3[3]=-qc->A_q*f3[3]+1.5*(qc->Omega_b*f3[0]+qc->Omega_q*muqK*f3[2])+St_q;
    for (int j=0;j<4;j++) dy[20+j]=fac*df3[j];

    for (int j=0;j<24;j++) {
        class_test(qcdm_qscf_finite_double(dy[j])==0,errmsg,
                   "Non-finite coupled QCDM F3 RHS: tau=%e component=%d y=%e dy=%e",
                   st->tau,j,y[j],dy[j]);
    }
    return _SUCCESS_;
}

static int qcdm_qscf_qs_integrate_F3_collapsed(
    struct background *pba,
    double tau_final,
    double k,
    double q,
    double mu,
    qcdm_qscf_kernel3_result *res,
    ErrorMsg errmsg) {

    double sperp=sqrt(MAX(1.0-mu*mu,0.0));
    qcdm_qscf_kvec3 p[3]={{k,0.0,0.0},{q*mu,q*sperp,0.0},{-q*mu,-q*sperp,0.0}};
    double F3eds,G3eds;
    qcdm_qscf_eds_F3G3_sym(p,&F3eds,&G3eds);

    int is_initial=_FALSE_;
    class_call(qcdm_qscf_qs_kernel_is_initial_time(pba,tau_final,&is_initial,errmsg),errmsg,errmsg);
    if (is_initial==_TRUE_) {
        memset(res,0,sizeof(*res));
        res->F3_m_red=F3eds;
        res->G3_m_red=G3eds;
        return _SUCCESS_;
    }

    double y[24]={0.0};
    for (int j=0;j<8;j++) y[j]=1.0;
    const int pair_labels[3][2]={{1,2},{0,2},{0,1}};
    for (int ipair=0;ipair<3;ipair++) {
        double F2eds,G2eds;
        qcdm_qscf_eds_F2G2_vec(p[pair_labels[ipair][0]],p[pair_labels[ipair][1]],&F2eds,&G2eds);
        y[8+4*ipair]=F2eds;
        y[9+4*ipair]=G2eds;
        y[10+4*ipair]=F2eds;
        y[11+4*ipair]=G2eds;
    }
    y[20]=F3eds; y[21]=G3eds; y[22]=F3eds; y[23]=G3eds;

    double tau_ini;
    class_call(qcdm_qscf_qs_kernel_tau_ini(pba,tau_final,&tau_ini,errmsg),errmsg,errmsg);
    class_test((tau_ini<=0.0)||(tau_final<=tau_ini),errmsg,
               "Invalid coupled QCDM F3 interval: tau_ini=%e tau_final=%e",tau_ini,tau_final);
    int Nstep=qcdm_qscf_kernel_steps_f3_runtime;
    double x0=log(tau_ini),h=(log(tau_final)-x0)/((double)Nstep);
    (void)x0;
    qcdm_qscf_rk_stage_cache *rkcache=NULL;
    class_call(qcdm_qscf_qs_get_rk_stage_cache(pba,tau_final,Nstep,&rkcache,errmsg),errmsg,errmsg);
    for (int n=0;n<Nstep;n++) {
        double k1v[24],k2v[24],k3v[24],k4v[24],yt[24];
        const qcdm_qscf_rk_stage *s0=&rkcache->stage[3*n];
        const qcdm_qscf_rk_stage *sm=&rkcache->stage[3*n+1];
        const qcdm_qscf_rk_stage *s1=&rkcache->stage[3*n+2];
        class_call(qcdm_qscf_qs_f3_coupled_rhs_stage(s0,p,y,k1v,errmsg),errmsg,errmsg);
        for (int j=0;j<24;j++) yt[j]=y[j]+0.5*h*k1v[j];
        class_call(qcdm_qscf_qs_f3_coupled_rhs_stage(sm,p,yt,k2v,errmsg),errmsg,errmsg);
        for (int j=0;j<24;j++) yt[j]=y[j]+0.5*h*k2v[j];
        class_call(qcdm_qscf_qs_f3_coupled_rhs_stage(sm,p,yt,k3v,errmsg),errmsg,errmsg);
        for (int j=0;j<24;j++) yt[j]=y[j]+h*k3v[j];
        class_call(qcdm_qscf_qs_f3_coupled_rhs_stage(s1,p,yt,k4v,errmsg),errmsg,errmsg);
        for (int j=0;j<24;j++) {
            y[j]+=h*(k1v[j]+2.0*k2v[j]+2.0*k3v[j]+k4v[j])/6.0;
            class_test(qcdm_qscf_finite_double(y[j])==0,errmsg,
                       "Non-finite coupled QCDM F3 state: step=%d k=%e q=%e mu=%e component=%d value=%e",
                       n,k,q,mu,j,y[j]);
        }
    }

    qcdm_qscf_qs_coefficients qc;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau_final,&qc,errmsg),errmsg,errmsg);
    double Om=qc.Omega_b+qc.Omega_q;
    double wb=(Om>0.0)?qc.Omega_b/Om:0.0;
    double wq=(Om>0.0)?qc.Omega_q/Om:0.0;
    double F1mk=wb*y[0]+wq*y[2];
    double F1mq=wb*y[4]+wq*y[6];
    double denom=F1mk*F1mq*F1mq;
    class_test(fabs(denom)<DBL_MIN,errmsg,
               "Vanishing QCDM F3 reduction denominator: k=%e q=%e mu=%e",k,q,mu);
    res->F3_m_red=(wb*y[20]+wq*y[22])/denom;
    res->G3_m_red=(wb*y[21]+wq*y[23])/denom;
    res->F3_r_red=(y[22]-y[20])/denom;
    res->G3_r_red=(y[23]-y[21])/denom;
    return _SUCCESS_;
}

static int qcdm_qscf_qs_f3_rhs_x(struct background *pba,double x,double k,double q,double mu,qcdm_qscf_kvec3 p[3],const double y[4],double dy[4],ErrorMsg errmsg) {
    double tau=exp(x);
    qcdm_qscf_qs_coefficients qc;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc,errmsg),errmsg,errmsg);
    double Sd_b,St_b,Sd_q,St_q;
    class_call(qcdm_qscf_qs_F3_sources(pba,tau,p,&Sd_b,&St_b,&Sd_q,&St_q,errmsg),errmsg,errmsg);
    class_test((qcdm_qscf_finite_double(Sd_b)==0) || (qcdm_qscf_finite_double(St_b)==0) ||
               (qcdm_qscf_finite_double(Sd_q)==0) || (qcdm_qscf_finite_double(St_q)==0),errmsg,
               "Non-finite QCDM F3 source: tau=%e k=%e q=%e mu=%e Sd_b=%e St_b=%e Sd_q=%e St_q=%e",
               tau,k,q,mu,Sd_b,St_b,Sd_q,St_q);
    double muqK=qcdm_qscf_qs_mu_q(&qc,k);
    double dF3b=y[1]+Sd_b;
    double dG3b=-qc.A_b*y[1]+1.5*(qc.Omega_b*y[0]+qc.Omega_q*y[2])+St_b;
    double dF3q=y[3]+Sd_q;
    double dG3q=-qc.A_q*y[3]+1.5*(qc.Omega_b*y[0]+qc.Omega_q*muqK*y[2])+St_q;
    double fac=tau*qc.Hconf;
    dy[0]=fac*dF3b; dy[1]=fac*dG3b; dy[2]=fac*dF3q; dy[3]=fac*dG3q;
    for(int i=0;i<4;i++){
        class_test(qcdm_qscf_finite_double(dy[i])==0,errmsg,
                   "Non-finite QCDM F3 RHS: tau=%e k=%e q=%e mu=%e component=%d y=%e dy=%e A_b=%e A_q=%e muqK=%e",
                   tau,k,q,mu,i,y[i],dy[i],qc.A_b,qc.A_q,muqK);
    }
    return _SUCCESS_;
}

static int qcdm_qscf_qs_solve_F1_species(struct background *pba,double tau_final,double k,
    double *Fb,double *Gb,double *Fq,double *Gq,ErrorMsg errmsg) {
    int is_initial = _FALSE_;
    class_call(qcdm_qscf_qs_kernel_is_initial_time(pba,tau_final,&is_initial,errmsg),errmsg,errmsg);
    if (is_initial == _TRUE_) {
        *Fb=1.0; *Gb=1.0; *Fq=1.0; *Gq=1.0;
        return _SUCCESS_;
    }
    double y[4]={1.0,1.0,1.0,1.0};
    class_call(qcdm_qscf_qs_rk4_f1(pba,tau_final,k,y,errmsg),errmsg,errmsg);
    *Fb=y[0]; *Gb=y[1]; *Fq=y[2]; *Gq=y[3];
    return _SUCCESS_;
}


static int qcdm_qscf_qs_solve_F1m(struct background *pba,double tau_final,double k,double *F1m,ErrorMsg errmsg) {
    if (qcdm_qscf_cache_get_f1m(tau_final,k,F1m) == _TRUE_) return _SUCCESS_;

    int is_initial = _FALSE_;
    class_call(qcdm_qscf_qs_kernel_is_initial_time(pba,tau_final,&is_initial,errmsg),errmsg,errmsg);
    if (is_initial == _TRUE_) {
        *F1m = 1.0;
        /* DW speed stage 3b: parallel real/RSD/basis workers must never
         * mutate the process-global kernel cache. */
        if (qcdm_qscf_cache_parallel_nostore!=_TRUE_)
            class_call(qcdm_qscf_cache_put_f1m(tau_final,k,*F1m,errmsg),errmsg,errmsg);
        return _SUCCESS_;
    }

    double y[4]={1.0,1.0,1.0,1.0};
    class_call(qcdm_qscf_qs_rk4_f1(pba,tau_final,k,y,errmsg),errmsg,errmsg);
    qcdm_qscf_qs_coefficients qcf;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau_final,&qcf,errmsg),errmsg,errmsg);
    double Om=qcf.Omega_b+qcf.Omega_q;
    *F1m=(Om>0.0)?(qcf.Omega_b*y[0]+qcf.Omega_q*y[2])/Om:1.0;
    class_test(qcdm_qscf_finite_double(*F1m)==0,errmsg,
               "Non-finite QCDM F1m final value: k=%e F1m=%e Fb=%e Fq=%e",k,*F1m,y[0],y[2]);
    if (qcdm_qscf_cache_parallel_nostore!=_TRUE_)
        class_call(qcdm_qscf_cache_put_f1m(tau_final,k,*F1m,errmsg),errmsg,errmsg);
    return _SUCCESS_;
}


static int qcdm_qscf_qs_solve_F2_full(struct background *pba,double tau_final,double k1,double k2,double mu12,qcdm_qscf_kernel2_result *res,ErrorMsg errmsg) {
    if (qcdm_qscf_cache_get_f2(tau_final,k1,k2,mu12,res) == _TRUE_) return _SUCCESS_;
    class_test(qcdm_qscf_cache_readonly_parallel==_TRUE_,errmsg,
               "QCDM parallel native-basis F2 cache miss: tau=%e k1=%e k2=%e mu=%e. "
               "The basis OpenMP path requires the initialization kernel cache to be warm.",
               tau_final,k1,k2,mu12);

    double k1sq=k1*k1,k2sq=k2*k2,dot12=k1*k2*mu12,ksq=k1sq+k2sq+2.0*dot12,K=sqrt(MAX(ksq,0.0));
    qcdm_qscf_kvec3 v1={k1,0.0,0.0}, v2={k2*mu12,k2*sqrt(MAX(1.0-mu12*mu12,0.0)),0.0};
    double alpha12=qcdm_qscf_alpha_vec(v1,v2), alpha21=qcdm_qscf_alpha_vec(v2,v1), beta12=qcdm_qscf_beta_vec(v1,v2);
    double r=(k2>0.0)?k1/k2:1.0;
    double F2eds=5.0/7.0+0.5*mu12*(r+1.0/r)+2.0/7.0*mu12*mu12;
    double G2eds=3.0/7.0+0.5*mu12*(r+1.0/r)+4.0/7.0*mu12*mu12;

    int is_initial = _FALSE_;
    class_call(qcdm_qscf_qs_kernel_is_initial_time(pba,tau_final,&is_initial,errmsg),errmsg,errmsg);
    if (is_initial == _TRUE_) {
        memset(res,0,sizeof(qcdm_qscf_kernel2_result));
        res->F2_b=F2eds; res->G2_b=G2eds; res->F2_q=F2eds; res->G2_q=G2eds;
        res->F1m_k1=1.0; res->F1m_k2=1.0;
        res->F1r_k1=0.0; res->F1r_k2=0.0;
        res->G1m_k1=1.0; res->G1m_k2=1.0;
        res->G1r_k1=0.0; res->G1r_k2=0.0;
        res->F2_m_red=F2eds; res->G2_m_red=G2eds;
        res->F2_r_red=0.0; res->G2_r_red=0.0;
        if (qcdm_qscf_cache_parallel_nostore!=_TRUE_)
            class_call(qcdm_qscf_cache_put_f2(tau_final,k1,k2,mu12,res,errmsg),errmsg,errmsg);
        return _SUCCESS_;
    }

    double y[12]={1.0,1.0,1.0,1.0, 1.0,1.0,1.0,1.0, F2eds,G2eds,F2eds,G2eds};
/* DW: use the user-selected RK4 resolution for the coupled two-fluid F2 evolution. */
    int Nstep=qcdm_qscf_kernel_steps_f2_runtime;
    double tau_ini;
    class_call(qcdm_qscf_qs_kernel_tau_ini(pba,tau_final,&tau_ini,errmsg),errmsg,errmsg);
    class_test((tau_ini<=0.0) || (tau_final<=tau_ini),errmsg,
               "Invalid QCDM F2 RK4 interval: tau_ini=%e tau_final=%e",tau_ini,tau_final);
    double x0=log(tau_ini), h=(log(tau_final)-x0)/((double)Nstep);
    (void)x0;
    qcdm_qscf_rk_stage_cache *rkcache=NULL;
    class_call(qcdm_qscf_qs_get_rk_stage_cache(pba,tau_final,Nstep,&rkcache,errmsg),errmsg,errmsg);
    for(int n=0;n<Nstep;n++){
        double k1v[12],k2v[12],k3v[12],k4v[12],yt[12];
        const qcdm_qscf_rk_stage *s0=&rkcache->stage[3*n];
        const qcdm_qscf_rk_stage *sm=&rkcache->stage[3*n+1];
        const qcdm_qscf_rk_stage *s1=&rkcache->stage[3*n+2];
        class_call(qcdm_qscf_qs_f2_rhs_stage(s0,k1,k2,K,alpha12,alpha21,beta12,y,k1v,errmsg),errmsg,errmsg);
        for(int i=0;i<12;i++) yt[i]=y[i]+0.5*h*k1v[i];
        class_call(qcdm_qscf_qs_f2_rhs_stage(sm,k1,k2,K,alpha12,alpha21,beta12,yt,k2v,errmsg),errmsg,errmsg);
        for(int i=0;i<12;i++) yt[i]=y[i]+0.5*h*k2v[i];
        class_call(qcdm_qscf_qs_f2_rhs_stage(sm,k1,k2,K,alpha12,alpha21,beta12,yt,k3v,errmsg),errmsg,errmsg);
        for(int i=0;i<12;i++) yt[i]=y[i]+h*k3v[i];
        class_call(qcdm_qscf_qs_f2_rhs_stage(s1,k1,k2,K,alpha12,alpha21,beta12,yt,k4v,errmsg),errmsg,errmsg);
        for(int i=0;i<12;i++) y[i]+=h*(k1v[i]+2.0*k2v[i]+2.0*k3v[i]+k4v[i])/6.0;
        for(int i=0;i<12;i++){
            class_test(qcdm_qscf_finite_double(y[i])==0,errmsg,
                       "Non-finite QCDM F2 RK4 state: n=%d k1=%e k2=%e mu=%e tau=%e component=%d value=%e",
                       n,k1,k2,mu12,s1->tau,i,y[i]);
        }
    }
    double F1b1=y[0],G1b1=y[1],F1q1=y[2],G1q1=y[3];
    double F1b2=y[4],G1b2=y[5],F1q2=y[6],G1q2=y[7];
    double F2b=y[8],G2b=y[9],F2q=y[10],G2q=y[11];

    qcdm_qscf_qs_coefficients qcf;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau_final,&qcf,errmsg),errmsg,errmsg);
    double Om=qcf.Omega_b+qcf.Omega_q, wb=(Om>0.0)?qcf.Omega_b/Om:0.0, wq=(Om>0.0)?qcf.Omega_q/Om:0.0;
    double F1m1=wb*F1b1+wq*F1q1, F1m2=wb*F1b2+wq*F1q2, denom=F1m1*F1m2;
    double G1m1=wb*G1b1+wq*G1q1, G1m2=wb*G1b2+wq*G1q2;
    res->F2_b=F2b; res->G2_b=G2b; res->F2_q=F2q; res->G2_q=G2q;
    res->F1m_k1=F1m1; res->F1m_k2=F1m2;
    res->F1r_k1=(fabs(F1m1)>0.0)?(F1q1-F1b1)/F1m1:0.0;
    res->F1r_k2=(fabs(F1m2)>0.0)?(F1q2-F1b2)/F1m2:0.0;
    res->G1m_k1=(fabs(F1m1)>0.0)?G1m1/F1m1:0.0;
    res->G1m_k2=(fabs(F1m2)>0.0)?G1m2/F1m2:0.0;
    res->G1r_k1=(fabs(F1m1)>0.0)?(G1q1-G1b1)/F1m1:0.0;
    res->G1r_k2=(fabs(F1m2)>0.0)?(G1q2-G1b2)/F1m2:0.0;
    res->F2_m_red=(fabs(denom)>0.0)?(wb*F2b+wq*F2q)/denom:0.0;
    res->G2_m_red=(fabs(denom)>0.0)?(wb*G2b+wq*G2q)/denom:0.0;
    res->F2_r_red=(fabs(denom)>0.0)?(F2q-F2b)/denom:0.0;
    res->G2_r_red=(fabs(denom)>0.0)?(G2q-G2b)/denom:0.0;
    class_test((qcdm_qscf_finite_double(res->F2_m_red)==0) || (qcdm_qscf_finite_double(res->G2_m_red)==0) ||
               (qcdm_qscf_finite_double(res->F2_r_red)==0) || (qcdm_qscf_finite_double(res->G2_r_red)==0),errmsg,
               "Non-finite QCDM F2 final reduced kernel: k1=%e k2=%e mu=%e denom=%e F2m=%e G2m=%e F2r=%e G2r=%e",
               k1,k2,mu12,denom,res->F2_m_red,res->G2_m_red,res->F2_r_red,res->G2_r_red);
    if (qcdm_qscf_cache_parallel_nostore!=_TRUE_)
        class_call(qcdm_qscf_cache_put_f2(tau_final,k1,k2,mu12,res,errmsg),errmsg,errmsg);
    return _SUCCESS_;
}

static int qcdm_qscf_qs_pair_kernel_at_tau(struct background *pba,double tau,qcdm_qscf_kvec3 a,qcdm_qscf_kvec3 b,qcdm_qscf_kernel2_result *res,ErrorMsg errmsg) {
    double ka=qcdm_qscf_norm(a), kb=qcdm_qscf_norm(b); if((ka<=0.0)||(kb<=0.0)){memset(res,0,sizeof(qcdm_qscf_kernel2_result)); return _SUCCESS_;}
    double mu=qcdm_qscf_dot(a,b)/(ka*kb); if(mu>1.0) mu=1.0; if(mu<-1.0) mu=-1.0; class_call(qcdm_qscf_qs_solve_F2_full(pba,tau,ka,kb,mu,res,errmsg),errmsg,errmsg); return _SUCCESS_;
}

static int qcdm_qscf_qs_F3_sources(struct background *pba,double tau,qcdm_qscf_kvec3 p[3],double *Sd_b,double *St_b,double *Sd_q,double *St_q,ErrorMsg errmsg) {
    int perms[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    *Sd_b=*St_b=*Sd_q=*St_q=0.0;

    for(int ip=0;ip<6;ip++){
        int i=perms[ip][0],j=perms[ip][1],l=perms[ip][2];
        qcdm_qscf_kvec3 pi=p[i],pj=p[j],pl=p[l];
        qcdm_qscf_kvec3 pjl=qcdm_qscf_add(pj,pl),pij=qcdm_qscf_add(pi,pj);
        qcdm_qscf_kernel2_result Kjl,Kij;
        double F1bi,G1bi,F1qi,G1qi,F1bl,G1bl,F1ql,G1ql;
        double ki=qcdm_qscf_norm(pi), kl=qcdm_qscf_norm(pl);

        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pj,pl,&Kjl,errmsg),errmsg,errmsg);
        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pi,pj,&Kij,errmsg),errmsg,errmsg);
        class_call(qcdm_qscf_qs_solve_F1_species(pba,tau,ki,&F1bi,&G1bi,&F1qi,&G1qi,errmsg),errmsg,errmsg);
        class_call(qcdm_qscf_qs_solve_F1_species(pba,tau,kl,&F1bl,&G1bl,&F1ql,&G1ql,errmsg),errmsg,errmsg);

        double ai_jl=qcdm_qscf_alpha_vec(pi,pjl);
        double bi_jl=qcdm_qscf_beta_vec(pi,pjl);
        double aij_l=qcdm_qscf_alpha_vec(pij,pl);
        double bij_l=qcdm_qscf_beta_vec(pij,pl);

        *Sd_b += ai_jl*G1bi*Kjl.F2_b + aij_l*Kij.G2_b*F1bl;
        *St_b += bi_jl*G1bi*Kjl.G2_b + bij_l*Kij.G2_b*G1bl;

        *Sd_q += ai_jl*G1qi*Kjl.F2_q + aij_l*Kij.G2_q*F1ql;
        *St_q += bi_jl*G1qi*Kjl.G2_q + bij_l*Kij.G2_q*G1ql;
    }

    *Sd_b/=6.0; *St_b/=6.0; *Sd_q/=6.0; *St_q/=6.0;
    return _SUCCESS_;
}


static int qcdm_qscf_qs_solve_F3_collapsed(struct background *pba,double tau_final,double k,double q,double mu,double *F3_m_red,ErrorMsg errmsg) {
    if (qcdm_qscf_cache_get_f3m(tau_final,k,q,mu,F3_m_red) == _TRUE_) return _SUCCESS_;

/* DW: reuse the coupled full-species collapsed F3 solution and cache its reduced total kernel. */
    qcdm_qscf_kernel3_result full;
    if (qcdm_qscf_cache_get_f3(tau_final,k,q,mu,&full) != _TRUE_) {
        class_call(qcdm_qscf_qs_integrate_F3_collapsed(pba,tau_final,k,q,mu,&full,errmsg),errmsg,errmsg);
        /* DW speed stage 3b: this lightweight P13 wrapper predates the
         * OpenMP no-store mode.  Honor it here too; otherwise real-space
         * workers concurrently grow/rehash the shared F3 cache. */
        if (qcdm_qscf_cache_parallel_nostore!=_TRUE_)
            class_call(qcdm_qscf_cache_put_f3(tau_final,k,q,mu,&full,errmsg),errmsg,errmsg);
    }
/* DW: use log-k linear interpolation for signed one-loop quantities to avoid spline overshoot. */
    *F3_m_red=full.F3_m_red;
    if (qcdm_qscf_cache_parallel_nostore!=_TRUE_)
        class_call(qcdm_qscf_cache_put_f3m(tau_final,k,q,mu,*F3_m_red,errmsg),errmsg,errmsg);
    return _SUCCESS_;
}

static double qcdm_qscf_interp_linear(int n,const double *x,const double *y,double xp);

static int qcdm_qscf_qs_solve_F1_reduced(struct background *pba,double tau,double k,qcdm_qscf_kernel1_result *res,ErrorMsg errmsg) {
    if (qcdm_qscf_cache_get_f1(tau,k,res) == _TRUE_) return _SUCCESS_;
    class_test(qcdm_qscf_cache_readonly_parallel==_TRUE_,errmsg,
               "QCDM parallel native-basis F1 cache miss: tau=%e k=%e. "
               "The basis OpenMP path requires the initialization kernel cache to be warm.",tau,k);

    double Fb,Gb,Fq,Gq;
    class_call(qcdm_qscf_qs_solve_F1_species(pba,tau,k,&Fb,&Gb,&Fq,&Gq,errmsg),errmsg,errmsg);
    qcdm_qscf_qs_coefficients qc;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc,errmsg),errmsg,errmsg);
    double Om=qc.Omega_b+qc.Omega_q, wb=(Om>0.0)?qc.Omega_b/Om:0.0, wq=(Om>0.0)?qc.Omega_q/Om:0.0;
    double Fm=wb*Fb+wq*Fq, Gm=wb*Gb+wq*Gq;
    res->F1_m=Fm;
    res->G1_m_red=(fabs(Fm)>0.0)?Gm/Fm:0.0;
    res->F1_r_red=(fabs(Fm)>0.0)?(Fq-Fb)/Fm:0.0;
    res->G1_r_red=(fabs(Fm)>0.0)?(Gq-Gb)/Fm:0.0;
    if (qcdm_qscf_cache_parallel_nostore!=_TRUE_)
        class_call(qcdm_qscf_cache_put_f1(tau,k,res,errmsg),errmsg,errmsg);
    return _SUCCESS_;
}

/* -------------------------------------------------------------------------
 * Standalone exact-QS F1 diagnostic for thinned-chain post-processing.
 *
 * QCDM_RELVEL_OUT=<unique CSV path> turns it on; QCDM_RELVEL_ZS and
 * QCDM_RELVEL_KHS are comma-separated z and k [h/Mpc] arrays.  This is
 * deliberately implemented INSIDE nonlinear_pt.c: it calls the exact same
 * F1 ODE, reduction, and background coefficient routines as the one-loop
 * spectra.  It does not read gauge-dependent CLASS velocity transfers.
 *
 * Must be called while nonlinear_pt.method == nlpt_none.  The usual
 * nonlinear_pt_init() no-PT path continues normally after this diagnostic.
 * ------------------------------------------------------------------------- */
static int qcdm_qscf_relvel_parse_grid(const char *source,
                                       const char *fallback,
                                       double *values,
                                       int *count,
                                       double minimum,
                                       ErrorMsg errmsg) {
    const char *s = (source && *source) ? source : fallback;
    int n=0;
    while (*s) {
        while (*s==' ' || *s=='\t' || *s==',' || *s==';') ++s;
        if (*s=='\0') break;
        char *end=NULL;
        double v=strtod(s,&end);
        class_test(end==s || !qcdm_qscf_finite_double(v) || v<minimum,errmsg,
                   "Invalid QCDM relative-velocity diagnostic grid near '%s'",s);
        class_test(n>=64,errmsg,"QCDM diagnostic grid contains more than 64 entries");
        values[n++]=v;
        s=end;
        while (*s==' ' || *s=='\t') ++s;
        class_test(*s!='\0' && *s!=',' && *s!=';',errmsg,
                   "Invalid QCDM diagnostic separator near '%s'",s);
    }
    class_test(n==0,errmsg,"Empty QCDM relative-velocity diagnostic grid");
    *count=n;
    return _SUCCESS_;
}

static int qcdm_qscf_dump_relvel_f1(struct background *pba,
                                     struct nonlinear_pt *pnlpt) {
    const char *path=getenv("QCDM_RELVEL_OUT");
    class_test(path==NULL || *path=='\0',pnlpt->error_message,
               "QCDM_RELVEL_OUT must name a nonempty CSV output path");
    class_test(pba->has_qcdm!=_TRUE_ || pba->has_qscf!=_TRUE_,pnlpt->error_message,
               "QCDM relative-velocity F1 diagnostic requires QCDM and QSCF");
    class_test(pnlpt->qcdm_qscf_kernels==qcdm_qscf_kernels_no,
               pnlpt->error_message,
               "QCDM relative-velocity diagnostic requires exact_qs kernels");

    double zs[64],khs[64];
    int nz=0,nk=0;
    class_call(qcdm_qscf_relvel_parse_grid(getenv("QCDM_RELVEL_ZS"),
                   "0.295364,0.509629,0.705796,0.918585,1.317070,1.490500",
                   zs,&nz,0.0,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    class_call(qcdm_qscf_relvel_parse_grid(getenv("QCDM_RELVEL_KHS"),
                   "0.02,0.05,0.10,0.15,0.20",
                   khs,&nk,1.e-15,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);

    /* Each posterior draw is a new cosmology, even when the same Python
     * process repeatedly constructs different Class instances. */
    qcdm_qscf_cache_readonly_parallel=_FALSE_;
    qcdm_qscf_cache_parallel_nostore=_FALSE_;
    qcdm_qscf_qs_cache_reset();

    /* input.c parses qcdm_kernel_z_ini / steps_f1 only when method==PT.
     * This diagnostic deliberately uses method==none.  Pass the exact
     * production F1 settings via environment variables instead; otherwise
     * we would silently use the PT defaults (99,128), not e.g. (99,24). */
    {
        const char *sz=getenv("QCDM_RELVEL_Z_INI");
        const char *ss=getenv("QCDM_RELVEL_F1_STEPS");
        class_test(sz==NULL || ss==NULL,pnlpt->error_message,
                   "Diagnostic requires QCDM_RELVEL_Z_INI and QCDM_RELVEL_F1_STEPS to match production");
        char *endz=NULL,*ends=NULL;
        double zi=strtod(sz,&endz);
        long ns=strtol(ss,&ends,10);
        class_test(endz==sz || *endz!='\0' || !qcdm_qscf_finite_double(zi) || zi<1.0,
                   pnlpt->error_message,"Invalid QCDM_RELVEL_Z_INI='%s'",sz);
        class_test(ends==ss || *ends!='\0' || ns<16 || ns>100000,
                   pnlpt->error_message,"Invalid QCDM_RELVEL_F1_STEPS='%s'",ss);
        pnlpt->qcdm_kernel_z_ini=zi;
        pnlpt->qcdm_kernel_steps_f1=(int)ns;
    }
    qcdm_qscf_qs_configure_runtime(pnlpt);

    FILE *fp=fopen(path,"w");
    class_test(fp==NULL,pnlpt->error_message,
               "Unable to create QCDM F1 diagnostic CSV '%s'",path);
    fprintf(fp,"# exact-QS F1 from production qcdm_qscf_qs_solve_F1_reduced\n");
    fprintf(fp,"# z_ini=%.12g steps_f1=%d; k_h in h/Mpc\n",
            qcdm_qscf_kernel_z_ini_runtime,qcdm_qscf_kernel_steps_f1_runtime);
    fprintf(fp,"z,k_h,wb,F1_m,G1_m_red,G1_r_red,R\n");

    for (int iz=0;iz<nz;iz++) {
        double tau;
        class_test(zs[iz]>=qcdm_qscf_kernel_z_ini_runtime,pnlpt->error_message,
                   "Diagnostic target z=%.8g must be less than kernel z_ini=%.8g",
                   zs[iz],qcdm_qscf_kernel_z_ini_runtime);
        class_call(background_tau_of_z(pba,zs[iz],&tau),
                   pba->error_message,pnlpt->error_message);
        qcdm_qscf_qs_coefficients qc;
        class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        double om=qc.Omega_b+qc.Omega_q;
        class_test(om<=0.0,pnlpt->error_message,
                   "Nonpositive baryon+QCDM background density at z=%.8g",zs[iz]);
        double wb=qc.Omega_b/om;
        for (int ik=0;ik<nk;ik++) {
            qcdm_qscf_kernel1_result K1;
            class_call(qcdm_qscf_qs_solve_F1_reduced(
                           pba,tau,khs[ik]*pba->h,&K1,pnlpt->error_message),
                       pnlpt->error_message,pnlpt->error_message);
            double gm=K1.G1_m_red,gr=K1.G1_r_red;
            class_test(fabs(gm)<1.e-14 || !qcdm_qscf_finite_double(gm) ||
                       !qcdm_qscf_finite_double(gr),pnlpt->error_message,
                       "Invalid F1 velocity reduction at z=%g k=%g: Gm=%g Gr=%g",
                       zs[iz],khs[ik],gm,gr);
            double R=fabs(wb*gr/gm);
            fprintf(fp,"%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                    zs[iz],khs[ik],wb,K1.F1_m,gm,gr,R);
        }
    }
    class_test(fclose(fp)!=0,pnlpt->error_message,
               "Failed closing QCDM F1 diagnostic CSV '%s'",path);
    return _SUCCESS_;
}

static int qcdm_qscf_qs_solve_F3_collapsed_full(struct background *pba,double tau_final,double k,double q,double mu,qcdm_qscf_kernel3_result *res,ErrorMsg errmsg) {
    if (qcdm_qscf_cache_get_f3(tau_final,k,q,mu,res) == _TRUE_) return _SUCCESS_;
    class_test(qcdm_qscf_cache_readonly_parallel==_TRUE_,errmsg,
               "QCDM parallel native-basis F3 cache miss: tau=%e k=%e q=%e mu=%e. "
               "The basis OpenMP path requires the initialization kernel cache to be warm.",
               tau_final,k,q,mu);
/* DW: evolve the full baryon/QCDM collapsed F3 system when it is absent from the cache. */
    class_call(qcdm_qscf_qs_integrate_F3_collapsed(pba,tau_final,k,q,mu,res,errmsg),errmsg,errmsg);
    if (qcdm_qscf_cache_parallel_nostore!=_TRUE_) {
        class_call(qcdm_qscf_cache_put_f3(tau_final,k,q,mu,res,errmsg),errmsg,errmsg);
        /* DW: mirror the reduced total F3 kernel into the lightweight P13 cache. */
        class_call(qcdm_qscf_cache_put_f3m(tau_final,k,q,mu,res->F3_m_red,errmsg),errmsg,errmsg);
    }
    return _SUCCESS_;
}

static double qcdm_tf_legendre(int ell,double mu) {
    if (ell==0) return 1.0;
    if (ell==2) return 0.5*(3.0*mu*mu-1.0);
    if (ell==4) { double m2=mu*mu; return (35.0*m2*m2-30.0*m2+3.0)/8.0; }
    if (ell==6) { double m2=mu*mu,m4=m2*m2,m6=m4*m2; return (231.0*m6-315.0*m4+105.0*m2-5.0)/16.0; }
    if (ell==8) { double m2=mu*mu,m4=m2*m2,m6=m4*m2,m8=m4*m4; return (6435.0*m8-12012.0*m6+6930.0*m4-1260.0*m2+35.0)/128.0; }
    return 0.0;
}

/* DW: generate a stable Gauss-Legendre angular grid for loop and multipole integration. */
static int qcdm_tf_gauss_legendre_grid(int n,double *x,double *w,ErrorMsg errmsg) {
    class_test(n<1,errmsg,"Gauss-Legendre order must be positive (got %d)",n);
    int m=(n+1)/2;
    for (int i=0;i<m;i++) {
        double z=cos(_PI_*(((double)i)+0.75)/(((double)n)+0.5));
        double zold=0.0,pp=0.0; int it=0;
        do {
            double p1=1.0,p2=0.0;
            for (int j=1;j<=n;j++) { double p3=p2; p2=p1; p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j; }
            pp=n*(z*p1-p2)/(z*z-1.0); zold=z; z=zold-p1/pp; it++;
            class_test(it>100,errmsg,"Gauss-Legendre root finder failed for n=%d",n);
        } while (fabs(z-zold)>2.e-15);
        x[i]=-z; x[n-1-i]=z; w[i]=2.0/((1.0-z*z)*pp*pp); w[n-1-i]=w[i];
    }
    return _SUCCESS_;
}

/* ===================================================================== */
/* DW: exact-QS anisotropic IR resummation                               */
/* ===================================================================== */
/*
 * The exact two-fluid kernel hierarchy evolves, species by species,
 *
 *   dF_s/d ln a = G_s,
 *
 * with Theta_s=-theta_s/Hconf.  Thus F_s is the linear displacement-
 * divergence transfer and G_s is the corresponding velocity-displacement
 * transfer in the conventions used by the Z_n kernels.  At the final time
 * the common mode used by the tracer is
 *
 *   F_m = w_b F_b + w_q F_q,
 *   G_m = w_b G_b + w_q G_q,
 *
 * so the physical smooth power P_nw supplies F_m^2 P_0, while the reduced
 * ratios G_m/F_m, (F_q-F_b)/F_m and (G_q-G_b)/F_m supply the velocity and
 * relative-displacement factors.  This remains true even when the energy
 * fractions w_b,w_q evolve: the RSD mapping needs the equal-time
 * displacement and velocity fields, not d(w_b F_b+w_q F_q)/d ln a.
 *
 * For scale-dependent growth, the common-mode redshift-space BAO variance is
 *
 * Sigma_s^2(mu) = Sigma_DD + 2 mu^2 Sigma_DV + mu^2 Sigma_VV
 *               + mu^2(mu^2-1) deltaSigma_VV,
 *
 * where Sigma_DD=I_S[P_nw], Sigma_DV=I_S[g_m P_nw],
 * Sigma_VV=I_S[g_m^2 P_nw], and deltaSigma_VV=I_delta[g_m^2 P_nw].
 * For g_m=f=const this reduces exactly to the standard CLASS-PT RSD
 * expression.
 */
typedef struct qcdm_tf_ir_state {
    int enabled;
    double rbao;
    double ks;

    double Sigma_DD;
    double deltaSigma_DD;
    double Sigma_DV;
    double Sigma_VV;
    double deltaSigma_VV;

    /* Relative-mode audit quantities.  These are diagnostic only: the
     * production exponent below resums the common displacement and leaves
     * relative operators perturbative. */
    double Sigma_rr_DD;
    double Sigma_rr_DV;
    double Sigma_rr_VV;
    double deltaSigma_rr_VV;

    double Sigma_mr_DD;
    double Sigma_mr_DVsum;
    double Sigma_mr_VV;
    double deltaSigma_mr_VV;

    double max_rr_over_mm;
    double max_mr_over_mm;
} qcdm_tf_ir_state;

typedef struct qcdm_tf_ir_fast_cache {
    int nmax;
    int nz;
    double *Pnw;
    double *Pw;
    qcdm_tf_ir_state *state;
    short *ready;
} qcdm_tf_ir_fast_cache;

static qcdm_tf_ir_fast_cache qcdm_tf_ir_cache = {0};

static void qcdm_tf_ir_fast_cache_reset(void) {
    free(qcdm_tf_ir_cache.Pnw);
    free(qcdm_tf_ir_cache.Pw);
    free(qcdm_tf_ir_cache.state);
    free(qcdm_tf_ir_cache.ready);
    memset(&qcdm_tf_ir_cache,0,sizeof(qcdm_tf_ir_cache));
}


/* DW: validation-only override for the IR soft/hard separation scale.
 * Production default remains kS = 0.20 h/Mpc. */
static double qcdm_tf_ir_ks_hmpc(void) {
    const char *s=getenv("QCDM_IR_KS_HMPC");
    if (s==NULL || s[0]=='\0') return 0.20;

    errno=0;
    char *end=NULL;
    double v=strtod(s,&end);

    if (errno!=0 || end==s || (end!=NULL && *end!='\0') ||
        !isfinite(v) || !(v>0.0)) {
        fprintf(stderr,
                "Invalid QCDM_IR_KS_HMPC='%s'; expected a positive number in h/Mpc.\n",
                s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static double qcdm_tf_ir_j2(double x) {
    double ax=fabs(x);
    if (ax<1.e-3) {
        double x2=x*x;
        return x2/15.0-x2*x2/210.0+x2*x2*x2/7560.0;
    }
    return ((3.0/(x*x*x)-1.0/x)*sin(x)-3.0*cos(x)/(x*x));
}

static double qcdm_tf_ir_sigma_kernel(double x) {
    double ax=fabs(x);
    if (ax<1.e-3) {
        double x2=x*x;
        return 0.3*x2-x2*x2/56.0+x2*x2*x2/2160.0;
    }
    return 1.0-sin(x)/x+2.0*qcdm_tf_ir_j2(x);
}

static double qcdm_tf_ir_sigma_common(const qcdm_tf_ir_state *s,double mu) {
    double mu2=mu*mu;
    return s->Sigma_DD
         + 2.0*mu2*s->Sigma_DV
         + mu2*s->Sigma_VV
         + mu2*(mu2-1.0)*s->deltaSigma_VV;
}

static double qcdm_tf_ir_sigma_rr(const qcdm_tf_ir_state *s,double mu) {
    double mu2=mu*mu;
    return s->Sigma_rr_DD
         + 2.0*mu2*s->Sigma_rr_DV
         + mu2*s->Sigma_rr_VV
         + mu2*(mu2-1.0)*s->deltaSigma_rr_VV;
}

static double qcdm_tf_ir_sigma_mr(const qcdm_tf_ir_state *s,double mu) {
    double mu2=mu*mu;
    return s->Sigma_mr_DD
         + mu2*s->Sigma_mr_DVsum
         + mu2*s->Sigma_mr_VV
         + mu2*(mu2-1.0)*s->deltaSigma_mr_VV;
}

static double qcdm_tf_ir_exponent(const qcdm_tf_ir_state *s,double k,double mu) {
    if ((s==NULL)||(s->enabled!=_TRUE_)) return 0.0;
    double sig=qcdm_tf_ir_sigma_common(s,mu);
    /* Sigma_s^2 is a covariance and must be non-negative.  Permit only a
     * roundoff-sized excursion below zero before clipping. */
    if (sig<0.0 && sig>-1.e-12*(1.0+fabs(s->Sigma_DD))) sig=0.0;
    return k*k*sig;
}

static int qcdm_tf_compute_ir_state(struct background *pba,
                                    struct nonlinear_pt *pnlpt,
                                    double tau,
                                    int Nmax,
                                    const double *kdisc,
                                    const double *Pnw,
                                    double rbao,
                                    double ks,
                                    qcdm_tf_ir_state *out) {
    memset(out,0,sizeof(*out));
    out->rbao=rbao;
    out->ks=ks;

    double qlo=MAX(kdisc[0],7.e-5);
    double qhi=MIN(kdisc[Nmax-1],ks);
    class_test(!(rbao>0.0) || !(qhi>qlo),pnlpt->error_message,
               "Invalid QCDM IR domain: rbao=%e qlo=%e qhi=%e",rbao,qlo,qhi);

    /* High-order GL in ln q.  This is done once per cosmology/redshift;
     * subsequent nuisance-only calls reuse the cached state. */
    const int Nir=256;
    double *x,*w;
    class_alloc(x,Nir*sizeof(double),pnlpt->error_message);
    class_alloc(w,Nir*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nir,x,w,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);

    double ylo=log(qlo),yhi=log(qhi);
    double ymid=0.5*(ylo+yhi),yhalf=0.5*(yhi-ylo);

    double S_DD=0.0,dS_DD=0.0,S_DV=0.0,S_VV=0.0,dS_VV=0.0;
    double S_rr_DD=0.0,S_rr_DV=0.0,S_rr_VV=0.0,dS_rr_VV=0.0;
    double S_mr_DD=0.0,S_mr_DVsum=0.0,S_mr_VV=0.0,dS_mr_VV=0.0;

    double gm_min=DBL_MAX,gm_max=-DBL_MAX;
    for (int i=0;i<Nir;i++) {
        double y=ymid+yhalf*x[i];
        double q=exp(y);
        double dq=q*yhalf*w[i];
        double P=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,q);
        class_test((qcdm_qscf_finite_double(P)==0)||(P<0.0),pnlpt->error_message,
                   "Invalid QCDM smooth power in IR covariance: q=%e Pnw=%e",q,P);

        qcdm_qscf_kernel1_result K1;
        class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,q,&K1,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        double gm=K1.G1_m_red;
        double fr=K1.F1_r_red;
        double gr=K1.G1_r_red;
        gm_min=MIN(gm_min,gm);
        gm_max=MAX(gm_max,gm);

        double qr=q*rbao;
        double KS=qcdm_tf_ir_sigma_kernel(qr);
        double KD=qcdm_tf_ir_j2(qr);
        double base=dq*P;

        S_DD += base*KS;
        dS_DD += base*KD;
        S_DV += base*KS*gm;
        S_VV += base*KS*gm*gm;
        dS_VV += base*KD*gm*gm;

        S_rr_DD += base*KS*fr*fr;
        S_rr_DV += base*KS*fr*gr;
        S_rr_VV += base*KS*gr*gr;
        dS_rr_VV += base*KD*gr*gr;

        S_mr_DD += base*KS*fr;
        S_mr_DVsum += base*KS*(gr+gm*fr);
        S_mr_VV += base*KS*gm*gr;
        dS_mr_VV += base*KD*gm*gr;
    }
    free(x); free(w);

    double inv6pi2=1.0/(6.0*_PI_*_PI_);
    double inv2pi2=1.0/(2.0*_PI_*_PI_);
    out->Sigma_DD=S_DD*inv6pi2;
    out->deltaSigma_DD=dS_DD*inv2pi2;
    out->Sigma_DV=S_DV*inv6pi2;
    out->Sigma_VV=S_VV*inv6pi2;
    out->deltaSigma_VV=dS_VV*inv2pi2;

    out->Sigma_rr_DD=S_rr_DD*inv6pi2;
    out->Sigma_rr_DV=S_rr_DV*inv6pi2;
    out->Sigma_rr_VV=S_rr_VV*inv6pi2;
    out->deltaSigma_rr_VV=dS_rr_VV*inv2pi2;

    out->Sigma_mr_DD=S_mr_DD*inv6pi2;
    out->Sigma_mr_DVsum=S_mr_DVsum*inv6pi2;
    out->Sigma_mr_VV=S_mr_VV*inv6pi2;
    out->deltaSigma_mr_VV=dS_mr_VV*inv2pi2;

    class_test((qcdm_qscf_finite_double(out->Sigma_DD)==0)||(out->Sigma_DD<=0.0) ||
               (qcdm_qscf_finite_double(out->Sigma_DV)==0) ||
               (qcdm_qscf_finite_double(out->Sigma_VV)==0) ||
               (qcdm_qscf_finite_double(out->deltaSigma_VV)==0),
               pnlpt->error_message,
               "Non-finite/invalid QCDM IR covariance: DD=%e DV=%e VV=%e dVV=%e",
               out->Sigma_DD,out->Sigma_DV,out->Sigma_VV,out->deltaSigma_VV);

    out->max_rr_over_mm=0.0;
    out->max_mr_over_mm=0.0;
    for (int j=0;j<=200;j++) {
        double mu=((double)j)/200.0;
        double mm=qcdm_tf_ir_sigma_common(out,mu);
        double rr=qcdm_tf_ir_sigma_rr(out,mu);
        double mr=qcdm_tf_ir_sigma_mr(out,mu);
        class_test(mm < -1.e-10*(1.0+fabs(out->Sigma_DD)),pnlpt->error_message,
                   "Negative common QCDM redshift-space displacement variance: mu=%e Sigma=%e",mu,mm);
        if (mm>1.e-30) {
            out->max_rr_over_mm=MAX(out->max_rr_over_mm,fabs(rr/mm));
            out->max_mr_over_mm=MAX(out->max_mr_over_mm,fabs(mr/mm));
        }
    }

    /* In the uncoupled limit the scale-dependent expression must collapse
     * to the standard CLASS-PT formula.  This is an audit, not an assumption
     * used by the production exponent. */
    qcdm_qscf_qs_coefficients qc;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    double uncoupled_resid=-1.0;
    if (fabs(qc.dalpha)<1.e-12) {
        double *pv;
        int last=0;
        class_alloc(pv,pba->bg_size*sizeof(double),pnlpt->error_message);
        class_call(background_at_tau(pba,tau,long_info,inter_normal,&last,pv),
                   pba->error_message,pnlpt->error_message);
        double fclass=pv[pba->index_bg_f];
        free(pv);
        uncoupled_resid=0.0;
        for (int j=0;j<=40;j++) {
            double mu=((double)j)/40.0;
            double mu2=mu*mu;
            double exact=qcdm_tf_ir_sigma_common(out,mu);
            double standard=(1.0+fclass*mu2*(2.0+fclass))*out->Sigma_DD
                           +fclass*fclass*mu2*(mu2-1.0)*out->deltaSigma_DD;
            double scale=MAX(1.e-12,MAX(fabs(exact),fabs(standard)));
            uncoupled_resid=MAX(uncoupled_resid,fabs(exact-standard)/scale);
        }
        class_test(uncoupled_resid>5.e-2,pnlpt->error_message,
                   "QCDM IR one-fluid-limit audit failed: max relative residual=%e, gm=[%e,%e]",
                   uncoupled_resid,gm_min,gm_max);
    }

    out->enabled=_TRUE_;
    if (pnlpt->nonlinear_pt_verbose>0) {
        double z=0.0;
        class_call(background_z_of_tau(pba,tau,&z),pba->error_message,pnlpt->error_message);
        printf("DW/QCDM-IR: z=%.8g rBAO=%.6g Mpc kS=%.6g 1/Mpc; Sigma_DD=%e Sigma_DV=%e Sigma_VV=%e deltaSigma_VV=%e Mpc^2.\n",
               z,rbao,ks,out->Sigma_DD,out->Sigma_DV,out->Sigma_VV,out->deltaSigma_VV);
        printf("DW/QCDM-IR relative-displacement audit: max |Sigma_rr/Sigma_mm|=%e, max |Sigma_mr/Sigma_mm|=%e; Gm/Fm over IR modes=[%e,%e]%s",
               out->max_rr_over_mm,out->max_mr_over_mm,gm_min,gm_max,
               (uncoupled_resid>=0.0)?"; one-fluid audit residual=":".\n");
        if (uncoupled_resid>=0.0) printf("%e.\n",uncoupled_resid);
    }
    return _SUCCESS_;
}

static int qcdm_tf_ir_fast_cache_store(struct background *pba,
                                       struct nonlinear_pt *pnlpt,
                                       double tau,
                                       int Nmax,
                                       const double *Pnw,
                                       const double *Pw,
                                       const qcdm_tf_ir_state *state) {
    if ((state==NULL)||(state->enabled!=_TRUE_)) return _SUCCESS_;

    double z=0.0;
    class_call(background_z_of_tau(pba,tau,&z),pba->error_message,pnlpt->error_message);
    int iz=0;
    double dzbest=fabs(z-pnlpt->z_pk[0]);
    for (int j=1;j<pnlpt->z_pk_num;j++) {
        double dz=fabs(z-pnlpt->z_pk[j]);
        if (dz<dzbest) {dzbest=dz;iz=j;}
    }
    class_test(dzbest>1.e-7*(1.0+fabs(z)),pnlpt->error_message,
               "Could not associate QCDM IR cache tau=%e (z=%e) with requested z_pk list",tau,z);

    if (qcdm_tf_ir_cache.nmax==0) {
        qcdm_tf_ir_cache.nmax=Nmax;
        qcdm_tf_ir_cache.nz=pnlpt->z_pk_num;
        class_alloc(qcdm_tf_ir_cache.Pnw,(size_t)Nmax*qcdm_tf_ir_cache.nz*sizeof(double),pnlpt->error_message);
        class_alloc(qcdm_tf_ir_cache.Pw,(size_t)Nmax*qcdm_tf_ir_cache.nz*sizeof(double),pnlpt->error_message);
        class_alloc(qcdm_tf_ir_cache.state,qcdm_tf_ir_cache.nz*sizeof(qcdm_tf_ir_state),pnlpt->error_message);
        class_alloc(qcdm_tf_ir_cache.ready,qcdm_tf_ir_cache.nz*sizeof(short),pnlpt->error_message);
        memset(qcdm_tf_ir_cache.Pnw,0,(size_t)Nmax*qcdm_tf_ir_cache.nz*sizeof(double));
        memset(qcdm_tf_ir_cache.Pw,0,(size_t)Nmax*qcdm_tf_ir_cache.nz*sizeof(double));
        memset(qcdm_tf_ir_cache.state,0,qcdm_tf_ir_cache.nz*sizeof(qcdm_tf_ir_state));
        memset(qcdm_tf_ir_cache.ready,0,qcdm_tf_ir_cache.nz*sizeof(short));
    }
    class_test(qcdm_tf_ir_cache.nmax!=Nmax || qcdm_tf_ir_cache.nz!=pnlpt->z_pk_num,
               pnlpt->error_message,
               "QCDM IR fast cache dimensions changed within one cosmology");
    memcpy(qcdm_tf_ir_cache.Pnw+(size_t)iz*Nmax,Pnw,Nmax*sizeof(double));
    memcpy(qcdm_tf_ir_cache.Pw +(size_t)iz*Nmax,Pw ,Nmax*sizeof(double));
    qcdm_tf_ir_cache.state[iz]=*state;
    qcdm_tf_ir_cache.ready[iz]=_TRUE_;
    return _SUCCESS_;
}


static double qcdm_tf_s2_vec(qcdm_qscf_kvec3 a,qcdm_qscf_kvec3 b) {
    double na=qcdm_qscf_norm(a), nb=qcdm_qscf_norm(b);
    if ((na<=0.0)||(nb<=0.0)) return 0.0;
    double mu=qcdm_qscf_dot(a,b)/(na*nb);
    return mu*mu-1.0/3.0;
}

static double qcdm_tf_los_mu(qcdm_qscf_kvec3 a,qcdm_qscf_kvec3 los) {
    double na=qcdm_qscf_norm(a);
    if (na<=0.0) return 0.0;
    return qcdm_qscf_dot(a,los)/na;
}

static double qcdm_tf_V1(struct nonlinear_pt *pnlpt,const qcdm_qscf_kernel1_result *K1) {
    return K1->G1_m_red + pnlpt->qcdm_tf_c_vr*K1->G1_r_red;
}

static double qcdm_tf_D1(struct nonlinear_pt *pnlpt,const qcdm_qscf_kernel1_result *K1) {
    return pnlpt->qcdm_tf_b_m + pnlpt->qcdm_tf_b_r*K1->F1_r_red + pnlpt->qcdm_tf_b_theta_r*K1->G1_r_red;
}

static double qcdm_tf_Z1(struct nonlinear_pt *pnlpt,const qcdm_qscf_kernel1_result *K1,double mu_los) {
    return qcdm_tf_D1(pnlpt,K1) + mu_los*mu_los*qcdm_tf_V1(pnlpt,K1);
}

/* DW speed: F2 evolves both linear legs internally. Reconstruct the reduced
 * F1 data for a leg directly from that exact solution instead of launching a
 * redundant F1 ODE solve. */
static void qcdm_tf_K1_from_K2_leg(const qcdm_qscf_kernel2_result *K2,int leg,
                                   qcdm_qscf_kernel1_result *K1) {
    if (leg==1) {
        K1->F1_m=K2->F1m_k1;
        K1->F1_r_red=K2->F1r_k1;
        K1->G1_m_red=K2->G1m_k1;
        K1->G1_r_red=K2->G1r_k1;
    }
    else {
        K1->F1_m=K2->F1m_k2;
        K1->F1_r_red=K2->F1r_k2;
        K1->G1_m_red=K2->G1m_k2;
        K1->G1_r_red=K2->G1r_k2;
    }
}

static double qcdm_tf_D2(struct nonlinear_pt *pnlpt,const qcdm_qscf_kernel2_result *K2,qcdm_qscf_kvec3 k1,qcdm_qscf_kvec3 k2) {
    double s2=qcdm_tf_s2_vec(k1,k2);
    double fr1=K2->F1r_k1, fr2=K2->F1r_k2;
    return pnlpt->qcdm_tf_b_m*K2->F2_m_red
         + pnlpt->qcdm_tf_b_r*K2->F2_r_red
         + pnlpt->qcdm_tf_b_theta_r*K2->G2_r_red
         + 0.5*pnlpt->qcdm_tf_b_mm
         + 0.5*pnlpt->qcdm_tf_b_mr*(fr1+fr2)
         + 0.5*pnlpt->qcdm_tf_b_rr*fr1*fr2
         + pnlpt->qcdm_tf_b_Gmm*s2
         + 0.5*pnlpt->qcdm_tf_b_Gmr*s2*(fr1+fr2)
         + pnlpt->qcdm_tf_b_Grr*s2*fr1*fr2;
}

/* DW: add nonlinear relative-velocity bias coefficients to the two-fluid tracer model. */
static double qcdm_tf_velocity_K_kernel(qcdm_qscf_kvec3 ktot,
                                               qcdm_qscf_kvec3 kdensity,
                                               qcdm_qscf_kvec3 kvelocity) {
    double kd2=qcdm_qscf_dot(kdensity,kdensity);
    double kv2=qcdm_qscf_dot(kvelocity,kvelocity);
    if ((kd2<=0.0)||(kv2<=0.0)) return 0.0;
    return qcdm_qscf_dot(ktot,kdensity)*qcdm_qscf_dot(kdensity,kvelocity)/(kd2*kv2)
         - qcdm_qscf_dot(ktot,kvelocity)/(3.0*kv2);
}

static double qcdm_tf_V2(struct nonlinear_pt *pnlpt,
                         const qcdm_qscf_kernel2_result *K2,
                         qcdm_qscf_kvec3 k1,
                         qcdm_qscf_kvec3 k2) {
    qcdm_qscf_kvec3 k=qcdm_qscf_add(k1,k2);
    double k1sq=qcdm_qscf_dot(k1,k1), k2sq=qcdm_qscf_dot(k2,k2);
    double conv_mr=0.0, conv_K=0.0;
    if (k1sq>0.0)
        conv_mr += 0.5*qcdm_qscf_dot(k,k1)/k1sq*K2->G1r_k1;
    if (k2sq>0.0)
        conv_mr += 0.5*qcdm_qscf_dot(k,k2)/k2sq*K2->G1r_k2;
    conv_K = 0.5*(qcdm_tf_velocity_K_kernel(k,k2,k1)*K2->G1r_k1
                 +qcdm_tf_velocity_K_kernel(k,k1,k2)*K2->G1r_k2);
    return K2->G2_m_red
         + pnlpt->qcdm_tf_c_vr*K2->G2_r_red
         + pnlpt->qcdm_tf_c_mr*conv_mr
         + pnlpt->qcdm_tf_c_K*conv_K;
}

static double qcdm_tf_Z2_from_kernels(struct nonlinear_pt *pnlpt,
                                         qcdm_qscf_kvec3 k1,
                                         qcdm_qscf_kvec3 k2,
                                         qcdm_qscf_kvec3 los,
                                         const qcdm_qscf_kernel2_result *K2,
                                         const qcdm_qscf_kernel1_result *K1a,
                                         const qcdm_qscf_kernel1_result *K1b) {
    double k1n=qcdm_qscf_norm(k1), k2n=qcdm_qscf_norm(k2);
    qcdm_qscf_kvec3 k=qcdm_qscf_add(k1,k2);
    double kn=qcdm_qscf_norm(k);
    double muk=qcdm_tf_los_mu(k,los);
    double mu1=qcdm_tf_los_mu(k1,los), mu2=qcdm_tf_los_mu(k2,los);
    double Z1a=qcdm_tf_Z1(pnlpt,K1a,mu1);
    double Z1b=qcdm_tf_Z1(pnlpt,K1b,mu2);
    double V1a=qcdm_tf_V1(pnlpt,K1a);
    double V1b=qcdm_tf_V1(pnlpt,K1b);
    double D2=qcdm_tf_D2(pnlpt,K2,k1,k2);
    double V2=qcdm_tf_V2(pnlpt,K2,k1,k2);
    double mapping=0.0;
    if (k1n>0.0) mapping += (mu1/k1n)*V1a*Z1b;
    if (k2n>0.0) mapping += (mu2/k2n)*V1b*Z1a;
    return D2 + muk*muk*V2 + 0.5*muk*kn*mapping;
}

static int qcdm_tf_Z2(struct background *pba,struct nonlinear_pt *pnlpt,double tau,qcdm_qscf_kvec3 k1,qcdm_qscf_kvec3 k2,qcdm_qscf_kvec3 los,double *Z2,ErrorMsg errmsg) {
    qcdm_qscf_kernel2_result K2;
    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,k1,k2,&K2,errmsg),errmsg,errmsg);
    double k1n=qcdm_qscf_norm(k1), k2n=qcdm_qscf_norm(k2);
    qcdm_qscf_kernel1_result K1a,K1b;
    class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,k1n,&K1a,errmsg),errmsg,errmsg);
    class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,k2n,&K1b,errmsg),errmsg,errmsg);
    *Z2=qcdm_tf_Z2_from_kernels(pnlpt,k1,k2,los,&K2,&K1a,&K1b);
    return _SUCCESS_;
}

/* DW: complete the third-order tracer density and nonlocal bias kernel for P13. */
static double qcdm_tf_pair_total_tidal(qcdm_qscf_kvec3 spectator,
                                       qcdm_qscf_kvec3 a,
                                       qcdm_qscf_kvec3 b,
                                       double trace_subtraction) {
    qcdm_qscf_kvec3 pair=qcdm_qscf_add(a,b);
    double ns=qcdm_qscf_norm(spectator), np=qcdm_qscf_norm(pair);
    if ((ns<=0.0)||(np<=0.0)) return 0.0;
    double mu=qcdm_qscf_dot(spectator,pair)/(ns*np);
    return mu*mu-trace_subtraction;
}

static double qcdm_tf_D3_collapsed(struct nonlinear_pt *pnlpt,
                                   const qcdm_qscf_kernel3_result *K3,
                                   qcdm_qscf_kvec3 p[3],
                                   qcdm_qscf_kernel2_result pair[3]) {
    /* pair[0]=(p1,p2), spectator p0; pair[1]=(p0,p2), spectator p1;
       pair[2]=(p0,p1), spectator p2.  This is the symmetrized third-order
       kernel of the renormalized two-fluid bias basis.  The Gamma3 term is
       evaluated from its defining density-velocity tidal difference, so it
       remains valid for exact-time, non-EdS kernels. */
    const int spec[3]={0,1,2};
    const int ia[3]={1,0,0};
    const int ib[3]={2,2,1};
    /* DW: map each spectator to its own linear relative kernel explicitly:
     * p0 from pair(0,2).k1, p1 from pair(0,1).k2, and p2 from
     * pair(0,2).k2.  The previous p1/p2 ordering was harmless only in the
     * collapsed |p1|=|p2| configuration but obscured the permutation logic. */
    double fr_spec[3] = {pair[1].F1r_k1, pair[2].F1r_k2, pair[1].F1r_k2};
/* DW: accumulate the renormalized third-order local, relative, tidal, and Gamma3 bias shapes. */
    double sum_F2m=0.0,sum_mr=0.0,sum_rr=0.0;
    double sum_Gmm=0.0,sum_Gmr=0.0,sum_Grr=0.0,sum_Gamma3=0.0;
    for(int i=0;i<3;i++){
        double Fm=pair[i].F2_m_red, Fr=pair[i].F2_r_red, fr=fr_spec[i];
/* DW: evaluate the trace-free tidal contraction for each spectator-pair partition. */
        double S=qcdm_tf_pair_total_tidal(p[spec[i]],p[ia[i]],p[ib[i]],1.0/3.0);
        double T=qcdm_tf_pair_total_tidal(p[spec[i]],p[ia[i]],p[ib[i]],1.0);
        sum_F2m += Fm;
        sum_mr += Fm*fr + Fr;
        sum_rr += Fr*fr;
/* DW: build total/relative tidal and exact-time Gamma3 contributions for Z3. */
        sum_Gmm += S*Fm;
        sum_Gmr += S*(Fm*fr + Fr);
        sum_Grr += S*Fr*fr;
        sum_Gamma3 += T*(Fm-pair[i].G2_m_red);
    }
    return pnlpt->qcdm_tf_b_m*K3->F3_m_red
         + pnlpt->qcdm_tf_b_r*K3->F3_r_red
         + pnlpt->qcdm_tf_b_theta_r*K3->G3_r_red
         + (pnlpt->qcdm_tf_b_mm/3.0)*sum_F2m
         + (pnlpt->qcdm_tf_b_mr/3.0)*sum_mr
         + (pnlpt->qcdm_tf_b_rr/3.0)*sum_rr
/* DW: add the third-order nonlocal Gamma3 bias and relative EFT parameters. */
         + (2.0*pnlpt->qcdm_tf_b_Gmm/3.0)*sum_Gmm
         + (pnlpt->qcdm_tf_b_Gmr/3.0)*sum_Gmr
/* DW: add nonlinear relative-velocity bias coefficients to the two-fluid tracer model. */
         + (2.0*pnlpt->qcdm_tf_b_Grr/3.0)*sum_Grr
         + (2.0*pnlpt->qcdm_tf_b_Gamma3/3.0)*sum_Gamma3;
}

static double qcdm_tf_V3_collapsed(struct nonlinear_pt *pnlpt,
                                   const qcdm_qscf_kernel3_result *K3,
                                   qcdm_qscf_kvec3 p[3],
                                   qcdm_qscf_kernel2_result pair[3]) {
    const int spec[3]={0,1,2};
    const int ia[3]={1,0,0};
    const int ib[3]={2,2,1};
    /* DW: keep the spectator-relative-velocity mapping permutation-correct. */
    double fr_spec[3] = {pair[1].G1r_k1, pair[2].G1r_k2, pair[1].G1r_k2};
    qcdm_qscf_kvec3 ktot=qcdm_qscf_add(qcdm_qscf_add(p[0],p[1]),p[2]);
    double sum_mr=0.0,sum_K=0.0;
    for(int i=0;i<3;i++){
        qcdm_qscf_kvec3 ps=p[spec[i]];
        qcdm_qscf_kvec3 pp=qcdm_qscf_add(p[ia[i]],p[ib[i]]);
        double nssq=qcdm_qscf_dot(ps,ps), npsq=qcdm_qscf_dot(pp,pp);
        if (npsq>0.0)
            sum_mr += qcdm_qscf_dot(ktot,pp)/npsq*pair[i].G2_r_red;
        if (nssq>0.0)
            sum_mr += qcdm_qscf_dot(ktot,ps)/nssq*fr_spec[i]*pair[i].F2_m_red;
        sum_K += qcdm_tf_velocity_K_kernel(ktot,ps,pp)*pair[i].G2_r_red;
        sum_K += qcdm_tf_velocity_K_kernel(ktot,pp,ps)*fr_spec[i]*pair[i].F2_m_red;
    }
    return K3->G3_m_red
         + pnlpt->qcdm_tf_c_vr*K3->G3_r_red
         + (pnlpt->qcdm_tf_c_mr/3.0)*sum_mr
         + (pnlpt->qcdm_tf_c_K/3.0)*sum_K;
}

static double qcdm_tf_Z3_collapsed_from_kernels(struct nonlinear_pt *pnlpt,
                                                    qcdm_qscf_kvec3 p[3],
                                                    qcdm_qscf_kvec3 los,
                                                    double muk,
                                                    const qcdm_qscf_kernel3_result *K3,
                                                    qcdm_qscf_kernel2_result pair[3],
                                                    const qcdm_qscf_kernel1_result K1[3],
                                                    double D3,double V3) {
    /* DW RENORM-P13: project standard local b2 out of Z3/P13.
     *
     * The second-order native coordinates obey
     *   b2_std  = b_mm + (4/3) b_Gmm,
     *   bG2_std = b_Gmm.
     *
     * Standard CLASS-PT's renormalized local b2 template is exhausted by
     * the P22-type contribution. Hence the independent local b2_std
     * component must not enter P13. The G2 direction is preserved exactly
     * because b2_std=0 for b_mm=-(4/3)b_Gmm.
     */
    const double b2_std_local =
        pnlpt->qcdm_tf_b_mm + (4.0/3.0)*pnlpt->qcdm_tf_b_Gmm;
    const double sum_F2m_p13 =
        pair[0].F2_m_red + pair[1].F2_m_red + pair[2].F2_m_red;
    const double D3_ren =
        D3 - (b2_std_local/3.0)*sum_F2m_p13;
    /* DW FIX: exact symmetrized third-order RSD mapping.
     *
     * Write A = (k_tot . zhat) = mu*k.  At third order the mapping contains
     *
     *   D3 + mu^2 V3
     * + A/3 sum_i x_i V1_i [D2_jl + mu_jl^2 V2_jl]
     * + A/3 sum_(ij|l) x_ij V2_ij Z1_l
     * + A^2/6 sum_i (x_j V1_j)(x_l V1_l) Z1_i,
     *
     * where x_i=mu_i/k_i and x_ij=mu_ij/k_ij.
     *
     * The previous implementation put the FULL Z2_jl in the first sum.
     * That folds the second-order velocity-mapping term into the wrong
     * momentum factor (k_j+k_l instead of the total external k), producing
     * an unphysical O(k^0) contribution to collapsed Z3(k,q,-q), and hence
     * an O(P_L(k)) contamination of P13 at k -> 0.
     */
    qcdm_qscf_kvec3 ktot=qcdm_qscf_add(qcdm_qscf_add(p[0],p[1]),p[2]);
    double kn=qcdm_qscf_norm(ktot);
    double A=muk*kn;

    double map_V1_base2=0.0;
    double map_V2_Z1=0.0;
    double map_cubic=0.0;

    /* V1 times the BASE second-order redshift kernel:
     * D2 + mu_pair^2 V2.  Do not use the full Z2 here. */
    for (int i=0;i<3;i++) {
        int j=(i+1)%3,l=(i+2)%3;

        double kin=qcdm_qscf_norm(p[i]);
        if (kin>0.0) {
            int pair_jl=qcdm_qscf_pair_id(j,l);
            qcdm_qscf_kvec3 pjl=qcdm_qscf_add(p[j],p[l]);
            double pjln=qcdm_qscf_norm(pjl);
            double mujl=(pjln>0.0)?qcdm_tf_los_mu(pjl,los):0.0;

            double D2jl=qcdm_tf_D2(pnlpt,&pair[pair_jl],p[j],p[l])
                       -0.5*b2_std_local;
            double V2jl=qcdm_tf_V2(pnlpt,&pair[pair_jl],p[j],p[l]);
            double Z2base_jl=D2jl+mujl*mujl*V2jl;

            map_V1_base2 += (qcdm_tf_los_mu(p[i],los)/kin)
                          *qcdm_tf_V1(pnlpt,&K1[i])
                          *Z2base_jl;
        }

        /* V2(pair) times Z1(spectator). */
        qcdm_qscf_kvec3 pij=qcdm_qscf_add(p[i],p[j]);
        double pijn=qcdm_qscf_norm(pij);
        if (pijn>0.0) {
            int pair_ij=qcdm_qscf_pair_id(i,j);
            double Z1l=qcdm_tf_Z1(pnlpt,&K1[l],
                                  qcdm_tf_los_mu(p[l],los));
            double V2ij=qcdm_tf_V2(pnlpt,&pair[pair_ij],p[i],p[j]);

            map_V2_Z1 += (qcdm_tf_los_mu(pij,los)/pijn)
                        *V2ij*Z1l;
        }
    }

    /* Explicit cubic velocity mapping with the TOTAL A^2=(mu*k)^2 factor. */
    for (int i=0;i<3;i++) {
        int j=(i+1)%3,l=(i+2)%3;
        double kjn=qcdm_qscf_norm(p[j]);
        double kln=qcdm_qscf_norm(p[l]);
        if ((kjn>0.0)&&(kln>0.0)) {
            double xjV1=(qcdm_tf_los_mu(p[j],los)/kjn)
                       *qcdm_tf_V1(pnlpt,&K1[j]);
            double xlV1=(qcdm_tf_los_mu(p[l],los)/kln)
                       *qcdm_tf_V1(pnlpt,&K1[l]);
            double Z1i=qcdm_tf_Z1(pnlpt,&K1[i],
                                  qcdm_tf_los_mu(p[i],los));
            map_cubic += xjV1*xlV1*Z1i;
        }
    }

    return D3_ren + muk*muk*V3
         + (A/3.0)*(map_V1_base2+map_V2_Z1)
         + (A*A/6.0)*map_cubic;
}

static int qcdm_tf_Z3_collapsed(struct background *pba,struct nonlinear_pt *pnlpt,double tau,double k,double q,double xq,double phi,double muk,double *Z3,ErrorMsg errmsg) {
    double sq=sqrt(MAX(1.0-xq*xq,0.0));
    qcdm_qscf_kvec3 p[3];
    p[0].x=k; p[0].y=0.0; p[0].z=0.0;
    p[1].x=q*xq; p[1].y=q*sq*cos(phi); p[1].z=q*sq*sin(phi);
    p[2].x=-p[1].x; p[2].y=-p[1].y; p[2].z=-p[1].z;
    qcdm_qscf_kvec3 los={muk,sqrt(MAX(1.0-muk*muk,0.0)),0.0};
    qcdm_qscf_kernel3_result K3;
    class_call(qcdm_qscf_qs_solve_F3_collapsed_full(pba,tau,k,q,xq,&K3,errmsg),errmsg,errmsg);
    qcdm_qscf_kernel2_result pair[3];
    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,p[1],p[2],&pair[0],errmsg),errmsg,errmsg);
    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,p[0],p[2],&pair[1],errmsg),errmsg,errmsg);
    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,p[0],p[1],&pair[2],errmsg),errmsg,errmsg);
    qcdm_qscf_kernel1_result K1[3];
    class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,qcdm_qscf_norm(p[0]),&K1[0],errmsg),errmsg,errmsg);
    class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,qcdm_qscf_norm(p[1]),&K1[1],errmsg),errmsg,errmsg);
    K1[2]=K1[1];
    double D3=qcdm_tf_D3_collapsed(pnlpt,&K3,p,pair);
    double V3=qcdm_tf_V3_collapsed(pnlpt,&K3,p,pair);
    *Z3=qcdm_tf_Z3_collapsed_from_kernels(pnlpt,p,los,muk,&K3,pair,K1,D3,V3);
    return _SUCCESS_;
}

/* DW: validate the H/Hconf normalization and the permutation symmetry of
 * the collapsed redshift-space kernel before starting the expensive loop
 * integration.  This guards the two convention errors that most directly
 * contaminate the Kaiser term and P13. */
static int qcdm_tf_validate_rsd_conventions(struct background *pba,
                                            struct nonlinear_pt *pnlpt,
                                            double tau,
                                            double kref,
                                            double qref,
                                            ErrorMsg errmsg) {
    qcdm_qscf_qs_coefficients qc;
    class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc,errmsg),errmsg,errmsg);
    class_test(qc.Hphys<=0.0 || qc.Hconf<=0.0,errmsg,
               "Invalid Hubble rates in QCDM RSD audit: H=%e Hconf=%e a=%e",
               qc.Hphys,qc.Hconf,qc.a);
    class_test(fabs(qc.Hconf-qc.a*qc.Hphys)>
               64.0*DBL_EPSILON*(1.0+fabs(qc.Hconf)),errmsg,
               "QCDM RSD Hubble convention failure: Hconf=%e but aH=%e",
               qc.Hconf,qc.a*qc.Hphys);

    /* In the uncoupled, scale-independent limit the exact linear velocity
     * kernel must equal the CLASS logarithmic growth rate.  The loose 2%
     * threshold allows a low-resolution smoke run and finite-z_ini matching,
     * while the old H-versus-Hconf implementation failed by O(1). */
    if (fabs(qc.dalpha)<1.e-12) {
        qcdm_qscf_kernel1_result K1;
        double *pvecback;
        int last_index=0;
        class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,kref,&K1,errmsg),errmsg,errmsg);
        class_alloc(pvecback,pba->bg_size*sizeof(double),errmsg);
        class_call(background_at_tau(pba,tau,long_info,inter_normal,&last_index,pvecback),
                   pba->error_message,errmsg);
        double fclass=pvecback[pba->index_bg_f];
        free(pvecback);
        double scale=MAX(1.e-8,MAX(fabs(fclass),fabs(K1.G1_m_red)));
        class_test(fabs(K1.G1_m_red-fclass)/scale>2.e-2,errmsg,
                   "QCDM RSD growth normalization failure: exact G1/F1=%e, CLASS f=%e at k=%e 1/Mpc",
                   K1.G1_m_red,fclass,kref);
        if (pnlpt->nonlinear_pt_verbose>1)
            printf("DW/QCDM-RSD audit: H=%e 1/Mpc, Hconf=aH=%e 1/Mpc, f_exact=%e, f_CLASS=%e.\n",
                   qc.Hphys,qc.Hconf,K1.G1_m_red,fclass);
    }

    /* DW: check the exact Z2 mapping independently of the loop integral.
     * The symmetrized kernel must be invariant under k1<->k2, and when the
     * total mode is transverse to the line of sight (mu=0) every RSD mapping
     * term vanishes so Z2 reduces exactly to the real-space tracer D2. */
    {
        const double x=0.371;
        double sq=sqrt(1.0-x*x);
        qcdm_qscf_kvec3 k1={kref,0.0,0.0};
        qcdm_qscf_kvec3 k2={qref*x,qref*sq,0.0};
        qcdm_qscf_kvec3 los={0.437,sqrt(1.0-0.437*0.437),0.0};
        double Z2a=0.0,Z2b=0.0;
        class_call(qcdm_tf_Z2(pba,pnlpt,tau,k1,k2,los,&Z2a,errmsg),errmsg,errmsg);
        class_call(qcdm_tf_Z2(pba,pnlpt,tau,k2,k1,los,&Z2b,errmsg),errmsg,errmsg);
        double scale=MAX(1.e-10,MAX(fabs(Z2a),fabs(Z2b)));
        class_test(fabs(Z2a-Z2b)/scale>2.e-8,errmsg,
                   "QCDM Z2 is not symmetric under k1<->k2: Z2a=%e Z2b=%e rel=%e",
                   Z2a,Z2b,fabs(Z2a-Z2b)/scale);

        qcdm_qscf_kernel2_result K2;
        qcdm_qscf_kvec3 los_transverse={0.0,0.0,1.0};
        double Z2trans=0.0;
        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,k1,k2,&K2,errmsg),errmsg,errmsg);
        class_call(qcdm_tf_Z2(pba,pnlpt,tau,k1,k2,los_transverse,&Z2trans,errmsg),errmsg,errmsg);
        double D2=qcdm_tf_D2(pnlpt,&K2,k1,k2);
        scale=MAX(1.e-10,MAX(fabs(Z2trans),fabs(D2)));
        class_test(fabs(Z2trans-D2)/scale>2.e-10,errmsg,
                   "Transverse QCDM Z2 does not reduce to D2: Z2=%e D2=%e rel=%e",
                   Z2trans,D2,fabs(Z2trans-D2)/scale);
        if (pnlpt->nonlinear_pt_verbose>1)
            printf("DW/QCDM-RSD audit: Z2 exchange residual=%e, transverse residual=%e.\n",
                   fabs(Z2a-Z2b)/MAX(1.e-10,MAX(fabs(Z2a),fabs(Z2b))),
                   fabs(Z2trans-D2)/scale);
    }

    /* DW: exchange q <-> -q leaves the symmetrized collapsed Z3 kernel
     * invariant. Represent the exchange by x->-x and phi->phi+pi. Also
     * verify the exact transverse limit Z3(mu=0)=D3, which checks every
     * factor of mu*k in the third-order redshift-space map. */
    {
        const double x=0.371,phi=0.713,muk=0.437;
        double Z3a=0.0,Z3b=0.0;
        class_call(qcdm_tf_Z3_collapsed(pba,pnlpt,tau,kref,qref,x,phi,muk,&Z3a,errmsg),errmsg,errmsg);
        class_call(qcdm_tf_Z3_collapsed(pba,pnlpt,tau,kref,qref,-x,phi+_PI_,muk,&Z3b,errmsg),errmsg,errmsg);
        double scale=MAX(1.e-10,MAX(fabs(Z3a),fabs(Z3b)));
        class_test(fabs(Z3a-Z3b)/scale>2.e-7,errmsg,
                   "Collapsed QCDM Z3 is not symmetric under q<->-q: Z3a=%e Z3b=%e rel=%e",
                   Z3a,Z3b,fabs(Z3a-Z3b)/scale);

        double sq=sqrt(1.0-x*x);
        qcdm_qscf_kvec3 p3[3]={{kref,0.0,0.0},
                               {qref*x,qref*sq*cos(phi),qref*sq*sin(phi)},
                               {-qref*x,-qref*sq*cos(phi),-qref*sq*sin(phi)}};
        qcdm_qscf_kernel3_result K3;
        qcdm_qscf_kernel2_result pair[3];
        class_call(qcdm_qscf_qs_solve_F3_collapsed_full(pba,tau,kref,qref,x,&K3,errmsg),errmsg,errmsg);
        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,p3[1],p3[2],&pair[0],errmsg),errmsg,errmsg);
        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,p3[0],p3[2],&pair[1],errmsg),errmsg,errmsg);
        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,p3[0],p3[1],&pair[2],errmsg),errmsg,errmsg);
        double D3=qcdm_tf_D3_collapsed(pnlpt,&K3,p3,pair);

        /* DW RENORM-P13 VALIDATOR: transverse Z3 must reduce to D3_ren.
         * The production Z3/P13 branch projects out
         *   b2_std_local = b_mm + (4/3) b_Gmm
         * so the transverse limit is D3_ren rather than D3.
         */
        double b2_std_local =
            pnlpt->qcdm_tf_b_mm + (4.0/3.0)*pnlpt->qcdm_tf_b_Gmm;
        double sum_F2m_p13 =
            pair[0].F2_m_red + pair[1].F2_m_red + pair[2].F2_m_red;
        double D3ren =
            D3 - (b2_std_local/3.0)*sum_F2m_p13;

        double Z3trans=0.0;
        class_call(qcdm_tf_Z3_collapsed(pba,pnlpt,tau,kref,qref,x,phi,0.0,&Z3trans,errmsg),errmsg,errmsg);
        double scale_trans=MAX(1.e-10,MAX(fabs(D3ren),fabs(Z3trans)));
        class_test(fabs(Z3trans-D3ren)/scale_trans>2.e-9,errmsg,
                   "Transverse collapsed QCDM Z3 does not reduce to D3_ren: Z3=%e D3ren=%e rel=%e",
                   Z3trans,D3ren,fabs(Z3trans-D3ren)/scale_trans);
        if (pnlpt->nonlinear_pt_verbose>1)
            printf("DW/QCDM-RSD audit: collapsed Z3 exchange residual=%e, transverse-renormalized residual=%e.\n",
                   fabs(Z3a-Z3b)/scale,fabs(Z3trans-D3ren)/scale_trans);
    }
    return _SUCCESS_;
}

/* DW: forward declaration for exact-time PNG templates used by the tracer
 * and RSD integrators below. */
static double qcdm_png_bphi_unit(int orthogonal,double P1,double P2,double P3);

static int qcdm_tf_exact_rsd_multipoles(struct background *pba,
                                        struct nonlinear_pt *pnlpt,
                                        double tau,
                                        int Nmax,
                                        const double *kdisc,
                                        const double *Pbin,
                                        const double *Pphi,
                                        const double *Mphi,
                                        double hratio,
                                        double Dratio,
                                        double cutoff,
                                        double *P0,double *P2,double *P4,
                                        double *P0eq,double *P2eq,double *P4eq,
                                        double *P0ortho,double *P2ortho,double *P4ortho) {
    /*
     * DW speed/physics note:
     *
     * The exact F_n/G_n kernels are rotationally invariant: for fixed
     * (k,q,x=khat.qhat) they do not depend on the azimuth phi or on the
     * line-of-sight direction.  The old implementation nevertheless entered
     * qcdm_tf_Z2/Z3 inside the (mu_LOS,phi) loops, forcing repeated cache
     * lookups (and, with AP enabled, repeated ODE solves at each observed mu).
     *
     * We now compute the exact no-AP redshift-space loop spectrum at the true
     * k-grid first.  At one loop the LOS dependence is an even polynomial
     * through mu^8, so ell=0,2,4,6,8 is a complete representation.  We then
     * apply the same AP map to this true-space spectrum and project back to
     * observed ell=0,2,4.  No kernel equation, bias operator, loop quadrature,
     * UV taper, or AP formula is altered.  Only the order of evaluation is
     * changed, with linear interpolation on the existing internal k-grid for
     * the AP remapping.
     */
    int Nq=pnlpt->qcdm_tf_Nq, Nx=pnlpt->qcdm_tf_Nx, Nphi=pnlpt->qcdm_tf_Nphi;
    int Nmu=MAX(pnlpt->qcdm_tf_Nmu,9); /* exact projection for mu^8 x L8 */
    const int nL=5;
    const int Lval[5]={0,2,4,6,8};
    double qmin=MAX(kdisc[0],pnlpt->qcdm_loop_q_min_h*pba->h);
    double qmax=MIN(kdisc[Nmax-1],pnlpt->qcdm_loop_q_max_h*pba->h);
    double kmax=pnlpt->qcdm_loop_k_max_h*pba->h;
    class_test(qmax<=qmin,pnlpt->error_message,
               "Invalid QCDM RSD loop interval after grid clipping: qmin=%e qmax=%e",qmin,qmax);

    {
        double kref=MIN(MAX(sqrt(kdisc[0]*kdisc[Nmax-1]),kdisc[1]),kdisc[Nmax-2]);
        double qref=sqrt(qmin*qmax);
        class_call(qcdm_tf_validate_rsd_conventions(pba,pnlpt,tau,kref,qref,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
    }

    /* DW convergence: use Gauss-Legendre in ln(q) and in
     * p=|k-q| rather than midpoint ln(q) x Gauss-Legendre in
     * x=khat.qhat.  The p change removes the square-root endpoint
     * p=sqrt(k^2+q^2-2kqx) from the integration coordinate.
     *
     * The q integral is split at q=k, where p_min=|k-q| has a kink.
     * Nq is the total number of GL nodes and is distributed between the
     * two logarithmic q panels.  Nx is retained as the user-facing name
     * for the inner angular order,
     * but in this branch it is now the Gauss-Legendre order in p. */
    double *pnode,*pweight,*qnode,*qweight,*qgl,*qgw,*munode,*muweight;
    qcdm_qscf_kvec3 *losnode;
    class_alloc(pnode,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(pweight,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(qnode,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qweight,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qgl,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qgw,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(munode,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(muweight,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(losnode,Nmu*sizeof(qcdm_qscf_kvec3),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nx,pnode,pweight,pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nmu,munode,muweight,pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
    for (int im=0;im<Nmu;im++)
        losnode[im]=(qcdm_qscf_kvec3){munode[im],sqrt(MAX(1.0-munode[im]*munode[im],0.0)),0.0};

    double *Lloop[5],*L22[5],*L13[5],*Leq[5],*Lortho[5];
    for (int il=0;il<nL;il++) {
        class_alloc(Lloop[il],Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(L22[il],Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(L13[il],Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(Leq[il],Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(Lortho[il],Nmax*sizeof(double),pnlpt->error_message);
        memset(Lloop[il],0,Nmax*sizeof(double));
        memset(L22[il],0,Nmax*sizeof(double));
        memset(L13[il],0,Nmax*sizeof(double));
        memset(Leq[il],0,Nmax*sizeof(double));
        memset(Lortho[il],0,Nmax*sizeof(double));
    }

    double *P22mu,*P13mu,*P12eqmu,*P12orthomu,*loopmu,*loop22mu,*loop13mu,*eqmu,*orthomu,*Z1k;
    class_alloc(P22mu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(P13mu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(P12eqmu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(P12orthomu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(loopmu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(loop22mu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(loop13mu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(eqmu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(orthomu,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(Z1k,Nmu*sizeof(double),pnlpt->error_message);

    int has_png=((pnlpt->fNL_equil_ortho_switch==fNL_equil_ortho_yes)&&(Pphi!=NULL)&&(Mphi!=NULL));
    double dphi=2.0*_PI_/((double)Nphi);
    double loopnorm=1.0/(8.0*_PI_*_PI_*_PI_);

    struct timespec ts_kernel0,ts_kernel1,ts_ap0,ts_ap1;
    clock_gettime(CLOCK_MONOTONIC,&ts_kernel0);

    /* ---- Exact true-space loop multipoles (no AP) --------------------- */
    for (int ik=0;ik<Nmax;ik++) {
        double kk=kdisc[ik];
        /* No loop kernel is requested above the user-selected PT cutoff. */
        if (kk>kmax) continue;
        if ((pnlpt->nonlinear_pt_verbose>0) && ((ik%10)==0 || ik==Nmax-1))
            printf("DW/QCDM-QSCF: RSD exact kernels k-index %d/%d, k=%.5e\n",ik+1,Nmax,kk);

        for (int im=0;im<Nmu;im++) {
            P22mu[im]=P13mu[im]=P12eqmu[im]=P12orthomu[im]=0.0;
        }

        double Pk=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,kk);
        qcdm_qscf_kernel1_result K1k;
        class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,kk,&K1k,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        for (int im=0;im<Nmu;im++) Z1k[im]=qcdm_tf_Z1(pnlpt,&K1k,munode[im]);

        double Mk=has_png?Mphi[ik]:0.0;
        double Pkphi=has_png?Pphi[ik]:0.0;

        if (kk<=kmax) {
            /* The transformed p integral has a non-analytic lower boundary
             * p_min=|k-q|.  A single high-order GL rule across the full
             * ln(q) interval therefore converges poorly because q=k is an
             * interior kink.  Make q=k an explicit panel boundary.
             *
             * Nq remains the TOTAL number of q nodes for this external k.
             * When both panels are present, distribute those nodes roughly
             * in proportion to their logarithmic lengths, while reserving
             * at least four nodes on each side for Nq>=8.  GL nodes never
             * include the panel endpoint, so q=k itself is not sampled. */
            int Nq_this=0;
            if ((kk>qmin) && (kk<qmax) && (Nq>=2)) {
                double ylo=log(qmin), ysplit=log(kk), yhi=log(qmax);
                double Lleft=ysplit-ylo, Lright=yhi-ysplit;
                double Ltot=Lleft+Lright;
                int min_side=(Nq>=8)?4:1;
                int nleft=(int)floor(((double)Nq)*Lleft/Ltot+0.5);
                if (nleft<min_side) nleft=min_side;
                if (nleft>Nq-min_side) nleft=Nq-min_side;
                int nright=Nq-nleft;

                class_call(qcdm_tf_gauss_legendre_grid(nleft,qgl,qgw,pnlpt->error_message),
                           pnlpt->error_message,pnlpt->error_message);
                {
                    double ymid=0.5*(ylo+ysplit), yhalf=0.5*(ysplit-ylo);
                    for (int j=0;j<nleft;j++) {
                        double y=ymid+yhalf*qgl[j];
                        qnode[Nq_this]=exp(y);
                        qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j];
                        Nq_this++;
                    }
                }

                class_call(qcdm_tf_gauss_legendre_grid(nright,qgl,qgw,pnlpt->error_message),
                           pnlpt->error_message,pnlpt->error_message);
                {
                    double ymid=0.5*(ysplit+yhi), yhalf=0.5*(yhi-ysplit);
                    for (int j=0;j<nright;j++) {
                        double y=ymid+yhalf*qgl[j];
                        qnode[Nq_this]=exp(y);
                        qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j];
                        Nq_this++;
                    }
                }
            }
            else {
                double ylo=log(qmin), yhi=log(qmax);
                class_call(qcdm_tf_gauss_legendre_grid(Nq,qgl,qgw,pnlpt->error_message),
                           pnlpt->error_message,pnlpt->error_message);
                {
                    double ymid=0.5*(ylo+yhi), yhalf=0.5*(yhi-ylo);
                    for (int j=0;j<Nq;j++) {
                        double y=ymid+yhalf*qgl[j];
                        qnode[Nq_this]=exp(y);
                        qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j];
                        Nq_this++;
                    }
                }
            }

            class_test(Nq_this!=Nq,pnlpt->error_message,
                       "Internal QCDM split-q quadrature error: built %d nodes, expected %d",
                       Nq_this,Nq);

            for (int iq=0;iq<Nq_this;iq++) {
                double q=qnode[iq], dq=qweight[iq];
                double Pq=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,q);
                double Pqphi=has_png?qcdm_qscf_interp_linear(Nmax,kdisc,Pphi,q):0.0;
                double Mq=has_png?qcdm_qscf_interp_linear(Nmax,kdisc,Mphi,q):0.0;

                /* q and -q are common to all x and phi at this q. */
                qcdm_qscf_kvec3 qp={q,0.0,0.0}, qm={-q,0.0,0.0};
                qcdm_qscf_kernel2_result pair0;
                class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qp,qm,&pair0,pnlpt->error_message),
                           pnlpt->error_message,pnlpt->error_message);
                qcdm_qscf_kernel1_result K1q;
                qcdm_tf_K1_from_K2_leg(&pair0,1,&K1q);

                /* Transform x=khat.qhat to p=|k-q|:
                 *   x=(k^2+q^2-p^2)/(2kq),
                 *   dx=-(p/(kq)) dp.
                 * Hence q^2 dq dx dphi -> (q p/k) dq dp dphi. */
                double pmin=fabs(kk-q);
                double pmax=kk+q;
                double pmid=0.5*(pmin+pmax);
                double phalf=0.5*(pmax-pmin);
                for (int ix=0;ix<Nx;ix++) {
                    double pnorm=pmid+phalf*pnode[ix];
                    double dp=phalf*pweight[ix];
                    double x=(kk*kk+q*q-pnorm*pnorm)/(2.0*kk*q);
                    if (x>1.0) x=1.0;
                    if (x<-1.0) x=-1.0;
                    double sq=sqrt(MAX(1.0-x*x,0.0));
                    int pvalid=((pnorm>kdisc[0])&&(pnorm<kdisc[Nmax-1]));

                    qcdm_qscf_kvec3 pbase[3];
                    pbase[0]=(qcdm_qscf_kvec3){kk,0.0,0.0};
                    pbase[1]=(qcdm_qscf_kvec3){q*x,q*sq,0.0};
                    pbase[2]=(qcdm_qscf_kvec3){-q*x,-q*sq,0.0};

                    qcdm_qscf_kernel3_result K3;
                    qcdm_qscf_kernel2_result pair[3];
                    pair[0]=pair0;
                    class_call(qcdm_qscf_qs_solve_F3_collapsed_full(pba,tau,kk,q,x,&K3,pnlpt->error_message),
                               pnlpt->error_message,pnlpt->error_message);
                    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pbase[0],pbase[2],&pair[1],pnlpt->error_message),
                               pnlpt->error_message,pnlpt->error_message);
                    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pbase[0],pbase[1],&pair[2],pnlpt->error_message),
                               pnlpt->error_message,pnlpt->error_message);
                    qcdm_qscf_kernel1_result K1coll[3]={K1k,K1q,K1q};
                    double D3=qcdm_tf_D3_collapsed(pnlpt,&K3,pbase,pair);
                    double V3=qcdm_tf_V3_collapsed(pnlpt,&K3,pbase,pair);

                    qcdm_qscf_kernel2_result K2qp;
                    qcdm_qscf_kernel1_result K1p;
                    double Pp=0.0,Ppphi=0.0,Mp=0.0;
                    if (pvalid) {
                        qcdm_qscf_kvec3 qv0={q*x,q*sq,0.0};
                        qcdm_qscf_kvec3 pv0={kk-qv0.x,-qv0.y,0.0};
                        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qv0,pv0,&K2qp,pnlpt->error_message),
                                   pnlpt->error_message,pnlpt->error_message);
                        qcdm_tf_K1_from_K2_leg(&K2qp,2,&K1p);
                        Pp=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,pnorm);
                        if (has_png) {
                            Ppphi=qcdm_qscf_interp_linear(Nmax,kdisc,Pphi,pnorm);
                            Mp=qcdm_qscf_interp_linear(Nmax,kdisc,Mphi,pnorm);
                        }
                    }

                    double measure_base=(q*pnorm/kk)*dq*dp*dphi;
                    for (int ip=0;ip<Nphi;ip++) {
                        double phi=2.0*_PI_*((((double)ip)+0.5)/((double)Nphi));
                        double cphi=cos(phi),sphi=sin(phi);
                        qcdm_qscf_kvec3 pvec[3];
                        pvec[0]=pbase[0];
                        pvec[1]=(qcdm_qscf_kvec3){q*x,q*sq*cphi,q*sq*sphi};
                        pvec[2]=(qcdm_qscf_kvec3){-pvec[1].x,-pvec[1].y,-pvec[1].z};
                        qcdm_qscf_kvec3 qv=pvec[1];
                        qcdm_qscf_kvec3 pv={kk-qv.x,-qv.y,-qv.z};

                        for (int im=0;im<Nmu;im++) {
                            if (pvalid) {
                                double Z2=qcdm_tf_Z2_from_kernels(pnlpt,qv,pv,losnode[im],&K2qp,&K1q,&K1p);
                                P22mu[im] += measure_base*Z2*Z2*Pq*Pp;
                                if (has_png) {
                                    double transfer=Mk*Mq*Mp;
                                    P12eqmu[im] += measure_base*Z2*transfer*qcdm_png_bphi_unit(0,Pkphi,Pqphi,Ppphi);
                                    P12orthomu[im] += measure_base*Z2*transfer*qcdm_png_bphi_unit(1,Pkphi,Pqphi,Ppphi);
                                }
                            }
                            double Z3=qcdm_tf_Z3_collapsed_from_kernels(pnlpt,pvec,losnode[im],munode[im],
                                                                        &K3,pair,K1coll,D3,V3);
                            P13mu[im] += measure_base*Z3*Pq;
                        }
                    }
                }
            }
        }

        double kc=kk/cutoff;
        double damp=exp(-pow(kc,6.0));
        for (int im=0;im<Nmu;im++) {
            loop22mu[im]=damp*(2.0*P22mu[im]*loopnorm);
            loop13mu[im]=damp*(6.0*Z1k[im]*Pk*P13mu[im]*loopnorm);
            loopmu[im]=loop22mu[im]+loop13mu[im];
            eqmu[im]=damp*2.0*Z1k[im]*P12eqmu[im]*loopnorm;
            orthomu[im]=damp*2.0*Z1k[im]*P12orthomu[im]*loopnorm;
        }
        for (int il=0;il<nL;il++) {
            int ell=Lval[il];
            double fac=0.5*(2.0*((double)ell)+1.0);
            double sl=0.0,s22=0.0,s13=0.0,se=0.0,so=0.0;
            for (int im=0;im<Nmu;im++) {
                double w=muweight[im]*qcdm_tf_legendre(ell,munode[im]);
                sl+=w*loopmu[im];
                s22+=w*loop22mu[im];
                s13+=w*loop13mu[im];
                se+=w*eqmu[im];
                so+=w*orthomu[im];
            }
            Lloop[il][ik]=fac*sl;
            L22[il][ik]=fac*s22;
            L13[il][ik]=fac*s13;
            Leq[il][ik]=fac*se;
            Lortho[il][ik]=fac*so;
        }
    }

    /* DW debug: expose the two one-loop RSD pieces separately.
     * These are TRUE-SPACE multipoles before AP projection. This block does
     * not modify the model prediction. */
    if (pnlpt->qcdm_write_multipoles==_TRUE_) {
        double *pvecback_dbg;
        int last_index_dbg=0;
        class_alloc(pvecback_dbg,pba->bg_size*sizeof(double),pnlpt->error_message);
        class_call(background_at_tau(pba,tau,long_info,inter_normal,
                                     &last_index_dbg,pvecback_dbg),
                   pba->error_message,pnlpt->error_message);
        double zdbg=1.0/pvecback_dbg[pba->index_bg_a]-1.0;
        free(pvecback_dbg);

        FileName dbgname;
        class_sprintf(dbgname,"%sqcdm_rsd_loop_split_z%.6f.dat",
                      pnlpt->qcdm_pt_output_root,zdbg);
        FILE *fdbg=fopen(dbgname,"w");
        class_test(fdbg==NULL,pnlpt->error_message,
                   "Could not open %s for writing",dbgname);

        fprintf(fdbg,"# exact-QS IR-off RSD loop split at z = %.12g\n",zdbg);
        fprintf(fdbg,"# true-space multipoles before AP; all P in (Mpc/h)^3\n");
        fprintf(fdbg,"# k[h/Mpc]");
        for (int il=0;il<nL;il++)
            fprintf(fdbg," P22_l%d P13_l%d Ploop_l%d",
                    Lval[il],Lval[il],Lval[il]);
        fprintf(fdbg,"\n");

        double h3dbg=pba->h*pba->h*pba->h;
        for (int ik=0;ik<Nmax;ik++) {
            fprintf(fdbg,"%.12e",kdisc[ik]/pba->h);
            for (int il=0;il<nL;il++)
                fprintf(fdbg," %.12e %.12e %.12e",
                        L22[il][ik]*h3dbg,
                        L13[il][ik]*h3dbg,
                        Lloop[il][ik]*h3dbg);
            fprintf(fdbg,"\n");
        }
        fclose(fdbg);
        if (pnlpt->nonlinear_pt_verbose>0)
            printf("DW/QCDM-QSCF: wrote RSD P22/P13 split to %s.\n",dbgname);
    }

    clock_gettime(CLOCK_MONOTONIC,&ts_kernel1);
    double t_kernel=(ts_kernel1.tv_sec-ts_kernel0.tv_sec)+1.e-9*(ts_kernel1.tv_nsec-ts_kernel0.tv_nsec);
    clock_gettime(CLOCK_MONOTONIC,&ts_ap0);

    /* Number of k-grid points carrying an explicitly integrated loop. */
    int Nloop=0;
    while (Nloop<Nmax && kdisc[Nloop]<=kmax) Nloop++;
    if (Nloop<2) Nloop=MIN(Nmax,2);

    /* ---- AP projection of cached true-space multipoles ---------------- */
    for (int ik=0;ik<Nmax;ik++) {
        double kobs=kdisc[ik];
        double ell0=0.0,ell2=0.0,ell4=0.0;
        double ell0eq=0.0,ell2eq=0.0,ell4eq=0.0;
        double ell0ortho=0.0,ell2ortho=0.0,ell4ortho=0.0;

        for (int im=0;im<Nmu;im++) {
            double muobs=munode[im],dmu=muweight[im];
            double ktrue=kobs,mutrue=muobs,jac=1.0;
            if (pnlpt->AP_effect==AP_effect_yes) {
                double invD2=1.0/(Dratio*Dratio);
                double apfac=sqrt(invD2+(hratio*hratio-invD2)*muobs*muobs);
                ktrue=kobs*apfac;
                mutrue=muobs*hratio/apfac;
                jac=hratio/(Dratio*Dratio);
            }

            /* Tree + EFT + stochastic pieces are cheap and are evaluated at
             * the exact AP-mapped (ktrue,mutrue), avoiding any endpoint issue. */
            qcdm_qscf_kernel1_result K1;
            class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,ktrue,&K1,pnlpt->error_message),
                       pnlpt->error_message,pnlpt->error_message);
            double Pcl=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,ktrue);
            double Z1=qcdm_tf_Z1(pnlpt,&K1,mutrue);
            double mu2=mutrue*mutrue,mu4=mu2*mu2;
            double Pmr=K1.F1_r_red*Pcl;
            double Pbase=Z1*Z1*Pcl
                        -(pnlpt->qcdm_tf_ct0+pnlpt->qcdm_tf_ct2*mu2+pnlpt->qcdm_tf_ct4*mu4)*ktrue*ktrue*Pcl
                        -(pnlpt->qcdm_tf_ct_r0+pnlpt->qcdm_tf_ct_r2*mu2+pnlpt->qcdm_tf_ct_r4*mu4)*ktrue*ktrue*Pmr
                        +pnlpt->qcdm_tf_stoch0+ktrue*ktrue*(pnlpt->qcdm_tf_stoch2+pnlpt->qcdm_tf_stoch2_mu2*mu2);

            double Ploop_true=0.0,Peq_true=0.0,Portho_true=0.0;
            if ((ktrue>=kdisc[0]) && (ktrue<=kmax)) {
                for (int il=0;il<nL;il++) {
                    double Leg=qcdm_tf_legendre(Lval[il],mutrue);
                    Ploop_true += qcdm_qscf_interp_linear(Nloop,kdisc,Lloop[il],ktrue)*Leg;
                    Peq_true += qcdm_qscf_interp_linear(Nloop,kdisc,Leq[il],ktrue)*Leg;
                    Portho_true += qcdm_qscf_interp_linear(Nloop,kdisc,Lortho[il],ktrue)*Leg;
                }
            }

            double Ppng_eq=Peq_true;
            double Ppng_ortho=Portho_true;
            double Ps=Pbase+Ploop_true
                     +pnlpt->qcdm_fNL_equilateral*Ppng_eq
                     +pnlpt->qcdm_fNL_orthogonal*Ppng_ortho;
            Ps*=jac; Ppng_eq*=jac; Ppng_ortho*=jac;

            ell0 += 0.5*dmu*Ps*qcdm_tf_legendre(0,muobs);
            ell2 += 2.5*dmu*Ps*qcdm_tf_legendre(2,muobs);
            ell4 += 4.5*dmu*Ps*qcdm_tf_legendre(4,muobs);
            ell0eq += 0.5*dmu*Ppng_eq*qcdm_tf_legendre(0,muobs);
            ell2eq += 2.5*dmu*Ppng_eq*qcdm_tf_legendre(2,muobs);
            ell4eq += 4.5*dmu*Ppng_eq*qcdm_tf_legendre(4,muobs);
            ell0ortho += 0.5*dmu*Ppng_ortho*qcdm_tf_legendre(0,muobs);
            ell2ortho += 2.5*dmu*Ppng_ortho*qcdm_tf_legendre(2,muobs);
            ell4ortho += 4.5*dmu*Ppng_ortho*qcdm_tf_legendre(4,muobs);
        }

        P0[ik]=ell0; P2[ik]=ell2; P4[ik]=ell4;
        P0eq[ik]=ell0eq; P2eq[ik]=ell2eq; P4eq[ik]=ell4eq;
        P0ortho[ik]=ell0ortho; P2ortho[ik]=ell2ortho; P4ortho[ik]=ell4ortho;
    }

    clock_gettime(CLOCK_MONOTONIC,&ts_ap1);
    double t_ap=(ts_ap1.tv_sec-ts_ap0.tv_sec)+1.e-9*(ts_ap1.tv_nsec-ts_ap0.tv_nsec);
    if (pnlpt->nonlinear_pt_verbose>0)
        printf("DW/QCDM-QSCF: exact RSD timing: true-space kernels %.3fs, AP projection %.3fs (Nmu=%d, ell<=8).\n",
               t_kernel,t_ap,Nmu);

    for (int il=0;il<nL;il++) {
        free(Lloop[il]); free(L22[il]); free(L13[il]); free(Leq[il]); free(Lortho[il]);
    }
    free(P22mu);free(P13mu);free(P12eqmu);free(P12orthomu);
    free(loopmu);free(loop22mu);free(loop13mu);free(eqmu);free(orthomu);free(Z1k);
    free(pnode);free(pweight);free(qnode);free(qweight);free(qgl);free(qgw);
    free(munode);free(muweight);free(losnode);
    return _SUCCESS_;
}

/* Proper anisotropic common-mode IR resummation for the exact-QS RSD path.
 *
 * The expensive undamped one-loop spectrum is decomposed into a smooth part
 * and a term with exactly one BAO-wiggle insertion.  Both remain finite even
 * polynomials in mu through mu^8, so they are cached as ell=0,2,4,6,8 true-
 * space multipoles.  Only the cheap final AP projection applies the
 * non-polynomial exp[-k^2 Sigma_s^2(mu)] factor, using 40-point GL quadrature.
 *
 * At NLO:
 *   P_tree^IR = Z1^2 [P_nw + e^-S (1+S) P_w]
 *   P_loop^IR = P_loop,nw + e^-S P_loop,w^(1 wiggle)
 *   P_ct^IR   uses P_LO = P_nw + e^-S P_w.
 *
 * Relative density/velocity operators remain inside Z_n perturbatively.  The
 * exponent resums only the common cosmological displacement; the size of the
 * relative displacement is audited by qcdm_tf_compute_ir_state().
 */
/* DW speed stage 2: one external-k slice of the exact anisotropic-IR RSD
 * contraction.  All quadrature/loop scratch is private to the caller, and
 * only Lloop_[nw,w][ell][ik] is written.  Kernel-cache misses are allowed in
 * qcdm_qscf_cache_parallel_nostore mode, where they are solved but not stored. */
static int qcdm_tf_exact_rsd_ir_one_k_parallel(
    struct background *pba, struct nonlinear_pt *pnlpt, double tau,
    int Nmax, const double *kdisc, const double *Pnw, const double *Pw,
    int Nq, int Nx, int Nphi, int Nmu_loop, int nL, const int *Lval,
    double qmin, double qmax, double kmax, double cutoff,
    const double *pnode, const double *pweight,
    const double *munode, const double *muweight,
    const qcdm_qscf_kvec3 *losnode, double dphi, double loopnorm,
    int ik, double **Lloop_nw, double **Lloop_w, ErrorMsg errmsg) {

    double kk=kdisc[ik];
    if (kk>kmax) return _SUCCESS_;

    if ((pnlpt->nonlinear_pt_verbose>0) && ((ik%10)==0 || ik==Nmax-1)) {
        #pragma omp critical(qcdm_rsd_progress)
        printf("DW/QCDM-IR: RSD exact kernels k-index %d/%d, k=%.5e [thread %d/%d]\n",
               ik+1,Nmax,kk,omp_get_thread_num(),omp_get_num_threads());
    }

    double qnode[Nq],qweight[Nq],qgl[Nq],qgw[Nq];
    double P22nwmu[Nmu_loop],P22wmu[Nmu_loop],I13nwmu[Nmu_loop],I13wmu[Nmu_loop];
    double loopnwmu[Nmu_loop],loopwmu[Nmu_loop],Z1k[Nmu_loop];
    for (int im=0;im<Nmu_loop;im++)
        P22nwmu[im]=P22wmu[im]=I13nwmu[im]=I13wmu[im]=0.0;

    double Pknw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,kk);
    double Pkw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,kk);
    qcdm_qscf_kernel1_result K1k;
    class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,kk,&K1k,errmsg),errmsg,errmsg);
    for (int im=0;im<Nmu_loop;im++) Z1k[im]=qcdm_tf_Z1(pnlpt,&K1k,munode[im]);

    int Nq_this=0;
    if ((kk>qmin)&&(kk<qmax)&&(Nq>=2)) {
        double ylo=log(qmin),ysplit=log(kk),yhi=log(qmax);
        double Lleft=ysplit-ylo,Lright=yhi-ysplit,Ltot=Lleft+Lright;
        int min_side=(Nq>=8)?4:1;
        int nleft=(int)floor(((double)Nq)*Lleft/Ltot+0.5);
        if (nleft<min_side) nleft=min_side;
        if (nleft>Nq-min_side) nleft=Nq-min_side;
        int nright=Nq-nleft;
        class_call(qcdm_tf_gauss_legendre_grid(nleft,qgl,qgw,errmsg),errmsg,errmsg);
        double ymid=0.5*(ylo+ysplit),yhalf=0.5*(ysplit-ylo);
        for (int j=0;j<nleft;j++) {
            double y=ymid+yhalf*qgl[j];
            qnode[Nq_this]=exp(y); qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j]; Nq_this++;
        }
        class_call(qcdm_tf_gauss_legendre_grid(nright,qgl,qgw,errmsg),errmsg,errmsg);
        ymid=0.5*(ysplit+yhi); yhalf=0.5*(yhi-ysplit);
        for (int j=0;j<nright;j++) {
            double y=ymid+yhalf*qgl[j];
            qnode[Nq_this]=exp(y); qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j]; Nq_this++;
        }
    } else {
        double ylo=log(qmin),yhi=log(qmax);
        class_call(qcdm_tf_gauss_legendre_grid(Nq,qgl,qgw,errmsg),errmsg,errmsg);
        double ymid=0.5*(ylo+yhi),yhalf=0.5*(yhi-ylo);
        for (int j=0;j<Nq;j++) {
            double y=ymid+yhalf*qgl[j];
            qnode[Nq_this]=exp(y); qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j]; Nq_this++;
        }
    }
    class_test(Nq_this!=Nq,errmsg,
               "Internal QCDM parallel IR-RSD split-q quadrature error: built %d nodes, expected %d",Nq_this,Nq);

    for (int iq=0;iq<Nq_this;iq++) {
        double q=qnode[iq],dq=qweight[iq];
        double Pqnw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,q);
        double Pqw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,q);
        qcdm_qscf_kvec3 qp={q,0.0,0.0},qm={-q,0.0,0.0};
        qcdm_qscf_kernel2_result pair0;
        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qp,qm,&pair0,errmsg),errmsg,errmsg);
        qcdm_qscf_kernel1_result K1q; qcdm_tf_K1_from_K2_leg(&pair0,1,&K1q);

        double pmin=fabs(kk-q),pmax=kk+q,pmid=0.5*(pmin+pmax),phalf=0.5*(pmax-pmin);
        for (int ix=0;ix<Nx;ix++) {
            double pnorm=pmid+phalf*pnode[ix],dp=phalf*pweight[ix];
            double x=(kk*kk+q*q-pnorm*pnorm)/(2.0*kk*q);
            if (x>1.0) x=1.0; if (x<-1.0) x=-1.0;
            double sq=sqrt(MAX(1.0-x*x,0.0));
            int pvalid=((pnorm>kdisc[0])&&(pnorm<kdisc[Nmax-1]));
            qcdm_qscf_kvec3 pbase[3];
            pbase[0]=(qcdm_qscf_kvec3){kk,0.0,0.0};
            pbase[1]=(qcdm_qscf_kvec3){q*x,q*sq,0.0};
            pbase[2]=(qcdm_qscf_kvec3){-q*x,-q*sq,0.0};
            qcdm_qscf_kernel3_result K3; qcdm_qscf_kernel2_result pair[3]; pair[0]=pair0;
            class_call(qcdm_qscf_qs_solve_F3_collapsed_full(pba,tau,kk,q,x,&K3,errmsg),errmsg,errmsg);
            class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pbase[0],pbase[2],&pair[1],errmsg),errmsg,errmsg);
            class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pbase[0],pbase[1],&pair[2],errmsg),errmsg,errmsg);
            qcdm_qscf_kernel1_result K1coll[3]={K1k,K1q,K1q};
            double D3=qcdm_tf_D3_collapsed(pnlpt,&K3,pbase,pair);
            double V3=qcdm_tf_V3_collapsed(pnlpt,&K3,pbase,pair);
            qcdm_qscf_kernel2_result K2qp; qcdm_qscf_kernel1_result K1p;
            double Ppnw=0.0,Ppw=0.0;
            if (pvalid) {
                qcdm_qscf_kvec3 qv0={q*x,q*sq,0.0},pv0={kk-qv0.x,-qv0.y,0.0};
                class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qv0,pv0,&K2qp,errmsg),errmsg,errmsg);
                qcdm_tf_K1_from_K2_leg(&K2qp,2,&K1p);
                Ppnw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,pnorm);
                Ppw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,pnorm);
            }
            double measure_base=(q*pnorm/kk)*dq*dp*dphi;
            for (int ip=0;ip<Nphi;ip++) {
                double phi=2.0*_PI_*((((double)ip)+0.5)/((double)Nphi));
                double cphi=cos(phi),sphi=sin(phi);
                qcdm_qscf_kvec3 pvec[3]; pvec[0]=pbase[0];
                pvec[1]=(qcdm_qscf_kvec3){q*x,q*sq*cphi,q*sq*sphi};
                pvec[2]=(qcdm_qscf_kvec3){-pvec[1].x,-pvec[1].y,-pvec[1].z};
                qcdm_qscf_kvec3 qv=pvec[1],pv={kk-qv.x,-qv.y,-qv.z};
                for (int im=0;im<Nmu_loop;im++) {
                    if (pvalid) {
                        double Z2=qcdm_tf_Z2_from_kernels(pnlpt,qv,pv,losnode[im],&K2qp,&K1q,&K1p);
                        double Z22=Z2*Z2;
                        P22nwmu[im]+=measure_base*Z22*Pqnw*Ppnw;
                        P22wmu[im] +=measure_base*Z22*(Pqw*Ppnw+Pqnw*Ppw);
                    }
                    double Z3=qcdm_tf_Z3_collapsed_from_kernels(pnlpt,pvec,losnode[im],munode[im],
                                                                  &K3,pair,K1coll,D3,V3);
                    I13nwmu[im]+=measure_base*Z3*Pqnw;
                    I13wmu[im] +=measure_base*Z3*Pqw;
                }
            }
        }
    }

    double kc=kk/cutoff,uv=exp(-pow(kc,6.0));
    for (int im=0;im<Nmu_loop;im++) {
        loopnwmu[im]=uv*(2.0*P22nwmu[im]*loopnorm+6.0*Z1k[im]*Pknw*I13nwmu[im]*loopnorm);
        loopwmu[im] =uv*(2.0*P22wmu[im]*loopnorm+6.0*Z1k[im]*(Pkw*I13nwmu[im]+Pknw*I13wmu[im])*loopnorm);
    }
    for (int il=0;il<nL;il++) {
        int ell=Lval[il]; double fac=0.5*(2.0*((double)ell)+1.0),snw=0.0,sw=0.0;
        for (int im=0;im<Nmu_loop;im++) {
            double wt=muweight[im]*qcdm_tf_legendre(ell,munode[im]);
            snw+=wt*loopnwmu[im]; sw+=wt*loopwmu[im];
        }
        Lloop_nw[il][ik]=fac*snw; Lloop_w[il][ik]=fac*sw;
    }
    return _SUCCESS_;
}

static int qcdm_tf_exact_rsd_multipoles_ir(struct background *pba,
                                           struct nonlinear_pt *pnlpt,
                                           double tau,
                                           int Nmax,
                                           const double *kdisc,
                                           const double *Pnw,
                                           const double *Pw,
                                           const qcdm_tf_ir_state *ir_state,
                                           double hratio,
                                           double Dratio,
                                           double cutoff,
                                           double *P0,double *P2,double *P4,
                                           double *P0eq,double *P2eq,double *P4eq,
                                           double *P0ortho,double *P2ortho,double *P4ortho) {
    class_test((ir_state==NULL)||(ir_state->enabled!=_TRUE_),pnlpt->error_message,
               "QCDM anisotropic IR RSD requested without an initialized IR state");
    class_test(fabs(pnlpt->qcdm_fNL_equilateral)>0.0 ||
               fabs(pnlpt->qcdm_fNL_orthogonal)>0.0,
               pnlpt->error_message,
               "QCDM exact anisotropic IR resummation is currently implemented for Gaussian initial conditions only; set fNL_equilateral=fNL_orthogonal=0. The c-c' DESI baseline already does this.");

    int Nq=pnlpt->qcdm_tf_Nq;
    int Nx=pnlpt->qcdm_tf_Nx;
    int Nphi=pnlpt->qcdm_tf_Nphi;
    int Nmu_loop=MAX(pnlpt->qcdm_tf_Nmu,9);
    const int Nmu_final=40;
    const int nL=5;
    const int Lval[5]={0,2,4,6,8};

    double qmin=MAX(kdisc[0],pnlpt->qcdm_loop_q_min_h*pba->h);
    double qmax=MIN(kdisc[Nmax-1],pnlpt->qcdm_loop_q_max_h*pba->h);
    double kmax=pnlpt->qcdm_loop_k_max_h*pba->h;
    class_test(qmax<=qmin,pnlpt->error_message,
               "Invalid QCDM IR-RSD loop interval after grid clipping: qmin=%e qmax=%e",qmin,qmax);

    {
        double kref=MIN(MAX(sqrt(kdisc[0]*kdisc[Nmax-1]),kdisc[1]),kdisc[Nmax-2]);
        double qref=sqrt(qmin*qmax);
        class_call(qcdm_tf_validate_rsd_conventions(pba,pnlpt,tau,kref,qref,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
    }

    double *pnode,*pweight,*qnode,*qweight,*qgl,*qgw;
    double *munode,*muweight,*mu_final,*w_final;
    qcdm_qscf_kvec3 *losnode;
    class_alloc(pnode,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(pweight,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(qnode,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qweight,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qgl,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qgw,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(munode,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(muweight,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(losnode,Nmu_loop*sizeof(qcdm_qscf_kvec3),pnlpt->error_message);
    class_alloc(mu_final,Nmu_final*sizeof(double),pnlpt->error_message);
    class_alloc(w_final,Nmu_final*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nx,pnode,pweight,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nmu_loop,munode,muweight,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nmu_final,mu_final,w_final,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    for (int im=0;im<Nmu_loop;im++)
        losnode[im]=(qcdm_qscf_kvec3){munode[im],sqrt(MAX(1.0-munode[im]*munode[im],0.0)),0.0};

    double *Lloop_nw[5],*Lloop_w[5];
    for (int il=0;il<nL;il++) {
        class_alloc(Lloop_nw[il],Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(Lloop_w[il],Nmax*sizeof(double),pnlpt->error_message);
        memset(Lloop_nw[il],0,Nmax*sizeof(double));
        memset(Lloop_w[il],0,Nmax*sizeof(double));
    }

    double *P22nwmu,*P22wmu,*I13nwmu,*I13wmu,*loopnwmu,*loopwmu,*Z1k;
    class_alloc(P22nwmu,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(P22wmu,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(I13nwmu,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(I13wmu,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(loopnwmu,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(loopwmu,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(Z1k,Nmu_loop*sizeof(double),pnlpt->error_message);

    const double dphi=2.0*_PI_/((double)Nphi);
    const double loopnorm=1.0/(8.0*_PI_*_PI_*_PI_);

    struct timespec ts_kernel0,ts_kernel1,ts_ap0,ts_ap1;
    clock_gettime(CLOCK_MONOTONIC,&ts_kernel0);
/* TEMPORARY DIAGNOSTIC: insert inside qcdm_tf_exact_rsd_multipoles_ir()
 * immediately after clock_gettime(CLOCK_MONOTONIC,&ts_kernel0);
 * Uses the existing statically-defined P+B kernel functions.
 * Run with OMP_NUM_THREADS=1 QCDM_KERNEL_AUDIT=1 (and IR resummation = yes).
 * Caution: in the absence of the environment variable this block does nothing.
 * Delete the temporary block after testing; keep the production binary unchanged.
 */
    if (getenv("QCDM_KERNEL_AUDIT") != NULL) {
        qcdm_qscf_kernel1_result r1;
        qcdm_qscf_kernel2_result r2;
        qcdm_qscf_kernel3_result r3;
        qcdm_qscf_qs_coefficients qc_audit;
        const double k1_audit=0.10*pba->h;  /* reference k1=0.10 h/Mpc */
        const double k2_audit=0.07*pba->h;  /* reference k2=0.07 h/Mpc */
        const double mu_audit=0.30;
        qcdm_qscf_cache_readonly_parallel=_FALSE_;
        qcdm_qscf_cache_parallel_nostore=_FALSE_;
        class_call(qcdm_qscf_qs_get_coefficients(pba,tau,&qc_audit,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,k1_audit,&r1,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        class_call(qcdm_qscf_qs_solve_F2_full(pba,tau,k1_audit,k2_audit,mu_audit,&r2,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        class_call(qcdm_qscf_qs_solve_F3_collapsed_full(pba,tau,k1_audit,k2_audit,mu_audit,&r3,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        FILE *audit_fp=fopen("kernel_audit.dat","a");
        class_test(audit_fp==NULL,pnlpt->error_message,"Cannot open kernel_audit.dat");
        fprintf(audit_fp,"# z F1m G1m_red F2m_red G2m_red F3m_red G3m_red F1r_red G1r_red F2r_red G2r_red F3r_red G3r_red\n");
        fprintf(audit_fp,
                "%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n",
                1.0/qc_audit.a-1.0,
                r1.F1_m,r1.G1_m_red,
                r2.F2_m_red,r2.G2_m_red,
                r3.F3_m_red,r3.G3_m_red,
                r1.F1_r_red,r1.G1_r_red,
                r2.F2_r_red,r2.G2_r_red,
                r3.F3_r_red,r3.G3_r_red);
        fclose(audit_fp);
    }


    /* ---- Undamped true-space smooth and one-wiggle loops ------------ */
    int rsd_use_nostore=(omp_get_max_threads()>1)?_TRUE_:_FALSE_;
    /* Prewarm all possible RK background-stage grids before threads enter.
     * Once warm, qcdm_qscf_qs_get_rk_stage_cache() is read-only.  With one
     * OpenMP thread we deliberately keep the old cache-filling semantics so
     * OMP=1 remains a strict numerical baseline. */
    if (rsd_use_nostore==_TRUE_) {
        class_call(qcdm_qscf_qs_prewarm_rk_for_tau(pba,tau,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
    }

    int rsd_omp_failed=0;
    char rsd_omp_error[4096]; rsd_omp_error[0]='\0';
    qcdm_qscf_cache_parallel_nostore=rsd_use_nostore;
    #pragma omp parallel for schedule(dynamic,1) if(rsd_use_nostore)
    for (int ik=0;ik<Nmax;ik++) {
        char local_errmsg[2048]; local_errmsg[0]='\0';
        int st=qcdm_tf_exact_rsd_ir_one_k_parallel(
            pba,pnlpt,tau,Nmax,kdisc,Pnw,Pw,Nq,Nx,Nphi,Nmu_loop,nL,Lval,
            qmin,qmax,kmax,cutoff,pnode,pweight,munode,muweight,losnode,
            dphi,loopnorm,ik,Lloop_nw,Lloop_w,local_errmsg);
        if (st!=_SUCCESS_) {
            #pragma omp critical(qcdm_rsd_error)
            {
                if (rsd_omp_failed==0) {
                    rsd_omp_failed=1;
                    snprintf(rsd_omp_error,sizeof(rsd_omp_error),
                             "initialization RSD OpenMP k-index %d failed: %s",ik,local_errmsg);
                }
            }
        }
    }
    qcdm_qscf_cache_parallel_nostore=_FALSE_;
    class_test(rsd_omp_failed!=0,pnlpt->error_message,"%s",rsd_omp_error);

    clock_gettime(CLOCK_MONOTONIC,&ts_kernel1);
    double t_kernel=(ts_kernel1.tv_sec-ts_kernel0.tv_sec)+1.e-9*(ts_kernel1.tv_nsec-ts_kernel0.tv_nsec);
    clock_gettime(CLOCK_MONOTONIC,&ts_ap0);

    int Nloop=0;
    while (Nloop<Nmax && kdisc[Nloop]<=kmax) Nloop++;
    if (Nloop<2) Nloop=MIN(Nmax,2);

    /* ---- Apply anisotropic damping at true (k,mu), then AP-project ---- */
    for (int ik=0;ik<Nmax;ik++) {
        double kobs=kdisc[ik];
        double ell0=0.0,ell2=0.0,ell4=0.0;

        for (int im=0;im<Nmu_final;im++) {
            double muobs=mu_final[im],dmu=w_final[im];
            double ktrue=kobs,mutrue=muobs,jac=1.0;
            if (pnlpt->AP_effect==AP_effect_yes) {
                double invD2=1.0/(Dratio*Dratio);
                double apfac=sqrt(invD2+(hratio*hratio-invD2)*muobs*muobs);
                ktrue=kobs*apfac;
                mutrue=muobs*hratio/apfac;
                jac=hratio/(Dratio*Dratio);
            }

            qcdm_qscf_kernel1_result K1;
            class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,ktrue,&K1,pnlpt->error_message),
                       pnlpt->error_message,pnlpt->error_message);
            double Pnwt=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,ktrue);
            double Pwt =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,ktrue);
            double S=qcdm_tf_ir_exponent(ir_state,ktrue,mutrue);
            class_test((qcdm_qscf_finite_double(S)==0)||(S<0.0),pnlpt->error_message,
                       "Invalid QCDM IR exponent at k=%e mu=%e: S=%e",ktrue,mutrue,S);
            double Dir=exp(-S);
            double Plo=Pnwt+Dir*Pwt;
            double Ptree_ir=Pnwt+Dir*(1.0+S)*Pwt;

            double Z1=qcdm_tf_Z1(pnlpt,&K1,mutrue);
            double mu2=mutrue*mutrue,mu4=mu2*mu2;
            double Pmr=K1.F1_r_red*Plo;
            double Pbase=Z1*Z1*Ptree_ir
                        -(pnlpt->qcdm_tf_ct0+pnlpt->qcdm_tf_ct2*mu2+pnlpt->qcdm_tf_ct4*mu4)*ktrue*ktrue*Plo
                        -(pnlpt->qcdm_tf_ct_r0+pnlpt->qcdm_tf_ct_r2*mu2+pnlpt->qcdm_tf_ct_r4*mu4)*ktrue*ktrue*Pmr
                        +pnlpt->qcdm_tf_stoch0+ktrue*ktrue*(pnlpt->qcdm_tf_stoch2+pnlpt->qcdm_tf_stoch2_mu2*mu2);

            double Ploop_nw_true=0.0,Ploop_w_true=0.0;
            if ((ktrue>=kdisc[0])&&(ktrue<=kmax)) {
                for (int il=0;il<nL;il++) {
                    double Leg=qcdm_tf_legendre(Lval[il],mutrue);
                    Ploop_nw_true += qcdm_qscf_interp_linear(Nloop,kdisc,Lloop_nw[il],ktrue)*Leg;
                    Ploop_w_true  += qcdm_qscf_interp_linear(Nloop,kdisc,Lloop_w[il],ktrue)*Leg;
                }
            }

            double Ps=jac*(Pbase+Ploop_nw_true+Dir*Ploop_w_true);
            ell0 += 0.5*dmu*Ps*qcdm_tf_legendre(0,muobs);
            ell2 += 2.5*dmu*Ps*qcdm_tf_legendre(2,muobs);
            ell4 += 4.5*dmu*Ps*qcdm_tf_legendre(4,muobs);
        }

        P0[ik]=ell0; P2[ik]=ell2; P4[ik]=ell4;
        P0eq[ik]=P2eq[ik]=P4eq[ik]=0.0;
        P0ortho[ik]=P2ortho[ik]=P4ortho[ik]=0.0;
    }

    clock_gettime(CLOCK_MONOTONIC,&ts_ap1);
    double t_ap=(ts_ap1.tv_sec-ts_ap0.tv_sec)+1.e-9*(ts_ap1.tv_nsec-ts_ap0.tv_nsec);
    if (pnlpt->nonlinear_pt_verbose>0)
        printf("DW/QCDM-IR: exact RSD timing: smooth+1wiggle kernels %.3fs, anisotropic IR+AP %.3fs (Nmu_loop=%d, Nmu_IR/AP=%d, ell<=8).\n",
               t_kernel,t_ap,Nmu_loop,Nmu_final);

    for (int il=0;il<nL;il++) {free(Lloop_nw[il]);free(Lloop_w[il]);}
    free(P22nwmu);free(P22wmu);free(I13nwmu);free(I13wmu);free(loopnwmu);free(loopwmu);free(Z1k);
    free(pnode);free(pweight);free(qnode);free(qweight);free(qgl);free(qgw);
    free(munode);free(muweight);free(losnode);free(mu_final);free(w_final);
    return _SUCCESS_;
}



/* ===================================================================== */
/* DW: native exact quadratic nuisance basis for the DESI M0 likelihood */
/* ===================================================================== */

/* Coefficient order returned by the basis API:
 *   0: 1
 *   1: b_m
 *   2: b_mm
 *   3: b_Gmm
 *   4: b_m^2
 *   5: b_mm^2
 *   6: b_Gmm^2
 *   7: b_m b_mm
 *   8: b_m b_Gmm
 *   9: b_mm b_Gmm
 *  10: ct0
 *  11: ct2
 *  12: ct4
 *  13: stoch0
 *  14: stoch2
 *  15: stoch2_mu2
 *  16: b_Gamma3
 *  17: b_m b_Gamma3
 *
 * Columns 0--15 retain their validated legacy definitions.  The final
 * two columns are exact P13 responses and introduce no b_Gamma3^2,
 * b_mm*b_Gamma3, or b_Gmm*b_Gamma3 terms.
 *
 * All other tracer coefficients are held fixed at the values supplied to
 * nonlinear_pt_qcdm_fast_pkpole_basis().  The first ten coefficients are
 * accumulated directly inside the exact q,p,phi,mu integration.  This is
 * algebraically equivalent to reconstructing the same polynomial from 10+
 * complete spectra, but the expensive cosmology-dependent kernel contraction
 * is performed only once.
 */

static void qcdm_tf_basis_make_views(const struct nonlinear_pt *src,
                                     struct nonlinear_pt view[5]) {
    for (int j=0;j<5;j++) {
        view[j]=*src; /* shallow copy: all cached cosmology pointers are read-only */
        view[j].qcdm_tf_b_m=0.0;
        view[j].qcdm_tf_b_mm=0.0;
        view[j].qcdm_tf_b_Gmm=0.0;
        view[j].qcdm_tf_b_Gamma3=0.0;
        view[j].qcdm_tf_ct0=0.0;
        view[j].qcdm_tf_ct2=0.0;
        view[j].qcdm_tf_ct4=0.0;
        view[j].qcdm_tf_stoch0=0.0;
        view[j].qcdm_tf_stoch2=0.0;
        view[j].qcdm_tf_stoch2_mu2=0.0;
        view[j].qcdm_fNL_equilateral=0.0;
        view[j].qcdm_fNL_orthogonal=0.0;
    }
    /* Legacy views 0--3 are unchanged apart from explicitly fixing Gamma3=0. */
    view[1].qcdm_tf_b_m=1.0;
    view[2].qcdm_tf_b_mm=1.0;
    view[3].qcdm_tf_b_Gmm=1.0;
    /* Dedicated cubic-bias response. Gamma3 enters Z3/D3 only. */
    view[4].qcdm_tf_b_Gamma3=1.0;
}

/* Convert values at the four bias views into affine coefficients
 * a(x)=a0+a1*b_m+a2*b_mm+a3*b_Gmm. */
static void qcdm_tf_basis_affine_from_views(const double v[4],double a[4]) {
    a[0]=v[0];
    a[1]=v[1]-v[0];
    a[2]=v[2]-v[0];
    a[3]=v[3]-v[0];
}

/* Coefficients of a(x)*b(x) in the ten-component quadratic ordering above. */
static void qcdm_tf_basis_affine_product(const double a[4],const double b[4],double c[10]) {
    c[0]=a[0]*b[0];
    c[1]=a[1]*b[0]+a[0]*b[1];
    c[2]=a[2]*b[0]+a[0]*b[2];
    c[3]=a[3]*b[0]+a[0]*b[3];
    c[4]=a[1]*b[1];
    c[5]=a[2]*b[2];
    c[6]=a[3]*b[3];
    c[7]=a[1]*b[2]+a[2]*b[1];
    c[8]=a[1]*b[3]+a[3]*b[1];
    c[9]=a[2]*b[3]+a[3]*b[2];
}

static void qcdm_tf_basis_affine_square(const double a[4],double c[10]) {
    qcdm_tf_basis_affine_product(a,a,c);
}

/* DW: one external-k slice of the native nuisance-basis loop.
 *
 * This helper is called from an OpenMP loop only after nonlinear_pt_init() has
 * filled the exact kernel cache.  All scratch arrays are stack-local, the
 * cached kernels are read-only, and this routine writes only the caller's
 * unique ik slice of Lloop_nw/Lloop_w.  Hence different k values are fully
 * independent. */
static int qcdm_tf_exact_rsd_basis_ir_one_k_cached(
    struct background *pba,
    struct nonlinear_pt view[5],
    double tau,
    int Nmax,
    const double *kdisc,
    const double *Pnw,
    const double *Pw,
    int Nq, int Nx, int Nphi, int Nmu_loop,
    int nL, const int *Lval,
    double qmin, double qmax, double kmax, double cutoff,
    const double *pnode, const double *pweight,
    const double *munode, const double *muweight,
    const qcdm_qscf_kvec3 *losnode,
    double dphi, double loopnorm,
    int ik,
    double *Lloop_nw, double *Lloop_w,
    double *Lgamma_nw, double *Lgamma_w,
    ErrorMsg errmsg) {

    const int NB=10;
    double kk=kdisc[ik];
    if (kk>kmax) return _SUCCESS_;

    if ((view[0].nonlinear_pt_verbose>0) && ((ik%10)==0 || ik==Nmax-1)) {
        #pragma omp critical(qcdm_basis_progress)
        printf("DW/QCDM-IR-BASIS: exact kernels k-index %d/%d, k=%.5e [thread %d/%d]\n",
               ik+1,Nmax,kk,omp_get_thread_num(),omp_get_num_threads());
    }

    /* Runtime sizes are small in production (Nq=64, Nmu_loop>=9), so VLAs
     * give each thread private scratch without heap allocation in the hot loop. */
    double qnode[Nq],qweight[Nq],qgl[Nq],qgw[Nq];
    double P22nw[NB*Nmu_loop],P22w[NB*Nmu_loop];
    double I13nw[4*Nmu_loop],I13w[4*Nmu_loop],Z1aff[4*Nmu_loop];
    double I13gamma_nw[Nmu_loop],I13gamma_w[Nmu_loop];
    memset(P22nw,0,sizeof(P22nw));
    memset(P22w ,0,sizeof(P22w));
    memset(I13nw,0,sizeof(I13nw));
    memset(I13w ,0,sizeof(I13w));
    memset(I13gamma_nw,0,sizeof(I13gamma_nw));
    memset(I13gamma_w ,0,sizeof(I13gamma_w));

    double Pknw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,kk);
    double Pkw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,kk);
    qcdm_qscf_kernel1_result K1k;
    class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,kk,&K1k,errmsg),errmsg,errmsg);
    for (int im=0;im<Nmu_loop;im++) {
        double zv[4],za[4];
        for (int j=0;j<4;j++) zv[j]=qcdm_tf_Z1(&view[j],&K1k,munode[im]);
        qcdm_tf_basis_affine_from_views(zv,za);
        for (int a=0;a<4;a++) Z1aff[(size_t)a*Nmu_loop+im]=za[a];
    }

    int Nq_this=0;
    if ((kk>qmin)&&(kk<qmax)&&(Nq>=2)) {
        double ylo=log(qmin),ysplit=log(kk),yhi=log(qmax);
        double Lleft=ysplit-ylo,Lright=yhi-ysplit,Ltot=Lleft+Lright;
        int min_side=(Nq>=8)?4:1;
        int nleft=(int)floor(((double)Nq)*Lleft/Ltot+0.5);
        if (nleft<min_side) nleft=min_side;
        if (nleft>Nq-min_side) nleft=Nq-min_side;
        int nright=Nq-nleft;

        class_call(qcdm_tf_gauss_legendre_grid(nleft,qgl,qgw,errmsg),errmsg,errmsg);
        double ymid=0.5*(ylo+ysplit),yhalf=0.5*(ysplit-ylo);
        for (int j=0;j<nleft;j++) {
            double y=ymid+yhalf*qgl[j];
            qnode[Nq_this]=exp(y);
            qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j];
            Nq_this++;
        }

        class_call(qcdm_tf_gauss_legendre_grid(nright,qgl,qgw,errmsg),errmsg,errmsg);
        ymid=0.5*(ysplit+yhi); yhalf=0.5*(yhi-ysplit);
        for (int j=0;j<nright;j++) {
            double y=ymid+yhalf*qgl[j];
            qnode[Nq_this]=exp(y);
            qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j];
            Nq_this++;
        }
    }
    else {
        double ylo=log(qmin),yhi=log(qmax);
        class_call(qcdm_tf_gauss_legendre_grid(Nq,qgl,qgw,errmsg),errmsg,errmsg);
        double ymid=0.5*(ylo+yhi),yhalf=0.5*(yhi-ylo);
        for (int j=0;j<Nq;j++) {
            double y=ymid+yhalf*qgl[j];
            qnode[Nq_this]=exp(y);
            qweight[Nq_this]=qnode[Nq_this]*yhalf*qgw[j];
            Nq_this++;
        }
    }
    class_test(Nq_this!=Nq,errmsg,
               "Internal QCDM parallel IR-basis split-q quadrature error: built %d nodes, expected %d",
               Nq_this,Nq);

    for (int iq=0;iq<Nq_this;iq++) {
        double q=qnode[iq],dq=qweight[iq];
        double Pqnw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,q);
        double Pqw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,q);

        qcdm_qscf_kvec3 qp={q,0.0,0.0},qm={-q,0.0,0.0};
        qcdm_qscf_kernel2_result pair0;
        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qp,qm,&pair0,errmsg),errmsg,errmsg);
        qcdm_qscf_kernel1_result K1q;
        qcdm_tf_K1_from_K2_leg(&pair0,1,&K1q);

        double pmin=fabs(kk-q),pmax=kk+q;
        double pmid=0.5*(pmin+pmax),phalf=0.5*(pmax-pmin);
        for (int ix=0;ix<Nx;ix++) {
            double pnorm=pmid+phalf*pnode[ix];
            double dp=phalf*pweight[ix];
            double x=(kk*kk+q*q-pnorm*pnorm)/(2.0*kk*q);
            if (x>1.0) x=1.0;
            if (x<-1.0) x=-1.0;
            double sq=sqrt(MAX(1.0-x*x,0.0));
            int pvalid=((pnorm>kdisc[0])&&(pnorm<kdisc[Nmax-1]));

            qcdm_qscf_kvec3 pbase[3];
            pbase[0]=(qcdm_qscf_kvec3){kk,0.0,0.0};
            pbase[1]=(qcdm_qscf_kvec3){q*x,q*sq,0.0};
            pbase[2]=(qcdm_qscf_kvec3){-q*x,-q*sq,0.0};

            qcdm_qscf_kernel3_result K3;
            qcdm_qscf_kernel2_result pair[3];
            pair[0]=pair0;
            class_call(qcdm_qscf_qs_solve_F3_collapsed_full(pba,tau,kk,q,x,&K3,errmsg),errmsg,errmsg);
            class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pbase[0],pbase[2],&pair[1],errmsg),errmsg,errmsg);
            class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pbase[0],pbase[1],&pair[2],errmsg),errmsg,errmsg);
            qcdm_qscf_kernel1_result K1coll[3]={K1k,K1q,K1q};

            double V3=qcdm_tf_V3_collapsed(&view[0],&K3,pbase,pair);
            double D3v[5];
            for (int j=0;j<5;j++) D3v[j]=qcdm_tf_D3_collapsed(&view[j],&K3,pbase,pair);

            qcdm_qscf_kernel2_result K2qp;
            qcdm_qscf_kernel1_result K1p;
            double Ppnw=0.0,Ppw=0.0;
            if (pvalid) {
                qcdm_qscf_kvec3 qv0={q*x,q*sq,0.0};
                qcdm_qscf_kvec3 pv0={kk-qv0.x,-qv0.y,0.0};
                class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qv0,pv0,&K2qp,errmsg),errmsg,errmsg);
                qcdm_tf_K1_from_K2_leg(&K2qp,2,&K1p);
                Ppnw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,pnorm);
                Ppw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,pnorm);
            }

            double measure_base=(q*pnorm/kk)*dq*dp*dphi;
            for (int ip=0;ip<Nphi;ip++) {
                double phi=2.0*_PI_*((((double)ip)+0.5)/((double)Nphi));
                double cphi=cos(phi),sphi=sin(phi);
                qcdm_qscf_kvec3 pvec[3];
                pvec[0]=pbase[0];
                pvec[1]=(qcdm_qscf_kvec3){q*x,q*sq*cphi,q*sq*sphi};
                pvec[2]=(qcdm_qscf_kvec3){-pvec[1].x,-pvec[1].y,-pvec[1].z};
                qcdm_qscf_kvec3 qv=pvec[1];
                qcdm_qscf_kvec3 pv={kk-qv.x,-qv.y,-qv.z};

                for (int im=0;im<Nmu_loop;im++) {
                    if (pvalid) {
                        double zv[4],za[4],sqcoef[10];
                        for (int j=0;j<4;j++)
                            zv[j]=qcdm_tf_Z2_from_kernels(&view[j],qv,pv,losnode[im],&K2qp,&K1q,&K1p);
                        qcdm_tf_basis_affine_from_views(zv,za);
                        qcdm_tf_basis_affine_square(za,sqcoef);
                        for (int c=0;c<NB;c++) {
                            P22nw[(size_t)c*Nmu_loop+im] += measure_base*sqcoef[c]*Pqnw*Ppnw;
                            P22w [(size_t)c*Nmu_loop+im] += measure_base*sqcoef[c]*(Pqw*Ppnw+Pqnw*Ppw);
                        }
                    }

                    double z3v[4],z3a[4];
                    for (int j=0;j<4;j++)
                        z3v[j]=qcdm_tf_Z3_collapsed_from_kernels(&view[j],pvec,losnode[im],munode[im],
                                                                  &K3,pair,K1coll,D3v[j],V3);
                    qcdm_tf_basis_affine_from_views(z3v,z3a);
                    for (int a=0;a<4;a++) {
                        I13nw[(size_t)a*Nmu_loop+im] += measure_base*z3a[a]*Pqnw;
                        I13w [(size_t)a*Nmu_loop+im] += measure_base*z3a[a]*Pqw;
                    }

                    /* Exact Gamma3 response of Z3.  Subtract the zero-Gamma3
                     * base view so fixed relative-sector pieces cancel. */
                    double z3gamma=qcdm_tf_Z3_collapsed_from_kernels(
                        &view[4],pvec,losnode[im],munode[im],
                        &K3,pair,K1coll,D3v[4],V3)-z3v[0];
                    I13gamma_nw[im] += measure_base*z3gamma*Pqnw;
                    I13gamma_w [im] += measure_base*z3gamma*Pqw;
                }
            }
        }
    }

    double kc=kk/cutoff;
    double uv=exp(-pow(kc,6.0));
    for (int im=0;im<Nmu_loop;im++) {
        double za[4],inw[4],iw[4],tmpa[4],prod_nw[10],prod_w[10];
        for (int a=0;a<4;a++) {
            za[a]=Z1aff[(size_t)a*Nmu_loop+im];
            inw[a]=I13nw[(size_t)a*Nmu_loop+im];
            iw[a]=I13w[(size_t)a*Nmu_loop+im];
            tmpa[a]=Pkw*inw[a]+Pknw*iw[a];
        }
        qcdm_tf_basis_affine_product(za,inw,prod_nw);
        qcdm_tf_basis_affine_product(za,tmpa,prod_w);

        double loopnw[10],loopw[10];
        for (int c=0;c<NB;c++) {
            loopnw[c]=uv*(2.0*P22nw[(size_t)c*Nmu_loop+im]*loopnorm
                        +6.0*Pknw*prod_nw[c]*loopnorm);
            loopw[c]=uv*(2.0*P22w[(size_t)c*Nmu_loop+im]*loopnorm
                       +6.0*prod_w[c]*loopnorm);
        }

        /* Gamma3 appears only in Z3.  Therefore P13=6 Z1 P11 int(Z3 P11)
         * produces exactly two new monomials because Z1 is affine only in b_m. */
        double ig_nw=I13gamma_nw[im];
        double ig_w =I13gamma_w[im];
        double ig_mix=Pkw*ig_nw+Pknw*ig_w;
        double loop_gamma_nw=uv*(6.0*Pknw*za[0]*ig_nw*loopnorm);
        double loop_mgamma_nw=uv*(6.0*Pknw*za[1]*ig_nw*loopnorm);
        double loop_gamma_w=uv*(6.0*za[0]*ig_mix*loopnorm);
        double loop_mgamma_w=uv*(6.0*za[1]*ig_mix*loopnorm);

        for (int il=0;il<nL;il++) {
            int ell=Lval[il];
            double wt=muweight[im]*qcdm_tf_legendre(ell,munode[im]);
            double fac=0.5*(2.0*((double)ell)+1.0);
            for (int c=0;c<NB;c++) {
                size_t idx=((size_t)c*nL+il)*Nmax+ik;
                Lloop_nw[idx] += fac*wt*loopnw[c];
                Lloop_w[idx]  += fac*wt*loopw[c];
            }
            size_t idxg=((size_t)0*nL+il)*Nmax+ik;
            size_t idxmg=((size_t)1*nL+il)*Nmax+ik;
            Lgamma_nw[idxg]  += fac*wt*loop_gamma_nw;
            Lgamma_w [idxg]  += fac*wt*loop_gamma_w;
            Lgamma_nw[idxmg] += fac*wt*loop_mgamma_nw;
            Lgamma_w [idxmg] += fac*wt*loop_mgamma_w;
        }
    }
    return _SUCCESS_;
}

/* Exact anisotropic-IR basis builder for the current DESI M0 nuisance model.
 * The underlying F_n/G_n kernels, quadrature, UV taper, anisotropic IR
 * resummation and AP mapping are identical to qcdm_tf_exact_rsd_multipoles_ir.
 * Only the tracer-bias algebra is vectorized into polynomial coefficients. */
static int qcdm_tf_exact_rsd_basis_ir(struct background *pba,
                                      struct nonlinear_pt *pnlpt,
                                      double tau,
                                      int Nmax,
                                      const double *kdisc,
                                      const double *Pnw,
                                      const double *Pw,
                                      const qcdm_tf_ir_state *ir_state,
                                      double hratio,
                                      double Dratio,
                                      double cutoff,
                                      double *B0,double *B2,double *B4) {
    class_test((ir_state==NULL)||(ir_state->enabled!=_TRUE_),pnlpt->error_message,
               "QCDM anisotropic IR basis requested without an initialized IR state");
    class_test(fabs(pnlpt->qcdm_fNL_equilateral)>0.0 ||
               fabs(pnlpt->qcdm_fNL_orthogonal)>0.0,
               pnlpt->error_message,
               "QCDM native nuisance basis currently supports the Gaussian M0 baseline only (fNL=0)");

    const int NB=10;
    const int NC=QCDM_TF_BASIS_NCOEFF;
    int Nq=pnlpt->qcdm_tf_Nq;
    int Nx=pnlpt->qcdm_tf_Nx;
    int Nphi=pnlpt->qcdm_tf_Nphi;
    int Nmu_loop=MAX(pnlpt->qcdm_tf_Nmu,9);
    const int Nmu_final=40;
    const int nL=5;
    const int Lval[5]={0,2,4,6,8};

    double qmin=MAX(kdisc[0],pnlpt->qcdm_loop_q_min_h*pba->h);
    double qmax=MIN(kdisc[Nmax-1],pnlpt->qcdm_loop_q_max_h*pba->h);
    double kmax=pnlpt->qcdm_loop_k_max_h*pba->h;
    class_test(qmax<=qmin,pnlpt->error_message,
               "Invalid QCDM IR-basis loop interval after grid clipping: qmin=%e qmax=%e",qmin,qmax);

    {
        double kref=MIN(MAX(sqrt(kdisc[0]*kdisc[Nmax-1]),kdisc[1]),kdisc[Nmax-2]);
        double qref=sqrt(qmin*qmax);
        class_call(qcdm_tf_validate_rsd_conventions(pba,pnlpt,tau,kref,qref,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
    }

    struct nonlinear_pt view[5];
    qcdm_tf_basis_make_views(pnlpt,view);

    double *pnode,*pweight,*qnode,*qweight,*qgl,*qgw;
    double *munode,*muweight,*mu_final,*w_final;
    qcdm_qscf_kvec3 *losnode;
    class_alloc(pnode,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(pweight,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(qnode,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qweight,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qgl,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qgw,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(munode,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(muweight,Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(losnode,Nmu_loop*sizeof(qcdm_qscf_kvec3),pnlpt->error_message);
    class_alloc(mu_final,Nmu_final*sizeof(double),pnlpt->error_message);
    class_alloc(w_final,Nmu_final*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nx,pnode,pweight,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nmu_loop,munode,muweight,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nmu_final,mu_final,w_final,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    for (int im=0;im<Nmu_loop;im++)
        losnode[im]=(qcdm_qscf_kvec3){munode[im],sqrt(MAX(1.0-munode[im]*munode[im],0.0)),0.0};

    /* Stored as [bias coefficient][ell index][k].  Keep the validated
     * ten legacy loop monomials separate from the two Gamma3 P13 responses. */
    double *Lloop_nw,*Lloop_w,*Lgamma_nw,*Lgamma_w;
    class_alloc(Lloop_nw,(size_t)NB*nL*Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(Lloop_w ,(size_t)NB*nL*Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(Lgamma_nw,(size_t)2*nL*Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(Lgamma_w ,(size_t)2*nL*Nmax*sizeof(double),pnlpt->error_message);
    memset(Lloop_nw,0,(size_t)NB*nL*Nmax*sizeof(double));
    memset(Lloop_w ,0,(size_t)NB*nL*Nmax*sizeof(double));
    memset(Lgamma_nw,0,(size_t)2*nL*Nmax*sizeof(double));
    memset(Lgamma_w ,0,(size_t)2*nL*Nmax*sizeof(double));

    /* P22 is quadratic; I13 and Z1 are affine. Arrays are [coeff][mu]. */
    double *P22nw,*P22w,*I13nw,*I13w;
    class_alloc(P22nw,(size_t)NB*Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(P22w ,(size_t)NB*Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(I13nw,(size_t)4*Nmu_loop*sizeof(double),pnlpt->error_message);
    class_alloc(I13w ,(size_t)4*Nmu_loop*sizeof(double),pnlpt->error_message);
    double *Z1aff;
    class_alloc(Z1aff,(size_t)4*Nmu_loop*sizeof(double),pnlpt->error_message);

    const double dphi=2.0*_PI_/((double)Nphi);
    const double loopnorm=1.0/(8.0*_PI_*_PI_*_PI_);

    struct timespec ts_kernel0,ts_kernel1,ts_ap0,ts_ap1;
    clock_gettime(CLOCK_MONOTONIC,&ts_kernel0);

    int basis_omp_failed=0;
    char basis_omp_error[4096];
    basis_omp_error[0]='\0';

    /* DW speed stage 2: distribute independent external-k contractions.
     * OMP=1 retains the validated read-only warm-cache baseline.  With more
     * than one thread, initialization used no-store mode, so basis misses are
     * likewise solved privately without mutating the global cache. */
    int basis_use_nostore=(omp_get_max_threads()>1)?_TRUE_:_FALSE_;
    if (basis_use_nostore==_TRUE_) {
        /* The basis is evaluated later, after all z_pk initialization passes.
         * Therefore the shared RK-stage cache generally contains another
         * redshift.  Rebuild/pin the three current-tau stage grids serially
         * before worker threads are allowed to solve cache misses. */
        class_call(qcdm_qscf_qs_prewarm_rk_for_tau(pba,tau,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
    }
    qcdm_qscf_cache_readonly_parallel=(basis_use_nostore==_TRUE_)?_FALSE_:_TRUE_;
    qcdm_qscf_cache_parallel_nostore=basis_use_nostore;
    #pragma omp parallel for schedule(dynamic,1) if(basis_use_nostore)
    for (int ik=0;ik<Nmax;ik++) {
        char local_errmsg[2048];
        local_errmsg[0]='\0';
        int st=qcdm_tf_exact_rsd_basis_ir_one_k_cached(
            pba,view,tau,Nmax,kdisc,Pnw,Pw,
            Nq,Nx,Nphi,Nmu_loop,nL,Lval,
            qmin,qmax,kmax,cutoff,pnode,pweight,munode,muweight,losnode,
            dphi,loopnorm,ik,Lloop_nw,Lloop_w,Lgamma_nw,Lgamma_w,local_errmsg);
        if (st!=_SUCCESS_) {
            #pragma omp critical(qcdm_basis_error)
            {
                if (basis_omp_failed==0) {
                    basis_omp_failed=1;
                    snprintf(basis_omp_error,sizeof(basis_omp_error),
                             "native-basis OpenMP k-index %d failed: %s",ik,local_errmsg);
                }
            }
        }
    }
    qcdm_qscf_cache_readonly_parallel=_FALSE_;
    qcdm_qscf_cache_parallel_nostore=_FALSE_;
    class_test(basis_omp_failed!=0,pnlpt->error_message,"%s",basis_omp_error);

    clock_gettime(CLOCK_MONOTONIC,&ts_kernel1);
    double t_kernel=(ts_kernel1.tv_sec-ts_kernel0.tv_sec)+1.e-9*(ts_kernel1.tv_nsec-ts_kernel0.tv_nsec);
    clock_gettime(CLOCK_MONOTONIC,&ts_ap0);

    int Nloop=0;
    while (Nloop<Nmax && kdisc[Nloop]<=kmax) Nloop++;
    if (Nloop<2) Nloop=MIN(Nmax,2);

    memset(B0,0,(size_t)NC*Nmax*sizeof(double));
    memset(B2,0,(size_t)NC*Nmax*sizeof(double));
    memset(B4,0,(size_t)NC*Nmax*sizeof(double));

    for (int ik=0;ik<Nmax;ik++) {
        double kobs=kdisc[ik];
        for (int im=0;im<Nmu_final;im++) {
            double muobs=mu_final[im],dmu=w_final[im];
            double ktrue=kobs,mutrue=muobs,jac=1.0;
            if (pnlpt->AP_effect==AP_effect_yes) {
                double invD2=1.0/(Dratio*Dratio);
                double apfac=sqrt(invD2+(hratio*hratio-invD2)*muobs*muobs);
                ktrue=kobs*apfac;
                mutrue=muobs*hratio/apfac;
                jac=hratio/(Dratio*Dratio);
            }

            qcdm_qscf_kernel1_result K1;
            class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,ktrue,&K1,pnlpt->error_message),
                       pnlpt->error_message,pnlpt->error_message);
            double Pnwt=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,ktrue);
            double Pwt =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,ktrue);
            double S=qcdm_tf_ir_exponent(ir_state,ktrue,mutrue);
            class_test((qcdm_qscf_finite_double(S)==0)||(S<0.0),pnlpt->error_message,
                       "Invalid QCDM IR-basis exponent at k=%e mu=%e: S=%e",ktrue,mutrue,S);
            double Dir=exp(-S);
            double Plo=Pnwt+Dir*Pwt;
            double Ptree_ir=Pnwt+Dir*(1.0+S)*Pwt;
            double mu2=mutrue*mutrue,mu4=mu2*mu2;
            double k2=ktrue*ktrue;

            double z1v[4],z1a[4],treec[10];
            for (int j=0;j<4;j++) z1v[j]=qcdm_tf_Z1(&view[j],&K1,mutrue);
            qcdm_tf_basis_affine_from_views(z1v,z1a);
            qcdm_tf_basis_affine_square(z1a,treec);
            for (int c=0;c<NB;c++) treec[c]*=Ptree_ir;

            /* Fixed relative-sector counterterms are part of the constant
             * coefficient; the six sampled M0 linear terms are explicit
             * basis columns below. */
            double Pmr=K1.F1_r_red*Plo;
            treec[0] += -(pnlpt->qcdm_tf_ct_r0+pnlpt->qcdm_tf_ct_r2*mu2+pnlpt->qcdm_tf_ct_r4*mu4)*k2*Pmr;

            double coeff[QCDM_TF_BASIS_NCOEFF];
            for (int c=0;c<NB;c++) {
                double lnw=0.0,lw=0.0;
                if ((ktrue>=kdisc[0])&&(ktrue<=kmax)) {
                    for (int il=0;il<nL;il++) {
                        double Leg=qcdm_tf_legendre(Lval[il],mutrue);
                        const double *arrnw=Lloop_nw+((size_t)c*nL+il)*Nmax;
                        const double *arrw =Lloop_w +((size_t)c*nL+il)*Nmax;
                        lnw += qcdm_qscf_interp_linear(Nloop,kdisc,arrnw,ktrue)*Leg;
                        lw  += qcdm_qscf_interp_linear(Nloop,kdisc,arrw ,ktrue)*Leg;
                    }
                }
                coeff[c]=treec[c]+lnw+Dir*lw;
            }
            coeff[10]=-k2*Plo;
            coeff[11]=-mu2*k2*Plo;
            coeff[12]=-mu4*k2*Plo;
            coeff[13]=1.0;
            coeff[14]=k2;
            coeff[15]=mu2*k2;

            double gamma_nw[2]={0.0,0.0},gamma_w[2]={0.0,0.0};
            if ((ktrue>=kdisc[0])&&(ktrue<=kmax)) {
                for (int g=0;g<2;g++) {
                    for (int il=0;il<nL;il++) {
                        double Leg=qcdm_tf_legendre(Lval[il],mutrue);
                        const double *arrnw=Lgamma_nw+((size_t)g*nL+il)*Nmax;
                        const double *arrw =Lgamma_w +((size_t)g*nL+il)*Nmax;
                        gamma_nw[g] += qcdm_qscf_interp_linear(Nloop,kdisc,arrnw,ktrue)*Leg;
                        gamma_w [g] += qcdm_qscf_interp_linear(Nloop,kdisc,arrw ,ktrue)*Leg;
                    }
                }
            }
            coeff[16]=gamma_nw[0]+Dir*gamma_w[0];
            coeff[17]=gamma_nw[1]+Dir*gamma_w[1];

            double w0=0.5*dmu*qcdm_tf_legendre(0,muobs);
            double w2=2.5*dmu*qcdm_tf_legendre(2,muobs);
            double w4=4.5*dmu*qcdm_tf_legendre(4,muobs);
            for (int c=0;c<NC;c++) {
                double ps=jac*coeff[c];
                B0[(size_t)c*Nmax+ik] += w0*ps;
                B2[(size_t)c*Nmax+ik] += w2*ps;
                B4[(size_t)c*Nmax+ik] += w4*ps;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC,&ts_ap1);
    double t_ap=(ts_ap1.tv_sec-ts_ap0.tv_sec)+1.e-9*(ts_ap1.tv_nsec-ts_ap0.tv_nsec);
    if (pnlpt->nonlinear_pt_verbose>0)
        printf("DW/QCDM-IR-BASIS: exact basis timing: kernels %.3fs, anisotropic IR+AP %.3fs (18 coefficients, Nmu_loop=%d, Nmu_IR/AP=%d).\n",
               t_kernel,t_ap,Nmu_loop,Nmu_final);

    free(Lloop_nw);free(Lloop_w);free(Lgamma_nw);free(Lgamma_w);
    free(P22nw);free(P22w);free(I13nw);free(I13w);free(Z1aff);
    free(pnode);free(pweight);free(qnode);free(qweight);free(qgl);free(qgw);
    free(munode);free(muweight);free(losnode);free(mu_final);free(w_final);
    return _SUCCESS_;
}

/* ===================================================================== */
/* DW: fast exact-QS tracer/nuisance layer                               */
/* ===================================================================== */

/* Store the cosmology-dependent one-redshift inputs needed by the exact
 * RSD contraction.  The expensive F_n/G_n kernel cache is retained
 * separately (and now survives across z_pk values within one cosmology).
 * These arrays are tiny compared with the kernel cache: O(Nz*Nmax). */
static int qcdm_tf_fast_cache_store(struct background *pba,
                                    struct nonlinear_pt *pnlpt,
                                    double tau,
                                    int Nmax,
                                    const double *kdisc,
                                    const double *Pfull,
                                    const double *Pphi,
                                    const double *Mphi) {
    double z=0.0;
    class_call(background_z_of_tau(pba,tau,&z),pba->error_message,pnlpt->error_message);

    int iz=0;
    double dzbest=fabs(z-pnlpt->z_pk[0]);
    for (int j=1;j<pnlpt->z_pk_num;j++) {
        double dz=fabs(z-pnlpt->z_pk[j]);
        if (dz<dzbest) { dzbest=dz; iz=j; }
    }
    class_test(dzbest>1.e-7*(1.0+fabs(z)),pnlpt->error_message,
               "Could not associate QCDM fast-tracer cache tau=%e (z=%e) with requested z_pk list",tau,z);

    if (pnlpt->qcdm_fast_nmax==0) {
        pnlpt->qcdm_fast_nmax=Nmax;
        class_alloc(pnlpt->qcdm_fast_kdisc,Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(pnlpt->qcdm_fast_Pbin,(size_t)pnlpt->z_pk_num*Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(pnlpt->qcdm_fast_Pphi,(size_t)pnlpt->z_pk_num*Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(pnlpt->qcdm_fast_Mphi,(size_t)pnlpt->z_pk_num*Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(pnlpt->qcdm_fast_has_png,pnlpt->z_pk_num*sizeof(short),pnlpt->error_message);
        class_alloc(pnlpt->qcdm_fast_z_ready,pnlpt->z_pk_num*sizeof(short),pnlpt->error_message);
        memcpy(pnlpt->qcdm_fast_kdisc,kdisc,Nmax*sizeof(double));
        memset(pnlpt->qcdm_fast_Pbin,0,(size_t)pnlpt->z_pk_num*Nmax*sizeof(double));
        memset(pnlpt->qcdm_fast_Pphi,0,(size_t)pnlpt->z_pk_num*Nmax*sizeof(double));
        memset(pnlpt->qcdm_fast_Mphi,0,(size_t)pnlpt->z_pk_num*Nmax*sizeof(double));
        memset(pnlpt->qcdm_fast_has_png,0,pnlpt->z_pk_num*sizeof(short));
        memset(pnlpt->qcdm_fast_z_ready,0,pnlpt->z_pk_num*sizeof(short));
    }
    else {
        class_test(pnlpt->qcdm_fast_nmax!=Nmax,pnlpt->error_message,
                   "QCDM fast-tracer cache Nmax changed within one cosmology (%d -> %d)",
                   pnlpt->qcdm_fast_nmax,Nmax);
        for (int ik=0;ik<Nmax;ik++) {
            class_test(fabs(pnlpt->qcdm_fast_kdisc[ik]-kdisc[ik])>
                       1.e-12*(1.0+fabs(kdisc[ik])),pnlpt->error_message,
                       "QCDM fast-tracer k grid changed within one cosmology at index %d",ik);
        }
    }

    /* Historical field name qcdm_fast_Pbin is retained in the public
     * structure, but for exact-QS it now stores the undamped physical P_L. */
    double *dstP=pnlpt->qcdm_fast_Pbin+(size_t)iz*Nmax;
    double *dstPhi=pnlpt->qcdm_fast_Pphi+(size_t)iz*Nmax;
    double *dstM=pnlpt->qcdm_fast_Mphi+(size_t)iz*Nmax;
    memcpy(dstP,Pfull,Nmax*sizeof(double));
    if ((Pphi!=NULL)&&(Mphi!=NULL)) {
        memcpy(dstPhi,Pphi,Nmax*sizeof(double));
        memcpy(dstM,Mphi,Nmax*sizeof(double));
        pnlpt->qcdm_fast_has_png[iz]=_TRUE_;
    }
    else {
        memset(dstPhi,0,Nmax*sizeof(double));
        memset(dstM,0,Nmax*sizeof(double));
        pnlpt->qcdm_fast_has_png[iz]=_FALSE_;
    }
    pnlpt->qcdm_fast_z_ready[iz]=_TRUE_;

    pnlpt->qcdm_fast_tracer_ready=_TRUE_;
    for (int j=0;j<pnlpt->z_pk_num;j++)
        if (pnlpt->qcdm_fast_z_ready[j]!=_TRUE_) pnlpt->qcdm_fast_tracer_ready=_FALSE_;

    if (pnlpt->nonlinear_pt_verbose>1)
        printf("DW/QCDM-QSCF: cached exact fast-tracer inputs for z=%.8g (%d/%d).\n",
               pnlpt->z_pk[iz],iz+1,pnlpt->z_pk_num);
    return _SUCCESS_;
}

static void qcdm_tf_fast_get_params(const struct nonlinear_pt *pnlpt,double p[QCDM_TF_FAST_NPARAM]) {
    p[0]=pnlpt->qcdm_tf_b_m;          p[1]=pnlpt->qcdm_tf_b_r;
    p[2]=pnlpt->qcdm_tf_b_theta_r;    p[3]=pnlpt->qcdm_tf_c_vr;
    p[4]=pnlpt->qcdm_tf_c_mr;         p[5]=pnlpt->qcdm_tf_c_K;
    p[6]=pnlpt->qcdm_tf_b_mm;         p[7]=pnlpt->qcdm_tf_b_mr;
    p[8]=pnlpt->qcdm_tf_b_rr;         p[9]=pnlpt->qcdm_tf_b_Gmm;
    p[10]=pnlpt->qcdm_tf_b_Gmr;       p[11]=pnlpt->qcdm_tf_b_Grr;
    p[12]=pnlpt->qcdm_tf_b_Gamma3;    p[13]=pnlpt->qcdm_tf_ct0;
    p[14]=pnlpt->qcdm_tf_ct2;         p[15]=pnlpt->qcdm_tf_ct4;
    p[16]=pnlpt->qcdm_tf_ct_r0;       p[17]=pnlpt->qcdm_tf_ct_r2;
    p[18]=pnlpt->qcdm_tf_ct_r4;       p[19]=pnlpt->qcdm_tf_stoch0;
    p[20]=pnlpt->qcdm_tf_stoch2;      p[21]=pnlpt->qcdm_tf_stoch2_mu2;
    p[22]=pnlpt->qcdm_fNL_equilateral;p[23]=pnlpt->qcdm_fNL_orthogonal;
}

static void qcdm_tf_fast_set_params(struct nonlinear_pt *pnlpt,const double p[QCDM_TF_FAST_NPARAM]) {
    pnlpt->qcdm_tf_b_m=p[0];          pnlpt->qcdm_tf_b_r=p[1];
    pnlpt->qcdm_tf_b_theta_r=p[2];    pnlpt->qcdm_tf_c_vr=p[3];
    pnlpt->qcdm_tf_c_mr=p[4];         pnlpt->qcdm_tf_c_K=p[5];
    pnlpt->qcdm_tf_b_mm=p[6];         pnlpt->qcdm_tf_b_mr=p[7];
    pnlpt->qcdm_tf_b_rr=p[8];         pnlpt->qcdm_tf_b_Gmm=p[9];
    pnlpt->qcdm_tf_b_Gmr=p[10];       pnlpt->qcdm_tf_b_Grr=p[11];
    pnlpt->qcdm_tf_b_Gamma3=p[12];    pnlpt->qcdm_tf_ct0=p[13];
    pnlpt->qcdm_tf_ct2=p[14];         pnlpt->qcdm_tf_ct4=p[15];
    pnlpt->qcdm_tf_ct_r0=p[16];       pnlpt->qcdm_tf_ct_r2=p[17];
    pnlpt->qcdm_tf_ct_r4=p[18];       pnlpt->qcdm_tf_stoch0=p[19];
    pnlpt->qcdm_tf_stoch2=p[20];      pnlpt->qcdm_tf_stoch2_mu2=p[21];
    pnlpt->qcdm_fNL_equilateral=p[22];pnlpt->qcdm_fNL_orthogonal=p[23];
}

/* ===================================================================== */
/* DW: exact-time QCDM/QSCF tree-level redshift-space bispectrum monopole */
/* ===================================================================== */

/* Apply the same AP map used by the power-spectrum multipoles, now at the
 * vector level.  hratio = H_true/H_fid = 1/q_parallel and
 * Dratio = D_A,true*h / D_A,fid = q_perp in the DESI convention used by
 * qcdm_classy_desi.py.  The line-of-sight direction is unchanged. */
static qcdm_qscf_kvec3 qcdm_tf_ap_map_vector(qcdm_qscf_kvec3 kobs,
                                              qcdm_qscf_kvec3 los,
                                              double hratio,
                                              double Dratio) {
    double kpar=qcdm_qscf_dot(kobs,los);
    double invD=1.0/Dratio;
    qcdm_qscf_kvec3 out;
    out.x=invD*kobs.x+(hratio-invD)*kpar*los.x;
    out.y=invD*kobs.y+(hratio-invD)*kpar*los.y;
    out.z=invD*kobs.z+(hratio-invD)*kpar*los.z;
    return out;
}

/* Tree-level deterministic redshift-space bispectrum for one oriented true
 * triangle.  This reuses the exact-time two-fluid Z1/Z2 kernels already used
 * by the one-loop power-spectrum calculation.  P1/P2/P3 are the (possibly
 * IR-resummed) linear common-mode spectra evaluated on the three true legs. */
static int qcdm_tf_tree_bispectrum_oriented(struct background *pba,
                                             struct nonlinear_pt *pnlpt,
                                             double tau,
                                             qcdm_qscf_kvec3 k1,
                                             qcdm_qscf_kvec3 k2,
                                             qcdm_qscf_kvec3 k3,
                                             qcdm_qscf_kvec3 los,
                                             double P1,
                                             double P2,
                                             double P3,
                                             double *Bout) {
    qcdm_qscf_kernel2_result K12,K23,K31;
    qcdm_qscf_kernel1_result K1,K2,K3;

    int status=qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,k1,k2,&K12,pnlpt->error_message);
    if (status==_FAILURE_) return _FAILURE_;
    status=qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,k2,k3,&K23,pnlpt->error_message);
    if (status==_FAILURE_) return _FAILURE_;
    status=qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,k3,k1,&K31,pnlpt->error_message);
    if (status==_FAILURE_) return _FAILURE_;

    /* Every exact F2 solve also evolves both linear legs.  Reuse those exact
     * solutions instead of launching redundant F1 ODE solves. */
    qcdm_tf_K1_from_K2_leg(&K12,1,&K1);
    qcdm_tf_K1_from_K2_leg(&K12,2,&K2);
    qcdm_tf_K1_from_K2_leg(&K23,2,&K3);

    double Z11=qcdm_tf_Z1(pnlpt,&K1,qcdm_tf_los_mu(k1,los));
    double Z12=qcdm_tf_Z1(pnlpt,&K2,qcdm_tf_los_mu(k2,los));
    double Z13=qcdm_tf_Z1(pnlpt,&K3,qcdm_tf_los_mu(k3,los));
    double Z212=qcdm_tf_Z2_from_kernels(pnlpt,k1,k2,los,&K12,&K1,&K2);
    double Z223=qcdm_tf_Z2_from_kernels(pnlpt,k2,k3,los,&K23,&K2,&K3);
    double Z231=qcdm_tf_Z2_from_kernels(pnlpt,k3,k1,los,&K31,&K3,&K1);

    double B=2.0*(Z11*Z12*Z212*P1*P2
                 +Z12*Z13*Z223*P2*P3
                 +Z13*Z11*Z231*P3*P1);
    class_test(qcdm_qscf_finite_double(B)==0,pnlpt->error_message,
               "Non-finite QCDM tree bispectrum for oriented triangle");
    *Bout=B;
    return _SUCCESS_;
}

/* Public bulk evaluator for the exact-time tree-level bispectrum monopole.
 *
 * Inputs
 * ------
 * k1_h,k2_h,k3_h : triangle-side arrays in h/Mpc (observed/fiducial coords)
 * ntri             : number of triangles
 * tracer_params    : the same 24 native QCDM nuisance values used by
 *                    nonlinear_pt_qcdm_fast_pkpoles().  Power-spectrum-only
 *                    counterterm/stochastic entries are intentionally ignored
 *                    by this deterministic tree-level bispectrum.
 * nmu,nphi         : LOS quadrature orders; pass <=0 to use the PT defaults.
 *
 * Output
 * ------
 * b0_h             : B0 in (Mpc/h)^6.
 *
 * Physics included
 * ----------------
 *  - exact-time two-fluid Z1 and Z2 kernels (no EdS fallback),
 *  - full orientation average over the line of sight,
 *  - DESI AP vector remapping and the bispectrum AP Jacobian,
 *  - leading-order anisotropic IR resummation when IR resummation=yes,
 *    implemented by P_L -> P_nw + exp[-S(k,mu)] P_w on each external leg.
 *
 * Not included yet
 * ----------------
 *  - primordial-bispectrum terms (therefore nonzero fNL is rejected),
 *  - GEO-FPT's simulation-calibrated nonlinear Z2 correction (deliberately
 *    not imported into the c-c' model),
 *  - bispectrum stochastic/shot-noise terms,
 *  - survey triangle-bin/window convolution (belongs in the likelihood).
 */
int nonlinear_pt_qcdm_fast_b0_desi(struct background *pba,
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
                                    double Dratio_override) {
    class_test(pnlpt->qcdm_fast_nmax<=0 || pnlpt->qcdm_fast_kdisc==NULL,
               pnlpt->error_message,
               "QCDM fast tracer cache is not initialized; run nonlinear_pt_init first");
    class_test(pnlpt->rsd!=rsd_yes,pnlpt->error_message,
               "QCDM fast B0 requires RSD=yes");
    class_test(qcdm_qscf_finite_double(sigma_B_h)==0 || sigma_B_h<0.0,
               pnlpt->error_message,
               "QCDM DESI B0 requires finite sigma_B>=0 in Mpc/h (got %g)",sigma_B_h);
    class_test(ntri<=0,pnlpt->error_message,"QCDM fast B0 requires ntri>0");
    class_test(n_tracer_params!=QCDM_TF_FAST_NPARAM,pnlpt->error_message,
               "QCDM fast B0 expected %d tracer parameters, got %d",
               QCDM_TF_FAST_NPARAM,n_tracer_params);
    class_test(fabs(tracer_params[22])>0.0 || fabs(tracer_params[23])>0.0,
               pnlpt->error_message,
               "QCDM fast tree B0 currently implements the Gaussian gravitational bispectrum only; set fNL_equilateral=fNL_orthogonal=0");
    if (use_ap_override) {
        class_test(pnlpt->AP_effect!=AP_effect_yes,pnlpt->error_message,
                   "External QCDM AP ratios requested for B0 but AP=no in the initial CLASS run");
        class_test(!(hratio_override>0.0) || !(Dratio_override>0.0),pnlpt->error_message,
                   "External QCDM B0 AP ratios must be positive (hratio=%g, Dratio=%g)",
                   hratio_override,Dratio_override);
    }

    int iz=0;
    double dzbest=fabs(z-pnlpt->z_pk[0]);
    for (int j=1;j<pnlpt->z_pk_num;j++) {
        double dz=fabs(z-pnlpt->z_pk[j]);
        if (dz<dzbest) { dzbest=dz; iz=j; }
    }
    class_test(dzbest>1.e-7*(1.0+fabs(z)),pnlpt->error_message,
               "QCDM fast B0 z=%.12g is not one of the cached z_pk values",z);
    class_test(pnlpt->qcdm_fast_z_ready==NULL || pnlpt->qcdm_fast_z_ready[iz]!=_TRUE_,
               pnlpt->error_message,"QCDM fast B0 cache is not ready at z=%.12g",z);

    const int Nmax=pnlpt->qcdm_fast_nmax;
    const double *kdisc=pnlpt->qcdm_fast_kdisc;
    const double *Pfull=pnlpt->qcdm_fast_Pbin+(size_t)iz*Nmax;
    const double hratio=use_ap_override ? hratio_override : pnlpt->hratio_array[iz];
    const double Dratio=use_ap_override ? Dratio_override : pnlpt->Dratio_array[iz];
    class_test(!(hratio>0.0) || !(Dratio>0.0),pnlpt->error_message,
               "Invalid cached QCDM B0 AP ratios (hratio=%g, Dratio=%g)",hratio,Dratio);

    const double *Pnw=NULL,*Pw=NULL;
    const qcdm_tf_ir_state *ir_state=NULL;
    if (pnlpt->irres==irres_yes) {
        class_test(qcdm_tf_ir_cache.nmax!=Nmax || qcdm_tf_ir_cache.nz!=pnlpt->z_pk_num ||
                   qcdm_tf_ir_cache.ready==NULL || qcdm_tf_ir_cache.ready[iz]!=_TRUE_,
                   pnlpt->error_message,
                   "QCDM fast B0 anisotropic-IR cache is not ready at z=%.12g",z);
        Pnw=qcdm_tf_ir_cache.Pnw+(size_t)iz*Nmax;
        Pw =qcdm_tf_ir_cache.Pw +(size_t)iz*Nmax;
        ir_state=&qcdm_tf_ir_cache.state[iz];
    }

    /* AP can stretch any leg by at most max(hratio,1/Dratio).  Check the
     * complete possible interval before changing nuisance parameters so all
     * error exits remain leak/state safe. */
    double amin=MIN(hratio,1.0/Dratio),amax=MAX(hratio,1.0/Dratio);
    for (int it=0;it<ntri;it++) {
        class_test(!(k1_h[it]>0.0) || !(k2_h[it]>0.0) || !(k3_h[it]>0.0),
                   pnlpt->error_message,
                   "QCDM fast B0 triangle %d has non-positive side(s): (%g,%g,%g) h/Mpc",
                   it,k1_h[it],k2_h[it],k3_h[it]);
        double kmin_obs=MIN(k1_h[it],MIN(k2_h[it],k3_h[it]))*pba->h;
        double kmax_obs=MAX(k1_h[it],MAX(k2_h[it],k3_h[it]))*pba->h;
        class_test(kmin_obs*amin<kdisc[0]*(1.0-1.e-12) ||
                   kmax_obs*amax>kdisc[Nmax-1]*(1.0+1.e-12),
                   pnlpt->error_message,
                   "QCDM fast B0 triangle %d can AP-map outside cached k grid: observed=(%g,%g,%g) h/Mpc, cached=[%g,%g] h/Mpc",
                   it,k1_h[it],k2_h[it],k3_h[it],kdisc[0]/pba->h,kdisc[Nmax-1]/pba->h);
        double a=k1_h[it],b=k2_h[it],c=k3_h[it];
        double cos12=(c*c-a*a-b*b)/(2.0*a*b);
        class_test(cos12 < -1.0-2.e-12 || cos12 > 1.0+2.e-12,pnlpt->error_message,
                   "QCDM fast B0 triangle %d does not close: (%g,%g,%g) h/Mpc",
                   it,a,b,c);
    }


    nmu=(nmu>0)?nmu:MAX(pnlpt->qcdm_tf_Nmu,12);
    nphi=(nphi>0)?nphi:MAX(pnlpt->qcdm_tf_Nphi,12);
    class_test(nmu<2 || nphi<2,pnlpt->error_message,
               "QCDM fast B0 quadrature requires nmu,nphi >= 2 (got %d,%d)",nmu,nphi);

    double *munode=NULL,*muweight=NULL;
    class_alloc(munode,nmu*sizeof(double),pnlpt->error_message);
    class_alloc(muweight,nmu*sizeof(double),pnlpt->error_message);
    int status=qcdm_tf_gauss_legendre_grid(nmu,munode,muweight,pnlpt->error_message);
    if (status==_FAILURE_) { free(munode); free(muweight); return _FAILURE_; }

    double tau=0.0;
    status=background_tau_of_z(pba,z,&tau);
    if (status==_FAILURE_) { free(munode); free(muweight); return _FAILURE_; }

    double oldp[QCDM_TF_FAST_NPARAM];
    qcdm_tf_fast_get_params(pnlpt,oldp);
    int old_verbose=pnlpt->nonlinear_pt_verbose;
    qcdm_tf_fast_set_params(pnlpt,tracer_params);
    pnlpt->nonlinear_pt_verbose=0;

    const double dphi=2.0*_PI_/((double)nphi);
    const double orient_norm=dphi/(4.0*_PI_);
    const double jacB=pow(hratio/(Dratio*Dratio),2.0);
    /* DESI sigma_B is quoted in Mpc/h, while exact-QS vectors below are
     * stored in physical CLASS units 1/Mpc. */
    const double sigma_B_phys=(pba->h>0.0)?sigma_B_h/pba->h:0.0;

    for (int it=0;it<ntri;it++) {
        const double a=k1_h[it]*pba->h;
        const double b=k2_h[it]*pba->h;
        const double c=k3_h[it]*pba->h;
        double cos12=(c*c-a*a-b*b)/(2.0*a*b);
        cos12=MAX(-1.0,MIN(1.0,cos12));
        double sin12=sqrt(MAX(0.0,1.0-cos12*cos12));

        qcdm_qscf_kvec3 ko1={a,0.0,0.0};
        qcdm_qscf_kvec3 ko2={b*cos12,b*sin12,0.0};
        qcdm_qscf_kvec3 ko3={-(ko1.x+ko2.x),-(ko1.y+ko2.y),0.0};
        double accum=0.0;

        for (int im=0;im<nmu;im++) {
            double mu=munode[im];
            double sperp=sqrt(MAX(0.0,1.0-mu*mu));
            for (int ip=0;ip<nphi;ip++) {
                double phi=(ip+0.5)*dphi;
                qcdm_qscf_kvec3 los={mu,sperp*cos(phi),sperp*sin(phi)};
                qcdm_qscf_kvec3 k1=ko1,k2=ko2,k3=ko3;
                if (pnlpt->AP_effect==AP_effect_yes) {
                    k1=qcdm_tf_ap_map_vector(ko1,los,hratio,Dratio);
                    k2=qcdm_tf_ap_map_vector(ko2,los,hratio,Dratio);
                    k3=qcdm_tf_ap_map_vector(ko3,los,hratio,Dratio);
                }

                double kt1=qcdm_qscf_norm(k1),kt2=qcdm_qscf_norm(k2),kt3=qcdm_qscf_norm(k3);
                double mu1=qcdm_tf_los_mu(k1,los);
                double mu2=qcdm_tf_los_mu(k2,los);
                double mu3=qcdm_tf_los_mu(k3,los);
                double P1,P2,P3;
                if (pnlpt->irres==irres_yes) {
                    double S1=qcdm_tf_ir_exponent(ir_state,kt1,mu1);
                    double S2=qcdm_tf_ir_exponent(ir_state,kt2,mu2);
                    double S3=qcdm_tf_ir_exponent(ir_state,kt3,mu3);
                    if (qcdm_qscf_finite_double(S1)==0 || qcdm_qscf_finite_double(S2)==0 ||
                        qcdm_qscf_finite_double(S3)==0 || S1<0.0 || S2<0.0 || S3<0.0) {
                        class_sprintf(pnlpt->error_message,
                                      "Invalid QCDM B0 IR exponent for triangle %d: (%e,%e,%e)",it,S1,S2,S3);
                        status=_FAILURE_;
                        goto qcdm_b0_cleanup;
                    }
                    P1=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,kt1)+exp(-S1)*qcdm_qscf_interp_linear(Nmax,kdisc,Pw,kt1);
                    P2=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,kt2)+exp(-S2)*qcdm_qscf_interp_linear(Nmax,kdisc,Pw,kt2);
                    P3=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,kt3)+exp(-S3)*qcdm_qscf_interp_linear(Nmax,kdisc,Pw,kt3);
                }
                else {
                    P1=qcdm_qscf_interp_linear(Nmax,kdisc,Pfull,kt1);
                    P2=qcdm_qscf_interp_linear(Nmax,kdisc,Pfull,kt2);
                    P3=qcdm_qscf_interp_linear(Nmax,kdisc,Pfull,kt3);
                }

                double Borient=0.0;
                status=qcdm_tf_tree_bispectrum_oriented(pba,pnlpt,tau,k1,k2,k3,los,P1,P2,P3,&Borient);
                if (status==_FAILURE_) goto qcdm_b0_cleanup;

                /* DESI DR1 P+B convention (Novell-Masot et al.):
                 *   D_FoG^B = [1 + 1/2 sigma_B^4
                 *     (k1^2 mu1^2 + k2^2 mu2^2 + k3^2 mu3^2)^2]^-2.
                 * Apply it after the exact AP vector map, hence all k_i and
                 * mu_i here are true-cosmology quantities. sigma_B=0 is
                 * exactly the previously validated tree-level path. */
                if (sigma_B_phys>0.0) {
                    double qfog=sigma_B_phys*sigma_B_phys
                               *(kt1*kt1*mu1*mu1
                                +kt2*kt2*mu2*mu2
                                +kt3*kt3*mu3*mu3);
                    double den=1.0+0.5*qfog*qfog;
                    double fogB=1.0/(den*den);
                    if (qcdm_qscf_finite_double(fogB)==0 || !(fogB>0.0) || fogB>1.0+1.e-14) {
                        class_sprintf(pnlpt->error_message,
                                      "Invalid QCDM DESI B0 FoG factor for triangle %d: %e",it,fogB);
                        status=_FAILURE_;
                        goto qcdm_b0_cleanup;
                    }
                    Borient*=fogB;
                }
                accum += muweight[im]*orient_norm*jacB*Borient;
            }
        }
        if (qcdm_qscf_finite_double(accum)==0) {
            class_sprintf(pnlpt->error_message,"Non-finite QCDM B0 for triangle %d",it);
            status=_FAILURE_;
            goto qcdm_b0_cleanup;
        }
        b0_h[it]=accum*pow(pba->h,6.0);
    }

qcdm_b0_cleanup:
    qcdm_tf_fast_set_params(pnlpt,oldp);
    pnlpt->nonlinear_pt_verbose=old_verbose;
    free(munode);
    free(muweight);
    return status;
}

/* Backward-compatible validated tree-level API. sigma_B=0 makes the DESI
 * FoG factor exactly unity, so all existing smoke/reference tests retain
 * bitwise-identical theory algebra apart from compiler roundoff. */
int nonlinear_pt_qcdm_fast_b0_tree(struct background *pba,
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
                                    double Dratio_override) {
    return nonlinear_pt_qcdm_fast_b0_desi(
        pba,pnlpt,z,k1_h,k2_h,k3_h,ntri,tracer_params,n_tracer_params,
        nmu,nphi,0.0,b0_h,use_ap_override,hratio_override,Dratio_override);
}

/* Public exact fast-nuisance evaluator.  This is deliberately not a
 * low-order single-fluid bias approximation: it reruns the full Z1/Z2/Z3
 * tracer contractions, AP map and multipole projection with the requested
 * nuisance parameters, while every cosmology-dependent exact kernel lookup
 * is a cache hit. */
int nonlinear_pt_qcdm_fast_pkpoles(struct background *pba,
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
                                   double Dratio_override) {
    class_test(pnlpt->qcdm_fast_nmax<=0 || pnlpt->qcdm_fast_kdisc==NULL,
               pnlpt->error_message,
               "QCDM fast tracer cache is not initialized; run nonlinear_pt_init first");
    class_test(pnlpt->rsd!=rsd_yes,pnlpt->error_message,
               "QCDM fast pkpoles requires RSD=yes");
    class_test(k_size<=0,pnlpt->error_message,"QCDM fast pkpoles requires k_size>0");
    class_test(n_tracer_params!=QCDM_TF_FAST_NPARAM,pnlpt->error_message,
               "QCDM fast pkpoles expected %d tracer parameters, got %d",
               QCDM_TF_FAST_NPARAM,n_tracer_params);
    if (use_ap_override) {
        class_test(pnlpt->AP_effect!=AP_effect_yes,pnlpt->error_message,
                   "External QCDM AP ratios requested but AP=no in the initial CLASS run");
        class_test(!(hratio_override>0.0) || !(Dratio_override>0.0),pnlpt->error_message,
                   "External QCDM AP ratios must be positive (hratio=%g, Dratio=%g)",
                   hratio_override,Dratio_override);
    }

    int iz=0;
    double dzbest=fabs(z-pnlpt->z_pk[0]);
    for (int j=1;j<pnlpt->z_pk_num;j++) {
        double dz=fabs(z-pnlpt->z_pk[j]);
        if (dz<dzbest) { dzbest=dz; iz=j; }
    }
    class_test(dzbest>1.e-7*(1.0+fabs(z)),pnlpt->error_message,
               "QCDM fast pkpoles z=%.12g is not one of the cached z_pk values",z);
    class_test(pnlpt->qcdm_fast_z_ready==NULL || pnlpt->qcdm_fast_z_ready[iz]!=_TRUE_,
               pnlpt->error_message,"QCDM fast pkpoles cache is not ready at z=%.12g",z);
    class_test((fabs(tracer_params[22])>0.0 || fabs(tracer_params[23])>0.0) &&
               (pnlpt->qcdm_fast_has_png[iz]!=_TRUE_),pnlpt->error_message,
               "Nonzero fast fNL requested at z=%.12g but initial CLASS run did not build PNG templates (set PNG=yes)",z);

    int Nmax=pnlpt->qcdm_fast_nmax;
    const double *kdisc=pnlpt->qcdm_fast_kdisc;

    /* Validate requested output k values before allocating work arrays or
     * performing the exact tracer contraction.  Besides producing a clearer
     * error, this guarantees the fast path cannot leak its temporary arrays
     * when a caller accidentally requests a k outside the cached PT grid. */
    for (int ik=0;ik<k_size;ik++) {
        double kval=k_h[ik]*pba->h;
        class_test(kval<kdisc[0] || kval>kdisc[Nmax-1],pnlpt->error_message,
                   "QCDM fast pkpoles k=%.12g h/Mpc lies outside cached grid [%.12g, %.12g] h/Mpc",
                   k_h[ik],kdisc[0]/pba->h,kdisc[Nmax-1]/pba->h);
    }

    const double *Pfull=pnlpt->qcdm_fast_Pbin+(size_t)iz*Nmax;
    const double *Pphi=(pnlpt->qcdm_fast_has_png[iz]==_TRUE_)?
                       pnlpt->qcdm_fast_Pphi+(size_t)iz*Nmax:NULL;
    const double *Mphi=(pnlpt->qcdm_fast_has_png[iz]==_TRUE_)?
                       pnlpt->qcdm_fast_Mphi+(size_t)iz*Nmax:NULL;

    double *tmp[9]={NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
    for (int j=0;j<9;j++) class_alloc(tmp[j],Nmax*sizeof(double),pnlpt->error_message);

    double oldp[QCDM_TF_FAST_NPARAM];
    qcdm_tf_fast_get_params(pnlpt,oldp);
    int old_verbose=pnlpt->nonlinear_pt_verbose;
    qcdm_tf_fast_set_params(pnlpt,tracer_params);
    pnlpt->nonlinear_pt_verbose=0;

    double tau=0.0;
    int status=background_tau_of_z(pba,z,&tau);
    if (status==_SUCCESS_) {
        const double hratio = use_ap_override ? hratio_override : pnlpt->hratio_array[iz];
        const double Dratio = use_ap_override ? Dratio_override : pnlpt->Dratio_array[iz];
        if (pnlpt->irres==irres_yes) {
            class_test(qcdm_tf_ir_cache.nmax!=Nmax || qcdm_tf_ir_cache.nz!=pnlpt->z_pk_num ||
                       qcdm_tf_ir_cache.ready==NULL || qcdm_tf_ir_cache.ready[iz]!=_TRUE_,
                       pnlpt->error_message,
                       "QCDM fast anisotropic-IR cache is not ready at z=%.12g",z);
            const double *Pnw_fast=qcdm_tf_ir_cache.Pnw+(size_t)iz*Nmax;
            const double *Pw_fast =qcdm_tf_ir_cache.Pw +(size_t)iz*Nmax;
            status=qcdm_tf_exact_rsd_multipoles_ir(pba,pnlpt,tau,Nmax,kdisc,
                                                    Pnw_fast,Pw_fast,&qcdm_tf_ir_cache.state[iz],
                                                    hratio,Dratio,3.0*pba->h,
                                                    tmp[0],tmp[1],tmp[2],
                                                    tmp[3],tmp[4],tmp[5],
                                                    tmp[6],tmp[7],tmp[8]);
        }
        else {
            status=qcdm_tf_exact_rsd_multipoles(pba,pnlpt,tau,Nmax,kdisc,Pfull,Pphi,Mphi,
                                                 hratio,Dratio,
                                                 3.0*pba->h,
                                                 tmp[0],tmp[1],tmp[2],
                                                 tmp[3],tmp[4],tmp[5],
                                                 tmp[6],tmp[7],tmp[8]);
        }
    }

    qcdm_tf_fast_set_params(pnlpt,oldp);
    pnlpt->nonlinear_pt_verbose=old_verbose;

    if (status==_FAILURE_) {
        for (int j=0;j<9;j++) free(tmp[j]);
        return _FAILURE_;
    }

    double h3=pba->h*pba->h*pba->h;
    for (int ik=0;ik<k_size;ik++) {
        double kval=k_h[ik]*pba->h;
        pk0_h[ik]=qcdm_qscf_interp_linear(Nmax,kdisc,tmp[0],kval)*h3;
        pk2_h[ik]=qcdm_qscf_interp_linear(Nmax,kdisc,tmp[1],kval)*h3;
        pk4_h[ik]=qcdm_qscf_interp_linear(Nmax,kdisc,tmp[2],kval)*h3;
    }
    for (int j=0;j<9;j++) free(tmp[j]);
    return _SUCCESS_;
}

/* Public native nuisance-basis evaluator for the DESI M0 configuration.
 * This performs one exact anisotropic-IR contraction and returns all ten
 * quadratic bias coefficients plus the six linear EFT/stochastic templates. */
int nonlinear_pt_qcdm_fast_pkpole_basis(struct background *pba,
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
                                        double Dratio_override) {
    class_test(pnlpt->qcdm_fast_nmax<=0 || pnlpt->qcdm_fast_kdisc==NULL,
               pnlpt->error_message,
               "QCDM fast tracer cache is not initialized; run nonlinear_pt_init first");
    class_test(pnlpt->rsd!=rsd_yes,pnlpt->error_message,
               "QCDM fast pkpole basis requires RSD=yes");
    class_test(pnlpt->irres!=irres_yes,pnlpt->error_message,
               "QCDM native pkpole basis currently targets the validated M0 anisotropic-IR path (IR resummation=yes)");
    class_test(k_size<=0,pnlpt->error_message,"QCDM fast pkpole basis requires k_size>0");
    class_test(n_tracer_params!=QCDM_TF_FAST_NPARAM,pnlpt->error_message,
               "QCDM fast pkpole basis expected %d tracer parameters, got %d",
               QCDM_TF_FAST_NPARAM,n_tracer_params);
    class_test(fabs(tracer_params[22])>0.0 || fabs(tracer_params[23])>0.0,
               pnlpt->error_message,
               "QCDM native pkpole basis currently supports the Gaussian M0 baseline only (fNL=0)");
    if (use_ap_override) {
        class_test(pnlpt->AP_effect!=AP_effect_yes,pnlpt->error_message,
                   "External QCDM AP ratios requested but AP=no in the initial CLASS run");
        class_test(!(hratio_override>0.0) || !(Dratio_override>0.0),pnlpt->error_message,
                   "External QCDM AP ratios must be positive (hratio=%g, Dratio=%g)",
                   hratio_override,Dratio_override);
    }

    int iz=0;
    double dzbest=fabs(z-pnlpt->z_pk[0]);
    for (int j=1;j<pnlpt->z_pk_num;j++) {
        double dz=fabs(z-pnlpt->z_pk[j]);
        if (dz<dzbest) { dzbest=dz; iz=j; }
    }
    class_test(dzbest>1.e-7*(1.0+fabs(z)),pnlpt->error_message,
               "QCDM fast pkpole basis z=%.12g is not one of the cached z_pk values",z);
    class_test(pnlpt->qcdm_fast_z_ready==NULL || pnlpt->qcdm_fast_z_ready[iz]!=_TRUE_,
               pnlpt->error_message,"QCDM fast pkpole basis cache is not ready at z=%.12g",z);

    int Nmax=pnlpt->qcdm_fast_nmax;
    const double *kdisc=pnlpt->qcdm_fast_kdisc;
    for (int ik=0;ik<k_size;ik++) {
        double kval=k_h[ik]*pba->h;
        class_test(kval<kdisc[0] || kval>kdisc[Nmax-1],pnlpt->error_message,
                   "QCDM fast pkpole basis k=%.12g h/Mpc lies outside cached grid [%.12g, %.12g] h/Mpc",
                   k_h[ik],kdisc[0]/pba->h,kdisc[Nmax-1]/pba->h);
    }

    class_test(qcdm_tf_ir_cache.nmax!=Nmax || qcdm_tf_ir_cache.nz!=pnlpt->z_pk_num ||
               qcdm_tf_ir_cache.ready==NULL || qcdm_tf_ir_cache.ready[iz]!=_TRUE_,
               pnlpt->error_message,
               "QCDM fast anisotropic-IR cache is not ready for native basis at z=%.12g",z);

    double *tmp0=NULL,*tmp2=NULL,*tmp4=NULL;
    class_alloc(tmp0,(size_t)QCDM_TF_BASIS_NCOEFF*Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(tmp2,(size_t)QCDM_TF_BASIS_NCOEFF*Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(tmp4,(size_t)QCDM_TF_BASIS_NCOEFF*Nmax*sizeof(double),pnlpt->error_message);

    double oldp[QCDM_TF_FAST_NPARAM];
    qcdm_tf_fast_get_params(pnlpt,oldp);
    int old_verbose=pnlpt->nonlinear_pt_verbose;
    qcdm_tf_fast_set_params(pnlpt,tracer_params);
    pnlpt->nonlinear_pt_verbose=0;

    double tau=0.0;
    int status=background_tau_of_z(pba,z,&tau);
    if (status==_SUCCESS_) {
        const double hratio=use_ap_override?hratio_override:pnlpt->hratio_array[iz];
        const double Dratio=use_ap_override?Dratio_override:pnlpt->Dratio_array[iz];
        const double *Pnw=qcdm_tf_ir_cache.Pnw+(size_t)iz*Nmax;
        const double *Pw =qcdm_tf_ir_cache.Pw +(size_t)iz*Nmax;
        status=qcdm_tf_exact_rsd_basis_ir(pba,pnlpt,tau,Nmax,kdisc,Pnw,Pw,
                                          &qcdm_tf_ir_cache.state[iz],
                                          hratio,Dratio,3.0*pba->h,
                                          tmp0,tmp2,tmp4);
    }

    qcdm_tf_fast_set_params(pnlpt,oldp);
    pnlpt->nonlinear_pt_verbose=old_verbose;

    if (status==_FAILURE_) {
        free(tmp0);free(tmp2);free(tmp4);
        return _FAILURE_;
    }

    double h3=pba->h*pba->h*pba->h;
    for (int c=0;c<QCDM_TF_BASIS_NCOEFF;c++) {
        const double *s0=tmp0+(size_t)c*Nmax;
        const double *s2=tmp2+(size_t)c*Nmax;
        const double *s4=tmp4+(size_t)c*Nmax;
        double *d0=basis0_h+(size_t)c*k_size;
        double *d2=basis2_h+(size_t)c*k_size;
        double *d4=basis4_h+(size_t)c*k_size;
        for (int ik=0;ik<k_size;ik++) {
            double kval=k_h[ik]*pba->h;
            d0[ik]=qcdm_qscf_interp_linear(Nmax,kdisc,s0,kval)*h3;
            d2[ik]=qcdm_qscf_interp_linear(Nmax,kdisc,s2,kval)*h3;
            d4[ik]=qcdm_qscf_interp_linear(Nmax,kdisc,s4,kval)*h3;
        }
    }

    free(tmp0);free(tmp2);free(tmp4);
    return _SUCCESS_;
}

/* DW audit: compute ONLY the unresummed real-space tracer P22 with
 * exactly the q-panel and transformed-p GL quadrature of the RSD integrator.
 * The density kernel, input spectrum, k/q limits and UV damping are unchanged.
 * No P13 or PNG contribution is evaluated or altered by this helper.
 */
static int qcdm_tf_realspace_p22_split_p(struct background *pba,
                                          struct nonlinear_pt *pnlpt,
                                          double tau,
                                          int Nmax,
                                          const double *kdisc,
                                          const double *Pbin,
                                          double kk,
                                          double qmin,
                                          double qmax,
                                          double *result) {
    int Nq=pnlpt->qcdm_tf_Nq;
    int Nx=pnlpt->qcdm_tf_Nx;
    class_test((Nq<2)||(Nx<1)||(kk<=0.0)||(qmax<=qmin),
               pnlpt->error_message,"Invalid split-p real-space P22 quadrature parameters");

    double *pnode,*pweight,*qgl,*qgw;
    class_alloc(pnode,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(pweight,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(qgl,Nq*sizeof(double),pnlpt->error_message);
    class_alloc(qgw,Nq*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nx,pnode,pweight,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);

    int npanel=1;
    int nleft=Nq;
    double ylo[2]={log(qmin),0.0};
    double yhi[2]={log(qmax),0.0};
    if ((kk>qmin)&&(kk<qmax)) {
        double ys=log(kk);
        double Lleft=ys-ylo[0],Lright=yhi[0]-ys;
        int min_side=(Nq>=8)?4:1;
        nleft=(int)floor((double)Nq*Lleft/(Lleft+Lright)+0.5);
        if (nleft<min_side) nleft=min_side;
        if (nleft>Nq-min_side) nleft=Nq-min_side;
        yhi[0]=ys;
        ylo[1]=ys;
        yhi[1]=log(qmax);
        npanel=2;
    }

    double I22=0.0;
    for (int panel=0;panel<npanel;panel++) {
        int n=(panel==0)?nleft:(Nq-nleft);
        double mid=0.5*(ylo[panel]+yhi[panel]);
        double half=0.5*(yhi[panel]-ylo[panel]);
        class_call(qcdm_tf_gauss_legendre_grid(n,qgl,qgw,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        for (int iq=0;iq<n;iq++) {
            double q=exp(mid+half*qgl[iq]);
            double dq=q*half*qgw[iq];
            double Pq=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,q);
            double pmin=fabs(kk-q),pmax=kk+q;
            double pmid=0.5*(pmin+pmax);
            double phalf=0.5*(pmax-pmin);
            for (int ix=0;ix<Nx;ix++) {
                double pnorm=pmid+phalf*pnode[ix];
                if ((pnorm<=kdisc[0])||(pnorm>=kdisc[Nmax-1])) continue;
                double dp=phalf*pweight[ix];
                double x=(kk*kk+q*q-pnorm*pnorm)/(2.0*kk*q);
                if (x>1.0) x=1.0;
                if (x< -1.0) x= -1.0;
                double sq=sqrt(MAX(1.0-x*x,0.0));
                qcdm_qscf_kvec3 qv={q*x,q*sq,0.0};
                qcdm_qscf_kvec3 pv={kk-qv.x,-qv.y,0.0};
                qcdm_qscf_kernel2_result K2;
                class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qv,pv,&K2,
                                                            pnlpt->error_message),
                           pnlpt->error_message,pnlpt->error_message);
                double D2=qcdm_tf_D2(pnlpt,&K2,qv,pv);
                double Pp=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,pnorm);
                I22+=(q*pnorm/kk)*dq*dp*D2*D2*Pq*Pp;
            }
        }
    }
    *result=2.0*I22/(4.0*_PI_*_PI_);
    free(pnode);free(pweight);free(qgl);free(qgw);
    return _SUCCESS_;
}

static int qcdm_tf_exact_realspace_tracer(struct background *pba,
                                          struct nonlinear_pt *pnlpt,
                                          double tau,
                                          int Nmax,
                                          const double *kdisc,
                                          const double *Pbin,
                                          /* DW: primordial Phi power and
                                           * Phi-to-clustering-density transfer
                                           * for exact PNG P12 templates. */
                                          const double *Pphi,
                                          const double *Mphi,
                                          double cutoff,
                                          double *Ptree,
                                          double *P22,
                                          double *P13,
                                          double *P12eq,
                                          double *P12ortho,
                                          double *Pct,
                                          double *Pstoch,
                                          double *Ptotal) {
    int Nq=pnlpt->qcdm_tf_Nq;
    int Nx=pnlpt->qcdm_tf_Nx;
    double qmin=MAX(kdisc[0],pnlpt->qcdm_loop_q_min_h*pba->h);
    double qmax=MIN(kdisc[Nmax-1],pnlpt->qcdm_loop_q_max_h*pba->h);
    double kmax=pnlpt->qcdm_loop_k_max_h*pba->h;
    class_test(qmax<=qmin,pnlpt->error_message,
               "Invalid QCDM real-tracer loop interval: qmin=%e qmax=%e",qmin,qmax);
    double *xnode,*xweight; class_alloc(xnode,Nx*sizeof(double),pnlpt->error_message); class_alloc(xweight,Nx*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nx,xnode,xweight,pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
    for (int ik=0;ik<Nmax;ik++) {
        double kk=kdisc[ik];
        double Pk=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,kk);
        qcdm_qscf_kernel1_result K1k;
        class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,kk,&K1k,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        double D1=qcdm_tf_D1(pnlpt,&K1k);
        Ptree[ik]=D1*D1*Pk;
        /* DW: initialize Gaussian and exact-time PNG tracer loop pieces independently. */
        P22[ik]=P13[ik]=P12eq[ik]=P12ortho[ik]=0.0;
        Pct[ik]=-kk*kk*(pnlpt->qcdm_tf_ct0*Pk
                       +pnlpt->qcdm_tf_ct_r0*K1k.F1_r_red*Pk);
        Pstoch[ik]=pnlpt->qcdm_tf_stoch0+pnlpt->qcdm_tf_stoch2*kk*kk;
        if (kk<=kmax) {
            /* DW: accumulate unit equilateral/orthogonal P12 integrals with the exact D2 kernel. */
            double I13=0.0,I12eq=0.0,I12ortho=0.0;
            int has_png=((pnlpt->fNL_equil_ortho_switch==fNL_equil_ortho_yes)&&(Pphi!=NULL)&&(Mphi!=NULL));
            double Mk=has_png?Mphi[ik]:0.0;
            double Pkphi=has_png?Pphi[ik]:0.0;
            for (int iq=0;iq<Nq;iq++) {
                double t=((double)iq+0.5)/((double)Nq);
                double lnq=log(qmin)+t*(log(qmax)-log(qmin));
                double q=exp(lnq);
                double dq=q*(log(qmax)-log(qmin))/((double)Nq);
                double Pq=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,q);
                for (int ix=0;ix<Nx;ix++) {
                    double x=xnode[ix]; double dx=xweight[ix];
                    double pnorm=sqrt(MAX(kk*kk+q*q-2.0*kk*q*x,0.0));
                    if (has_png && (pnorm>kdisc[0]) && (pnorm<kdisc[Nmax-1])) {
                        double sq=sqrt(MAX(1.0-x*x,0.0));
                        qcdm_qscf_kvec3 qv={q*x,q*sq,0.0};
                        qcdm_qscf_kvec3 kv={kk,0.0,0.0};
                        qcdm_qscf_kvec3 pv={kk-qv.x,-qv.y,0.0};
                        qcdm_qscf_kernel2_result K2;
                        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qv,pv,&K2,pnlpt->error_message),
                                   pnlpt->error_message,pnlpt->error_message);
                        double D2=qcdm_tf_D2(pnlpt,&K2,qv,pv);
                        /* P22 is evaluated separately with the split-q, transformed-p GL rule. */
                        /* DW: add exact-time real-space tracer P12 using the
                         * same two-fluid D2 bias kernel as the Gaussian loop. */
                        if (has_png) {
                            double Pqphi=qcdm_qscf_interp_linear(Nmax,kdisc,Pphi,q);
                            double Ppphi=qcdm_qscf_interp_linear(Nmax,kdisc,Pphi,pnorm);
                            double Mq=qcdm_qscf_interp_linear(Nmax,kdisc,Mphi,q);
                            double Mp=qcdm_qscf_interp_linear(Nmax,kdisc,Mphi,pnorm);
                            double transfer=Mk*Mq*Mp;
                            double measure=q*q*dq*dx;
                            I12eq+=measure*D2*transfer*qcdm_png_bphi_unit(0,Pkphi,Pqphi,Ppphi);
                            I12ortho+=measure*D2*transfer*qcdm_png_bphi_unit(1,Pkphi,Pqphi,Ppphi);
                        }
                    }
                    qcdm_qscf_kvec3 pvec[3];
                    double sq=sqrt(MAX(1.0-x*x,0.0));
                    pvec[0]=(qcdm_qscf_kvec3){kk,0.0,0.0};
                    pvec[1]=(qcdm_qscf_kvec3){q*x,q*sq,0.0};
                    pvec[2]=(qcdm_qscf_kvec3){-q*x,-q*sq,0.0};
                    qcdm_qscf_kernel3_result K3;
                    qcdm_qscf_kernel2_result pair[3];
                    class_call(qcdm_qscf_qs_solve_F3_collapsed_full(pba,tau,kk,q,x,&K3,pnlpt->error_message),
                               pnlpt->error_message,pnlpt->error_message);
                    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pvec[1],pvec[2],&pair[0],pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
                    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pvec[0],pvec[2],&pair[1],pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
                    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pvec[0],pvec[1],&pair[2],pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
                    double D3=qcdm_tf_D3_collapsed(pnlpt,&K3,pvec,pair);
                    I13 += q*q*dq*dx*D3*Pq;
                }
            }
            double kc=kk/cutoff;
            double damp=exp(-pow(kc,6.0));
            class_call(qcdm_tf_realspace_p22_split_p(pba,pnlpt,tau,Nmax,kdisc,Pbin,
                                                       kk,qmin,qmax,&P22[ik]),
                       pnlpt->error_message,pnlpt->error_message);
            P22[ik] *= damp;
            P13[ik]=6.0*D1*Pk*I13/(4.0*_PI_*_PI_)*damp;
            /* DW: normalize the tracer P12 cross spectrum as 2<D1,D2>. */
            P12eq[ik]=2.0*D1*I12eq/(4.0*_PI_*_PI_)*damp;
            P12ortho[ik]=2.0*D1*I12ortho/(4.0*_PI_*_PI_)*damp;
        }
        /* DW: add user amplitudes only after preserving both unit PNG templates. */
        Ptotal[ik]=Ptree[ik]+P22[ik]+P13[ik]
                  +pnlpt->qcdm_fNL_equilateral*P12eq[ik]
                  +pnlpt->qcdm_fNL_orthogonal*P12ortho[ik]
                  +Pct[ik]+Pstoch[ik];
        class_test((qcdm_qscf_finite_double(Ptotal[ik])==0),pnlpt->error_message,
                   "Non-finite QCDM real tracer spectrum at k=%e",kk);
    }
    free(xnode);free(xweight);
    return _SUCCESS_;
}

/* Proper NLO IR resummation for the exact-QS real-space matter loop.
 * This is used by the matter diagnostics/lensing side of nonlinear_pt so
 * IR=yes does not mix an IR-resummed tree with loops evaluated on an
 * isotropically pre-damped input spectrum. */
/* DW speed stage 3: one external-k slice of the proper-IR real-space matter
 * loop.  All quadrature inputs are read-only and each worker writes only
 * P22/P13/P1loop[ik].  In OpenMP mode kernel-cache misses are solved under
 * qcdm_qscf_cache_parallel_nostore, so the process-global F1/F2/F3 hash
 * tables are never mutated concurrently. */
static int qcdm_qscf_qs_exact_realspace_ir_one_k_parallel(
    struct background *pba,
    struct nonlinear_pt *pnlpt,
    double tau,
    int Nmax,
    const double *kdisc,
    const double *Pnw,
    const double *Pw,
    const qcdm_tf_ir_state *ir_state,
    double cutoff,
    int Nq,
    int Nmu,
    double qmin,
    double qmax,
    double kmax,
    const double *munode,
    const double *muweight,
    int ik,
    double *P22,
    double *P13,
    double *P1loop,
    ErrorMsg errmsg) {

    double kk=kdisc[ik];
    if (kk>kmax) {
        P22[ik]=P13[ik]=P1loop[ik]=0.0;
        return _SUCCESS_;
    }

    double I22nw=0.0,I22w=0.0,I13nw=0.0,I13w=0.0;
    double Pknw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,kk);
    double Pkw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,kk);

    for (int iq=0;iq<Nq;iq++) {
        double xq=((double)iq+0.5)/((double)Nq);
        double lnq=log(qmin)+xq*(log(qmax)-log(qmin));
        double q=exp(lnq);
        double dq=q*(log(qmax)-log(qmin))/((double)Nq);
        double Pqnw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,q);
        double Pqw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,q);

        for (int imu=0;imu<Nmu;imu++) {
            double mu=munode[imu],dmu=muweight[imu];
            double p=sqrt(MAX(kk*kk+q*q-2.0*kk*q*mu,0.0));
            if ((p>kdisc[0])&&(p<kdisc[Nmax-1])) {
                double Ppnw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,p);
                double Ppw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,p);
                double mu_qp=(q*p>0.0)?(kk*mu-q)/p:0.0;
                if (mu_qp>1.0) mu_qp=1.0;
                if (mu_qp<-1.0) mu_qp=-1.0;
                qcdm_qscf_kernel2_result K2;
                class_call(qcdm_qscf_qs_solve_F2_full(pba,tau,q,p,mu_qp,&K2,errmsg),
                           errmsg,errmsg);
                double measure=q*q*dq*dmu*K2.F2_m_red*K2.F2_m_red;
                I22nw += measure*Pqnw*Ppnw;
                I22w  += measure*(Pqw*Ppnw+Pqnw*Ppw);
            }

            double F3m;
            class_call(qcdm_qscf_qs_solve_F3_collapsed(pba,tau,kk,q,mu,&F3m,errmsg),
                       errmsg,errmsg);
            double measure13=q*q*dq*dmu*F3m;
            I13nw += measure13*Pqnw;
            I13w  += measure13*Pqw;
        }
    }

    double S=kk*kk*ir_state->Sigma_DD;
    double Dir=exp(-S);
    double uv=exp(-pow(kk/cutoff,6.0));
    P22[ik]=2.0*(I22nw+Dir*I22w)/(4.0*_PI_*_PI_)*uv;
    P13[ik]=6.0*(Pknw*I13nw+Dir*(Pkw*I13nw+Pknw*I13w))/(4.0*_PI_*_PI_)*uv;
    P1loop[ik]=P22[ik]+P13[ik];

    class_test((qcdm_qscf_finite_double(P22[ik])==0)||
               (qcdm_qscf_finite_double(P13[ik])==0)||
               (qcdm_qscf_finite_double(P1loop[ik])==0),errmsg,
               "Non-finite QCDM real-space IR loop at k=%e: P22=%e P13=%e",
               kk,P22[ik],P13[ik]);
    return _SUCCESS_;
}

/* Proper NLO IR resummation for the exact-QS real-space matter loop.
 * This is used by the matter diagnostics/lensing side of nonlinear_pt so
 * IR=yes does not mix an IR-resummed tree with loops evaluated on an
 * isotropically pre-damped input spectrum.
 *
 * DW speed stage 3: with OMP>1 the external-k loop is parallel.  Before the
 * region starts, all RK background-stage grids needed by F1/F2/F3 are rebuilt
 * serially for this tau.  Kernel-hash misses are computed privately without
 * insertion, exactly as in the validated stage-2b RSD path. */
static int qcdm_qscf_qs_exact_realspace_loops_ir(struct background *pba,
                                                  struct nonlinear_pt *pnlpt,
                                                  double tau,
                                                  int Nmax,
                                                  const double *kdisc,
                                                  const double *Pnw,
                                                  const double *Pw,
                                                  const qcdm_tf_ir_state *ir_state,
                                                  double cutoff,
                                                  double *P22,
                                                  double *P13,
                                                  double *P1loop) {
    int Nq=pnlpt->qcdm_tf_Nq;
    int Nmu=pnlpt->qcdm_tf_Nmu;
    double qmin=MAX(kdisc[0],pnlpt->qcdm_loop_q_min_h*pba->h);
    double qmax=MIN(kdisc[Nmax-1],pnlpt->qcdm_loop_q_max_h*pba->h);
    double kmax=pnlpt->qcdm_loop_k_max_h*pba->h;
    class_test((ir_state==NULL)||(ir_state->enabled!=_TRUE_),pnlpt->error_message,
               "QCDM real-space IR loop requested without IR state");
    class_test((Nq<=0)||(Nmu<=0)||(qmax<=qmin),pnlpt->error_message,
               "Invalid QCDM real-space IR loop grid: Nq=%d Nmu=%d q=[%e,%e]",Nq,Nmu,qmin,qmax);

    double *munode,*muweight;
    class_alloc(munode,Nmu*sizeof(double),pnlpt->error_message);
    class_alloc(muweight,Nmu*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nmu,munode,muweight,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);

    int real_use_nostore=(omp_get_max_threads()>1)?_TRUE_:_FALSE_;
    if (real_use_nostore==_TRUE_) {
        class_call(qcdm_qscf_qs_prewarm_rk_for_tau(pba,tau,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
    }

    int real_omp_failed=0;
    char real_omp_error[4096]; real_omp_error[0]='\0';
    qcdm_qscf_cache_parallel_nostore=real_use_nostore;

    #pragma omp parallel for schedule(dynamic,1) if(real_use_nostore)
    for (int ik=0;ik<Nmax;ik++) {
        char local_errmsg[2048]; local_errmsg[0]='\0';
        int st=qcdm_qscf_qs_exact_realspace_ir_one_k_parallel(
            pba,pnlpt,tau,Nmax,kdisc,Pnw,Pw,ir_state,cutoff,
            Nq,Nmu,qmin,qmax,kmax,munode,muweight,ik,P22,P13,P1loop,
            local_errmsg);
        if (st!=_SUCCESS_) {
            #pragma omp critical(qcdm_real_error)
            {
                if (real_omp_failed==0) {
                    real_omp_failed=1;
                    snprintf(real_omp_error,sizeof(real_omp_error),
                             "initialization real-space OpenMP k-index %d failed: %s",
                             ik,local_errmsg);
                }
            }
        }
    }

    qcdm_qscf_cache_parallel_nostore=_FALSE_;
    class_test(real_omp_failed!=0,pnlpt->error_message,"%s",real_omp_error);

    free(munode);free(muweight);
    return _SUCCESS_;
}

/* IR-resummed exact real-space tracer diagnostic.  Gaussian IC only; the
 * DESI c-c' baseline fixes fNL=0. */
static int qcdm_tf_exact_realspace_tracer_ir(struct background *pba,
                                             struct nonlinear_pt *pnlpt,
                                             double tau,
                                             int Nmax,
                                             const double *kdisc,
                                             const double *Pnw,
                                             const double *Pw,
                                             const qcdm_tf_ir_state *ir_state,
                                             double cutoff,
                                             double *Ptree,
                                             double *P22,
                                             double *P13,
                                             double *P12eq,
                                             double *P12ortho,
                                             double *Pct,
                                             double *Pstoch,
                                             double *Ptotal) {
    class_test(fabs(pnlpt->qcdm_fNL_equilateral)>0.0 || fabs(pnlpt->qcdm_fNL_orthogonal)>0.0,
               pnlpt->error_message,
               "QCDM real-space proper IR tracer diagnostic currently requires fNL=0");
    int Nq=pnlpt->qcdm_tf_Nq;
    int Nx=pnlpt->qcdm_tf_Nx;
    double qmin=MAX(kdisc[0],pnlpt->qcdm_loop_q_min_h*pba->h);
    double qmax=MIN(kdisc[Nmax-1],pnlpt->qcdm_loop_q_max_h*pba->h);
    double kmax=pnlpt->qcdm_loop_k_max_h*pba->h;
    double *xnode,*xweight;
    class_alloc(xnode,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(xweight,Nx*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nx,xnode,xweight,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);

    for (int ik=0;ik<Nmax;ik++) {
        double kk=kdisc[ik];
        double Pknw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,kk);
        double Pkw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,kk);
        double S=kk*kk*ir_state->Sigma_DD;
        double Dir=exp(-S);
        double Plo=Pknw+Dir*Pkw;
        double Ptree_ir=Pknw+Dir*(1.0+S)*Pkw;

        qcdm_qscf_kernel1_result K1k;
        class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,kk,&K1k,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        double D1=qcdm_tf_D1(pnlpt,&K1k);
        Ptree[ik]=D1*D1*Ptree_ir;
        P22[ik]=P13[ik]=P12eq[ik]=P12ortho[ik]=0.0;
        Pct[ik]=-kk*kk*(pnlpt->qcdm_tf_ct0*Plo+pnlpt->qcdm_tf_ct_r0*K1k.F1_r_red*Plo);
        Pstoch[ik]=pnlpt->qcdm_tf_stoch0+pnlpt->qcdm_tf_stoch2*kk*kk;

        if (kk<=kmax) {
            double I22nw=0.0,I22w=0.0,I13nw=0.0,I13w=0.0;
            for (int iq=0;iq<Nq;iq++) {
                double t=((double)iq+0.5)/((double)Nq);
                double lnq=log(qmin)+t*(log(qmax)-log(qmin));
                double q=exp(lnq);
                double dq=q*(log(qmax)-log(qmin))/((double)Nq);
                double Pqnw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,q);
                double Pqw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,q);
                for (int ix=0;ix<Nx;ix++) {
                    double x=xnode[ix],dx=xweight[ix];
                    double pnorm=sqrt(MAX(kk*kk+q*q-2.0*kk*q*x,0.0));
                    if ((pnorm>kdisc[0])&&(pnorm<kdisc[Nmax-1])) {
                        double sq=sqrt(MAX(1.0-x*x,0.0));
                        qcdm_qscf_kvec3 qv={q*x,q*sq,0.0};
                        qcdm_qscf_kvec3 pv={kk-qv.x,-qv.y,0.0};
                        qcdm_qscf_kernel2_result K2;
                        class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qv,pv,&K2,pnlpt->error_message),
                                   pnlpt->error_message,pnlpt->error_message);
                        double D2=qcdm_tf_D2(pnlpt,&K2,qv,pv);
                        double Ppnw=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,pnorm);
                        double Ppw =qcdm_qscf_interp_linear(Nmax,kdisc,Pw ,pnorm);
                        double measure=q*q*dq*dx*D2*D2;
                        I22nw += measure*Pqnw*Ppnw;
                        I22w  += measure*(Pqw*Ppnw+Pqnw*Ppw);
                    }
                    double sq=sqrt(MAX(1.0-x*x,0.0));
                    qcdm_qscf_kvec3 pvec[3]={{kk,0.0,0.0},
                                             {q*x,q*sq,0.0},
                                             {-q*x,-q*sq,0.0}};
                    qcdm_qscf_kernel3_result K3;
                    qcdm_qscf_kernel2_result pair[3];
                    class_call(qcdm_qscf_qs_solve_F3_collapsed_full(pba,tau,kk,q,x,&K3,pnlpt->error_message),
                               pnlpt->error_message,pnlpt->error_message);
                    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pvec[1],pvec[2],&pair[0],pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
                    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pvec[0],pvec[2],&pair[1],pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
                    class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,pvec[0],pvec[1],&pair[2],pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
                    double D3=qcdm_tf_D3_collapsed(pnlpt,&K3,pvec,pair);
                    double measure13=q*q*dq*dx*D3;
                    I13nw += measure13*Pqnw;
                    I13w  += measure13*Pqw;
                }
            }
            double uv=exp(-pow(kk/cutoff,6.0));
            P22[ik]=2.0*(I22nw+Dir*I22w)/(4.0*_PI_*_PI_)*uv;
            P13[ik]=6.0*D1*(Pknw*I13nw+Dir*(Pkw*I13nw+Pknw*I13w))/(4.0*_PI_*_PI_)*uv;
        }
        Ptotal[ik]=Ptree[ik]+P22[ik]+P13[ik]+Pct[ik]+Pstoch[ik];
        class_test(qcdm_qscf_finite_double(Ptotal[ik])==0,pnlpt->error_message,
                   "Non-finite QCDM IR real tracer spectrum at k=%e",kk);
    }
    free(xnode);free(xweight);
    return _SUCCESS_;
}


static int qcdm_qscf_write_real_tracer_at_tau(struct background *pba,
                                               struct nonlinear_pt *pnlpt,
                                               double tau,
                                               int Nmax,
                                               const double *kdisc,
                                               const double *Pbin,
                                               const double *Pnw,
                                               const double *Pw,
                                               const qcdm_tf_ir_state *ir_state,
                                               /* DW: pass primordial Phi power and its density transfer to the tracer writer. */
                                               const double *Pphi,
                                               const double *Mphi,
                                               double cutoff) {
    if ((pnlpt->bias!=bias_yes)||(pnlpt->qcdm_write_tracer_real!=_TRUE_)) return _SUCCESS_;
    /* DW: retain Gaussian, PNG, EFT, and stochastic tracer pieces in separate buffers. */
    double *Ptree,*P22,*P13,*P12eq,*P12ortho,*Pct,*Pstoch,*Ptotal;
    class_alloc(Ptree,Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(P22,Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(P13,Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(P12eq,Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(P12ortho,Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(Pct,Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(Pstoch,Nmax*sizeof(double),pnlpt->error_message);
    class_alloc(Ptotal,Nmax*sizeof(double),pnlpt->error_message);
    if ((ir_state!=NULL)&&(ir_state->enabled==_TRUE_)) {
        class_call(qcdm_tf_exact_realspace_tracer_ir(pba,pnlpt,tau,Nmax,kdisc,Pnw,Pw,ir_state,cutoff,
                                                      Ptree,P22,P13,P12eq,P12ortho,Pct,Pstoch,Ptotal),
                   pnlpt->error_message,pnlpt->error_message);
    }
    else {
        class_call(qcdm_tf_exact_realspace_tracer(pba,pnlpt,tau,Nmax,kdisc,Pbin,Pphi,Mphi,cutoff,
                                                   Ptree,P22,P13,P12eq,P12ortho,Pct,Pstoch,Ptotal),
                   pnlpt->error_message,pnlpt->error_message);
    }
    double z;
    class_call(background_z_of_tau(pba,tau,&z),pba->error_message,pnlpt->error_message);
    int iz_best=0;
    double dz_best=fabs(z-pnlpt->z_pk[0]);
    for (int iz=1;iz<pnlpt->z_pk_num;iz++) {
        double dz=fabs(z-pnlpt->z_pk[iz]);
        if (dz<dz_best) {dz_best=dz;iz_best=iz;}
    }
    char zsuffix[16];
    if (pnlpt->z_pk_num==1) zsuffix[0]='\0';
    else class_sprintf(zsuffix,"z%d_",iz_best+1);
    FileName fname;
    class_sprintf(fname,"%s%sqcdm_pt_tracer_real.dat",pnlpt->qcdm_pt_output_root,zsuffix);
    FILE *fp=fopen(fname,"w");
    class_test(fp==NULL,pnlpt->error_message,"Could not open %s for writing",fname);
    double h3=pba->h*pba->h*pba->h;
    fprintf(fp,"# QCDM/QSCF exact-time two-fluid real-space tracer spectrum at z = %.12g\n",z);
    fprintf(fp,"# DW: P12eq_unit and P12ortho_unit are unit-fNL exact-time templates; PPNG uses fNL_equilateral=%.12g and fNL_orthogonal=%.12g.\n",
            pnlpt->qcdm_fNL_equilateral,pnlpt->qcdm_fNL_orthogonal);
    fprintf(fp,"# k[h/Mpc] Ptree P22 P13 P12eq_unit P12ortho_unit PPNG Pct Pstoch Ptotal  [all P in (Mpc/h)^3]\n");
    for (int ik=0;ik<Nmax;ik++)
        fprintf(fp,"%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n",
                kdisc[ik]/pba->h,Ptree[ik]*h3,P22[ik]*h3,P13[ik]*h3,
                P12eq[ik]*h3,P12ortho[ik]*h3,
                (pnlpt->qcdm_fNL_equilateral*P12eq[ik]+pnlpt->qcdm_fNL_orthogonal*P12ortho[ik])*h3,
                Pct[ik]*h3,Pstoch[ik]*h3,Ptotal[ik]*h3);
    fclose(fp);
    free(Ptree);free(P22);free(P13);free(P12eq);free(P12ortho);free(Pct);free(Pstoch);free(Ptotal);
    if (pnlpt->nonlinear_pt_verbose>0)
        printf("DW/QCDM-QSCF: wrote exact-time real-space tracer spectrum to %s.\n",fname);
    return _SUCCESS_;
}


/* DW: write a science-audit-friendly matter EFTofLSS decomposition on the
 * native loop grid. This separates the raw CLASS linear spectrum, the
 * IR-resummed tree contribution, P22, P13, the fitted matter counterterm,
 * the conservation-compatible k^4 matter stochastic term, and cumulative
 * sums. Coefficients with the suffix _h are interpreted in h-based units. */
/* DW: compute the fraction of late-time matter carried by the nonlinear
 * clustering sector represented by the exact kernels.  In the intended
 * c-c' setup this is baryons plus QCDM; ordinary CDM is also included if
 * present. Massive ncdm remains linear and is excluded from the numerator. */
static int qcdm_qscf_clustering_fraction_at_tau(struct background *pba,
                                                double tau,
                                                double *fcluster,
                                                ErrorMsg errmsg) {
    double *pvecback;
    int last_index=0;
    class_alloc(pvecback,pba->bg_size*sizeof(double),errmsg);
    class_call(background_at_tau(pba,tau,long_info,inter_normal,&last_index,pvecback),
               pba->error_message,errmsg);
    double rho_cluster=pvecback[pba->index_bg_rho_b];
    if (pba->has_cdm==_TRUE_) rho_cluster+=pvecback[pba->index_bg_rho_cdm];
    if (pba->has_qcdm==_TRUE_) rho_cluster+=pvecback[pba->index_bg_rho_qcdm];
    double rho_m=rho_cluster;
    if (pba->has_ncdm==_TRUE_) {
        for (int n=0;n<pba->N_ncdm;n++)
            rho_m+=pvecback[pba->index_bg_rho_ncdm1+n];
    }
    free(pvecback);
    class_test((rho_cluster<=0.0)||(rho_m<=0.0),errmsg,
               "Invalid clustering/total matter densities at tau=%e: rho_cluster=%e rho_m=%e",
               tau,rho_cluster,rho_m);
    *fcluster=rho_cluster/rho_m;
    return _SUCCESS_;
}

static int qcdm_qscf_write_matter_breakdown_at_tau(
    struct background *pba,
    struct nonlinear_pt *pnlpt,
    double tau,
    int Nmax,
    const double *kdisc,
    const double *Plinear_cluster_raw,
    const double *Plinear_total_raw,
    const double *Ptree_cluster_ir,
    const double *P22,
    const double *P13,
    const double *P12_equilateral,
    const double *P12_orthogonal,
    const double *Pct_basis) {

    if (pnlpt->qcdm_write_matter_breakdown != _TRUE_) return _SUCCESS_;

    double z;
    class_call(background_z_of_tau(pba,tau,&z),pba->error_message,pnlpt->error_message);
    int iz_best=0;
    double dz_best=fabs(z-pnlpt->z_pk[0]);
    for (int iz=1;iz<pnlpt->z_pk_num;iz++) {
        double dz=fabs(z-pnlpt->z_pk[iz]);
        if (dz<dz_best) {dz_best=dz;iz_best=iz;}
    }

    char zsuffix[16];
    if (pnlpt->z_pk_num==1) zsuffix[0]='\0';
    else class_sprintf(zsuffix,"z%d_",iz_best+1);

    FileName fname;
    class_sprintf(fname,"%s%sqcdm_pt_matter_breakdown.dat",pnlpt->qcdm_pt_output_root,zsuffix);
    FILE *fp=fopen(fname,"w");
    class_test(fp==NULL,pnlpt->error_message,"Could not open %s for writing",fname);

    const double h=pba->h;
    const double h2=h*h;
    const double h3=h2*h;
    double fcluster=1.0;
    class_call(qcdm_qscf_clustering_fraction_at_tau(pba,tau,&fcluster,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    double fcluster2=fcluster*fcluster;
    fprintf(fp,"# QCDM/QSCF exact-time matter EFTofLSS decomposition at z = %.12g\n",z);
    fprintf(fp,"# DW: cb=%s. The exact kernels evolve the clustering sector; f_cluster=%.12g.\n",
            pnlpt->cb==_TRUE_?"yes":"no",fcluster);
    fprintf(fp,"# DW: For cb=yes, total-matter nonlinear corrections use the standard hybrid approximation P_m=P_m,L+f_cluster^2(P_cluster,NL-P_cluster,L), while ncdm remains linear.\n");
    fprintf(fp,"# DW: P_ct,m = -qcdm_matter_ct_k2_h k_h^2 P_mm,bin with qcdm_matter_ct_k2_h = %.12g (Mpc/h)^2.\n",pnlpt->qcdm_matter_ct_k2_h);
    fprintf(fp,"# DW: P_ct,r = -qcdm_matter_ct_r_k2_h k_h^2 P_mr,bin with qcdm_matter_ct_r_k2_h = %.12g (Mpc/h)^2.\n",pnlpt->qcdm_matter_ct_r_k2_h);
    fprintf(fp,"# DW: P_stoch = qcdm_matter_stoch_k4_h k_h^4 with qcdm_matter_stoch_k4_h = %.12g (Mpc/h)^7.\n",pnlpt->qcdm_matter_stoch_k4_h);
    fprintf(fp,"# DW: Matter stochasticity starts at k^4 because total mass and momentum are conserved; set this nuisance coefficient to zero unless it is explicitly fitted.\n");
    fprintf(fp,"# DW: PNG columns are unit-amplitude templates and their amplitude-weighted sum, with fNL_equilateral=%.12g and fNL_orthogonal=%.12g.\n",
            pnlpt->qcdm_fNL_equilateral,pnlpt->qcdm_fNL_orthogonal);
    fprintf(fp,"# Columns:\n");
    fprintf(fp,"# 1 k[h/Mpc] 2 P_cluster_linear 3 P_total_linear 4 P_cluster_tree_IR 5 P_total_tree_IR 6 Delta_IR_total 7 P22_cluster 8 P13_cluster 9 P1loop_total 10 P12eq_unit_cluster 11 P12ortho_unit_cluster 12 PPNG_total 13 P_after_loops_PNG 14 P_ct_mm 15 P_ct_mr 16 P_ct_all 17 P_after_ct 18 P_stoch_k4 19 P_total_EFT 20 frac_IR 21 frac_1loop 22 frac_PNG 23 frac_ct_all 24 frac_stoch 25 total_over_linear\n");
    fprintf(fp,"# All power-spectrum columns are in (Mpc/h)^3. Fractional columns are dimensionless.\n");
    fprintf(fp,"# DW: Rows are restricted to k <= qcdm_loop_k_max_h/Mpc = %.12g, where the direct one-loop calculation is enabled.\n",pnlpt->qcdm_loop_k_max_h);

    for (int ik=0;ik<Nmax;ik++) {
        double k_h=kdisc[ik]/h;
        /* DW: do not present counterterm/stochastic extrapolations beyond the configured direct-loop validity range. */
        if (k_h > pnlpt->qcdm_loop_k_max_h*(1.0+64.0*DBL_EPSILON)) break;
        double Pcluster_raw=Plinear_cluster_raw[ik];
        double Ptotal_raw=Plinear_total_raw[ik];
        double Pcluster_ir=Ptree_cluster_ir[ik];
        double Ptotal_ir=Ptotal_raw+fcluster2*(Pcluster_ir-Pcluster_raw);
        double dIR=Ptotal_ir-Ptotal_raw;
        double P1_cluster=P22[ik]+P13[ik];
        double P1_total=fcluster2*P1_cluster;
        double Ppng_cluster=pnlpt->qcdm_fNL_equilateral*P12_equilateral[ik]
                           +pnlpt->qcdm_fNL_orthogonal*P12_orthogonal[ik];
        double Ppng_total=fcluster2*Ppng_cluster;
        double Pafter1=Ptotal_ir+P1_total+Ppng_total;
        double Pct_mm=-fcluster2*(pnlpt->qcdm_matter_ct_k2_h/h2)*Pct_basis[ik];
        qcdm_qscf_kernel1_result K1red;
        /* DW: obtain P_mr/P_mm from the exact-time reduced relative linear kernel. */
        class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,kdisc[ik],&K1red,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        double Pct_mr=-fcluster2*(pnlpt->qcdm_matter_ct_r_k2_h/h2)*K1red.F1_r_red*Pct_basis[ik];
        double Pct=Pct_mm+Pct_mr;
        double Pafterct=Pafter1+Pct;
        double Pstoch=pnlpt->qcdm_matter_stoch_k4_h*pow(k_h,4.0)/h3;
        double Ptotal=Pafterct+Pstoch;
        double inv=(fabs(Ptotal_raw)>DBL_MIN)?1.0/Ptotal_raw:0.0;

        class_test((qcdm_qscf_finite_double(k_h)==0) ||
                   (qcdm_qscf_finite_double(Pcluster_raw)==0) ||
                   (qcdm_qscf_finite_double(Ptotal_raw)==0) ||
                   (qcdm_qscf_finite_double(Pcluster_ir)==0) ||
                   (qcdm_qscf_finite_double(P22[ik])==0) ||
                   (qcdm_qscf_finite_double(P13[ik])==0) ||
                   (qcdm_qscf_finite_double(Pct_mm)==0) ||
                   (qcdm_qscf_finite_double(Pct_mr)==0) ||
                   (qcdm_qscf_finite_double(Pct)==0) ||
                   (qcdm_qscf_finite_double(Pstoch)==0) ||
                   (qcdm_qscf_finite_double(Ptotal)==0),
                   pnlpt->error_message,
                   "Non-finite QCDM matter breakdown at k index %d: k_h=%e Pcluster=%e PtotalL=%e Ptreecluster=%e P22=%e P13=%e Pct_mm=%e Pct_mr=%e Pstoch=%e Ptotal=%e",
                   ik,k_h,Pcluster_raw,Ptotal_raw,Pcluster_ir,P22[ik],P13[ik],Pct_mm,Pct_mr,Pstoch,Ptotal);

        fprintf(fp,"%.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e %.12e\n",
                k_h,
                Pcluster_raw*h3,
                Ptotal_raw*h3,
                Pcluster_ir*h3,
                Ptotal_ir*h3,
                dIR*h3,
                P22[ik]*h3,
                P13[ik]*h3,
                P1_total*h3,
                P12_equilateral[ik]*h3,
                P12_orthogonal[ik]*h3,
                Ppng_total*h3,
                Pafter1*h3,
                Pct_mm*h3,
                Pct_mr*h3,
                Pct*h3,
                Pafterct*h3,
                Pstoch*h3,
                Ptotal*h3,
                dIR*inv,
                P1_total*inv,
                Ppng_total*inv,
                Pct*inv,
                Pstoch*inv,
                Ptotal*inv);
    }
    fclose(fp);
    if (pnlpt->nonlinear_pt_verbose>0)
        printf("DW: wrote exact-time matter EFTofLSS decomposition to %s.\n",fname);
    return _SUCCESS_;
}

static double qcdm_qscf_interp_linear(int n,const double *x,const double *y,double xp){
    /* Interpolate y linearly in ln(x).  The CLASS-PT k grids are logarithmic
     * and all x values here are positive wavenumbers.  This is robust for
     * signed loop arrays because only the abscissa is logarithmic; y itself is
     * not logged. */
    if(xp<=x[0]) return y[0];
    if(xp>=x[n-1]) return y[n-1];
    int lo=0,hi=n-1;
    while(hi-lo>1){int mid=(lo+hi)/2; if(x[mid]>xp) hi=mid; else lo=mid;}
    double lxlo=log(x[lo]);
    double lxhi=log(x[hi]);
    double lxp=log(xp);
    double t=(lxp-lxlo)/(lxhi-lxlo);
    return y[lo]*(1.0-t)+y[hi]*t;
}

/* Positive-spectrum interpolation for the BAO smooth/wiggle split.  Unlike
 * qcdm_qscf_interp_linear(), this interpolates log P in log k whenever both
 * bracketing values are positive.  The exact-QS branch uses the physical
 * Pdisc array directly so isolated floors in the legacy ln P bookkeeping
 * cannot contaminate the DST no-wiggle extraction. */
static double qcdm_qscf_interp_positive_loglog(int n,
                                                const double *x,
                                                const double *y,
                                                double xp) {
    if (n<2) return MAX(y[0],DBL_MIN);
    int lo,hi;
    if (xp<=x[0]) { lo=0; hi=1; }
    else if (xp>=x[n-1]) { lo=n-2; hi=n-1; }
    else {
        lo=0; hi=n-1;
        while (hi-lo>1) {
            int mid=(lo+hi)/2;
            if (x[mid]>xp) hi=mid; else lo=mid;
        }
    }
    double lx0=log(x[lo]), lx1=log(x[hi]), lxp=log(xp);
    double t=(lxp-lx0)/(lx1-lx0);
    if ((y[lo]>0.0)&&(y[hi]>0.0)) {
        return exp((1.0-t)*log(y[lo])+t*log(y[hi]));
    }
    return MAX((1.0-t)*y[lo]+t*y[hi],DBL_MIN);
}


/* DW: unit-amplitude primordial potential bispectra used by the exact-time
 * PNG P12 integrals.  The convention is the standard equilateral/orthogonal
 * template for the Bardeen potential Phi, with Phi=(3/5)R in matter domination.
 * The returned quantity has units of P_Phi^2 and must be multiplied by the
 * three Phi-to-density transfer functions. */
static double qcdm_png_bphi_unit(int orthogonal,double P1,double P2,double P3) {
    P1=MAX(P1,DBL_MIN); P2=MAX(P2,DBL_MIN); P3=MAX(P3,DBL_MIN);
    double pair=P1*P2+P2*P3+P3*P1;
    double c1=cbrt(P1),c2=cbrt(P2),c3=cbrt(P3);
    double prod23=(c1*c2*c3)*(c1*c2*c3);
    double mixed=
        c1*c2*c2*P3 + c1*c3*c3*P2 +
        c2*c1*c1*P3 + c2*c3*c3*P1 +
        c3*c1*c1*P2 + c3*c2*c2*P1;
    if (orthogonal)
        return 6.0*(-3.0*pair-8.0*prod23+3.0*mixed);
    return 6.0*(-pair-2.0*prod23+mixed);
}

/* DW: compute exact-time unit-fNL matter P12 templates using the model's
 * F2_m kernel rather than the legacy EdS FFTLog M12 matrices:
 *   P12(k)=2 int d^3q/(2pi)^3 F2(q,k-q) B_delta(k,q,|k-q|).
 */
static int qcdm_qscf_qs_exact_png_matter(struct background *pba,
                                         struct nonlinear_pt *pnlpt,
                                         double tau,
                                         int Nmax,
                                         const double *kdisc,
                                         const double *Pphi,
                                         const double *Mphi,
                                         double cutoff,
                                         double *P12eq,
                                         double *P12ortho) {
    int Nq=pnlpt->qcdm_tf_Nq;
    int Nx=pnlpt->qcdm_tf_Nx;
    double qmin=MAX(kdisc[0],pnlpt->qcdm_loop_q_min_h*pba->h);
    double qmax=MIN(kdisc[Nmax-1],pnlpt->qcdm_loop_q_max_h*pba->h);
    double kmax=pnlpt->qcdm_loop_k_max_h*pba->h;
    double *xnode,*xweight;
    class_alloc(xnode,Nx*sizeof(double),pnlpt->error_message);
    class_alloc(xweight,Nx*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nx,xnode,xweight,pnlpt->error_message),
               pnlpt->error_message,pnlpt->error_message);
    for (int ik=0;ik<Nmax;ik++) {
        double k=kdisc[ik];
        P12eq[ik]=0.0; P12ortho[ik]=0.0;
        if (k>kmax) continue;
        double Mk=Mphi[ik],Pkphi=Pphi[ik];
        double Ieq=0.0,Iortho=0.0;
        for (int iq=0;iq<Nq;iq++) {
            double t=((double)iq+0.5)/(double)Nq;
            double q=exp(log(qmin)+t*(log(qmax)-log(qmin)));
            double dq=q*(log(qmax)-log(qmin))/(double)Nq;
            double Pqphi=qcdm_qscf_interp_linear(Nmax,kdisc,Pphi,q);
            double Mq=qcdm_qscf_interp_linear(Nmax,kdisc,Mphi,q);
            for (int ix=0;ix<Nx;ix++) {
                double x=xnode[ix],dx=xweight[ix];
                double p=sqrt(MAX(k*k+q*q-2.0*k*q*x,0.0));
                if ((p<=kdisc[0])||(p>=kdisc[Nmax-1])) continue;
                double sq=sqrt(MAX(1.0-x*x,0.0));
                qcdm_qscf_kvec3 qv={q*x,q*sq,0.0};
                qcdm_qscf_kvec3 pv={k-qv.x,-qv.y,0.0};
                qcdm_qscf_kernel2_result K2;
                class_call(qcdm_qscf_qs_pair_kernel_at_tau(pba,tau,qv,pv,&K2,pnlpt->error_message),
                           pnlpt->error_message,pnlpt->error_message);
                double Ppphi=qcdm_qscf_interp_linear(Nmax,kdisc,Pphi,p);
                double Mp=qcdm_qscf_interp_linear(Nmax,kdisc,Mphi,p);
                double transfer=Mk*Mq*Mp;
                double measure=q*q*dq*dx;
                Ieq+=measure*K2.F2_m_red*transfer*qcdm_png_bphi_unit(0,Pkphi,Pqphi,Ppphi);
                Iortho+=measure*K2.F2_m_red*transfer*qcdm_png_bphi_unit(1,Pkphi,Pqphi,Ppphi);
            }
        }
        double damp=exp(-pow(k/cutoff,6.0));
        P12eq[ik]=2.0*Ieq/(4.0*_PI_*_PI_)*damp;
        P12ortho[ik]=2.0*Iortho/(4.0*_PI_*_PI_)*damp;
    }
    free(xnode); free(xweight);
    return _SUCCESS_;
}

static int qcdm_qscf_qs_exact_realspace_loops(struct background *pba,struct nonlinear_pt *pnlpt,double tau,int Nmax,const double *kdisc,const double *Pbin,double cutoff,double *P22,double *P13,double *P1loop) {
    int Nq=pnlpt->qcdm_tf_Nq;
    int Nmu=pnlpt->qcdm_tf_Nmu;
/* DW: expose and validate direct-loop integration limits in h/Mpc. */
    double qmin=MAX(kdisc[0],pnlpt->qcdm_loop_q_min_h*pba->h);
    double qmax=MIN(kdisc[Nmax-1],pnlpt->qcdm_loop_q_max_h*pba->h);
    double kmax=pnlpt->qcdm_loop_k_max_h*pba->h;

    class_test((Nq <= 0) || (Nmu <= 0) || (Nmax <= 1),
               pnlpt->error_message,
               "Invalid QCDM exact-QS loop grid: Nmax=%d Nq=%d Nmu=%d",
               Nmax,Nq,Nmu);
    class_test((qcdm_qscf_finite_double(qmin) == 0) || (qcdm_qscf_finite_double(qmax) == 0) ||
               (qmin <= 0.0) || (qmax <= qmin) ||
               (qcdm_qscf_finite_double(cutoff) == 0) || (cutoff <= 0.0),
               pnlpt->error_message,
               "Invalid QCDM exact-QS loop domain: qmin=%e qmax=%e cutoff=%e",
               qmin,qmax,cutoff);

    if (pnlpt->nonlinear_pt_verbose > 0) {
/* DW: replace midpoint angular sums with Gauss-Legendre quadrature. */
        printf("DW/QCDM-QSCF: real-space exact-QS loop grid Nmax=%d, Nq=%d, Nmu=%d, q=[%.4e, %.4e], kmax=%.4e 1/Mpc\n",Nmax,Nq,Nmu,qmin,qmax,kmax);
        fflush(stdout);
    }
/* DW: replace midpoint angular sums with Gauss-Legendre quadrature. */
    double *munode,*muweight; class_alloc(munode,Nmu*sizeof(double),pnlpt->error_message); class_alloc(muweight,Nmu*sizeof(double),pnlpt->error_message);
    class_call(qcdm_tf_gauss_legendre_grid(Nmu,munode,muweight,pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);

    for(int ik=0;ik<Nmax;ik++){
        double kk=kdisc[ik], P22k=0.0, P13k=0.0;
/* DW: replace midpoint angular sums with Gauss-Legendre quadrature. */
        if (kk>kmax) {
            P22[ik]=P13[ik]=P1loop[ik]=0.0;
            continue;
        }
        if ((pnlpt->nonlinear_pt_verbose > 1) && ((ik % 10) == 0 || ik == Nmax-1)) {
            printf("DW/QCDM-QSCF: real-space loops k-index %d/%d, k=%.5e\n",ik+1,Nmax,kk);
            fflush(stdout);
        }
        for(int iq=0;iq<Nq;iq++){
            double xq=((double)iq+0.5)/((double)Nq), lnq=log(qmin)+xq*(log(qmax)-log(qmin)), q=exp(lnq), dq=q*(log(qmax)-log(qmin))/((double)Nq), Pq=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,q);
            class_test((qcdm_qscf_finite_double(q) == 0) || (qcdm_qscf_finite_double(dq) == 0) ||
                       (qcdm_qscf_finite_double(Pq) == 0) || (Pq <= 0.0),
                       pnlpt->error_message,
                       "Invalid Pq in QCDM exact-QS loop: ik=%d iq=%d k=%e q=%e dq=%e Pq=%e",
                       ik,iq,kk,q,dq,Pq);
            for(int imu=0;imu<Nmu;imu++){
/* DW: use log-k linear interpolation for signed one-loop quantities to avoid spline overshoot. */
                double mu=munode[imu], dmu=muweight[imu], p=sqrt(MAX(kk*kk+q*q-2.0*kk*q*mu,0.0));
                if((p>kdisc[0])&&(p<kdisc[Nmax-1])){
                    double Pp=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,p), mu_qp=(q*p>0.0)?(kk*mu-q)/p:0.0;
                    class_test((qcdm_qscf_finite_double(p) == 0) || (qcdm_qscf_finite_double(Pp) == 0) || (Pp <= 0.0),
                               pnlpt->error_message,
                               "Invalid Pp in QCDM exact-QS P22 loop: ik=%d iq=%d imu=%d k=%e q=%e p=%e Pp=%e",
                               ik,iq,imu,kk,q,p,Pp);
                    if(mu_qp>1.0) mu_qp=1.0;
                    if(mu_qp<-1.0) mu_qp=-1.0;
                    qcdm_qscf_kernel2_result K2;
                    class_call(qcdm_qscf_qs_solve_F2_full(pba,tau,q,p,mu_qp,&K2,pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
                    class_test(qcdm_qscf_finite_double(K2.F2_m_red) == 0,
                               pnlpt->error_message,
                               "Non-finite F2_m_red in QCDM exact-QS P22 loop: ik=%d iq=%d imu=%d k=%e q=%e p=%e mu=%e",
                               ik,iq,imu,kk,q,p,mu_qp);
                    {
                        double integrand22 = q*q*dq*dmu*K2.F2_m_red*K2.F2_m_red*Pq*Pp;
                        class_test(qcdm_qscf_finite_double(integrand22) == 0,
                                   pnlpt->error_message,
                                   "Non-finite P22 integrand in QCDM exact-QS loop: ik=%d iq=%d imu=%d k=%e q=%e p=%e mu=%e F2=%e Pq=%e Pp=%e",
                                   ik,iq,imu,kk,q,p,mu_qp,K2.F2_m_red,Pq,Pp);
                        P22k+=integrand22;
                    }
                }
                double F3m;
                class_call(qcdm_qscf_qs_solve_F3_collapsed(pba,tau,kk,q,mu,&F3m,pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
                class_test(qcdm_qscf_finite_double(F3m) == 0,
                           pnlpt->error_message,
                           "Non-finite F3_m in QCDM exact-QS P13 loop: ik=%d iq=%d imu=%d k=%e q=%e mu=%e",
                           ik,iq,imu,kk,q,mu);
                {
                    double integrand13 = q*q*dq*dmu*F3m*Pq;
                    class_test(qcdm_qscf_finite_double(integrand13) == 0,
                               pnlpt->error_message,
                               "Non-finite P13 integrand in QCDM exact-QS loop: ik=%d iq=%d imu=%d k=%e q=%e mu=%e F3=%e Pq=%e",
                               ik,iq,imu,kk,q,mu,F3m,Pq);
                    P13k+=integrand13;
                }
            }
        }
        class_test((qcdm_qscf_finite_double(Pbin[ik]) == 0) || (Pbin[ik] <= 0.0),
                   pnlpt->error_message,
                   "Invalid Pbin in QCDM exact-QS loop finalization: ik=%d k=%e Pbin=%e",
                   ik,kk,Pbin[ik]);
        P22[ik]=2.0*P22k/(4.0*_PI_*_PI_);
        P13[ik]=6.0*Pbin[ik]*P13k/(4.0*_PI_*_PI_);
        double kc=kk/cutoff,kc6=kc*kc*kc*kc*kc*kc,damp=exp(-kc6);
        class_test((qcdm_qscf_finite_double(kc) == 0) || (qcdm_qscf_finite_double(kc6) == 0) || (qcdm_qscf_finite_double(damp) == 0),
                   pnlpt->error_message,
                   "Non-finite damping in QCDM exact-QS loop: ik=%d k=%e cutoff=%e kc=%e kc6=%e damp=%e",
                   ik,kk,cutoff,kc,kc6,damp);
        P22[ik]*=damp;
        P13[ik]*=damp;
        class_test((qcdm_qscf_finite_double(P22[ik]) == 0) || (qcdm_qscf_finite_double(P13[ik]) == 0),
                   pnlpt->error_message,
                   "Non-finite QCDM exact-QS P22/P13 after damping: ik=%d k=%e P22=%e P13=%e P22raw=%e P13raw=%e",
                   ik,kk,P22[ik],P13[ik],P22k,P13k);
        P1loop[ik]=P22[ik]+P13[ik];
        class_test(qcdm_qscf_finite_double(P1loop[ik]) == 0,
                   pnlpt->error_message,
                   "Non-finite QCDM exact-QS P1loop: ik=%d k=%e P22=%e P13=%e",
                   ik,kk,P22[ik],P13[ik]);
    }
    if (pnlpt->nonlinear_pt_verbose > 0) {
        qcdm_qscf_qs_cache_print_stats();
        fflush(stdout);
    }
/* DW: release the direct-loop angular quadrature allocated for the exact-QS matter integrals. */
    free(munode);free(muweight);
    return _SUCCESS_;
}

/** Assemble P12: P12[j] = T(k) * k^3 * f * exp(-cutoff) (fNL transfer) */
static void assemble_P12(int nk, const double *k, const double *f,
                         const double *Tbin, double cutoff, double *P12) {
    double inv_cutoff = 1.0 / cutoff;
    for (int j = 0; j < nk; j++) {
        double kc = k[j] * inv_cutoff; double kc2 = kc * kc; double kc6 = kc2 * kc2 * kc2;
        P12[j] = Tbin[j] * k[j] * k[j] * k[j] * f[j] * exp(-kc6);
    }
}

/* ==================================================================
 * MACROS FOR REPETITIVE PT COMPUTATION PATTERNS
 *
 * These macros eliminate ~200+ near-identical code blocks that appear
 * in the RSD, IR resummation, and bias computation sections. Each macro
 * encapsulates a common pattern involving batch BLAS operations, spline
 * interpolation, or Legendre projection.
 *
 * MACRO REFERENCE:
 *   COMPUTE_P13(suffix, M13, uv)     - P13 one-loop via batch dot product
 *   COMPUTE_P22(suffix, M22)         - P22 one-loop via batch quadratic form
 *   COMPUTE_P12(suffix, M12)         - P12 fNL correction (if has_fNL==1)
 *   SPLINE_SETUP(x,n,data,dd,mode)   - Compute spline derivative table
 *   SPLINE_EVAL(x,n,data,dd,xv,li,o) - Evaluate spline at single point
 *   AP_SPLINE_SETUP(name)             - Alloc + spline for AP projection
 *   AP_INTERP(name)                   - Interpolate AP-rescaled k
 *   AP_SPLINE_FREE(name)              - Free AP spline derivative array
 *   AP_SPLINE_SETUP_EX(label, arr)    - Extended AP setup (decoupled names)
 *   AP_INTERP_EX(label, arr)          - Extended AP interpolation
 *   COMPUTE_NW_P22_P12(...)           - IR no-wiggle P22+P12 assembly
 *   COMPUTE_NW_P22_ONLY(...)          - IR no-wiggle P22 only
 *   COMPUTE_W_P22_P12(...)            - IR wiggle P22+P12 assembly
 *   COMPUTE_W_P22_ONLY(...)           - IR wiggle P22 only
 *   SPLINE_INTERP_OUTPUT(...)         - Spline + interpolate to output grid
 *   SPLINE_INTERP_BATCH(...)          - Batch version of SPLINE_INTERP_OUTPUT
 *   COMPUTE_P12_BIAS_PAIR(...)        - P12 pair (base+ortho) for bias
 *   LEGENDRE_PROJECT(val, P0,P2,P4)   - Project into ell=0,2,4 multipoles
 *   ASSEMBLE_P22_BIAS(M22, P_out)     - k^3-scaled P22 for bias operators
 *   TIDAL_P22(M, Xr, Xi, P, post)     - Self-contained tidal P22 integral
 *   COMPUTE_FNL_P12_PAIR(...)         - fNL P12 pair (base+ortho)
 *   FILL_M12_BIAS(...)                - Fill M12 matrix + compute P12 pair
 *   FILL_M22_BIAS(...)                - Fill M22 matrix + assemble P22
 *   FFT_TO_CMSYM(P, b, eta, cmsym)    - Full FFTLog decomposition pipeline
 *
 * Free variables assumed in scope (set up in nonlinear_pt_loop):
 *   Nmax, Np1, kdisc, Pbin, Tbin, cutoff, has_fNL
 *   Xr, Xi, Xr_transfer, Xi_transfer (FFTLog X matrices)
 *   Mr_ws, Mi_ws, Y1_ws, Y2_ws (BLAS workspace)
 *   pnlpt->error_message (for class_call/class_alloc)
 * ================================================================== */

/** Compute P13 for a given matrix and UV counterterm formula.
 *  @param suffix   name suffix (e.g. 0_vv) -> P13_0_vv, P13UV_0_vv
 *  @param M13_matrix  complex M13 kernel vector
 *  @param uv_formula  C expression for P13UV[j] (may reference j, Pbin, kdisc, sigmav, f) */
#define COMPUTE_P13(suffix, M13_matrix, uv_formula)                                     \
    do                                                                                  \
    {                                                                                   \
        double *f13_tmp = malloc(Nmax * sizeof(double));                                \
        TIMER_START(blas);                                                               \
        batch_dot_product(Np1, Nmax, Xr, Xi, M13_matrix, f13_tmp, dot_mr_ws, dot_mi_ws);                      \
        TIMER_ADD(blas);                                                                 \
        for (int j = 0; j < Nmax; j++)                                                  \
            P13UV_##suffix[j] = uv_formula;                                             \
        assemble_P13(Nmax, kdisc, f13_tmp, Pbin, P13UV_##suffix, cutoff, P13_##suffix); \
        free(f13_tmp);                                                                  \
    } while (0)

/** Compute P22 one-loop integral via batch quadratic form.
 *  @param suffix      name suffix -> P22_{suffix}
 *  @param M22_matrix  packed complex symmetric M22 kernel matrix */
#define COMPUTE_P22(suffix, M22_matrix)                            \
    do                                                             \
    {                                                              \
        double *f22_tmp = malloc(Nmax * sizeof(double));           \
        TIMER_START(blas);                                          \
        batch_quadratic_form(Np1, Nmax, M22_matrix, Xr, Xi,        \
                             f22_tmp, Mr_ws, Mi_ws, Y1_ws, Y2_ws); \
        TIMER_ADD(blas);                                            \
        assemble_P22(Nmax, kdisc, f22_tmp, cutoff, P22_##suffix);  \
        free(f22_tmp);                                             \
    } while (0)

/** Compute P12 fNL correction via batch quadratic form with transfer X.
 *  Original: f12[j] = x_transfer^T * M12 * x_transfer (quadratic form).
 *  Only executes when has_fNL (fNL enabled).
 *  @param suffix      name suffix -> P12_{suffix}
 *  @param M12_matrix  complex M12 kernel matrix */
#define COMPUTE_P12(suffix, M12_matrix)                                     \
    do                                                                      \
    {                                                                       \
        if (has_fNL)                                              \
        {                                                                   \
            double *f12_tmp = malloc(Nmax * sizeof(double));                \
            TIMER_START(blas);                                               \
            batch_quadratic_form(Np1, Nmax, M12_matrix,                     \
                                 Xr_transfer, Xi_transfer,                  \
                                 f12_tmp, Mr_ws, Mi_ws, Y1_ws, Y2_ws);     \
            TIMER_ADD(blas);                                                 \
            assemble_P12(Nmax, kdisc, f12_tmp, Tbin, cutoff, P12_##suffix); \
            free(f12_tmp);                                                  \
        }                                                                   \
    } while (0)

/* AP spline setup: allocate dd array, spline, declare output variable */
#define AP_SPLINE_SETUP(name)                                                      \
    double *dd_##name;                                                             \
    class_alloc(dd_##name, sizeof(double) * Nmax, pnlpt->error_message);           \
    class_call(array_spline_table_columns(kdisc, Nmax, name, 1, dd_##name,         \
                                          _SPLINE_NATURAL_, pnlpt->error_message), \
               pnlpt->error_message, pnlpt->error_message);                        \
    double name##_ap_out = 0

/* AP spline interpolation at ktrue (slow version — individual binary search) */
#define AP_INTERP(name)                                                                               \
    class_call(array_interpolate_spline(kdisc, Nmax, name, dd_##name, 1,                              \
                                        ktrue, &last_index, &name##_ap_out, 1, pnlpt->error_message), \
               pnlpt->error_message, pnlpt->error_message)

/* Fast AP interpolation: reuses precomputed binary search result.
 * Before using, compute ap_inf/ap_a/ap_b/ap_a3a/ap_b3b/ap_h2_6 once per ktrue. */
#define AP_INTERP_FAST(name) \
    name##_ap_out = ap_a_ * name[ap_inf_] + ap_b_ * name[ap_sup_] \
                  + (ap_a3a_ * dd_##name[ap_inf_] + ap_b3b_ * dd_##name[ap_sup_]) * ap_h2_6_

/* Fast version for extended labels */
#define AP_INTERP_EX_FAST(label, arr) \
    label##_ap_out = ap_a_ * arr[ap_inf_] + ap_b_ * arr[ap_sup_] \
                   + (ap_a3a_ * dd_##label[ap_inf_] + ap_b3b_ * dd_##label[ap_sup_]) * ap_h2_6_

/* Binary search on kdisc to find interval for ktrue, then precompute cubic spline coefficients.
 * Must be called once before AP_INTERP_FAST calls. */
#define AP_BSEARCH_SETUP()                                                   \
    int ap_inf_ = 0, ap_sup_ = Nmax - 1;                                    \
    while (ap_sup_ - ap_inf_ > 1) {                                         \
        int ap_mid_ = (ap_inf_ + ap_sup_) >> 1;                             \
        if (ktrue < kdisc[ap_mid_]) ap_sup_ = ap_mid_; else ap_inf_ = ap_mid_; \
    }                                                                        \
    double ap_h_ = kdisc[ap_sup_] - kdisc[ap_inf_];                         \
    double ap_b_ = (ktrue - kdisc[ap_inf_]) / ap_h_;                        \
    double ap_a_ = 1. - ap_b_;                                              \
    double ap_a3a_ = ap_a_ * ap_a_ * ap_a_ - ap_a_;                        \
    double ap_b3b_ = ap_b_ * ap_b_ * ap_b_ - ap_b_;                        \
    double ap_h2_6_ = ap_h_ * ap_h_ / 6.

/* AP spline cleanup */
#define AP_SPLINE_FREE(name) free(dd_##name)

/* Extended AP macros: label controls dd_/ap_out names, arr is the data array */
#define AP_SPLINE_SETUP_EX(label, arr)                                             \
    double *dd_##label;                                                            \
    class_alloc(dd_##label, sizeof(double) * Nmax, pnlpt->error_message);          \
    class_call(array_spline_table_columns(kdisc, Nmax, arr, 1, dd_##label,         \
                                          _SPLINE_NATURAL_, pnlpt->error_message), \
               pnlpt->error_message, pnlpt->error_message);                        \
    double label##_ap_out = 0

#define AP_INTERP_EX(label, arr)                                                                       \
    class_call(array_interpolate_spline(kdisc, Nmax, arr, dd_##label, 1,                               \
                                        ktrue, &last_index, &label##_ap_out, 1, pnlpt->error_message), \
               pnlpt->error_message, pnlpt->error_message)


/* IR resummation: NW P22 + P12 (with ortho) for a given mu-power */

#define COMPUTE_NW_P22_P12(suffix, M22_mat, M12_mat, M12_ortho_mat)                    \
    batch_quadratic_form(Np1, Nmax, M22_mat, Xr_nw, Xi_nw,                             \
                         f22_tmp, Mr_ws, Mi_ws, Y1_ws, Y2_ws);                         \
    assemble_P22(Nmax, kdisc, f22_tmp, cutoff, P22_##suffix);                          \
    if (has_fNL)                                                             \
    {                                                                                  \
        batch_quadratic_form(Np1, Nmax, M12_mat, Xr_nw_transfer, Xi_nw_transfer,       \
                             f12_tmp, Mr_ws, Mi_ws, Y1_ws, Y2_ws);                     \
        assemble_P12(Nmax, kdisc, f12_tmp, Tnw, cutoff, P12_##suffix);                 \
        batch_quadratic_form(Np1, Nmax, M12_ortho_mat, Xr_nw_transfer, Xi_nw_transfer, \
                             f12_tmp, Mr_ws, Mi_ws, Y1_ws, Y2_ws);                     \
        assemble_P12(Nmax, kdisc, f12_tmp, Tnw, cutoff, P12_##suffix##_ortho);         \
    }

/* IR resummation: NW P22 only (no P12) */
#define COMPUTE_NW_P22_ONLY(suffix, M22_mat)                   \
    batch_quadratic_form(Np1, Nmax, M22_mat, Xr_nw, Xi_nw,     \
                         f22_tmp, Mr_ws, Mi_ws, Y1_ws, Y2_ws); \
    assemble_P22(Nmax, kdisc, f22_tmp, cutoff, P22_##suffix)

/* IR resummation: wiggle P22 + P12 (with ortho) for a given mu-power */
#define COMPUTE_W_P22_P12(suffix, M22_mat, M12_mat, M12_ortho_mat)                           \
    batch_bilinear_form(Np1, Nmax, M22_mat, Xr_nw, Xi_nw, Xr_2w, Xi_2w,                      \
                        f22_w_tmp, Mr_ws, Mi_ws, Y1_ws);                                     \
    assemble_P22(Nmax, kdisc, f22_w_tmp, cutoff, P22_##suffix##_w);                          \
    if (has_fNL)                                                                   \
    {                                                                                        \
        batch_bilinear_form(Np1, Nmax, M12_mat, Xr_nw_transfer, Xi_nw_transfer,              \
                            Xr_2w_transfer, Xi_2w_transfer, f12_w_tmp, Mr_ws, Mi_ws, Y1_ws); \
        assemble_P12(Nmax, kdisc, f12_w_tmp, Tnw, cutoff, P12_##suffix##_w);                 \
        batch_bilinear_form(Np1, Nmax, M12_ortho_mat, Xr_nw_transfer, Xi_nw_transfer,        \
                            Xr_2w_transfer, Xi_2w_transfer, f12_w_tmp, Mr_ws, Mi_ws, Y1_ws); \
        assemble_P12(Nmax, kdisc, f12_w_tmp, Tnw, cutoff, P12_##suffix##_w_ortho);           \
    }

/* IR resummation: wiggle P22 only (no P12) */
#define COMPUTE_W_P22_ONLY(suffix, M22_mat)                             \
    batch_bilinear_form(Np1, Nmax, M22_mat, Xr_nw, Xi_nw, Xr_2w, Xi_2w, \
                        f22_w_tmp, Mr_ws, Mi_ws, Y1_ws);                \
    assemble_P22(Nmax, kdisc, f22_w_tmp, cutoff, P22_##suffix##_w)

/* Output spline interpolation: transfer results from the FFTLog grid (kdisc[])
 * to the output k-grid (pnlpt->k[]). Inside [kmin_b, kmax_b], values are
 * spline-interpolated; outside, the extrapolation expression is used.
 *
 * @param P_in        input array on kdisc grid (size Nmax)
 * @param pk_out      output array on pnlpt->k grid (size pnlpt->k_size)
 * @param kmin_b      lower bound of valid interpolation range
 * @param kmax_b      upper bound of valid interpolation range
 * @param interp_expr C expression for interpolated value (may use out_tmp_)
 * @param extrap_expr C expression for extrapolated value
 * @param do_free     if nonzero, free(P_in) after interpolation */
#define SPLINE_INTERP_OUTPUT(P_in, pk_out, kmin_b, kmax_b, interp_expr, extrap_expr, do_free)     \
    do                                                                                            \
    {                                                                                             \
        double *dd_sio_;                                                                          \
        double out_tmp_ = 0.;                                                                     \
        class_alloc(dd_sio_, sizeof(double) * Nmax, pnlpt->error_message);                        \
        class_call(array_spline_table_columns(kdisc, Nmax, P_in, 1, dd_sio_,                      \
                                              _SPLINE_NATURAL_, pnlpt->error_message),            \
                   pnlpt->error_message, pnlpt->error_message);                                   \
        last_index = 0;                                                                           \
        for (index_k = 0; index_k < pnlpt->k_size; index_k++)                                     \
        {                                                                                         \
            if (pnlpt->k[index_k] <= (kmax_b) && pnlpt->k[index_k] >= (kmin_b))                   \
            {                                                                                     \
                class_call(array_interpolate_spline(kdisc, Nmax, P_in, dd_sio_, 1,                \
                                                    pnlpt->k[index_k], &last_index, &out_tmp_, 1, \
                                                    pnlpt->error_message),                        \
                           pnlpt->error_message, pnlpt->error_message);                           \
                pk_out[index_k] = (interp_expr);                                                  \
            }                                                                                     \
            else                                                                                  \
            {                                                                                     \
                pk_out[index_k] = (extrap_expr);                                                  \
            }                                                                                     \
        }                                                                                         \
        free(dd_sio_);                                                                            \
        if (do_free)                                                                              \
            free(P_in);                                                                           \
    } while (0)

/* Batch spline-interpolate: applies SPLINE_INTERP_OUTPUT to n_pairs arrays at once.
 * All pairs share the same k-range and interpolation/extrapolation expressions.
 *
 * @param p_in_arr   array of double* pointers (inputs on kdisc grid)
 * @param p_out_arr  array of double* pointers (outputs on pnlpt->k grid)
 * @param n_pairs    number of (input, output) pairs to process
 * @param do_free    if nonzero, free each input array after interpolation */
#define SPLINE_INTERP_BATCH(p_in_arr, p_out_arr, n_pairs, kmin_b, kmax_b,     \
                            interp_expr, extrap_expr, do_free)                 \
    do {                                                                       \
        for (int sib_i_ = 0; sib_i_ < (n_pairs); sib_i_++) {                  \
            double *sib_pin_ = (p_in_arr)[sib_i_];                            \
            double *sib_pout_ = (p_out_arr)[sib_i_];                          \
            double *dd_sio_;                                                   \
            double out_tmp_ = 0.;                                              \
            class_alloc(dd_sio_, sizeof(double) * Nmax, pnlpt->error_message); \
            class_call(array_spline_table_columns(kdisc, Nmax, sib_pin_, 1,    \
                       dd_sio_, _SPLINE_NATURAL_, pnlpt->error_message),       \
                       pnlpt->error_message, pnlpt->error_message);           \
            last_index = 0;                                                    \
            for (index_k = 0; index_k < pnlpt->k_size; index_k++) {           \
                if (pnlpt->k[index_k] <= (kmax_b) &&                          \
                    pnlpt->k[index_k] >= (kmin_b)) {                          \
                    class_call(array_interpolate_spline(kdisc, Nmax,           \
                               sib_pin_, dd_sio_, 1, pnlpt->k[index_k],       \
                               &last_index, &out_tmp_, 1,                      \
                               pnlpt->error_message),                         \
                               pnlpt->error_message, pnlpt->error_message);   \
                    sib_pout_[index_k] = (interp_expr);                        \
                } else {                                                       \
                    sib_pout_[index_k] = (extrap_expr);                        \
                }                                                              \
            }                                                                  \
            free(dd_sio_);                                                     \
            if (do_free) free(sib_pin_);                                       \
        }                                                                      \
    } while (0)

/* Compute P12 pair (base + ortho) from transfer matrices, with k^3*T(k) scaling.
 * Allocates/frees X transfer matrix and f12 workspace internally. */
#define COMPUTE_P12_BIAS_PAIR(M12_base, M12_ortho, P12_out, P12_out_ortho,     \
                              cmsym_t, kb_t_cache, sc_c_tab, sc_s_tab)          \
    do {                                                                        \
        double *Xr_t_, *Xi_t_;                                                 \
        class_alloc(Xr_t_, Np1 * Nmax * sizeof(double), pnlpt->error_message); \
        class_alloc(Xi_t_, Np1 * Nmax * sizeof(double), pnlpt->error_message); \
        TIMER_START(buildx);                                                    \
        build_X_from_tables(Nmax, Np1, cmsym_t, kb_t_cache,                    \
                            sc_c_tab, sc_s_tab, Xr_t_, Xi_t_);                \
        TIMER_ADD(buildx);                                                      \
        double *f12_t_ = malloc(Nmax * sizeof(double));                        \
        TIMER_START(blas);                                                      \
        batch_quadratic_form(Np1, Nmax, M12_base, Xr_t_, Xi_t_,               \
                             f12_t_, Mr_ws, Mi_ws, Y1_ws, Y2_ws);             \
        TIMER_ADD(blas);                                                        \
        for (int j_ = 0; j_ < Nmax; j_++)                                     \
            P12_out[j_] = Tbin[j_] * kdisc[j_] * kdisc[j_] * kdisc[j_] * f12_t_[j_]; \
        TIMER_START(blas);                                                      \
        batch_quadratic_form(Np1, Nmax, M12_ortho, Xr_t_, Xi_t_,              \
                             f12_t_, Mr_ws, Mi_ws, Y1_ws, Y2_ws);             \
        TIMER_ADD(blas);                                                        \
        for (int j_ = 0; j_ < Nmax; j_++)                                     \
            P12_out_ortho[j_] = Tbin[j_] * kdisc[j_] * kdisc[j_] * kdisc[j_] * f12_t_[j_]; \
        free(f12_t_); free(Xr_t_); free(Xi_t_);                               \
    } while (0)

/* Project scalar into Legendre multipoles: ell=0,2,4 */
#define LEGENDRE_PROJECT(val, P0, P2, P4)                                      \
    P0[index_j] += (val) * LegendreP0 / 2.;                                   \
    P2[index_j] += (val) * LegendreP2 * 2.5;                                  \
    P4[index_j] += (val) * LegendreP4 * 4.5

/* Compute P22 from bias matrix and apply k^3 scaling.
 * f22_tmp and Xr2/Xi2 must already be in scope. */
#define ASSEMBLE_P22_BIAS(M22_mat, P22_out)                                    \
    TIMER_START(blas);                                                          \
    batch_quadratic_form(Np1, Nmax, M22_mat, Xr2, Xi2,                        \
                         f22_tmp, Mr_ws, Mi_ws, Y1_ws, Y2_ws);                \
    TIMER_ADD(blas);                                                            \
    for (int j_ = 0; j_ < Nmax; j_++)                                         \
        P22_out[j_] = kdisc[j_] * kdisc[j_] * kdisc[j_] * f22_tmp[j_]

/* Self-contained tidal P22: alloc workspace, quadratic form, k^3 scale, free.
 * post_fn is applied to k^3*f result: use IDENTITY for plain, fabs for abs. */
#define TIDAL_P22(M_mat, Xr_, Xi_, P_out, post_fn)                            \
    do {                                                                       \
        double *_f22_ = malloc(Nmax * sizeof(double));                         \
        batch_quadratic_form(Np1, Nmax, M_mat, Xr_, Xi_,                      \
                             _f22_, Mr_ws, Mi_ws, Y1_ws, Y2_ws);              \
        for (int j_ = 0; j_ < Nmax; j_++)                                     \
            P_out[j_] = post_fn(kdisc[j_] * kdisc[j_] * kdisc[j_] * _f22_[j_]); \
        free(_f22_);                                                           \
    } while (0)
#define IDENTITY(x) (x)

/* Compute fNL P12 pair (base + ortho) from a single X matrix.
 * Allocates workspace, runs two quadratic forms, applies Tbin*k^3, frees. */
#define COMPUTE_FNL_P12_PAIR(M_base, M_ortho, Xr_, Xi_, P_out, P_out_ortho)   \
    do {                                                                       \
        double *_f12_ = malloc(Nmax * sizeof(double));                         \
        batch_quadratic_form(Np1, Nmax, M_base, Xr_, Xi_,                     \
                             _f12_, Mr_ws, Mi_ws, Y1_ws, Y2_ws);              \
        for (int j_ = 0; j_ < Nmax; j_++)                                     \
            P_out[j_] = Tbin[j_] * kdisc[j_] * kdisc[j_] * kdisc[j_] * _f12_[j_]; \
        batch_quadratic_form(Np1, Nmax, M_ortho, Xr_, Xi_,                    \
                             _f12_, Mr_ws, Mi_ws, Y1_ws, Y2_ws);              \
        for (int j_ = 0; j_ < Nmax; j_++)                                     \
            P_out_ortho[j_] = Tbin[j_] * kdisc[j_] * kdisc[j_] * kdisc[j_] * _f12_[j_]; \
        free(_f12_);                                                           \
    } while (0)

/* Fill M12 bias matrix from source with coefficient, then compute P12 pair.
 * Only runs when has_fNL (fNL enabled). */
#define FILL_M12_BIAS(M_dst, M_dst_o, coeff, M_src, M_src_o, P_out, P_out_o, cs, kb_t_c, sc_c_tab, sc_s_tab) \
    count2 = 0;                                                                \
    if (has_fNL) {                                                   \
        for (index_l = 0; index_l < Nmax + 1; index_l++)                       \
            for (index_i = index_l; index_i < Nmax + 1; index_i++) {           \
                (M_dst)[count2] = (coeff) * (M_src)[count2];                   \
                (M_dst_o)[count2] = (coeff) * (M_src_o)[count2];              \
                count2++;                                                      \
            }                                                                  \
        COMPUTE_P12_BIAS_PAIR(M_dst, M_dst_o, P_out, P_out_o, cs, kb_t_c, sc_c_tab, sc_s_tab);\
    }

/* Fill M22 bias matrix with kernel expression (in nu1, nu2), then assemble P22.
 * nu1 = -0.5*etam2[index_i], nu2 = -0.5*etam2[index_l] are set automatically. */
#define FILL_M22_BIAS(M_out, kernel_expr, P_out)                               \
    {                                                                          \
        int _Nb_ = Nmax + 1;                                                   \
        _Pragma("omp parallel for schedule(static)") \
        for (int _lb_ = 0; _lb_ < _Nb_; _lb_++) {                             \
            int _cbb_ = _lb_ * _Nb_ - _lb_ * (_lb_ - 1) / 2;                 \
            for (int _ib_ = _lb_; _ib_ < _Nb_; _ib_++) {                      \
                int _idx_ = _cbb_ + (_ib_ - _lb_);                            \
                double complex nu1 = -0.5 * etam2[_ib_];                       \
                double complex nu2 = -0.5 * etam2[_lb_];                       \
                (M_out)[_idx_] = pnlpt->M22basic_oneline_complex[_idx_] * (kernel_expr); \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    ASSEMBLE_P22_BIAS(M_out, P_out)

/* FFTLog decomposition: the full pipeline from spectrum P(k) to symmetric
 * FFTLog coefficients c_m, implementing the algorithm of Hamilton (2000):
 *
 *   1. Bias subtraction: input_j = P(k_j) * exp(-j * b * Delta)
 *   2. FFT of biased input to get Fourier coefficients
 *   3. Reorder and apply k_min^{-eta} normalization
 *   4. Halve endpoints (trapezoidal rule correction)
 *
 * Output: cmsym_out[0..Nmax] with symmetrized coefficients.
 * Allocates cmsym_out internally; temporary FFT arrays are freed.
 *
 * @param P_src      power spectrum on kdisc grid (size Nmax)
 * @param b_exp      FFTLog bias exponent (e.g. -0.3 for matter, -1.6 for bias)
 * @param eta_arr    complex frequency array eta_m (size Nmax+1)
 * @param cmsym_out  output: allocated array of Nmax+1 complex coefficients
 *
 * Free variables: Nmax, Nmaxd, kmin, Delta, stepsize, pnlpt->error_message */
#define FFT_TO_CMSYM(P_src, b_exp, eta_arr, cmsym_out)                             \
    do {                                                                             \
        double *_ir, *_ii, *_or, *_oi;                                               \
        class_alloc(_ir, Nmax * sizeof(double), pnlpt->error_message);               \
        class_alloc(_ii, Nmax * sizeof(double), pnlpt->error_message);               \
        class_alloc(_or, Nmax * sizeof(double), pnlpt->error_message);               \
        class_alloc(_oi, Nmax * sizeof(double), pnlpt->error_message);               \
        for (int _k = 0; _k < Nmax; _k++) {                                         \
            _ir[_k] = (P_src)[_k] * exp(-1. * _k * (b_exp) * Delta);                \
            _ii[_k] = 0.;                                                            \
        }                                                                            \
        FFT(_ir, _ii, _or, _oi, Nmax, stepsize);                                    \
        class_alloc(cmsym_out, (Nmax + 1) * sizeof(complex double),                 \
                    pnlpt->error_message);                                           \
        /* kmin^(-eta) = kmin^(-b) * exp(-i*imag(eta)*ln(kmin))                      \
         * Use sincos to avoid expensive cpow */                                     \
        double _km_neg_b = pow(kmin, -(b_exp));                                      \
        double _lnkmin = log(kmin);                                                  \
        for (int _k = 0; _k < Nmax + 1; _k++) {                                     \
            int _idx = (_k < Nmax / 2) ? (Nmax / 2 - _k) : (_k - Nmax / 2);        \
            double _sign = (_k < Nmax / 2) ? -1. : 1.;                              \
            double _omega = -cimag((eta_arr)[_k]) * _lnkmin;                         \
            double _s, _c;                                                           \
            sincos(_omega, &_s, &_c);                                                \
            double complex _kpow = _km_neg_b * (_c + _Complex_I * _s);              \
            (cmsym_out)[_k] = _kpow *                                                \
                (_or[_idx] + _sign * _Complex_I * _oi[_idx]) / Nmaxd;               \
        }                                                                            \
        (cmsym_out)[0] /= 2.;                                                       \
        (cmsym_out)[Nmax] /= 2.;                                                    \
        free(_ir); free(_ii); free(_or); free(_oi);                                  \
    } while (0)

/* ====================================================================== */
/* IR RESUMMATION (Senatore & Zaldarriaga 2015, arXiv:1404.5954)           */
/*                                                                          */
/* Extracts the smooth (no-wiggle) power spectrum P_nw(k) from P_L(k)     */
/* using a DST-II transform of log(k*P), then computes the BAO damping    */
/* scales Sigma^2 and deltaSigma^2, and assembles the IR-resummed spectra: */
/*                                                                          */
/*   P_tree(k) = P_nw(k) + (1 + Sigma^2*k^2) * P_w(k) * exp(-Sigma^2*k^2)*/
/*   P_bin(k)  = P_nw(k) + P_w(k) * exp(-Sigma^2*k^2)                     */
/*                                                                          */
/* where P_w(k) = P_L(k) - P_nw(k) is the oscillatory (wiggle) component.  */
/*                                                                          */
/* The algorithm:                                                           */
/*   1. DST-II of log(k*P) on a fine grid (Nir=65536)                     */
/*   2. Zero Fourier harmonics between Nleft=120 and Nright=240            */
/*      (these contain the BAO bump in Fourier space)                      */
/*   3. Inverse DST to recover P_nw(k)                                    */
/*   4. Integrate (P_L - P_nw) to get Sigma^2 = int P_nw j0(kr)/(6pi^2)  */
/*   5. Interpolate P_nw onto the FFTLog grid and build resummed spectra   */
/* ====================================================================== */
static int nonlinear_pt_ir_resummation(
    struct nonlinear_pt *pnlpt,
    struct background *pba,
    struct primordial *ppm,
    struct thermodynamics *pth,
    int Nmax,
    double *lnk_l,
    double *lnpk_l,
    double *myddlnpk,
    double *kdisc,
    double *Pdisc,
    double *PPRIMdisc,
    /* outputs: pre-allocated arrays of size Nmax */
    double *Pnw,
    double *Pw,
    double *Pbin,
    double *Ptree,
    double *Tnw,
    double *Tw,
    double *Tbin,
    /* output scalars */
    double *SigmaBAO_out,
    double *deltaSigmaBAO_out) {

    int last_index = 0, last_index2 = 0;
    double lnpk_out;

    if (pnlpt->nonlinear_pt_verbose > 0)
        printf("Performing IR resummation...\n");

    int qcdm_direct_power=((pba->has_qcdm==_TRUE_) &&
                           (pnlpt->qcdm_qscf_kernels!=qcdm_qscf_kernels_no));

    /* Step 1: Extract smooth (no-wiggle) P(k) via DST-II of log(k*P) */

    /* DST grid for BAO extraction (Nir must match PT matrix resolution) */
    int Nir = 65536;
    int Nirover2 = Nir / 2;      /* half-grid for odd/even decomposition */
    int Nirby4 = 4 * Nir;        /* zero-padded FFT size */
    double Nird = (double)Nir;
    double kmin2 = 0.00007;       /* k range for DST interpolation [1/Mpc] */
    double kmax2 = 7;
    /* Allocate FFT buffers once, reuse for both forward and inverse transforms */
    double *logkPdiscr, *fft_ir, *fft_ii, *fft_or, *fft_oi;
    double logPbin2, kbin2;
    class_alloc(logkPdiscr, Nir * sizeof(double), pnlpt->error_message);
    class_alloc(fft_ir, Nirby4 * sizeof(double), pnlpt->error_message);
    class_alloc(fft_ii, Nirby4 * sizeof(double), pnlpt->error_message);
    class_alloc(fft_or, Nirby4 * sizeof(double), pnlpt->error_message);
    class_alloc(fft_oi, Nirby4 * sizeof(double), pnlpt->error_message);

    last_index = 0;
    int index_ir = 0;
    /* Zero the imaginary input and real even-indexed slots upfront */
    memset(fft_ii, 0, Nirby4 * sizeof(double));
    memset(fft_ir, 0, Nirby4 * sizeof(double));
    double kstep = (kmax2 - kmin2) / (Nird - 1.);
    double lnk0 = lnk_l[0];
    double exp_lnk0 = exp(lnk0);
    int _ksz_ir = pnlpt->k_size;
    double _lnk_hi = lnk_l[_ksz_ir - 1];
    double _exp_lnk_hi = exp(_lnk_hi);
    double _slope_hi = (_ksz_ir >= 2) ? (lnpk_l[_ksz_ir-1] - lnpk_l[_ksz_ir-2]) / (lnk_l[_ksz_ir-1] - lnk_l[_ksz_ir-2]) : ppm->n_s;
    #pragma omp parallel
    {
        int _last_idx = 0;
        #pragma omp for schedule(static)
        for (index_ir = 0; index_ir < Nir; index_ir++) {
            double _kbin2 = kmin2 + index_ir * kstep;

            if (qcdm_direct_power) {
                double _Pphys=qcdm_qscf_interp_positive_loglog(Nmax,kdisc,Pdisc,_kbin2);
                logkPdiscr[index_ir]=log(_kbin2)+log(MAX(_Pphys,DBL_MIN));
            }
            else if (_kbin2 < exp_lnk0) {
                logkPdiscr[index_ir] = log(_kbin2) + lnpk_l[0] + (ppm->n_s) * (log(_kbin2) - lnk0);
            } else if (_kbin2 <= _exp_lnk_hi) {
                double _logPbin2;
                array_interpolate_spline(lnk_l, pnlpt->k_size, lnpk_l, myddlnpk, 1, log(_kbin2),
                                         &_last_idx, &_logPbin2, 1, pnlpt->error_message);
                logkPdiscr[index_ir] = log(_kbin2) + _logPbin2;
            } else {
                /* High-k power-law extrapolation */
                double _dlnk = log(_kbin2) - _lnk_hi;
                logkPdiscr[index_ir] = log(_kbin2) + lnpk_l[_ksz_ir-1] + _slope_hi * _dlnk;
            }

            /* DST-II symmetry: alternating sign for odd-indexed real FFT input */
            double val = ((index_ir % 2 == 0) ? 1. : -1.) * logkPdiscr[index_ir];
            fft_ir[2 * index_ir + 1] = val;
            fft_ir[Nirby4 - 2 * index_ir - 1] = val;
        }
    }

    int stepsize = 1;
    FFT(fft_ir, fft_ii, fft_or, fft_oi, Nirby4, stepsize);

    double *out_ir;
    class_alloc(out_ir, Nir * sizeof(double), pnlpt->error_message);
    int index_ir2 = 0;
    for (index_ir2 = 0; index_ir2 < Nir; index_ir2++)
        out_ir[index_ir2] = fft_or[Nir - index_ir2 - 1];

    /* Step 2: Erase BAO bump from Fourier harmonics and spline-interpolate */

    double *cmodd, *cmeven;
    int *ivar;
    class_alloc(cmodd, Nirover2 * sizeof(double), pnlpt->error_message);
    class_alloc(cmeven, Nirover2 * sizeof(double), pnlpt->error_message);
    class_alloc(ivar, Nirover2 * sizeof(int), pnlpt->error_message);

    for (index_ir = 0; index_ir < Nirover2; index_ir++) {
        ivar[index_ir] = index_ir + 1;
        cmodd[index_ir] = out_ir[2 * index_ir];
        cmeven[index_ir] = out_ir[2 * index_ir + 1];
    }

    /* Fourier harmonic window containing BAO feature: modes [Nleft, Nright)
     * are removed to produce the no-wiggle spectrum P_nw */
    int Nleft = 120;
    int Nright = 240;
    int Nthrow = Nright - Nleft;
    int Nnew = Nirover2 - Nthrow;
    double *cmoddnw, *cmevennw, *inew;
    class_alloc(cmoddnw, Nnew * sizeof(double), pnlpt->error_message);
    class_alloc(cmevennw, Nnew * sizeof(double), pnlpt->error_message);
    class_alloc(inew, Nnew * sizeof(double), pnlpt->error_message);

    for (index_ir = 0; index_ir < Nleft; index_ir++) {
        cmoddnw[index_ir] = cmodd[index_ir];
        cmevennw[index_ir] = cmeven[index_ir];
        inew[index_ir] = index_ir + 1.;
    }
    for (index_ir = Nleft; index_ir < Nnew; index_ir++) {
        cmoddnw[index_ir] = cmodd[index_ir + Nthrow];
        cmevennw[index_ir] = cmeven[index_ir + Nthrow];
        inew[index_ir] = index_ir + 1. + Nthrow;
    }

    last_index = 0;
    double *dd_cmoddnw, *dd_cmevennw;
    double cmodd_newval, cmeven_newval;
    class_alloc(dd_cmoddnw, sizeof(double) * Nnew, pnlpt->error_message);
    SPLINE_SETUP(inew, Nnew, cmoddnw, dd_cmoddnw, _SPLINE_NATURAL_);
    class_alloc(dd_cmevennw, sizeof(double) * Nnew, pnlpt->error_message);
    SPLINE_SETUP(inew, Nnew, cmevennw, dd_cmevennw, _SPLINE_NATURAL_);

    double *cmnew;
    class_alloc(cmnew, Nir * sizeof(double), pnlpt->error_message);
    last_index = 0;
    last_index2 = 0;
    for (index_ir = 0; index_ir < Nirover2; index_ir++) {
        SPLINE_EVAL(inew, Nnew, cmoddnw, dd_cmoddnw, (double)(index_ir + 1), &last_index, &cmodd_newval);
        SPLINE_EVAL(inew, Nnew, cmevennw, dd_cmevennw, (double)(index_ir + 1), &last_index2, &cmeven_newval);
        cmnew[index_ir * 2] = cmodd_newval;
        cmnew[index_ir * 2 + 1] = cmeven_newval;
    }

    /* Step 3: Inverse DST to get smooth P_nw(k) and spline-interpolate */

    /* Reuse FFT buffers for the inverse transform (zero imaginary input) */
    double *out_2;
    class_alloc(out_2, Nir * sizeof(double), pnlpt->error_message);
    fft_ir[0] = cmnew[Nir - 1] * 0.5;
    fft_ir[Nir] = 0.;
    fft_ir[2 * Nir] = -1. * cmnew[Nir - 1] * 0.5;
    fft_ir[3 * Nir] = 0.;

    fft_ii[0] = 0.;
    fft_ii[Nir] = 0.;
    fft_ii[2 * Nir] = 0.;
    fft_ii[3 * Nir] = 0.;

    for (index_ir = 1; index_ir < Nir; index_ir++) {
        double half_cm = 0.5 * cmnew[Nir - 1 - index_ir];
        fft_ir[index_ir] = half_cm;
        fft_ir[4 * Nir - index_ir] = half_cm;
        fft_ir[2 * Nir - index_ir] = -half_cm;
        fft_ir[2 * Nir + index_ir] = -half_cm;

        fft_ii[index_ir] = 0.;
        fft_ii[4 * Nir - index_ir] = 0.;
        fft_ii[2 * Nir - index_ir] = 0.;
        fft_ii[2 * Nir + index_ir] = 0.;
    }

    FFT(fft_ir, fft_ii, fft_or, fft_oi, Nirby4, stepsize);

    for (index_ir = 0; index_ir < Nir; index_ir++)
        out_2[index_ir] = ((index_ir % 2 == 0) ? 1. : -1.) * fft_or[2 * index_ir + 1];

    /* Free FFT buffers (no longer needed) */
    free(fft_ir);
    free(fft_ii);
    free(fft_or);
    free(fft_oi);

    double *Pnw_ir, *knw_ir;
    class_alloc(Pnw_ir, Nir * sizeof(double), pnlpt->error_message);
    class_alloc(knw_ir, Nir * sizeof(double), pnlpt->error_message);
    for (index_ir = 0; index_ir < Nir; index_ir++) {
        knw_ir[index_ir] = kmin2 + index_ir * (kmax2 - kmin2) / (Nird - 1.);
        Pnw_ir[index_ir] = exp(out_2[index_ir] / (2 * Nird)) / (knw_ir[index_ir]);
    }

    free(dd_cmoddnw);
    free(dd_cmevennw);

    double *ddPnw;
    double Pnwval;
    class_alloc(ddPnw, sizeof(double) * Nir, pnlpt->error_message);
    SPLINE_SETUP(knw_ir, Nir, Pnw_ir, ddPnw, _SPLINE_NATURAL_);

    /* Step 4: Compute BAO damping factor Sigma^2 from P_nw */

    double rbao = pth->rs_d;
    int Nint2 = 500;

    double *qint2, *IntegrandBAO, *IntegrandBAO2;
    class_alloc(qint2, (Nint2 + 1) * sizeof(double), pnlpt->error_message);
    class_alloc(IntegrandBAO, (Nint2 + 1) * sizeof(double), pnlpt->error_message);
    class_alloc(IntegrandBAO2, (Nint2 + 1) * sizeof(double), pnlpt->error_message);
    double ks = qcdm_tf_ir_ks_hmpc() * pba->h;
    #pragma omp parallel
    {
        int _last_idx = 0;
        #pragma omp for schedule(static)
        for (index_ir = 0; index_ir < Nint2 + 1; index_ir++) {
            double _Pnwval;
            qint2[index_ir] = kmin2 * exp(index_ir * log(ks / kmin2) / (Nint2));
            array_interpolate_spline(knw_ir, Nir, Pnw_ir, ddPnw, 1, qint2[index_ir],
                                     &_last_idx, &_Pnwval, 1, pnlpt->error_message);
            { double qr = qint2[index_ir] * rbao;
              double qr2 = qr * qr, qr3 = qr2 * qr;
              double sinqr = sin(qr), cosqr = cos(qr);
            IntegrandBAO[index_ir] = _Pnwval * (1. - 3. * sinqr / qr + 6. * (sinqr / qr3 - cosqr / qr2));
            IntegrandBAO2[index_ir] = -1. * _Pnwval * (3. * cosqr * qr + (-3. + qr2) * sinqr) / qr3; }
        }
    }

    double SigmaBAO = 0., deltaSigmaBAO = 0.;
    for (index_ir = 0; index_ir < Nint2; index_ir++) {
        SigmaBAO += (log(qint2[index_ir + 1]) - log(qint2[index_ir])) * (qint2[index_ir + 1] * IntegrandBAO[index_ir + 1] + qint2[index_ir] * IntegrandBAO[index_ir]) / 2. / (6. * (M_PI * M_PI));
        deltaSigmaBAO += (log(qint2[index_ir + 1]) - log(qint2[index_ir])) * (qint2[index_ir + 1] * IntegrandBAO2[index_ir + 1] + qint2[index_ir] * IntegrandBAO2[index_ir]) / 2. / (2. * (M_PI * M_PI));
    }

    /* Step 5: Compute LO IR-resummed spectrum: P_nw + P_w * exp(-k^2 * Sigma^2) */

    #pragma omp parallel
    {
    int _last_idx = 0;
    #pragma omp for schedule(static)
    for (int i = 0; i < Nmax; i++) {
        double _Pnwval2, _Pnwval_rescaled, _Pwval_rescaled;
        if (kdisc[i] <= kmax2 && kdisc[i] >= kmin2 && pnlpt->alpha_rs * kdisc[i] <= kmax2 && pnlpt->alpha_rs * kdisc[i] >= kmin2) {
            array_interpolate_spline(knw_ir, Nir, Pnw_ir, ddPnw, 1, kdisc[i],
                                     &_last_idx, &_Pnwval2, 1, pnlpt->error_message);
            array_interpolate_spline(knw_ir, Nir, Pnw_ir, ddPnw, 1, pnlpt->alpha_rs * kdisc[i],
                                     &_last_idx, &_Pnwval_rescaled, 1, pnlpt->error_message);

            {
                double _keval=pnlpt->alpha_rs*kdisc[i];
                double _lnk_eval=log(_keval);
                if (qcdm_direct_power) {
                    _Pwval_rescaled=qcdm_qscf_interp_positive_loglog(Nmax,kdisc,Pdisc,_keval);
                }
                else if (_lnk_eval < lnk_l[0]) {
                    _Pwval_rescaled = exp(lnpk_l[0]) * pow(_keval / exp(lnk_l[0]), ppm->n_s);
                } else if (_lnk_eval <= _lnk_hi) {
                    double _lnpk_out;
                    array_interpolate_spline(lnk_l, pnlpt->k_size, lnpk_l, myddlnpk, 1,
                                             _lnk_eval, &_last_idx, &_lnpk_out, 1, pnlpt->error_message);
                    _Pwval_rescaled = exp(_lnpk_out);
                } else {
                    double _dlnk = _lnk_eval - _lnk_hi;
                    _Pwval_rescaled = exp(lnpk_l[_ksz_ir - 1] + _slope_hi * _dlnk);
                }
            }

            Pnw[i] = _Pnwval2;
            Tnw[i] = sqrt(_Pnwval2 / PPRIMdisc[i]) * (5. / 3.);
            Pw[i] = _Pwval_rescaled - _Pnwval_rescaled;
            Tw[i] = Tnw[i] * (Pw[i] / _Pnwval2 / 2. - Pw[i] * Pw[i] / _Pnwval2 / _Pnwval2 / 8. + Pw[i] * Pw[i] * Pw[i] / _Pnwval2 / _Pnwval2 / _Pnwval2 / 16.);

            if (pnlpt->no_wiggle)
                Pw[i] = 0.;
            if (pnlpt->wiggle_only)
                Pnw[i] = 0.;

            Pbin[i] = Pnw[i] + Pw[i] * exp(-SigmaBAO * kdisc[i] * kdisc[i]);
            Tbin[i] = Tnw[i] + Tw[i] * exp(-SigmaBAO * kdisc[i] * kdisc[i]);
        } else {
            Pnw[i] = Pdisc[i];
            Pw[i] = 0.;
            Pbin[i] = Pdisc[i];
            Tnw[i] = sqrt(Pdisc[i] / PPRIMdisc[i]) * (5. / 3.);
            Tw[i] = 0.;
            Tbin[i] = Tnw[i];
        }
        Ptree[i] = Pnw[i] + Pw[i] * exp(-SigmaBAO * kdisc[i] * kdisc[i]) * (1. + SigmaBAO * kdisc[i] * kdisc[i]);
    }
    } /* end omp parallel */

    *SigmaBAO_out = SigmaBAO;
    *deltaSigmaBAO_out = deltaSigmaBAO;

    { void *_f[] = {
        ddPnw, out_ir, out_2, cmnew, logkPdiscr,
        cmeven, cmodd, ivar, inew, cmoddnw,
        cmevennw, knw_ir, Pnw_ir, qint2, IntegrandBAO,
        IntegrandBAO2,
    };
    for (int fi_ = 0; fi_ < 16; fi_++) free(_f[fi_]); }

    return _SUCCESS_;
}

/**
 * Compute all one-loop power spectra at a single redshift.
 *
 * ALGORITHM OUTLINE:
 *   1. Set up FFTLog grid: k_j = kmin * exp(j*Delta), j=0..Nmax-1
 *   2. Interpolate linear P_L(k) and primordial P_prim(k) onto the grid
 *   3. IR resummation: split P_L = P_nw + P_w, compute BAO damping Sigma^2
 *   4. FFTLog decomposition: P_L(k) -> c_m coefficients via FFT
 *   5. One-loop integrals via BLAS:
 *      - P13[k] = sum_m c_m * M13_m * k^{eta_m}  (dot product)
 *      - P22[k] = c^T * M22 * c                   (quadratic form)
 *      - P12[k] = c_transfer^T * M12 * c           (bilinear, fNL only)
 *   6. Assemble EFT counterterm P_CTR = k^2 * P_L
 *   7. Interpolate 1-loop results to output k-grid
 *   8. RSD multipoles (ell=0,2,4): compute P_{ell}^{vv,vd,dd}
 *      - Fill M22/M13 kernels with RSD factors (f-dependent polynomials)
 *      - IR resummation: separate NW and wiggle P22/P13/P12
 *      - Alcock-Paczynski: Gauss-Legendre integration over mu
 *   9. Biased tracers: I[d2d2], I[d2G2], I[G2G2], I[FG2], fNL bias
 *
 * @param tau     conformal time at redshift z
 * @param f       logarithmic growth rate d(ln D)/d(ln a)
 * DW: AP ratios below follow the true-to-fiducial coordinate map used by both the legacy and exact-time paths.
 * @param hratio  H_true/H_fid for Alcock-Paczynski rescaling
 * @param Dratio  D_M,true/D_M,fid for Alcock-Paczynski rescaling
 * @param pk_l    linear power spectrum P_L(k) on output grid (input)
 * @param pk_l_X_Y  output arrays for 1-loop multipoles (see below)
 */
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
    /* DW: retain the physical total-matter linear spectrum alongside the
       clustering spectrum selected by cb for exact-QS matter reconstruction. */
    double *pk_l_total,
    double *pPRIMk_l,
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
    double *pk_nl_fNL,
    double *pk_fNLd2,
    double *pk_fNLG2,
    double *pk_nl_fNL_ortho,
    double *pk_fNLd2_ortho,
    double *pk_fNLG2_ortho,
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
    double *lnk_l,
    double *lnpk_l,
    double *lnpPRIMk_l) {
    int index_k = 0;
    int index_j = 0;
    int index_i = 0;
    int index_l = 0;

    double sigmav = 0.;
    double *pvecback;

    int last_index = 0;
    int last_index2 = 0;

    TIMER_DECL(blas); TIMER_DECL(fft); TIMER_DECL(buildx);
    TIMER_DECL(ap_spline); TIMER_DECL(rsd_ap); TIMER_DECL(bias_ap);
    TIMER_DECL(spline_out); TIMER_DECL(m22fill);
    TIMER_DECL(alloc); TIMER_DECL(ir); TIMER_DECL(macro);
    TIMER_DECL(qcdm_real); TIMER_DECL(qcdm_rsd);
    struct timespec _ts_total0, _ts_total1;
    clock_gettime(CLOCK_MONOTONIC, &_ts_total0);

    class_alloc(pvecback, pba->bg_size * sizeof(double), pnlpt->error_message);

    class_call(background_at_tau(pba, tau, long_info, inter_normal, &last_index, pvecback),
               pba->error_message,
               pnlpt->error_message);

    double qcdm_timing_z = 1. / pvecback[pba->index_bg_a] - 1.;
    if (pnlpt->nonlinear_pt_verbose > 0)
        printf("Computing one-loop power spectra at z=%e\n", qcdm_timing_z); /* a_today=1 in v3 */

    free(pvecback);

    /* --- FFTLog discretization parameters --- */
    int Nmax = ppr->nmax_nlpt; /* Number of FFTLog sampling points */
    int Nmaxf = ppr->nmax_nlpt + 1;
    double Nmaxd = Nmax * 1.;
    double kmin = 0.00005 * pba->h; /* FFTLog k-range (must match PT matrices) */
    double kmax = 100. * pba->h;
    double *js, *kdisc, *Pbin, *Pdisc, *Ptotaldisc, *Ptree, *PPRIMdisc, *Tbin;
    double *P10, *P10b1, *P12;
    double *Pnw, *Pw, *dd_Pnw, *dd_Pw, *Tnw, *Tw, *dd_Tnw, *dd_Tw;
    class_alloc(js, (Nmax + 1) * sizeof(double), pnlpt->error_message);
    { /* Allocate FFTLog grid arrays */
        /* DW: Pdisc is the clustering spectrum selected by cb; Ptotaldisc is
         * always the full linear matter spectrum used for hybrid diagnostics. */
        double **_ptrs[] = {&kdisc, &Pbin, &Pdisc, &Ptotaldisc, &Ptree, &PPRIMdisc, &Tbin,
                            &P10, &P10b1, &P12, &Pnw, &Pw, &dd_Pnw, &dd_Pw,
                            &Tnw, &Tw, &dd_Tnw, &dd_Tw};
        for (int ai_ = 0; ai_ < 18; ai_++)
            class_alloc(*_ptrs[ai_], Nmax * sizeof(double), pnlpt->error_message);
    }
    double SigmaBAO = 0.;
    double deltaSigmaBAO = 0.;
    qcdm_tf_ir_state qcdm_ir_state;
    memset(&qcdm_ir_state,0,sizeof(qcdm_ir_state));
    double Delta = log(kmax / kmin) / (Nmaxd - 1);
/* DW: activate the exact-time coupled-fluid path consistently throughout the loop calculation. */
    int qcdm_qscf_qs_active = ((pba->has_qcdm == _TRUE_) &&
                               (pnlpt->qcdm_qscf_kernels != qcdm_qscf_kernels_no));
    /* DW: massive-neutrino full-shape calculations must evolve the cb
     * clustering spectrum through pressureless nonlinear kernels. Feeding
     * total matter with cb=no would incorrectly nonlinearize free-streaming
     * ncdm, so reject that combination before expensive loop integration. */
    class_test((qcdm_qscf_qs_active==_TRUE_)&&(pba->has_ncdm==_TRUE_)&&(pnlpt->cb!=_TRUE_),
               pnlpt->error_message,
               "DW: exact c-c' EFTofLSS with massive ncdm requires cb=yes.");

    /* DW: the exact-QS branch now implements the full anisotropic common-mode
     * RSD IR operator below.  Keep the historical diagnostic flag readable
     * for old input files, but it no longer selects the isotropic shortcut. */
    if ((qcdm_qscf_qs_active==_TRUE_)&&(pnlpt->rsd==rsd_yes)&&
        (pnlpt->irres==irres_yes)&&
        (pnlpt->qcdm_allow_isotropic_rsd_ir_approximation==_TRUE_)&&
        (pnlpt->nonlinear_pt_verbose>0)) {
        printf("DW/QCDM-IR: qcdm_allow_isotropic_rsd_ir_approximation is deprecated and ignored; using proper anisotropic RSD IR resummation.\n");
    }

    /* Interpolate linear P(k) onto FFTLog grid */

    double lnpk_out = 0.;
    double *myddlnpk;
    double lnpPRIMk_out = 0.;
    double *myddlnpPRIMk;
    class_alloc(myddlnpk, sizeof(double) * pnlpt->k_size, pnlpt->error_message);
    SPLINE_SETUP(lnk_l, pnlpt->k_size, lnpk_l, myddlnpk, _SPLINE_NATURAL_);
    class_alloc(myddlnpPRIMk, sizeof(double) * pnlpt->k_size, pnlpt->error_message);
    SPLINE_SETUP(lnk_l, pnlpt->k_size, lnpPRIMk_l, myddlnpPRIMk, _SPLINE_NATURAL_);

    last_index = 0;
    int i_kdisc = 0;

    /* Compute local spectral index at high-k end for power-law extrapolation */
    int ksz = pnlpt->k_size;
    double ln_pk_slope_hi = (lnpk_l[ksz-1] - lnpk_l[ksz-2]) / (lnk_l[ksz-1] - lnk_l[ksz-2]);
    double ln_pPRIMk_slope_hi = (lnpPRIMk_l[ksz-1] - lnpPRIMk_l[ksz-2]) / (lnk_l[ksz-1] - lnk_l[ksz-2]);

/* DW: activate the exact-time coupled-fluid path consistently throughout the loop calculation. */
    /* In the coupled model the total-matter transfer can cross through zero
     * at isolated samples.  The physical power remains non-negative, but its
     * logarithmic companion is then floored for bookkeeping.  Feeding that
     * floor into FFTLog produced an artificial 1e-300 spectrum over a broad
     * k interval.  The exact-QS branch must therefore interpolate the actual
     * linear power arrays supplied to this routine. */
    double qcdm_pk_slope_hi=ln_pk_slope_hi;
    double qcdm_total_pk_slope_hi=ln_pk_slope_hi;
    double qcdm_prim_slope_hi=ln_pPRIMk_slope_hi;
    if (qcdm_qscf_qs_active == _TRUE_ && ksz>=2) {
        qcdm_pk_slope_hi=(log(MAX(pk_l[ksz-1],1.e-300))-log(MAX(pk_l[ksz-2],1.e-300)))/
                          (lnk_l[ksz-1]-lnk_l[ksz-2]);
        qcdm_total_pk_slope_hi=(log(MAX(pk_l_total[ksz-1],1.e-300))-log(MAX(pk_l_total[ksz-2],1.e-300)))/
                                (lnk_l[ksz-1]-lnk_l[ksz-2]);
        qcdm_prim_slope_hi=(log(MAX(pPRIMk_l[ksz-1],1.e-300))-log(MAX(pPRIMk_l[ksz-2],1.e-300)))/
                            (lnk_l[ksz-1]-lnk_l[ksz-2]);
    }

    for (i_kdisc = 0; i_kdisc < Nmax; i_kdisc++) {
        kdisc[i_kdisc] = kmin * exp(i_kdisc * Delta);
        double lnk_disc = log(kdisc[i_kdisc]);

/* DW: activate the exact-time coupled-fluid path consistently throughout the loop calculation. */
        if (qcdm_qscf_qs_active == _TRUE_) {
            if (kdisc[i_kdisc] < pnlpt->k[0]) {
                Pdisc[i_kdisc]=pk_l[0]*pow(kdisc[i_kdisc]/pnlpt->k[0],ppm->n_s);
                Ptotaldisc[i_kdisc]=pk_l_total[0]*pow(kdisc[i_kdisc]/pnlpt->k[0],ppm->n_s);
                PPRIMdisc[i_kdisc]=pPRIMk_l[0]*pow(kdisc[i_kdisc]/pnlpt->k[0],ppm->n_s);
            }
            else if (kdisc[i_kdisc] <= pnlpt->k[ksz-1]) {
                Pdisc[i_kdisc]=qcdm_qscf_interp_positive_loglog(ksz,pnlpt->k,pk_l,kdisc[i_kdisc]);
                Ptotaldisc[i_kdisc]=qcdm_qscf_interp_linear(ksz,pnlpt->k,pk_l_total,kdisc[i_kdisc]);
                PPRIMdisc[i_kdisc]=qcdm_qscf_interp_linear(ksz,pnlpt->k,pPRIMk_l,kdisc[i_kdisc]);
            }
            else {
                double dlnk=lnk_disc-lnk_l[ksz-1];
                Pdisc[i_kdisc]=MAX(pk_l[ksz-1],1.e-300)*exp(qcdm_pk_slope_hi*dlnk);
                Ptotaldisc[i_kdisc]=MAX(pk_l_total[ksz-1],1.e-300)*exp(qcdm_total_pk_slope_hi*dlnk);
                PPRIMdisc[i_kdisc]=MAX(pPRIMk_l[ksz-1],1.e-300)*exp(qcdm_prim_slope_hi*dlnk);
            }
        }
        else if (lnk_disc < lnk_l[0]) {
            /* Low-k extrapolation: power-law with n_s */
            Pdisc[i_kdisc] = exp(lnpk_l[0]) * pow(kdisc[i_kdisc] / exp(lnk_l[0]), ppm->n_s);
            Ptotaldisc[i_kdisc] = Pdisc[i_kdisc];
            PPRIMdisc[i_kdisc] = exp(lnpPRIMk_l[0]) * pow(kdisc[i_kdisc] / exp(lnk_l[0]), ppm->n_s);
        }
        else if (lnk_disc <= lnk_l[ksz-1]) {
            /* Within CLASS range: spline interpolation */
            SPLINE_EVAL(lnk_l, pnlpt->k_size, lnpk_l, myddlnpk, lnk_disc, &last_index, &lnpk_out);
            SPLINE_EVAL(lnk_l, pnlpt->k_size, lnpPRIMk_l, myddlnpPRIMk, lnk_disc, &last_index, &lnpPRIMk_out);
            Pdisc[i_kdisc] = exp(lnpk_out);
            Ptotaldisc[i_kdisc] = Pdisc[i_kdisc];
            PPRIMdisc[i_kdisc] = exp(lnpPRIMk_out);
        }
        else {
            /* High-k extrapolation: power-law using local slope at boundary */
            double dlnk = lnk_disc - lnk_l[ksz-1];
            Pdisc[i_kdisc] = exp(lnpk_l[ksz-1] + ln_pk_slope_hi * dlnk);
            Ptotaldisc[i_kdisc] = Pdisc[i_kdisc];
            PPRIMdisc[i_kdisc] = exp(lnpPRIMk_l[ksz-1] + ln_pPRIMk_slope_hi * dlnk);
        }
    }

    int Nside;

    /*
     To be safe we're shifting the range of kmaxes and kmins used in the analysis in order to be able to compute the AP,
     this should be increased if needed
     */

    if (pnlpt->AP_effect == AP_effect_yes) {
        Nside = 10;
    } else {
        Nside = 1;
    }
    double kmaxnew = kdisc[Nmax - 1 - Nside];
    double kminnew = kdisc[Nside];

    /* --- Compute velocity dispersion sigma_v^2 = int P_L(k)/(6pi^2) dk --- */
    /* Used for Fingers-of-God damping in IR resummation */
    sigmav = 0.;
    for (index_k = 0; index_k < pnlpt->k_size - 1; index_k++) {
/* DW: activate the exact-time coupled-fluid path consistently throughout the loop calculation. */
        double Plo=(qcdm_qscf_qs_active==_TRUE_)?pk_l[index_k]:exp(lnpk_l[index_k]);
        double Phi=(qcdm_qscf_qs_active==_TRUE_)?pk_l[index_k+1]:exp(lnpk_l[index_k+1]);
        sigmav += (lnk_l[index_k + 1] - lnk_l[index_k]) *
                  (exp(lnk_l[index_k + 1]) * Phi + Plo * exp(lnk_l[index_k])) /
                  2. / (6. * (M_PI * M_PI));
    }

    /* --- IR Resummation (Senatore & Zaldarriaga method) --- */
    /* irindex=0: no-wiggle contributions only
     * irindex=1: wiggly (BAO) contributions included via damped oscillations */
    int irindex = 0;

/* DW: allow the configured CLASS-PT IR-resummation path in the coupled-fluid calculation. */
    if (pnlpt->irres == irres_yes) {
        irindex = 1;
        TIMER_START(ir);
        class_call(nonlinear_pt_ir_resummation(
                       pnlpt, pba, ppm, pth, Nmax,
                       lnk_l, lnpk_l, myddlnpk, kdisc, Pdisc, PPRIMdisc,
                       Pnw, Pw, Pbin, Ptree, Tnw, Tw, Tbin,
                       &SigmaBAO, &deltaSigmaBAO),
                   pnlpt->error_message, pnlpt->error_message);
        TIMER_ADD(ir);
    } /* End of IR resummation conditional expression */

    else
    {
/* DW: allow the configured CLASS-PT IR-resummation path in the coupled-fluid calculation. */
        if (pnlpt->nonlinear_pt_verbose > 0)
            printf("IR resummation skipped.\n");
        int index_kd = 0;
        for (index_kd = 0; index_kd < Nmax; index_kd++) {
            Pbin[index_kd] = Pdisc[index_kd];
            Ptree[index_kd] = Pbin[index_kd];
            Pnw[index_kd] = Pdisc[index_kd];
            Pw[index_kd] = 0.;
            Tbin[index_kd] = pow(Pdisc[index_kd] / PPRIMdisc[index_kd], 0.5) * (5. / 3.);
            Tnw[index_kd] = Tbin[index_kd];
            Tw[index_kd] = 0.;
        }
    }

    if ((qcdm_qscf_qs_active==_TRUE_)&&(pnlpt->irres==irres_yes)) {
        class_call(qcdm_tf_compute_ir_state(pba,pnlpt,tau,Nmax,kdisc,Pnw,
                                             pth->rs_d,qcdm_tf_ir_ks_hmpc()*pba->h,&qcdm_ir_state),
                   pnlpt->error_message,pnlpt->error_message);
        if (pnlpt->nonlinear_pt_verbose>0) {
            double rel=fabs(qcdm_ir_state.Sigma_DD-SigmaBAO)/MAX(1.e-12,fabs(SigmaBAO));
            printf("DW/QCDM-IR: smooth-displacement cross-check Sigma_DD vs legacy SigmaBAO: %e vs %e (rel=%e).\n",
                   qcdm_ir_state.Sigma_DD,SigmaBAO,rel);
        }
    }

    class_call(array_spline_table_columns(kdisc, Nmax, Pnw, 1, dd_Pnw, _SPLINE_NATURAL_, pnlpt->error_message), pnlpt->error_message, pnlpt->error_message);
    last_index = 0;
    double Pnw_ap_out = 0;

    class_call(array_spline_table_columns(kdisc, Nmax, Pw, 1, dd_Pw, _SPLINE_NATURAL_, pnlpt->error_message), pnlpt->error_message, pnlpt->error_message);
    double Pw_ap_out = 0;

    class_call(array_spline_table_columns(kdisc, Nmax, Tnw, 1, dd_Tnw, _SPLINE_NATURAL_, pnlpt->error_message), pnlpt->error_message, pnlpt->error_message);
    last_index = 0;
    double Tnw_ap_out = 0;

    class_call(array_spline_table_columns(kdisc, Nmax, Tw, 1, dd_Tw, _SPLINE_NATURAL_, pnlpt->error_message), pnlpt->error_message, pnlpt->error_message);
    double Tw_ap_out = 0;

    /* --- FFTLog basis: decompose P(k) = sum_j c_j * k^{b + 2*pi*i*j/(N*Delta)} --- */
    /* etam[j] = b + 2*pi*i*j/(N*Delta) are the complex FFTLog exponents;
     * b is the bias parameter controlling convergence */

    double b = -0.3;
    double b_transfer = -0.8;

    /* DW speed: exact-QS bypasses the EdS FFTLog matrices completely. */
    double complex *etam = NULL;
    double complex *etam_transfer = NULL;
    double complex *cmsym = NULL;
    double complex *cmsym_transfer = NULL;
    int index_c = 0;
    int stepsize = 1;

    if (qcdm_qscf_qs_active != _TRUE_) {
        class_alloc(etam, (Nmax + 1) * sizeof(complex double), pnlpt->error_message);
        class_alloc(etam_transfer, (Nmax + 1) * sizeof(complex double), pnlpt->error_message);
        for (int index_c_local = 0; index_c_local < Nmax + 1; index_c_local++) {
            js[index_c_local] = index_c_local - Nmaxd / 2;
            etam[index_c_local] = b + 2. * M_PI * _Complex_I * js[index_c_local] / Nmaxd / Delta;
            etam_transfer[index_c_local] = b_transfer + 2. * M_PI * _Complex_I * js[index_c_local] / Nmaxd / Delta;
        }
        TIMER_START(fft);
        FFT_TO_CMSYM(Pbin, b, etam, cmsym);
        FFT_TO_CMSYM(Tbin, b_transfer, etam_transfer, cmsym_transfer);
        TIMER_ADD(fft);
    }


    /* Load precomputed PT kernel matrices into complex arrays */

    double complex nu1, nu2, nu12;

    /* Offset constants for log-safe storage: output arrays store
     * P(k) + offset so that values remain positive for log interpolation
     * in the spectra module. Different offsets suit different magnitude ranges:
     *   large_for_logs_big    = 1e6  (1-loop RSD multipoles, bias P22)
     *   large_for_logs_matter = 5e4  (matter 1-loop)
     *   large_for_logs_fNL    = 5e4  (fNL P12 corrections)
     *   large_for_logs_small  = 100  (tidal field integrals Id2, IG2, IFG2)
     *   epsilon_for_logs_fNL  = 1e-6 (floor for extrapolation regions) */
    double epsilon_for_logs_fNL = 1.e-6;
    double large_for_logs_fNL = 5.e4;
    double large_for_logs_matter = 5.e4;
    double large_for_logs_big = 1000000.;
    double large_for_logs_small = 100.;
/* DW: activate the exact-time coupled-fluid path consistently throughout the loop calculation. */

    if (qcdm_qscf_qs_active == _TRUE_) {
/* DW: activate the exact-time coupled-fluid path consistently throughout the loop calculation. */
        qcdm_qscf_qs_configure_runtime(pnlpt);
        if (pnlpt->qcdm_kernel_cache_started == _FALSE_) {
            qcdm_qscf_qs_cache_reset();
            pnlpt->qcdm_kernel_cache_started = _TRUE_;
            if (pnlpt->nonlinear_pt_verbose > 0)
                printf("DW/QCDM-QSCF: exact-QS kernel cache initialized for this cosmology; retaining all requested redshifts for fast tracer reweighting.\n");
        }
        else if (pnlpt->nonlinear_pt_verbose > 1) {
            printf("DW/QCDM-QSCF: reusing exact-QS kernel cache across requested redshifts.\n");
        }
    }

    /* ================================================================== */
    /* === MATTER ONE-LOOP INTEGRALS (P13 + P22) === */
    /* ================================================================== */
    /* The one-loop SPT integrals are computed via matrix multiplication
     * on the FFTLog basis coefficients c_m:
     *   f13[k] = sum_m c_m * M13_m * k^{eta_m}  (vector dot product)
     *   f22[k] = c^T * M22 * c                   (symmetric quadratic form)
     * where M13, M22 are precomputed kernel matrices encoding the F_2 and
     * F_3 symmetrized perturbation theory kernels (EdS approximation).
     *
     * P13UV is the UV counterterm that ensures UV convergence.
     * P_CTR = k^2 * P_L is the leading EFT counterterm (speed of sound).
     * cutoff = 3*h applies exp(-(k/cutoff)^6) to suppress numerical noise. */
    double cutoff = 3. * pba->h;
    double *P13;
    double *P13UV;
    double *P1loop;
    double *f13; /* Now real-valued */
    class_alloc(P13, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(P13UV, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(P1loop, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(f13, Nmax * sizeof(double), pnlpt->error_message);

    double *f22; /* Now real-valued */
    double *P22;
    double *P_CTR;
    class_alloc(f22, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(P22, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(P_CTR, Nmax * sizeof(double), pnlpt->error_message);

    double *P12_fNL;
    double *f12; /* Now real-valued */
    class_alloc(P12_fNL, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(f12, Nmax * sizeof(double), pnlpt->error_message);

    double *P12_fNL_ortho;
    class_alloc(P12_fNL_ortho, Nmax * sizeof(double), pnlpt->error_message);

    int count = 0;

    /* has_fNL: controls whether fNL (equilateral/orthogonal) P12 corrections
     * and their orthogonal variants are computed. Set to 1 when fNL is enabled. */
    int has_fNL = 0;
    if (pnlpt->fNL_equil_ortho_switch == fNL_equil_ortho_yes)
        has_fNL = 1;

    /* DW: build the physical primordial Phi power and its transfer to the
     * selected clustering density for exact-time PNG integrals.  This path is
     * restricted to the adiabatic scalar initial condition used by the c-c'
     * full-shape setup. */
    double *qcdm_Pphi=NULL,*qcdm_Mphi=NULL;
    if ((has_fNL==1)&&(qcdm_qscf_qs_active==_TRUE_)) {
        class_test(ppm->ic_size[pnlpt->index_md_scalars] != 1,
                   pnlpt->error_message,
                   "QCDM exact-time PNG currently requires one adiabatic scalar initial condition");
        class_alloc(qcdm_Pphi,Nmax*sizeof(double),pnlpt->error_message);
        class_alloc(qcdm_Mphi,Nmax*sizeof(double),pnlpt->error_message);
        double *primordial_pk_qcdm;
        class_alloc(primordial_pk_qcdm,ppm->ic_ic_size[pnlpt->index_md_scalars]*sizeof(double),pnlpt->error_message);
        for (int j=0;j<Nmax;j++) {
            class_call(primordial_spectrum_at_k(ppm,pnlpt->index_md_scalars,linear,kdisc[j],primordial_pk_qcdm),
                       ppm->error_message,pnlpt->error_message);
            double PRdim=2.0*_PI_*_PI_*primordial_pk_qcdm[0]/pow(kdisc[j],3.0);
            qcdm_Pphi[j]=(9.0/25.0)*PRdim;
            class_test((qcdm_Pphi[j]<=0.0)||(Pdisc[j]<0.0),pnlpt->error_message,
                       "Invalid primordial/density power for exact PNG at k=%e: Pphi=%e Plinear=%e",
                       kdisc[j],qcdm_Pphi[j],Pdisc[j]);
            qcdm_Mphi[j]=sqrt(Pdisc[j]/qcdm_Pphi[j]);
        }
        free(primordial_pk_qcdm);
    }
    else {
        memset(P12_fNL,0,Nmax*sizeof(double));
        memset(P12_fNL_ortho,0,Nmax*sizeof(double));
    }

    /* DW: snapshot the cosmology-dependent inputs needed for later fast
     * tracer/nuisance reweighting.  The exact kernel ODE cache itself is
     * retained across all z_pk values by the reset policy above. */
    if (qcdm_qscf_qs_active==_TRUE_) {
        class_call(qcdm_tf_fast_cache_store(pba,pnlpt,tau,Nmax,kdisc,Pdisc,
                                             qcdm_Pphi,qcdm_Mphi),
                   pnlpt->error_message,pnlpt->error_message);
        if (qcdm_ir_state.enabled==_TRUE_) {
            class_call(qcdm_tf_ir_fast_cache_store(pba,pnlpt,tau,Nmax,Pnw,Pw,&qcdm_ir_state),
                       pnlpt->error_message,pnlpt->error_message);
        }
    }

    int Np1 = Nmax + 1; /* Number of FFTLog coefficients */

    /* DW speed: legacy FFTLog/BLAS workspaces are unused by exact-QS. */
    double complex *x = NULL, *x_w = NULL;
    double complex *x_transfer = NULL, *x_w_transfer = NULL;
    double *Xr = NULL, *Xi = NULL, *Xr_transfer = NULL, *Xi_transfer = NULL;
    double **ws_Mr = NULL, **ws_Mi = NULL, **ws_Y1 = NULL, **ws_Y2 = NULL;
    double **ws_dot_mr = NULL, **ws_dot_mi = NULL;
    double *Mr_ws = NULL, *Mi_ws = NULL, *Y1_ws = NULL, *Y2_ws = NULL;
    double *dot_mr_ws = NULL, *dot_mi_ws = NULL;
    double *lnk_cache = NULL, *kb_cache = NULL, *kb_transfer_cache = NULL;
    double *sc_cos = NULL, *sc_sin = NULL, *sc_cos_t = NULL, *sc_sin_t = NULL;
    int _blas_threads_saved = classpt_blas_get_num_threads();
    int _n_threads = 0;

    if (qcdm_qscf_qs_active != _TRUE_) {
        class_alloc(x, (Nmax + 1) * sizeof(complex double), pnlpt->error_message);
        class_alloc(x_w, (Nmax + 1) * sizeof(complex double), pnlpt->error_message);
        class_alloc(x_transfer, (Nmax + 1) * sizeof(complex double), pnlpt->error_message);
        class_alloc(x_w_transfer, (Nmax + 1) * sizeof(complex double), pnlpt->error_message);

        _n_threads = omp_get_max_threads();
        classpt_blas_set_num_threads(1);
        ws_Mr = (double **)malloc(_n_threads * sizeof(double *));
        ws_Mi = (double **)malloc(_n_threads * sizeof(double *));
        ws_Y1 = (double **)malloc(_n_threads * sizeof(double *));
        ws_Y2 = (double **)malloc(_n_threads * sizeof(double *));
        ws_dot_mr = (double **)malloc(_n_threads * sizeof(double *));
        ws_dot_mi = (double **)malloc(_n_threads * sizeof(double *));
        for (int _t = 0; _t < _n_threads; _t++) {
            ws_Mr[_t]     = (double *)calloc(Np1 * Np1, sizeof(double));
            ws_Mi[_t]     = (double *)calloc(Np1 * Np1, sizeof(double));
            ws_Y1[_t]     = (double *)calloc(Np1 * Nmax, sizeof(double));
            ws_Y2[_t]     = (double *)calloc(Np1 * Nmax, sizeof(double));
            ws_dot_mr[_t] = (double *)calloc(Np1, sizeof(double));
            ws_dot_mi[_t] = (double *)calloc(Np1, sizeof(double));
        }
        Mr_ws = ws_Mr[0]; Mi_ws = ws_Mi[0];
        Y1_ws = ws_Y1[0]; Y2_ws = ws_Y2[0];
        dot_mr_ws = ws_dot_mr[0]; dot_mi_ws = ws_dot_mi[0];

        class_alloc(Xr, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(Xi, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(Xr_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(Xi_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(lnk_cache, Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(kb_cache, Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(kb_transfer_cache, Nmax * sizeof(double), pnlpt->error_message);
        for (int j = 0; j < Nmax; j++) {
            lnk_cache[j] = log(kdisc[j]);
            kb_cache[j] = pow(kdisc[j], b);
            kb_transfer_cache[j] = pow(kdisc[j], b_transfer);
        }
        class_alloc(sc_cos, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(sc_sin, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(sc_cos_t, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(sc_sin_t, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        precompute_sincos_table(Nmax, Np1, etam, lnk_cache, sc_cos, sc_sin);
        precompute_sincos_table(Nmax, Np1, etam_transfer, lnk_cache, sc_cos_t, sc_sin_t);

        TIMER_START(buildx);
        build_X_from_tables(Nmax, Np1, cmsym, kb_cache, sc_cos, sc_sin, Xr, Xi);
        build_X_from_tables(Nmax, Np1, cmsym_transfer, kb_transfer_cache, sc_cos_t, sc_sin_t, Xr_transfer, Xi_transfer);
        TIMER_ADD(buildx);
    }

    /* Compute real-space matter P13/P22.
     * DW: QCDM-QSCF: if qcdm is present, bypass the EdS FFTLog matrices and
     * use quasi-static exact kernels for baryons + qcdm. */
    if (qcdm_qscf_qs_active == _TRUE_) {
        TIMER_START(qcdm_real);
        if (pnlpt->nonlinear_pt_verbose > 0) {
            printf("DW/QCDM-QSCF: using quasi-static exact real-space kernels for P22/P13.\n");
        }
        if (qcdm_ir_state.enabled==_TRUE_) {
            class_call(qcdm_qscf_qs_exact_realspace_loops_ir(pba,pnlpt,tau,Nmax,kdisc,
                                                              Pnw,Pw,&qcdm_ir_state,cutoff,
                                                              P22,P13,P1loop),
                       pnlpt->error_message,pnlpt->error_message);
        }
        else {
            class_call(qcdm_qscf_qs_exact_realspace_loops(pba,
                                                          pnlpt,
                                                          tau,
                                                          Nmax,
                                                          kdisc,
                                                          Pbin,
                                                          cutoff,
                                                          P22,
                                                          P13,
                                                          P1loop),
                       pnlpt->error_message,
                       pnlpt->error_message);
        }
        for (index_j = 0; index_j < Nmax; index_j++) {
            P13UV[index_j] = 0.0;
        }
        /* DW: replace the legacy EdS M12 PNG templates by direct exact-time
         * equilateral/orthogonal P12 integrals in the coupled branch. */
        if (has_fNL==1) {
            class_call(qcdm_qscf_qs_exact_png_matter(pba,pnlpt,tau,Nmax,kdisc,
                                                     qcdm_Pphi,qcdm_Mphi,cutoff,
                                                     P12_fNL,P12_fNL_ortho),
                       pnlpt->error_message,pnlpt->error_message);
        }
/* DW: write the exact-time tracer tree, loop, EFT and stochastic decomposition. */
        class_call(qcdm_qscf_write_real_tracer_at_tau(pba,pnlpt,tau,Nmax,kdisc,Pbin,
                                                       Pnw,Pw,&qcdm_ir_state,
                                                       qcdm_Pphi,qcdm_Mphi,cutoff),
                   pnlpt->error_message,pnlpt->error_message);
        TIMER_ADD(qcdm_real);
    }
    else {
        /* Standard CLASS-PT EdS FFTLog branch. */
        {
            double *f13_real = malloc(Nmax * sizeof(double));
            TIMER_START(blas);
            batch_dot_product(Np1, Nmax, Xr, Xi, pnlpt->M13_oneline_complex, f13_real, dot_mr_ws, dot_mi_ws);
            TIMER_ADD(blas);
            for (index_j = 0; index_j < Nmax; index_j++) {
                P13UV[index_j] = -61. * Pbin[index_j] * kdisc[index_j] * kdisc[index_j] * sigmav / 105.;
                { double kc_ = kdisc[index_j] / cutoff; double kc2_ = kc_ * kc_; double kc6_ = kc2_ * kc2_ * kc2_;
                P13[index_j] = (kdisc[index_j] * kdisc[index_j] * kdisc[index_j] * f13_real[index_j] * Pbin[index_j] + P13UV[index_j]) * exp(-kc6_); }
            }
            free(f13_real);
        }
        index_j = 0;
        count = 0;
        {
            double *f22_real = malloc(Nmax * sizeof(double));
            TIMER_START(blas);
            batch_quadratic_form(Np1, Nmax, pnlpt->M22_oneline_complex, Xr, Xi, f22_real, Mr_ws, Mi_ws, Y1_ws, Y2_ws);
            TIMER_ADD(blas);
            for (index_j = 0; index_j < Nmax; index_j++) {
                P22[index_j] = kdisc[index_j] * kdisc[index_j] * kdisc[index_j] * f22_real[index_j] * exp(-pow(kdisc[index_j] / cutoff, 6.));
                P1loop[index_j] = P13[index_j] + P22[index_j];
            }
            free(f22_real);
        }
    }

    /* DW: use legacy EdS FFTLog PNG matrices only outside the exact c-c' branch. */
    if (has_fNL && qcdm_qscf_qs_active != _TRUE_) {
        double *f12_real = malloc(Nmax * sizeof(double));
        TIMER_START(blas);
        batch_quadratic_form(Np1, Nmax, pnlpt->M12_oneline_complex, Xr_transfer, Xi_transfer, f12_real, Mr_ws, Mi_ws, Y1_ws, Y2_ws);
        TIMER_ADD(blas);
        for (index_j = 0; index_j < Nmax; index_j++) {
            P12_fNL[index_j] = Tbin[index_j] * kdisc[index_j] * kdisc[index_j] * kdisc[index_j] * f12_real[index_j] * exp(-pow(kdisc[index_j] / cutoff, 6.));
        }

        TIMER_START(blas);
        batch_quadratic_form(Np1, Nmax, pnlpt->M12_oneline_complex_ortho, Xr_transfer, Xi_transfer, f12_real, Mr_ws, Mi_ws, Y1_ws, Y2_ws);
        TIMER_ADD(blas);
        for (index_j = 0; index_j < Nmax; index_j++) {
            P12_fNL_ortho[index_j] = Tbin[index_j] * kdisc[index_j] * kdisc[index_j] * kdisc[index_j] * f12_real[index_j] * exp(-pow(kdisc[index_j] / cutoff, 6.));
        }
        free(f12_real);
    }

    /* --- Assemble EFT counterterm and spline-interpolate all 1-loop results --- */
    /* DW: construct the standard k^2 P_bin matter counterterm basis before
     * applying the user-fitted coefficient in the diagnostic breakdown. */
    for (index_j = 0; index_j < Nmax; index_j++) {
        P_CTR[index_j] = kdisc[index_j] * kdisc[index_j] * Pbin[index_j];
    }

    /* DW: write every cumulative matter contribution while P22 and P13 are
     * still separately available on the native loop grid. */
    if (qcdm_qscf_qs_active == _TRUE_) {
        class_call(qcdm_qscf_write_matter_breakdown_at_tau(pba,pnlpt,tau,Nmax,kdisc,
                                                            Pdisc,Ptotaldisc,Ptree,P22,P13,
                                                            P12_fNL,P12_fNL_ortho,P_CTR),
                   pnlpt->error_message,pnlpt->error_message);
    }

    double *ddpk_nl;
    double *ddpk_nl_fNL;
    double *ddpk_nl_fNL_ortho;
    double *ddpk_CTR;
    double *ddpk_Tree;
    class_alloc(ddpk_nl, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(ddpk_nl_fNL, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(ddpk_nl_fNL_ortho, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(ddpk_CTR, Nmax * sizeof(double), pnlpt->error_message);
    class_alloc(ddpk_Tree, Nmax * sizeof(double), pnlpt->error_message);

    /* DW: QCDM: for exact-QS outputs, avoid cubic splines on raw signed loop
     * arrays. The signed P_1loop can cross zero and can be noisy at the edges;
     * cubic interpolation can overshoot badly and, in some builds, produce NaNs.
     * We use explicit linear interpolation in log(k) below for the QCDM branch. */
    if (qcdm_qscf_qs_active != _TRUE_) {
        SPLINE_SETUP(kdisc, Nmax, P1loop, ddpk_nl, _SPLINE_NATURAL_);
        if (has_fNL) {
            SPLINE_SETUP(kdisc, Nmax, P12_fNL, ddpk_nl_fNL, _SPLINE_NATURAL_);
            SPLINE_SETUP(kdisc, Nmax, P12_fNL_ortho, ddpk_nl_fNL_ortho, _SPLINE_NATURAL_);
        }
        SPLINE_SETUP(kdisc, Nmax, P_CTR, ddpk_CTR, _SPLINE_NATURAL_);
        SPLINE_SETUP(kdisc, Nmax, Ptree, ddpk_Tree, _SPLINE_NATURAL_);
    }

    double pk_nl_out;
    double pk_nl_fNL_out;
    double pk_nl_fNL_out_ortho;
    double pk_CTR_out;
    double pk_Tree_out;

    last_index = 0;
    last_index2 = 0;
    int last_index3 = 0;
    int last_index4 = 0;

    int last_index4_ortho = 0;

    /* --- Interpolate matter 1-loop results to the output k-grid --- */
    for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
        if (pnlpt->k[index_k] >= kmin && pnlpt->k[index_k] <= kmax) {
            if (qcdm_qscf_qs_active == _TRUE_) {
                /* Science-run QCDM path: log-k linear interpolation.
                 * Ptree and P_CTR are positive but P1loop is signed, so use
                 * direct linear interpolation for all three. This avoids the
                 * cubic spline NaNs/overshoots that were contaminating the
                 * custom output files. */
                pk_nl_out = qcdm_qscf_interp_linear(Nmax,kdisc,P1loop,pnlpt->k[index_k]);
                pk_CTR_out = qcdm_qscf_interp_linear(Nmax,kdisc,P_CTR,pnlpt->k[index_k]);
                pk_Tree_out = qcdm_qscf_interp_linear(Nmax,kdisc,Ptree,pnlpt->k[index_k]);
                class_test((qcdm_qscf_finite_double(pk_nl_out) == 0) ||
                           (qcdm_qscf_finite_double(pk_CTR_out) == 0) ||
                           (qcdm_qscf_finite_double(pk_Tree_out) == 0),
                           pnlpt->error_message,
                           "Non-finite QCDM exact-QS interpolation to output grid: index_k=%d k=%e Ploop=%e Pct=%e Ptree=%e",
                           index_k,pnlpt->k[index_k],pk_nl_out,pk_CTR_out,pk_Tree_out);
            }
            else {
                SPLINE_EVAL(kdisc, Nmax, P1loop, ddpk_nl, pnlpt->k[index_k], &last_index2, &pk_nl_out);
                SPLINE_EVAL(kdisc, Nmax, P_CTR, ddpk_CTR, pnlpt->k[index_k], &last_index, &pk_CTR_out);
                SPLINE_EVAL(kdisc, Nmax, Ptree, ddpk_Tree, pnlpt->k[index_k], &last_index3, &pk_Tree_out);
            }

            pk_nl[index_k] = pk_nl_out + large_for_logs_matter;

            if (has_fNL) {
                if (qcdm_qscf_qs_active == _TRUE_) {
                    /* DW: interpolate the exact-time c-c' equilateral and
                     * orthogonal unit-f_NL P12 templates instead of erasing
                     * them as the previous exact-QS branch did. */
                    pk_nl_fNL_out=qcdm_qscf_interp_linear(Nmax,kdisc,P12_fNL,pnlpt->k[index_k]);
                    pk_nl_fNL_out_ortho=qcdm_qscf_interp_linear(Nmax,kdisc,P12_fNL_ortho,pnlpt->k[index_k]);
                }
                else {
                    SPLINE_EVAL(kdisc, Nmax, P12_fNL, ddpk_nl_fNL, pnlpt->k[index_k], &last_index4, &pk_nl_fNL_out);
                    SPLINE_EVAL(kdisc, Nmax, P12_fNL_ortho, ddpk_nl_fNL_ortho, pnlpt->k[index_k], &last_index4_ortho, &pk_nl_fNL_out_ortho);
                }
                pk_nl_fNL[index_k] = pk_nl_fNL_out + large_for_logs_fNL;
                pk_nl_fNL_ortho[index_k] = pk_nl_fNL_out_ortho + large_for_logs_fNL;
            } else {
                pk_nl_fNL[index_k] = large_for_logs_fNL;
                pk_nl_fNL_ortho[index_k] = large_for_logs_fNL;
            }

            if (pk_CTR_out <= 0)
                pk_CTR_out = 1e-16;
            if (pk_Tree_out <= 0)
                pk_Tree_out = 1e-16;
            pk_CTR[index_k] = pk_CTR_out;
            pk_Tree[index_k] = pk_Tree_out;
        } else {
            if (pnlpt->k[index_k] < kmin) {
                if (qcdm_qscf_qs_active == _TRUE_) {
                    pk_nl[index_k] = large_for_logs_matter;
                }
                else {
                    pk_nl[index_k] = large_for_logs_matter - 61. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav / 105.;
                }
            }
            if (pnlpt->k[index_k] > kmax) {
                pk_nl[index_k] = large_for_logs_matter;
            }
            pk_nl_fNL[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_nl_fNL_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
/* DW: activate the exact-time coupled-fluid path consistently throughout the loop calculation. */
            if (qcdm_qscf_qs_active == _TRUE_) {
                pk_CTR[index_k] = pk_l[index_k]*pnlpt->k[index_k]*pnlpt->k[index_k];
                pk_Tree[index_k] = pk_l[index_k];
            }
            else {
                pk_CTR[index_k] = exp(lnpk_l[index_k] + 2. * lnk_l[index_k]);
                pk_Tree[index_k] = exp(lnpk_l[index_k]);
            }
        }

        class_test((qcdm_qscf_finite_double(pk_nl[index_k]) == 0) ||
                   (qcdm_qscf_finite_double(pk_CTR[index_k]) == 0) ||
                   (qcdm_qscf_finite_double(pk_Tree[index_k]) == 0),
                   pnlpt->error_message,
                   "Non-finite matter output arrays after interpolation: index_k=%d k=%e pk_nl=%e pk_CTR=%e pk_Tree=%e",
                   index_k,pnlpt->k[index_k],pk_nl[index_k],pk_CTR[index_k],pk_Tree[index_k]);
    }

    if (qcdm_qscf_qs_active == _TRUE_) {
        /* DW: exact nonlinear kernels describe the clustering baryon+CDM+
         * QCDM sector. With massive neutrinos, require cb=yes and reconstruct
         * total matter by keeping ncdm linear:
         * P_m^NL=P_m,L+f_cl^2(P_cl^NL-P_cl,L). */
        class_test((pba->has_ncdm==_TRUE_)&&(pnlpt->cb!=_TRUE_),
                   pnlpt->error_message,
                   "DW: exact c-c' EFTofLSS with massive ncdm requires cb=yes; cb=no would feed the non-clustering neutrino spectrum into pressureless fluid kernels.");
        double fcluster=1.0;
        class_call(qcdm_qscf_clustering_fraction_at_tau(pba,tau,&fcluster,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);
        double fcluster2=fcluster*fcluster;
        for (index_k=0; index_k<pnlpt->k_size; index_k++) {
            double Pcl=pk_l[index_k];
            double Ptot=pk_l_total[index_k];
            double Ptree_cl=pk_Tree[index_k];
            double Ploop_cl=pk_nl[index_k]-large_for_logs_matter;
            pk_Tree[index_k]=Ptot+fcluster2*(Ptree_cl-Pcl);
            pk_nl[index_k]=large_for_logs_matter+fcluster2*Ploop_cl;
            pk_CTR[index_k]*=fcluster2;
            if (has_fNL) {
                pk_nl_fNL[index_k]=large_for_logs_fNL+fcluster2*(pk_nl_fNL[index_k]-large_for_logs_fNL);
                pk_nl_fNL_ortho[index_k]=large_for_logs_fNL+fcluster2*(pk_nl_fNL_ortho[index_k]-large_for_logs_fNL);
            }
        }
    }

    /* ================================================================== */
    /* === REDSHIFT-SPACE DISTORTIONS (RSD) MULTIPOLES === */
    /* ================================================================== */
    /* Compute P_ell(k) for ell=0,2,4 (monopole, quadrupole, hexadecapole).
     * Each multipole is decomposed into velocity-velocity (vv),
     * velocity-density (vd), and density-density (dd) components,
     * with powers of the growth rate f from the RSD kernel Z_n.
     * Two IR passes: irindex=0 uses P_nw, irindex=1 adds P_w corrections. */

    if ((pnlpt->rsd == rsd_yes) && (qcdm_qscf_qs_active != _TRUE_)) {
        if (pnlpt->nonlinear_pt_verbose > 0)
            printf("Computing RSD...\n");
        if (pnlpt->nonlinear_pt_verbose > 0)
            printf("Logarithmic growth factor f=%.16e\n", f);

        /* RSD 1-loop power spectrum arrays: tree, P13, P13UV, P1loop, P22, P12 per (ell, component) */
        double *Ptree_0_vv, *Ptree_0_vd, *Ptree_0_dd, *Ptree_2_vv, *Ptree_2_vd, *Ptree_2_dd;
        double *Ptree_4_vv, *Ptree_4_vd;
        double *P13_0_vv, *P13UV_0_vv, *P1loop_0_vv, *P22_0_vv, *P12_0_vv, *P12_0_vv_ortho;
        double *P13_0_vd, *P13UV_0_vd, *P1loop_0_vd, *P22_0_vd, *P12_0_vd, *P12_0_vd_ortho;
        double *P13_0_dd, *P13UV_0_dd, *P1loop_0_dd, *P22_0_dd, *P12_0_dd, *P12_0_dd_ortho;
        double *P13_2_vv, *P13UV_2_vv, *P1loop_2_vv, *P22_2_vv, *P12_2_vv, *P12_2_vv_ortho;
        double *P13_2_vd, *P13UV_2_vd, *P1loop_2_vd, *P22_2_vd, *P12_2_vd, *P12_2_vd_ortho;
        double *P13_2_dd, *P13UV_2_dd, *P1loop_2_dd, *P22_2_dd, *P12_2_dd, *P12_2_dd_ortho;
        double *P13_4_vv, *P13UV_4_vv, *P1loop_4_vv, *P22_4_vv, *P12_4_vv, *P12_4_vv_ortho;
        double *P13_4_vd, *P13UV_4_vd, *P1loop_4_vd, *P22_4_vd, *P12_4_vd, *P12_4_vd_ortho;
        double *P22_4_dd, *P1loop_4_dd, *P12_4_dd, *P12_4_dd_ortho;
        double *P_CTR_0, *P_CTR_2, *P_CTR_4;
        { /* Batch-allocate all RSD multipole arrays (each Nmax doubles) */
            double **_rsd_ptrs[] = {
                /* P13, P13UV, P1loop */
                &P13_0_vv, &P13UV_0_vv, &P1loop_0_vv, &P13_0_vd, &P13UV_0_vd, &P1loop_0_vd,
                &P13_0_dd, &P13UV_0_dd, &P1loop_0_dd, &P13_2_vv, &P13UV_2_vv, &P1loop_2_vv,
                &P13_2_vd, &P13UV_2_vd, &P1loop_2_vd, &P13_2_dd, &P13UV_2_dd, &P1loop_2_dd,
                &P13_4_vv, &P13UV_4_vv, &P1loop_4_vv, &P13_4_vd, &P13UV_4_vd, &P1loop_4_vd,
                &P22_4_dd, &P1loop_4_dd, &P12_4_dd, &P12_4_dd_ortho,
                &P_CTR_0, &P_CTR_2, &P_CTR_4,
                /* Ptree */
                &Ptree_0_vv, &Ptree_0_vd, &Ptree_0_dd, &Ptree_2_vv, &Ptree_2_vd, &Ptree_2_dd,
                &Ptree_4_vv, &Ptree_4_vd,
                /* P22 */
                &P22_0_vv, &P22_0_vd, &P22_0_dd, &P22_2_vv, &P22_2_vd, &P22_2_dd,
                &P22_4_vv, &P22_4_vd,
                /* P12 */
                &P12_0_vv, &P12_0_vd, &P12_0_dd, &P12_2_vv, &P12_2_vd, &P12_2_dd,
                &P12_4_vv, &P12_4_vd,
                /* P12_ortho */
                &P12_0_vv_ortho, &P12_0_vd_ortho, &P12_0_dd_ortho,
                &P12_2_vv_ortho, &P12_2_vd_ortho, &P12_2_dd_ortho,
                &P12_4_vv_ortho, &P12_4_vd_ortho};
            for (int ai_ = 0; ai_ < 63; ai_++)
                class_alloc(*_rsd_ptrs[ai_], Nmax * sizeof(double), pnlpt->error_message);
        }

        if (irindex == 0) {
            /* --- Fill RSD M22 kernels with f-dependent polynomials ---
             *
             * The M22 kernel matrices for each RSD mu-power are derived from the
             * general M22 kernel by multiplying with rational polynomials in
             * nu1 = -eta_i/2, nu2 = -eta_l/2 (the FFTLog exponents) and powers
             * of the growth rate f. These expressions encode the angular
             * dependence of the redshift-space Z_2 kernel:
             *
             *   Z_2(k1,k2) = b1*F_2(k1,k2) + f*mu^2*k/k1/k2*[...] + ...
             *
             * Each M22 mu-power captures the coefficient of mu^{2n} in the
             * one-loop integrand (see Eq. 2.12-2.15 of arXiv:2004.10607).
             *
             * M12 kernels (fNL) are precomputed and simply combined with
             * powers of f here. */

            /* === FUSED M22 FILL: all 9 multipole M22 arrays + M12 fNL arrays filled in a single pass === */
            TIMER_START(m22fill);
            {
                int _N = Nmax + 1;
                double f2 = f * f, f3 = f2 * f, f4 = f3 * f;
                #pragma omp parallel for schedule(static)
                for (int _l = 0; _l < _N; _l++) {
                    int _count_base = _l * _N - _l * (_l - 1) / 2;
                    for (int _i = _l; _i < _N; _i++) {
                        int _c = _count_base + (_i - _l);
                        double complex _nu1 = -0.5 * etam[_i];
                        double complex _nu2 = -0.5 * etam[_l];
                        double complex _nu12 = _nu1 + _nu2;

                        /* Shared powers */
                        double complex nu1_sq = _nu1 * _nu1, nu1_cu = nu1_sq * _nu1;
                        double complex nu2_sq = _nu2 * _nu2, nu2_cu = nu2_sq * _nu2;
                        double complex nu12_sq = _nu12 * _nu12;

                        /* Shared denominator and base factor */
                        double complex DENOM = 98. * _nu1 * _nu2 * nu12_sq - 91. * nu12_sq + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.;
                        double complex base = pnlpt->M22_oneline_complex[_c] * 196. / DENOM;

                        /* Shared sub-expressions used across multiple multipoles */
                        double complex A0 = 50. - 9. * _nu2 + 98. * nu1_cu * _nu2 - 35. * nu2_sq + 7. * nu1_sq * (-5. - 18. * _nu2 + 28. * nu2_sq) + _nu1 * (-9. - 66. * _nu2 - 126. * nu2_sq + 98. * nu2_cu);
                        double complex B0 = 46. + 13. * _nu2 + 98. * nu1_cu * _nu2 - 63. * nu2_sq + 7. * nu1_sq * (-9. - 10. * _nu2 + 28. * nu2_sq) + _nu1 * (13. - 138. * _nu2 - 70. * nu2_sq + 98. * nu2_cu);
                        double complex C0 = 10. - _nu2 + 14. * nu1_cu * _nu2 - 17. * nu2_sq + nu1_sq * (-17. + 6. * _nu2 + 28. * nu2_sq) + _nu1 * (-1. - 22. * _nu2 + 6. * nu2_sq + 14. * nu2_cu);
                        double complex D0 = 58. + 3. * _nu2 + 98. * nu1_cu * _nu2 - 91. * nu2_sq + 7. * nu1_sq * (-13. - 2. * _nu2 + 28. * nu2_sq) + _nu1 * (3. - 146. * _nu2 - 14. * nu2_sq + 98. * nu2_cu);

                        /* 0_vv: monopole velocity-velocity */
                        {
                            double complex vv0_f4 = 14. * (24. - 8. * _nu2 - 15. * nu2_sq + 5. * nu2_cu + 5. * nu1_cu * (1. + 7. * _nu2) + 5. * nu1_sq * (-3. - 10. * _nu2 + 14. * nu2_sq) + _nu1 * (-8. - 24. * _nu2 - 50. * nu2_sq + 35. * nu2_cu));
                            double complex vv0_f3 = 18. * (36. - 8. * _nu2 + 70. * nu1_cu * _nu2 - 23. * nu2_sq + nu1_sq * (-23. - 94. * _nu2 + 140. * nu2_sq) + _nu1 * (-8. - 42. * _nu2 - 94. * nu2_sq + 70. * nu2_cu));
                            pnlpt->M22_oneline_0_vv_complex[_c] = base * f2 * (f2 * vv0_f4 + f * vv0_f3 + 9. * A0) / 8820.;
                        }

                        /* 0_vd: monopole velocity-density */
                        {
                            double complex vd0_f3 = 21. * (6. + 3. * _nu2 - 10. * nu2_sq + 2. * nu2_cu + 2. * nu1_cu * (1. + 5. * _nu2) + 2. * nu1_sq * (-5. - 2. * _nu2 + 10. * nu2_sq) + _nu1 * (3. - 24. * _nu2 - 4. * nu2_sq + 10. * nu2_cu));
                            double complex vd0_f2 = 14. * (18. + 11. * _nu2 + 42. * nu1_cu * _nu2 - 31. * nu2_sq + nu1_sq * (-31. - 22. * _nu2 + 84. * nu2_sq) + _nu1 * (11. - 74. * _nu2 - 22. * nu2_sq + 42. * nu2_cu));
                            pnlpt->M22_oneline_0_vd_complex[_c] = base * f * (f2 * vd0_f3 + f * vd0_f2 + 5. * B0) / 1470.;
                        }

                        /* 0_dd: monopole density-density */
                        {
                            double complex dd0_f2 = 98. * (4. - 2. * _nu2 - 5. * nu2_sq + nu2_cu + nu1_cu * (1. + 3. * _nu2) + nu1_sq * (-5. + 2. * _nu2 + 6. * nu2_sq) + _nu1 * (-2. - 4. * _nu2 + 2. * nu2_sq + 3. * nu2_cu));
                            pnlpt->M22_oneline_0_dd_complex[_c] = base * (f2 * dd0_f2 + 70. * f * C0 + 15. * D0) / 2940.;
                        }

                        /* 2_vv: quadrupole velocity-velocity */
                        {
                            double complex qvv_f2 = 231. * (142. - 21. * _nu2 + 280. * nu1_cu * _nu2 - 106. * nu2_sq + 2. * nu1_sq * (-53. - 174. * _nu2 + 280. * nu2_sq) + _nu1 * (-21. - 204. * _nu2 - 348. * nu2_sq + 280. * nu2_cu));
                            double complex qvv_f4 = 49. * (336. - 62. * _nu2 - 255. * nu2_sq + 50. * nu2_cu + 10. * nu1_cu * (5. + 56. * _nu2) + 5. * nu1_sq * (-51. - 142. * _nu2 + 224. * nu2_sq) + _nu1 * (-62. - 486. * _nu2 - 710. * nu2_sq + 560. * nu2_cu));
                            pnlpt->M22_oneline_2_vv_complex[_c] = base * f2 * (396. * A0 + f * qvv_f2 + f2 * qvv_f4) / 135828.;
                        }

                        /* 2_vd: quadrupole velocity-density */
                        {
                            double complex qvd_f3 = 7. * (22. + 11. * _nu2 - 40. * nu2_sq + 4. * nu2_cu + nu1_cu * (4. + 40. * _nu2) + 8. * nu1_sq * (-5. - _nu2 + 10. * nu2_sq) + _nu1 * (11. - 88. * _nu2 - 8. * nu2_sq + 40. * nu2_cu));
                            double complex qvd_f2 = 306. + 161. * _nu2 + 672. * nu1_cu * _nu2 - 538. * nu2_sq + 2. * nu1_sq * (-269. - 134. * _nu2 + 672. * nu2_sq) + _nu1 * (161. - 1196. * _nu2 - 268. * nu2_sq + 672. * nu2_cu);
                            pnlpt->M22_oneline_2_vd_complex[_c] = base * f * (f2 * qvd_f3 + 4. * B0 + f * qvd_f2) / 588.;
                        }

                        /* 2_dd: quadrupole density-density */
                        {
                            double complex qdd_f2 = 26. - 13. * _nu2 - 37. * nu2_sq + 2. * nu2_cu + nu1_cu * (2. + 24. * _nu2) + nu1_sq * (-37. + 22. * _nu2 + 48. * nu2_sq) + _nu1 * (-13. - 26. * _nu2 + 22. * nu2_sq + 24. * nu2_cu);
                            pnlpt->M22_oneline_2_dd_complex[_c] = base * f * (4. * C0 + f * qdd_f2) / 84.;
                        }

                        /* 4_vv: hexadecapole velocity-velocity */
                        {
                            double complex hvv_f2 = 728. * (206. + 420. * nu1_cu * _nu2 + (7. - 208. * _nu2) * _nu2 + 8. * nu1_sq * (-26. + _nu2 * (-53. + 105. * _nu2)) + _nu1 * (7. + 4. * _nu2 * (-108. + _nu2 * (-106. + 105. * _nu2))));
                            double complex hvv_f4 = 147. * (483. + 40. * nu1_cu * (-1. + 28. * _nu2) - 2. * _nu2 * (-57. + 10. * _nu2 * (29. + 2. * _nu2)) + 20. * nu1_sq * (-29. + 2. * _nu2 * (-25. + 56. * _nu2)) + 2. * _nu1 * (57. + 2. * _nu2 * (-327. + 10. * _nu2 * (-25. + 28. * _nu2))));
                            pnlpt->M22_oneline_4_vv_complex[_c] = base * f2 * (1144. * A0 + f * hvv_f2 + f2 * hvv_f4) / 980980.;
                        }

                        /* 4_vd: hexadecapole velocity-density */
                        {
                            double complex hvd_f2 = 14. * (26. + 13. * _nu2 - 60. * nu2_sq - 8. * nu2_cu + nu1_cu * (-8. + 60. * _nu2) + 4. * nu1_sq * (-15. + 4. * _nu2 + 30. * nu2_sq) + _nu1 * (13. - 104. * _nu2 + 16. * nu2_sq + 60. * nu2_cu));
                            double complex hvd_f1 = 11. * (58. + 21. * _nu2 + 112. * nu1_cu * _nu2 - 106. * nu2_sq + 2. * nu1_sq * (-53. - 6. * _nu2 + 112. * nu2_sq) + _nu1 * (21. - 204. * _nu2 - 12. * nu2_sq + 112. * nu2_cu));
                            pnlpt->M22_oneline_4_vd_complex[_c] = base * f2 * (f * hvd_f2 + hvd_f1) / 2695.;
                        }

                        /* 4_dd: hexadecapole density-density */
                        pnlpt->M22_oneline_4_dd_complex[_c] = base * f2 * (2. * _nu1 - 1.) * (2. * _nu2 - 1.) * (1. + _nu12) * (2. + _nu12) / 35.;

                        /* fNL M12 arrays (all multipoles) */
                        if (has_fNL) {
                            pnlpt->M12_oneline_0_vv_complex[_c] = f2 * pnlpt->M12_oneline_complex_matter_multipoles_vv0_f2[_c] + f3 * pnlpt->M12_oneline_complex_matter_multipoles_vv0_f3[_c];
                            pnlpt->M12_oneline_0_vv_complex_ortho[_c] = f2 * pnlpt->M12_oneline_complex_matter_multipoles_vv0_f2_ortho[_c] + f3 * pnlpt->M12_oneline_complex_matter_multipoles_vv0_f3_ortho[_c];
                            pnlpt->M12_oneline_0_vd_complex[_c] = f * pnlpt->M12_oneline_complex_matter_multipoles_vd0_f1[_c] + f2 * pnlpt->M12_oneline_complex_matter_multipoles_vd0_f2[_c];
                            pnlpt->M12_oneline_0_vd_complex_ortho[_c] = f * pnlpt->M12_oneline_complex_matter_multipoles_vd0_f1_ortho[_c] + f2 * pnlpt->M12_oneline_complex_matter_multipoles_vd0_f2_ortho[_c];
                            pnlpt->M12_oneline_0_dd_complex[_c] = pnlpt->M12_oneline_complex_matter_multipoles_dd0_f0[_c] + f * pnlpt->M12_oneline_complex_matter_multipoles_dd0_f1[_c];
                            pnlpt->M12_oneline_0_dd_complex_ortho[_c] = pnlpt->M12_oneline_complex_matter_multipoles_dd0_f0_ortho[_c] + f * pnlpt->M12_oneline_complex_matter_multipoles_dd0_f1_ortho[_c];
                            pnlpt->M12_oneline_2_vv_complex[_c] = 20. / 7. * f2 * pnlpt->M12_oneline_complex_matter_multipoles_vv0_f2[_c] + f3 * pnlpt->M12_oneline_complex_matter_multipoles_vv2_f3[_c];
                            pnlpt->M12_oneline_2_vv_complex_ortho[_c] = 20. / 7. * f2 * pnlpt->M12_oneline_complex_matter_multipoles_vv0_f2_ortho[_c] + f3 * pnlpt->M12_oneline_complex_matter_multipoles_vv2_f3_ortho[_c];
                            pnlpt->M12_oneline_2_vd_complex[_c] = 2. * f * pnlpt->M12_oneline_complex_matter_multipoles_vd0_f1[_c] + f2 * pnlpt->M12_oneline_complex_matter_multipoles_vd2_f2[_c];
                            pnlpt->M12_oneline_2_vd_complex_ortho[_c] = 2. * f * pnlpt->M12_oneline_complex_matter_multipoles_vd0_f1_ortho[_c] + f2 * pnlpt->M12_oneline_complex_matter_multipoles_vd2_f2_ortho[_c];
                            pnlpt->M12_oneline_2_dd_complex[_c] = 2. * f * pnlpt->M12_oneline_complex_matter_multipoles_dd0_f1[_c];
                            pnlpt->M12_oneline_2_dd_complex_ortho[_c] = 2. * f * pnlpt->M12_oneline_complex_matter_multipoles_dd0_f1_ortho[_c];
                            pnlpt->M12_oneline_4_vv_complex[_c] = 8. / 7. * f2 * pnlpt->M12_oneline_complex_matter_multipoles_vv0_f2[_c] + f3 * pnlpt->M12_oneline_complex_matter_multipoles_vv4_f3[_c];
                            pnlpt->M12_oneline_4_vv_complex_ortho[_c] = 8. / 7. * f2 * pnlpt->M12_oneline_complex_matter_multipoles_vv0_f2_ortho[_c] + f3 * pnlpt->M12_oneline_complex_matter_multipoles_vv4_f3_ortho[_c];
                            pnlpt->M12_oneline_4_vd_complex[_c] = f2 * pnlpt->M12_oneline_complex_matter_multipoles_vd4_f2[_c];
                            pnlpt->M12_oneline_4_vd_complex_ortho[_c] = f2 * pnlpt->M12_oneline_complex_matter_multipoles_vd4_f2_ortho[_c];
                        }
                    }
                }
            }
            TIMER_ADD(m22fill);

            /* === FUSED M13 FILL: all 8 M13 multipole arrays filled in a single pass === */
            for (index_i = 0; index_i < Nmax + 1; index_i++) {
                nu1 = -0.5 * etam[index_i];
                double complex m13_base = pnlpt->M13_oneline_complex[index_i] * 112. / (1. + 9. * nu1);
                pnlpt->M13_0_vv_oneline_complex[index_i] = m13_base * (3. * (f * f) * (7. * (-5. + 3. * nu1) + 6. * f * (-7. + 5. * nu1))) / 3920.;
                pnlpt->M13_0_vd_oneline_complex[index_i] = m13_base * (f * (-35. - 18. * f + 45. * nu1 + 54. * f * nu1)) / 840.;
                pnlpt->M13_0_dd_oneline_complex[index_i] = pnlpt->M13_oneline_complex[index_i] / (1. + 9. * nu1) * (1. + 9. * nu1 + 6. * f * (1. + nu1));
                pnlpt->M13_2_vv_oneline_complex[index_i] = m13_base * (3. * f * f * (-5. + 3. * nu1 + f * (-6. + 5. * nu1))) / 196.;
                pnlpt->M13_2_vd_oneline_complex[index_i] = m13_base * (f * (-49. - 9. * f + 63. * nu1 + 108. * f * nu1)) / 588.;
                pnlpt->M13_2_dd_oneline_complex[index_i] = m13_base * (3. * f * (1. + nu1)) / 28.;
                pnlpt->M13_4_vv_oneline_complex[index_i] = m13_base * (3. * f * f * (-55. + 33. * nu1 + f * (-66. + 90. * nu1))) / 5390.;
                pnlpt->M13_4_vd_oneline_complex[index_i] = m13_base * 9. * (f * f * (1. + 2. * nu1)) / 245.;
            }

            /* === P13 computations (cheap, sequential) === */
            COMPUTE_P13(0_vv, pnlpt->M13_0_vv_oneline_complex,
                        -1. * Pbin[j] * kdisc[j] * kdisc[j] * sigmav * (f * f * (441. + 566. * f + 175. * f * f) / 1225.));
            COMPUTE_P13(0_vd, pnlpt->M13_0_vd_oneline_complex,
                        -1. * Pbin[j] * kdisc[j] * kdisc[j] * sigmav * (2. * f * (625. + 558. * f + 315. * f * f) / 1575.));
            COMPUTE_P13(0_dd, pnlpt->M13_0_dd_oneline_complex,
                        -1. * Pbin[j] * kdisc[j] * kdisc[j] * sigmav * ((61. - 2. * f + 35. * f * f) / 105.));
            COMPUTE_P13(2_vv, pnlpt->M13_2_vv_oneline_complex,
                        -1. * Pbin[j] * kdisc[j] * kdisc[j] * sigmav * (2. * f * f * (54. + 74. * f + 25. * f * f) / 105.));
            COMPUTE_P13(2_vd, pnlpt->M13_2_vd_oneline_complex,
                        -1. * Pbin[j] * kdisc[j] * kdisc[j] * sigmav * 4. * f * (175. + 180. * f + 126. * f * f) / 441.);
            COMPUTE_P13(2_dd, pnlpt->M13_2_dd_oneline_complex,
                        -1. * Pbin[j] * kdisc[j] * kdisc[j] * sigmav * (2. * f * (35. * f - 2.) / 105.));
            COMPUTE_P13(4_vv, pnlpt->M13_4_vv_oneline_complex,
                        -1. * Pbin[j] * kdisc[j] * kdisc[j] * sigmav * (24. * f * f * (33. + 58. * f + 25. * f * f) / 1925.));
            COMPUTE_P13(4_vd, pnlpt->M13_4_vd_oneline_complex,
                        -1. * Pbin[j] * kdisc[j] * kdisc[j] * sigmav * 16. * f * f * (22. + 35. * f) / 1225.);

            /* === Parallel P22 + P12 dispatch (expensive BLAS, 9 independent tasks) === */
            {
                /* M22 matrices and output P22 arrays for each multipole component */
                const double complex *_rsd_M22[] = {
                    pnlpt->M22_oneline_0_vv_complex, pnlpt->M22_oneline_0_vd_complex,
                    pnlpt->M22_oneline_0_dd_complex, pnlpt->M22_oneline_2_vv_complex,
                    pnlpt->M22_oneline_2_vd_complex, pnlpt->M22_oneline_2_dd_complex,
                    pnlpt->M22_oneline_4_vv_complex, pnlpt->M22_oneline_4_vd_complex,
                    pnlpt->M22_oneline_4_dd_complex};
                double *_rsd_P22_out[] = {
                    P22_0_vv, P22_0_vd, P22_0_dd, P22_2_vv, P22_2_vd, P22_2_dd,
                    P22_4_vv, P22_4_vd, P22_4_dd};
                /* M12 matrices for fNL (regular + ortho), NULL if no P12 for that component */
                const double complex *_rsd_M12[] = {
                    pnlpt->M12_oneline_0_vv_complex, pnlpt->M12_oneline_0_vd_complex,
                    pnlpt->M12_oneline_0_dd_complex, pnlpt->M12_oneline_2_vv_complex,
                    pnlpt->M12_oneline_2_vd_complex, pnlpt->M12_oneline_2_dd_complex,
                    pnlpt->M12_oneline_4_vv_complex, pnlpt->M12_oneline_4_vd_complex,
                    NULL};
                const double complex *_rsd_M12_ortho[] = {
                    pnlpt->M12_oneline_0_vv_complex_ortho, pnlpt->M12_oneline_0_vd_complex_ortho,
                    pnlpt->M12_oneline_0_dd_complex_ortho, pnlpt->M12_oneline_2_vv_complex_ortho,
                    pnlpt->M12_oneline_2_vd_complex_ortho, pnlpt->M12_oneline_2_dd_complex_ortho,
                    pnlpt->M12_oneline_4_vv_complex_ortho, pnlpt->M12_oneline_4_vd_complex_ortho,
                    NULL};
                double *_rsd_P12_out[] = {
                    P12_0_vv, P12_0_vd, P12_0_dd, P12_2_vv, P12_2_vd, P12_2_dd,
                    P12_4_vv, P12_4_vd, NULL};
                double *_rsd_P12_ortho_out[] = {
                    P12_0_vv_ortho, P12_0_vd_ortho, P12_0_dd_ortho, P12_2_vv_ortho,
                    P12_2_vd_ortho, P12_2_dd_ortho, P12_4_vv_ortho, P12_4_vd_ortho,
                    NULL};

                TIMER_START(blas);
                #pragma omp parallel for schedule(dynamic)
                for (int _i = 0; _i < 9; _i++) {
                    int _tid = omp_get_thread_num();
                    double *_f22 = (double *)malloc(Nmax * sizeof(double));
                    batch_quadratic_form(Np1, Nmax, _rsd_M22[_i], Xr, Xi,
                                         _f22, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid], ws_Y2[_tid]);
                    assemble_P22(Nmax, kdisc, _f22, cutoff, _rsd_P22_out[_i]);
                    if (has_fNL && _rsd_M12[_i] != NULL) {
                        double *_f12 = (double *)malloc(Nmax * sizeof(double));
                        batch_quadratic_form(Np1, Nmax, _rsd_M12[_i],
                                             Xr_transfer, Xi_transfer,
                                             _f12, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid], ws_Y2[_tid]);
                        assemble_P12(Nmax, kdisc, _f12, Tbin, cutoff, _rsd_P12_out[_i]);
                        batch_quadratic_form(Np1, Nmax, _rsd_M12_ortho[_i],
                                             Xr_transfer, Xi_transfer,
                                             _f12, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid], ws_Y2[_tid]);
                        assemble_P12(Nmax, kdisc, _f12, Tbin, cutoff, _rsd_P12_ortho_out[_i]);
                        free(_f12);
                    }
                    free(_f22);
                }
                TIMER_ADD(blas);
            }

            /* === Assembly loops (cheap, sequential) === */
            for (int j = 0; j < Nmax; j++) {
                P1loop_0_vv[j] = P13_0_vv[j] + P22_0_vv[j];
                P_CTR_0[j] = kdisc[j] * kdisc[j] * Pbin[j];
                Ptree_0_vv[j] = Pbin[j] * (f * f / 5.);
                P1loop_0_vd[j] = P13_0_vd[j] + P22_0_vd[j];
                Ptree_0_vd[j] = Pbin[j] * (f * 2. / 3.);
                P1loop_0_dd[j] = P13_0_dd[j] + P22_0_dd[j];
                Ptree_0_dd[j] = Pbin[j];
                P1loop_2_vv[j] = P13_2_vv[j] + P22_2_vv[j];
                P_CTR_2[j] = kdisc[j] * kdisc[j] * Pbin[j] * f * 2. / 3.;
                Ptree_2_vv[j] = Pbin[j] * (f * f * 4. / 7.);
                P1loop_2_vd[j] = P13_2_vd[j] + P22_2_vd[j];
                Ptree_2_vd[j] = Pbin[j] * (f * 4. / 3.);
                P1loop_2_dd[j] = P13_2_dd[j] + P22_2_dd[j];
                Ptree_2_dd[j] = Pbin[j];
                P1loop_4_vv[j] = P13_4_vv[j] + P22_4_vv[j];
                P_CTR_4[j] = kdisc[j] * kdisc[j] * Pbin[j] * (f * f * 8. / 35.);
                Ptree_4_vv[j] = Pbin[j] * (f * f * 8. / 35.);
                P1loop_4_vd[j] = P13_4_vd[j] + P22_4_vd[j];
                Ptree_4_vd[j] = Pbin[j] * (f * f * 4. / 7.);
                P1loop_4_dd[j] = P22_4_dd[j]; /* no P13 for 4_dd */
            }

            double complex *f22_4_dd;
            class_alloc(f22_4_dd, Nmax * sizeof(complex double), pnlpt->error_message);

        } /* end of IR resummation expression */

        if (irindex == 1) {
            /* Computing FFT for wiggly and non-wiggly parts */

            /* FFTLog decomposition of NW and wiggle spectra */
            TIMER_START(fft);
            double complex *cmsym_nw, *cmsym_nw_transfer, *cmsym_w, *cmsym_w_transfer;
            FFT_TO_CMSYM(Pnw, b, etam, cmsym_nw);
            FFT_TO_CMSYM(Tnw, b_transfer, etam_transfer, cmsym_nw_transfer);
            FFT_TO_CMSYM(Pw, b, etam, cmsym_w);
            FFT_TO_CMSYM(Tw, b_transfer, etam_transfer, cmsym_w_transfer);
            TIMER_ADD(fft);

            /* --- IR resummation: NW and wiggle one-loop integrals ---
             *
             * With IR resummation (irindex=1), we split each one-loop integral
             * into no-wiggle (NW) and wiggle (W) parts:
             *   P_1loop = P_1loop^{NW} + P_1loop^{W} * exp(-Sigma^2 * k^2)
             *
             * The NW part uses P_nw(k) in the FFTLog basis (quadratic form),
             * while the W part uses a bilinear form between P_nw and P_w.
             *
             * mu-power decomposition for Alcock-Paczynski projection:
             * P(k,mu) = sum_{n=0,2,4,6,8} P_mu^{2n}(k) * mu^{2n}
             * These are later integrated over mu with Legendre polynomials
             * and the AP rescaling k -> k(mu), to get P_ell(k). */

            double *P13_mu0_dd, *P13UV_mu0_dd, *P13_mu2_dd, *P13UV_mu2_dd;
            double *P13_mu2_vd, *P13UV_mu2_vd, *P13_mu4_vd, *P13UV_mu4_vd;
            double *P13_mu4_vv, *P13UV_mu4_vv, *P13_mu6, *P13UV_mu6;
            { /* Batch-allocate P13/P13UV IR multipole arrays */
                double **_p13ir[] = {
                    &P13_mu0_dd, &P13UV_mu0_dd, &P13_mu2_dd, &P13UV_mu2_dd,
                    &P13_mu2_vd, &P13UV_mu2_vd, &P13_mu4_vd, &P13UV_mu4_vd,
                    &P13_mu4_vv, &P13UV_mu4_vv, &P13_mu6, &P13UV_mu6};
                for (int ai_ = 0; ai_ < 12; ai_++)
                    class_alloc(*_p13ir[ai_], Nmax * sizeof(double), pnlpt->error_message);
            }
            double *P22_mu0_dd, *P22_mu2_vd, *P22_mu2_dd, *P22_mu4_vv, *P22_mu4_vd;
            double *P22_mu4_dd, *P22_mu6_vv, *P22_mu6_vd, *P22_mu8;
            double *P12_mu0_dd, *P12_mu2_vd, *P12_mu2_dd;
            double *P12_mu4_vv, *P12_mu4_vd, *P12_mu6_vv;
            double *P12_mu0_dd_ortho, *P12_mu2_vd_ortho, *P12_mu2_dd_ortho;
            double *P12_mu4_vv_ortho, *P12_mu4_vd_ortho, *P12_mu6_vv_ortho;
            { /* Batch-allocate P22/P12 NW arrays */
                double **_p22p12[] = {
                    &P22_mu0_dd, &P22_mu2_vd, &P22_mu2_dd, &P22_mu4_vv, &P22_mu4_vd,
                    &P22_mu4_dd, &P22_mu6_vv, &P22_mu6_vd, &P22_mu8,
                    &P12_mu0_dd, &P12_mu2_vd, &P12_mu2_dd,
                    &P12_mu4_vv, &P12_mu4_vd, &P12_mu6_vv,
                    &P12_mu0_dd_ortho, &P12_mu2_vd_ortho, &P12_mu2_dd_ortho,
                    &P12_mu4_vv_ortho, &P12_mu4_vd_ortho, &P12_mu6_vv_ortho};
                for (int ai_ = 0; ai_ < 21; ai_++)
                    *_p22p12[ai_] = malloc(Nmax * sizeof(double));
            }

            /* M22/M12 kernel fill: parallelize over outer index of triangular loop.
             * count = l*(Nmax+1) - l*(l-1)/2 + (i - l) for (l, i) with i >= l. */
            TIMER_START(m22fill);
            {
                int _N = Nmax + 1;
                #pragma omp parallel for schedule(static)
                for (int _l = 0; _l < _N; _l++) {
                    int _count_base = _l * _N - _l * (_l - 1) / 2;
                    for (int _i = _l; _i < _N; _i++) {
                        int _c = _count_base + (_i - _l);
                        double complex _nu1 = -0.5 * etam[_i];
                        double complex _nu2 = -0.5 * etam[_l];
                        double complex _nu12 = _nu1 + _nu2;

                        pnlpt->M22_oneline_mu2_vd_complex[_c] = (pnlpt->M22_oneline_complex[_c] * 196. / (98. * _nu1 * _nu2 * _nu12 * _nu12 - 91. * _nu12 * _nu12 + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.)) * (-1.) * f * (7. * f * (-1. + 2. * _nu1) * (-1. + 2. * _nu2) * (6. + 7. * _nu12) - 4. * (46. + 13. * _nu2 + 98. * _nu1 * _nu1 * _nu1 * _nu2 - 63. * _nu2 * _nu2 + 7. * _nu1 * _nu1 * (-9. - 10. * _nu2 + 28. * _nu2 * _nu2) + _nu1 * (13. - 138. * _nu2 - 70. * _nu2 * _nu2 + 98. * _nu2 * _nu2 * _nu2))) / 392.;

                        pnlpt->M22_oneline_mu2_dd_complex[_c] = (pnlpt->M22_oneline_complex[_c] * 196. / (98. * _nu1 * _nu2 * _nu12 * _nu12 - 91. * _nu12 * _nu12 + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.)) * f * (7. * f * (2. + 2. * _nu1 * _nu1 * _nu1 - _nu2 - _nu2 * _nu2 + 2. * _nu2 * _nu2 * _nu2 - _nu1 * _nu1 * (1. + 2. * _nu2) - _nu1 * (1. + 2. * _nu2 + 2. * _nu2 * _nu2)) + 4. * (10. - _nu2 + 14. * _nu1 * _nu1 * _nu1 * _nu2 - 17. * _nu2 * _nu2 + _nu1 * _nu1 * (-17. + 6. * _nu2 + 28. * _nu2 * _nu2) + _nu1 * (-1. - 22. * _nu2 + 6. * _nu2 * _nu2 + 14. * _nu2 * _nu2 * _nu2))) / 56.;

                        pnlpt->M22_oneline_mu4_vv_complex[_c] = (pnlpt->M22_oneline_complex[_c] * 196. / (98. * _nu1 * _nu2 * _nu12 * _nu12 - 91. * _nu12 * _nu12 + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.)) * f * f * (147. * f * f * (-1. + 2. * _nu1) * (-1. + 2. * _nu2) - 28. * f * (-1. + 2. * _nu1) * (-1. + 2. * _nu2) * (-2. + 7. * _nu12) + 8. * (50. - 9. * _nu2 + 98 * _nu1 * _nu1 * _nu1 * _nu2 - 35. * _nu2 * _nu2 + 7. * _nu1 * _nu1 * (-5. - 18. * _nu2 + 28 * _nu2 * _nu2) + _nu1 * (-9. - 66. * _nu2 - 126 * _nu2 * _nu2 + 98. * _nu2 * _nu2 * _nu2))) / 1568.;

                        pnlpt->M22_oneline_mu4_vd_complex[_c] = (pnlpt->M22_oneline_complex[_c] * 196. / (98. * _nu1 * _nu2 * _nu12 * _nu12 - 91. * _nu12 * _nu12 + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.)) * f * f * (58. + 21. * _nu2 + 112. * _nu1 * _nu1 * _nu1 * _nu2 - 106. * _nu2 * _nu2 + 2. * _nu1 * _nu1 * (-53. - 6. * _nu2 + 112. * _nu2 * _nu2) + 7. * f * (2. + _nu1 + 4. * _nu1 * _nu1 * _nu1 + _nu2 - 8. * _nu1 * _nu2 - 8. * _nu1 * _nu1 * _nu2 - 8. * _nu1 * _nu2 * _nu2 + 4. * _nu2 * _nu2 * _nu2) + _nu1 * (21. - 204. * _nu2 - 12. * _nu2 * _nu2 + 112. * _nu2 * _nu2 * _nu2)) / 56.;

                        pnlpt->M22_oneline_mu4_dd_complex[_c] = (pnlpt->M22_oneline_complex[_c] * 196. / (98. * _nu1 * _nu2 * _nu12 * _nu12 - 91. * _nu12 * _nu12 + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.)) * f * f * (2. * _nu1 - 1.) * (2. * _nu2 - 1.) * (2. + _nu1 * _nu1 + 3. * _nu2 + _nu2 * _nu2 + _nu1 * (3. + 2. * _nu2)) / 8.;

                        pnlpt->M22_oneline_mu6_vv_complex[_c] = (pnlpt->M22_oneline_complex[_c] * 196. / (98. * _nu1 * _nu2 * _nu12 * _nu12 - 91. * _nu12 * _nu12 + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.)) * f * f * f * (7. * f * (1. + 4. * _nu1 * _nu1 * _nu1 + _nu1 * _nu1 * (2. - 12. * _nu2) + 2. * _nu2 + 2. * _nu2 * _nu2 + 4. * _nu2 * _nu2 * _nu2 - 2. * _nu1 * (-1. + 4. * _nu2 + 6. * _nu2 * _nu2)) + 2. * (26. + 9. * _nu2 + 56. * _nu1 * _nu1 * _nu1 * _nu2 - 38. * _nu2 * _nu2 + 2. * _nu1 * _nu1 * (-19. - 18. * _nu2 + 56. * _nu2 * _nu2) + _nu1 * (9. - 84. * _nu2 - 36. * _nu2 * _nu2 + 56. * _nu2 * _nu2 * _nu2))) / 112.;

                        pnlpt->M22_oneline_mu6_vd_complex[_c] = (pnlpt->M22_oneline_complex[_c] * 196. / (98. * _nu1 * _nu2 * _nu12 * _nu12 - 91. * _nu12 * _nu12 + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.)) * f * f * f * (2. * _nu1 - 1.) * (2. * _nu2 - 1.) * (2. + 2. * _nu1 * _nu1 + 5. * _nu2 + 2. * _nu2 * _nu2 + _nu1 * (5. + 4. * _nu2)) / 8.;

                        pnlpt->M22_oneline_mu8_complex[_c] = (pnlpt->M22_oneline_complex[_c] * 196. / (98. * _nu1 * _nu2 * _nu12 * _nu12 - 91. * _nu12 * _nu12 + 36. * _nu1 * _nu2 - 14. * _nu1 * _nu2 * _nu12 + 3. * _nu12 + 58.)) * f * f * f * f * (2. * _nu1 - 1.) * (2. * _nu2 - 1.) * (3. + 4. * _nu1 * _nu1 + 8. * _nu2 + 4. * _nu2 * _nu2 + 8. * _nu1 * (1. + _nu2)) / 32.;

                        if (has_fNL) {
                            pnlpt->M12_oneline_mu2_vd_complex[_c] = f * (pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f1[_c]) + f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f2[_c]);
                            pnlpt->M12_oneline_mu2_vd_complex_ortho[_c] = f * (pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f1_ortho[_c]) + f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f2_ortho[_c]);
                            pnlpt->M12_oneline_mu2_dd_complex[_c] = f * (pnlpt->M12_oneline_complex_matter_mu_powers_dd2_f1[_c]);
                            pnlpt->M12_oneline_mu2_dd_complex_ortho[_c] = f * (pnlpt->M12_oneline_complex_matter_mu_powers_dd2_f1_ortho[_c]);
                            pnlpt->M12_oneline_mu4_vv_complex[_c] = f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vv4_f2[_c]) + f * f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f2[_c]);
                            pnlpt->M12_oneline_mu4_vv_complex_ortho[_c] = f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vv4_f2_ortho[_c]) + f * f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vd2_f2_ortho[_c]);
                            pnlpt->M12_oneline_mu4_vd_complex[_c] = f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vd4_f2[_c]);
                            pnlpt->M12_oneline_mu4_vd_complex_ortho[_c] = f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vd4_f2_ortho[_c]);
                            pnlpt->M12_oneline_mu6_vv_complex[_c] = f * f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vv6_f3[_c]);
                            pnlpt->M12_oneline_mu6_vv_complex_ortho[_c] = f * f * f * (pnlpt->M12_oneline_complex_matter_mu_powers_vv6_f3_ortho[_c]);
                        }
                    }
                }
            }
            TIMER_ADD(m22fill);

            for (index_i = 0; index_i < Nmax + 1; index_i++) {
                nu1 = -0.5 * etam[index_i];
                pnlpt->M13_mu2_dd_oneline_complex[index_i] = pnlpt->M13_oneline_complex[index_i] * 2. / (1. + 9. * nu1) * 9. * f * (1. + nu1);
                pnlpt->M13_mu2_vd_oneline_complex[index_i] = pnlpt->M13_oneline_complex[index_i] * 2. / (1. + 9. * nu1) * (-1.) * (f * (7. + 9. * f - 9. * nu1));
                pnlpt->M13_mu4_vv_oneline_complex[index_i] = pnlpt->M13_oneline_complex[index_i] / (1. + 9. * nu1) * (-1.) * 3. * f * f * (5. + 6. * f - 3. * nu1);
                pnlpt->M13_mu4_vd_oneline_complex[index_i] = pnlpt->M13_oneline_complex[index_i] * 2. / (1. + 9. * nu1) * 9. * f * f * (1. + 2. * nu1);
                pnlpt->M13_mu6_oneline_complex[index_i] = pnlpt->M13_oneline_complex[index_i] * 2. / (1. + 9. * nu1) * 9. * f * f * f * nu1;
            }

            /* f13/f22/f12 for all mu-powers computed inline by batch BLAS macros */

            /* Build X matrix for NW once and compute all P13 mu-powers */
            double *Xr_nw, *Xi_nw;
            class_alloc(Xr_nw, Np1 * Nmax * sizeof(double), pnlpt->error_message);
            class_alloc(Xi_nw, Np1 * Nmax * sizeof(double), pnlpt->error_message);
            TIMER_START(buildx);
            build_X_from_tables(Nmax, Np1, cmsym_nw, kb_cache, sc_cos, sc_sin, Xr_nw, Xi_nw);
            TIMER_ADD(buildx);

            /* P13 NW components: batch_dot_product + UV formula + assemble (parallel) */
            {
                double complex *_m13nw[] = {
                    pnlpt->M13_oneline_complex, pnlpt->M13_mu2_dd_oneline_complex,
                    pnlpt->M13_mu2_vd_oneline_complex, pnlpt->M13_mu4_vv_oneline_complex,
                    pnlpt->M13_mu4_vd_oneline_complex, pnlpt->M13_mu6_oneline_complex};
                double *_p13_out[] = {P13_mu0_dd, P13_mu2_dd, P13_mu2_vd,
                                      P13_mu4_vv, P13_mu4_vd, P13_mu6};
                double *_p13uv[] = {P13UV_mu0_dd, P13UV_mu2_dd, P13UV_mu2_vd,
                                    P13UV_mu4_vv, P13UV_mu4_vd, P13UV_mu6};
                double _uv_coeff[] = {
                    61. / 105.,
                    f * (105. * f - 6.) / 105.,
                    f * (250. + 144. * f) / 105.,
                    f * f * (63. + 48. * f) / 35.,
                    f * f * (44. + 70. * f) / 35.,
                    f * f * f * (46. + 35. * f) / 35.};
                #pragma omp parallel for schedule(static)
                for (int pi_ = 0; pi_ < 6; pi_++) {
                    int _tid = omp_get_thread_num();
                    double *_f13 = (double *)malloc(Nmax * sizeof(double));
                    batch_dot_product(Np1, Nmax, Xr_nw, Xi_nw, _m13nw[pi_], _f13, ws_dot_mr[_tid], ws_dot_mi[_tid]);
                    for (int j = 0; j < Nmax; j++)
                        _p13uv[pi_][j] = -1. * Pnw[j] * kdisc[j] * kdisc[j] * sigmav * _uv_coeff[pi_];
                    assemble_P13(Nmax, kdisc, _f13, Pnw, _p13uv[pi_], cutoff, _p13_out[pi_]);
                    free(_f13);
                }
            }

            /* Build X matrices for NW transfer once */
            double *Xr_nw_transfer, *Xi_nw_transfer;
            if (has_fNL) {
                class_alloc(Xr_nw_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
                class_alloc(Xi_nw_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
                TIMER_START(buildx);
                build_X_from_tables(Nmax, Np1, cmsym_nw_transfer, kb_transfer_cache, sc_cos_t, sc_sin_t, Xr_nw_transfer, Xi_nw_transfer);
                TIMER_ADD(buildx);
            }

            /* Compute all P22 and P12 NW components in parallel.
             * 9 independent quadratic form computations, each with its own workspace. */
            {
                const double complex *_nw_M22[] = {
                    pnlpt->M22_oneline_complex, pnlpt->M22_oneline_mu2_vd_complex,
                    pnlpt->M22_oneline_mu2_dd_complex, pnlpt->M22_oneline_mu4_vv_complex,
                    pnlpt->M22_oneline_mu4_vd_complex, pnlpt->M22_oneline_mu4_dd_complex,
                    pnlpt->M22_oneline_mu6_vv_complex, pnlpt->M22_oneline_mu6_vd_complex,
                    pnlpt->M22_oneline_mu8_complex};
                double *_nw_P22_out[] = {
                    P22_mu0_dd, P22_mu2_vd, P22_mu2_dd, P22_mu4_vv, P22_mu4_vd,
                    P22_mu4_dd, P22_mu6_vv, P22_mu6_vd, P22_mu8};
                /* M12 matrices: NULL for P22-only entries (indices 5,7,8) */
                const double complex *_nw_M12[] = {
                    pnlpt->M12_oneline_complex, pnlpt->M12_oneline_mu2_vd_complex,
                    pnlpt->M12_oneline_mu2_dd_complex, pnlpt->M12_oneline_mu4_vv_complex,
                    pnlpt->M12_oneline_mu4_vd_complex, NULL,
                    pnlpt->M12_oneline_mu6_vv_complex, NULL, NULL};
                const double complex *_nw_M12_ortho[] = {
                    pnlpt->M12_oneline_complex_ortho, pnlpt->M12_oneline_mu2_vd_complex_ortho,
                    pnlpt->M12_oneline_mu2_dd_complex_ortho, pnlpt->M12_oneline_mu4_vv_complex_ortho,
                    pnlpt->M12_oneline_mu4_vd_complex_ortho, NULL,
                    pnlpt->M12_oneline_mu6_vv_complex_ortho, NULL, NULL};
                double *_nw_P12_out[] = {
                    P12_mu0_dd, P12_mu2_vd, P12_mu2_dd, P12_mu4_vv, P12_mu4_vd,
                    NULL, P12_mu6_vv, NULL, NULL};
                double *_nw_P12_ortho_out[] = {
                    P12_mu0_dd_ortho, P12_mu2_vd_ortho, P12_mu2_dd_ortho,
                    P12_mu4_vv_ortho, P12_mu4_vd_ortho, NULL,
                    P12_mu6_vv_ortho, NULL, NULL};

                #pragma omp parallel for schedule(dynamic)
                for (int _i = 0; _i < 9; _i++) {
                    int _tid = omp_get_thread_num();
                    double *_f22 = (double *)malloc(Nmax * sizeof(double));
                    /* P22: quadratic form with NW X matrices */
                    batch_quadratic_form(Np1, Nmax, _nw_M22[_i], Xr_nw, Xi_nw,
                                         _f22, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid], ws_Y2[_tid]);
                    assemble_P22(Nmax, kdisc, _f22, cutoff, _nw_P22_out[_i]);
                    /* P12: only for entries with fNL and non-NULL M12 */
                    if (has_fNL && _nw_M12[_i] != NULL) {
                        double *_f12 = (double *)malloc(Nmax * sizeof(double));
                        batch_quadratic_form(Np1, Nmax, _nw_M12[_i], Xr_nw_transfer, Xi_nw_transfer,
                                             _f12, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid], ws_Y2[_tid]);
                        assemble_P12(Nmax, kdisc, _f12, Tnw, cutoff, _nw_P12_out[_i]);
                        batch_quadratic_form(Np1, Nmax, _nw_M12_ortho[_i], Xr_nw_transfer, Xi_nw_transfer,
                                             _f12, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid], ws_Y2[_tid]);
                        assemble_P12(Nmax, kdisc, _f12, Tnw, cutoff, _nw_P12_ortho_out[_i]);
                        free(_f12);
                    }
                    free(_f22);
                }
            }
            /* Xr_nw_transfer/Xi_nw_transfer freed later — still needed by W dispatch */

            /* Wiggle (BAO) P13/P22/P12 arrays */
            double *P13_mu0_dd_w, *P22_mu0_dd_w;
            double *P13_mu2_dd_w, *P13_mu2_vd_w, *P13_mu4_vd_w, *P13_mu4_vv_w, *P13_mu6_w;
            double *P22_mu2_vd_w, *P22_mu2_dd_w, *P22_mu4_vv_w, *P22_mu4_vd_w, *P22_mu4_dd_w;
            double *P22_mu6_vv_w, *P22_mu6_vd_w, *P22_mu8_w;
            double *P12_mu0_dd_w, *P12_mu2_vd_w, *P12_mu2_dd_w;
            double *P12_mu4_vv_w, *P12_mu4_vd_w, *P12_mu6_vv_w;
            double *P12_mu0_dd_w_ortho, *P12_mu2_vd_w_ortho, *P12_mu2_dd_w_ortho;
            double *P12_mu4_vv_w_ortho, *P12_mu4_vd_w_ortho, *P12_mu6_vv_w_ortho;
            { /* Batch-allocate wiggle P13/P22/P12 arrays */
                double **_pwiggle[] = {
                    &P13_mu0_dd_w, &P22_mu0_dd_w,
                    &P13_mu2_dd_w, &P13_mu2_vd_w, &P13_mu4_vd_w, &P13_mu4_vv_w, &P13_mu6_w,
                    &P22_mu2_vd_w, &P22_mu2_dd_w, &P22_mu4_vv_w, &P22_mu4_vd_w, &P22_mu4_dd_w,
                    &P22_mu6_vv_w, &P22_mu6_vd_w, &P22_mu8_w,
                    &P12_mu0_dd_w, &P12_mu2_vd_w, &P12_mu2_dd_w,
                    &P12_mu4_vv_w, &P12_mu4_vd_w, &P12_mu6_vv_w,
                    &P12_mu0_dd_w_ortho, &P12_mu2_vd_w_ortho, &P12_mu2_dd_w_ortho,
                    &P12_mu4_vv_w_ortho, &P12_mu4_vd_w_ortho, &P12_mu6_vv_w_ortho};
                for (int ai_ = 0; ai_ < 27; ai_++)
                    *_pwiggle[ai_] = malloc(Nmax * sizeof(double));
            }

            /* Build X matrix for wiggle P13 components */
            double *Xr_w, *Xi_w;
            class_alloc(Xr_w, Np1 * Nmax * sizeof(double), pnlpt->error_message);
            class_alloc(Xi_w, Np1 * Nmax * sizeof(double), pnlpt->error_message);
            TIMER_START(buildx);
            build_X_from_tables(Nmax, Np1, cmsym_w, kb_cache, sc_cos, sc_sin, Xr_w, Xi_w);
            TIMER_ADD(buildx);

            /* Compute P13 wiggle components in parallel */
            {
                double complex *_m13w[] = {
                    pnlpt->M13_oneline_complex, pnlpt->M13_mu2_dd_oneline_complex,
                    pnlpt->M13_mu2_vd_oneline_complex, pnlpt->M13_mu4_vv_oneline_complex,
                    pnlpt->M13_mu4_vd_oneline_complex, pnlpt->M13_mu6_oneline_complex};
                double *_p13w_out[] = {
                    P13_mu0_dd_w, P13_mu2_dd_w, P13_mu2_vd_w,
                    P13_mu4_vv_w, P13_mu4_vd_w, P13_mu6_w};
                #pragma omp parallel for schedule(static)
                for (int wi_ = 0; wi_ < 6; wi_++) {
                    int _tid = omp_get_thread_num();
                    double *_f13w = (double *)malloc(Nmax * sizeof(double));
                    batch_dot_product(Np1, Nmax, Xr_w, Xi_w, _m13w[wi_], _f13w, ws_dot_mr[_tid], ws_dot_mi[_tid]);
                    for (int j = 0; j < Nmax; j++)
                        _p13w_out[wi_][j] = kdisc[j] * kdisc[j] * kdisc[j] * _f13w[j] * Pnw[j] * exp(-pow(kdisc[j] / cutoff, 6.));
                    free(_f13w);
                }
            }

            /* Build X matrices for wiggle (scaled by 2 for P22/P12) */
            double *Xr_2w, *Xi_2w;
            class_alloc(Xr_2w, Np1 * Nmax * sizeof(double), pnlpt->error_message);
            class_alloc(Xi_2w, Np1 * Nmax * sizeof(double), pnlpt->error_message);
            for (int k = 0; k < Np1 * Nmax; k++) {
                Xr_2w[k] = 2. * Xr_w[k];
                Xi_2w[k] = 2. * Xi_w[k];
            }

            double *Xr_w_transfer, *Xi_w_transfer, *Xr_2w_transfer, *Xi_2w_transfer;
            if (has_fNL) {
                class_alloc(Xr_w_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
                class_alloc(Xi_w_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
                TIMER_START(buildx);
                build_X_from_tables(Nmax, Np1, cmsym_w_transfer, kb_transfer_cache, sc_cos_t, sc_sin_t, Xr_w_transfer, Xi_w_transfer);
                TIMER_ADD(buildx);
                class_alloc(Xr_2w_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
                class_alloc(Xi_2w_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
                for (int k = 0; k < Np1 * Nmax; k++) {
                    Xr_2w_transfer[k] = 2. * Xr_w_transfer[k];
                    Xi_2w_transfer[k] = 2. * Xi_w_transfer[k];
                }
            }

            /* Compute all P22/P12 wiggle components in parallel (bilinear forms: nw x w) */
            {
                const double complex *_w_M22[] = {
                    pnlpt->M22_oneline_complex, pnlpt->M22_oneline_mu2_vd_complex,
                    pnlpt->M22_oneline_mu2_dd_complex, pnlpt->M22_oneline_mu4_vv_complex,
                    pnlpt->M22_oneline_mu4_vd_complex, pnlpt->M22_oneline_mu4_dd_complex,
                    pnlpt->M22_oneline_mu6_vv_complex, pnlpt->M22_oneline_mu6_vd_complex,
                    pnlpt->M22_oneline_mu8_complex};
                double *_w_P22_out[] = {
                    P22_mu0_dd_w, P22_mu2_vd_w, P22_mu2_dd_w, P22_mu4_vv_w, P22_mu4_vd_w,
                    P22_mu4_dd_w, P22_mu6_vv_w, P22_mu6_vd_w, P22_mu8_w};
                const double complex *_w_M12[] = {
                    pnlpt->M12_oneline_complex, pnlpt->M12_oneline_mu2_vd_complex,
                    pnlpt->M12_oneline_mu2_dd_complex, pnlpt->M12_oneline_mu4_vv_complex,
                    pnlpt->M12_oneline_mu4_vd_complex, NULL,
                    pnlpt->M12_oneline_mu6_vv_complex, NULL, NULL};
                const double complex *_w_M12_ortho[] = {
                    pnlpt->M12_oneline_complex_ortho, pnlpt->M12_oneline_mu2_vd_complex_ortho,
                    pnlpt->M12_oneline_mu2_dd_complex_ortho, pnlpt->M12_oneline_mu4_vv_complex_ortho,
                    pnlpt->M12_oneline_mu4_vd_complex_ortho, NULL,
                    pnlpt->M12_oneline_mu6_vv_complex_ortho, NULL, NULL};
                double *_w_P12_out[] = {
                    P12_mu0_dd_w, P12_mu2_vd_w, P12_mu2_dd_w, P12_mu4_vv_w, P12_mu4_vd_w,
                    NULL, P12_mu6_vv_w, NULL, NULL};
                double *_w_P12_ortho_out[] = {
                    P12_mu0_dd_w_ortho, P12_mu2_vd_w_ortho, P12_mu2_dd_w_ortho,
                    P12_mu4_vv_w_ortho, P12_mu4_vd_w_ortho, NULL,
                    P12_mu6_vv_w_ortho, NULL, NULL};

                #pragma omp parallel for schedule(dynamic)
                for (int _i = 0; _i < 9; _i++) {
                    int _tid = omp_get_thread_num();
                    double *_f22w = (double *)malloc(Nmax * sizeof(double));
                    /* P22 wiggle: bilinear form with nw x 2w */
                    batch_bilinear_form(Np1, Nmax, _w_M22[_i], Xr_nw, Xi_nw, Xr_2w, Xi_2w,
                                        _f22w, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid]);
                    assemble_P22(Nmax, kdisc, _f22w, cutoff, _w_P22_out[_i]);
                    /* P12 wiggle: only for entries with fNL and non-NULL M12 */
                    if (has_fNL && _w_M12[_i] != NULL) {
                        double *_f12w = (double *)malloc(Nmax * sizeof(double));
                        batch_bilinear_form(Np1, Nmax, _w_M12[_i], Xr_nw_transfer, Xi_nw_transfer,
                                            Xr_2w_transfer, Xi_2w_transfer, _f12w, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid]);
                        assemble_P12(Nmax, kdisc, _f12w, Tnw, cutoff, _w_P12_out[_i]);
                        batch_bilinear_form(Np1, Nmax, _w_M12_ortho[_i], Xr_nw_transfer, Xi_nw_transfer,
                                            Xr_2w_transfer, Xi_2w_transfer, _f12w, ws_Mr[_tid], ws_Mi[_tid], ws_Y1[_tid]);
                        assemble_P12(Nmax, kdisc, _f12w, Tnw, cutoff, _w_P12_ortho_out[_i]);
                        free(_f12w);
                    }
                    free(_f22w);
                }
            }
            free(Xr_w);
            free(Xi_w);
            free(Xr_2w);
            free(Xi_2w);
            free(Xr_nw);
            free(Xi_nw);
            if (has_fNL) {
                free(Xr_nw_transfer);
                free(Xi_nw_transfer);
                free(Xr_w_transfer);
                free(Xi_w_transfer);
                free(Xr_2w_transfer);
                free(Xi_2w_transfer);
            }

            /* AP spline setup for all NW/wiggle components */
            TIMER_START(ap_spline);
            AP_SPLINE_SETUP(P13_mu4_vv);
            AP_SPLINE_SETUP(P22_mu4_vv);
            AP_SPLINE_SETUP(P22_mu4_vv_w);
            AP_SPLINE_SETUP(P13_mu4_vv_w);
            AP_SPLINE_SETUP(P13_mu6);
            AP_SPLINE_SETUP(P22_mu6_vv);
            AP_SPLINE_SETUP(P22_mu6_vv_w);
            AP_SPLINE_SETUP(P13_mu6_w);
            AP_SPLINE_SETUP(P22_mu8);
            AP_SPLINE_SETUP(P22_mu8_w);
            AP_SPLINE_SETUP(P13_mu0_dd);
            AP_SPLINE_SETUP(P22_mu0_dd);
            AP_SPLINE_SETUP(P13_mu0_dd_w);
            AP_SPLINE_SETUP(P22_mu0_dd_w);
            AP_SPLINE_SETUP(P22_mu2_dd);
            AP_SPLINE_SETUP(P13_mu2_dd);
            AP_SPLINE_SETUP(P22_mu2_dd_w);
            AP_SPLINE_SETUP(P13_mu2_dd_w);
            AP_SPLINE_SETUP(P22_mu4_dd);
            AP_SPLINE_SETUP(P22_mu4_dd_w);
            AP_SPLINE_SETUP(P13_mu2_vd);
            AP_SPLINE_SETUP(P22_mu2_vd);
            AP_SPLINE_SETUP(P22_mu2_vd_w);
            AP_SPLINE_SETUP(P13_mu2_vd_w);
            AP_SPLINE_SETUP(P13_mu4_vd);
            AP_SPLINE_SETUP(P22_mu4_vd);
            AP_SPLINE_SETUP(P22_mu4_vd_w);
            AP_SPLINE_SETUP(P13_mu4_vd_w);
            AP_SPLINE_SETUP(P22_mu6_vd);
            AP_SPLINE_SETUP(P22_mu6_vd_w);
            AP_SPLINE_SETUP(P12_mu6_vv);
            AP_SPLINE_SETUP(P12_mu4_vd);
            AP_SPLINE_SETUP(P12_mu4_vv);
            AP_SPLINE_SETUP(P12_mu2_vd);
            AP_SPLINE_SETUP(P12_mu2_dd);
            AP_SPLINE_SETUP(P12_mu0_dd);
            AP_SPLINE_SETUP(P12_mu0_dd_w);
            AP_SPLINE_SETUP(P12_mu2_vd_w);
            AP_SPLINE_SETUP(P12_mu2_dd_w);
            AP_SPLINE_SETUP(P12_mu4_vv_w);
            AP_SPLINE_SETUP(P12_mu4_vd_w);
            AP_SPLINE_SETUP(P12_mu6_vv_w);
            AP_SPLINE_SETUP(P12_mu6_vv_ortho);
            AP_SPLINE_SETUP(P12_mu4_vd_ortho);
            AP_SPLINE_SETUP(P12_mu4_vv_ortho);
            AP_SPLINE_SETUP(P12_mu2_vd_ortho);
            AP_SPLINE_SETUP(P12_mu2_dd_ortho);
            AP_SPLINE_SETUP(P12_mu0_dd_ortho);
            AP_SPLINE_SETUP(P12_mu0_dd_w_ortho);
            AP_SPLINE_SETUP(P12_mu2_vd_w_ortho);
            AP_SPLINE_SETUP(P12_mu2_dd_w_ortho);
            AP_SPLINE_SETUP(P12_mu4_vv_w_ortho);
            AP_SPLINE_SETUP(P12_mu4_vd_w_ortho);
            AP_SPLINE_SETUP(P12_mu6_vv_w_ortho);
            TIMER_ADD(ap_spline);

            /* Numerical integration over mu */

            double Sigmatot = 0., Exp = 0., mu = 0., ktrue = 0., mutrue = 0.;
            double LegendreP0 = 1., LegendreP2 = 0., LegendreP4 = 0.;
            double P1loopvv = 0., P1loopvd = 0., P1loopdd = 0.;
            double P12vv = 0., P12vd = 0., P12dd = 0.;
            double P12vv_ortho = 0., P12vd_ortho = 0., P12dd_ortho = 0.;
            double p_tree = 0., p_tree_vv = 0., p_tree_vd = 0., p_tree_dd = 0.;
            double P13ratio = 0., P12ratio = 0.;
            double Pctr0 = 0., Pctr2 = 0., Pctr4 = 0.;
            double P1 = 0., P1b1 = 0.;
            double P1loopvv_ap_ir = 0., P1loopvd_ap_ir = 0., P1loopdd_ap_ir = 0.;
            double P12vv_ap_ir = 0., P12vd_ap_ir = 0., P12dd_ap_ir = 0.;
            double P12vv_ap_ir_ortho = 0., P12vd_ap_ir_ortho = 0., P12dd_ap_ir_ortho = 0.;
            int index_gauss2 = 0;
            last_index = 0;

            /* ============================================================
             * ALCOCK-PACZYNSKI PROJECTION VIA GAUSS-LEGENDRE QUADRATURE
             *
             * The AP effect distorts observed (k, mu) when the fiducial
             * cosmology differs from the true one. The mapping is:
             *   k_true = k_fid * sqrt(1/D^2 + (h^2 - 1/D^2) * mu^2)
             *   mu_true = mu * h / sqrt(...)
             * DW: Keep the legacy AP documentation consistent with the implemented true-to-fiducial ratios.
             * where h = H_true/H_fid, D = D_M,true/D_M,fid.
             *
             * For each k on the FFTLog grid, we integrate over mu using
             * 40-point Gauss-Legendre quadrature. At each mu, we:
             *   1. Compute k_true(mu) and mu_true(mu)
             *   2. Interpolate all P(k) arrays at k_true via splines
             *   3. Build P(k,mu) from mu-power decomposition
             *   4. Project onto Legendre multipoles P_ell(k)
             *
             * The gauss_x[] and gauss_w[] arrays are precomputed and stored
             * in pnlpt during initialization.
             * ============================================================ */

            /* Zero all RSD AP projection arrays */
            { void *_z[] = {
                P1loop_0_vv, P1loop_0_dd, P1loop_0_vd, P1loop_2_vv, P1loop_2_dd, P1loop_2_vd,
                P1loop_4_vv, P1loop_4_dd, P1loop_4_vd,
                P12_0_vv, P12_0_dd, P12_0_vd, P12_2_vv, P12_2_dd, P12_2_vd,
                P12_4_vv, P12_4_dd, P12_4_vd,
                P12_0_vv_ortho, P12_0_dd_ortho, P12_0_vd_ortho,
                P12_2_vv_ortho, P12_2_dd_ortho, P12_2_vd_ortho,
                P12_4_vv_ortho, P12_4_dd_ortho, P12_4_vd_ortho,
                P_CTR_0, P_CTR_2, P_CTR_4,
                Ptree_0_vv, Ptree_0_vd, Ptree_0_dd, Ptree_2_vv, Ptree_2_vd, Ptree_4_vv,
                P10b1, P10, P12
            };
            for (int zi_ = 0; zi_ < 39; zi_++) memset(_z[zi_], 0, sizeof(double) * Nmax); }

            TIMER_START(rsd_ap);
            /* Precompute AP constants outside loops */
            double ap_inv_Dr2_ = 1. / (Dratio * Dratio);
            double ap_hr2_minus_inv_Dr2_ = hratio * hratio - ap_inv_Dr2_;
            double ap_w_hr_Dr2_ = hratio / (Dratio * Dratio);

            for (index_j = Nside; index_j < Nmax - Nside; index_j++) {
                for (index_gauss2 = 0; index_gauss2 < 40; index_gauss2++) {
                    mu = pnlpt->gauss_x[index_gauss2];

                    if (pnlpt->AP_effect == AP_effect_yes) {
                        /* AP rescaling: k_true(mu) and mu_true(mu) from fiducial -> true cosmology */
                        double ap_fac_ = sqrt(ap_inv_Dr2_ + ap_hr2_minus_inv_Dr2_ * mu * mu);
                        mutrue = mu * hratio / ap_fac_;
                        ktrue = kdisc[index_j] * ap_fac_;
                    } else {
                        mutrue = mu;
                        ktrue = kdisc[index_j];
                    }

                    /* Single binary search for all spline interpolations at this ktrue */
                    AP_BSEARCH_SETUP();

                    AP_INTERP_FAST(Pnw);
                    AP_INTERP_FAST(Pw);

                    if (has_fNL) {
                        AP_INTERP_FAST(Tnw);
                        AP_INTERP_FAST(Tw);
                    }

                    AP_INTERP_FAST(P22_mu4_vv);
                    AP_INTERP_FAST(P13_mu4_vv);
                    AP_INTERP_FAST(P22_mu4_vv_w);
                    AP_INTERP_FAST(P13_mu4_vv_w);
                    AP_INTERP_FAST(P13_mu6);
                    AP_INTERP_FAST(P22_mu6_vv);
                    AP_INTERP_FAST(P22_mu6_vv_w);
                    AP_INTERP_FAST(P13_mu6_w);
                    AP_INTERP_FAST(P22_mu8);
                    AP_INTERP_FAST(P22_mu8_w);
                    AP_INTERP_FAST(P22_mu0_dd);
                    AP_INTERP_FAST(P13_mu0_dd);
                    AP_INTERP_FAST(P13_mu0_dd_w);
                    AP_INTERP_FAST(P22_mu0_dd_w);
                    AP_INTERP_FAST(P22_mu2_dd);
                    AP_INTERP_FAST(P13_mu2_dd);
                    AP_INTERP_FAST(P22_mu2_dd_w);
                    AP_INTERP_FAST(P13_mu2_dd_w);
                    AP_INTERP_FAST(P22_mu4_dd);
                    AP_INTERP_FAST(P22_mu4_dd_w);
                    AP_INTERP_FAST(P13_mu2_vd);
                    AP_INTERP_FAST(P22_mu2_vd);
                    AP_INTERP_FAST(P22_mu2_vd_w);
                    AP_INTERP_FAST(P13_mu2_vd_w);
                    AP_INTERP_FAST(P13_mu4_vd);
                    AP_INTERP_FAST(P22_mu4_vd);
                    AP_INTERP_FAST(P22_mu4_vd_w);
                    AP_INTERP_FAST(P13_mu4_vd_w);
                    AP_INTERP_FAST(P22_mu6_vd);
                    AP_INTERP_FAST(P22_mu6_vd_w);

                    if (has_fNL) {
                        AP_INTERP_FAST(P12_mu0_dd);
                        AP_INTERP_FAST(P12_mu2_vd);
                        AP_INTERP_FAST(P12_mu2_dd);
                        AP_INTERP_FAST(P12_mu4_vv);
                        AP_INTERP_FAST(P12_mu4_vd);
                        AP_INTERP_FAST(P12_mu6_vv);
                        AP_INTERP_FAST(P12_mu0_dd_ortho);
                        AP_INTERP_FAST(P12_mu2_vd_ortho);
                        AP_INTERP_FAST(P12_mu2_dd_ortho);
                        AP_INTERP_FAST(P12_mu4_vv_ortho);
                        AP_INTERP_FAST(P12_mu4_vd_ortho);
                        AP_INTERP_FAST(P12_mu6_vv_ortho);
                        AP_INTERP_FAST(P12_mu0_dd_w);
                        AP_INTERP_FAST(P12_mu2_vd_w);
                        AP_INTERP_FAST(P12_mu2_dd_w);
                        AP_INTERP_FAST(P12_mu4_vv_w);
                        AP_INTERP_FAST(P12_mu4_vd_w);
                        AP_INTERP_FAST(P12_mu6_vv_w);
                        AP_INTERP_FAST(P12_mu0_dd_w_ortho);
                        AP_INTERP_FAST(P12_mu2_vd_w_ortho);
                        AP_INTERP_FAST(P12_mu2_dd_w_ortho);
                        AP_INTERP_FAST(P12_mu4_vv_w_ortho);
                        AP_INTERP_FAST(P12_mu4_vd_w_ortho);
                        AP_INTERP_FAST(P12_mu6_vv_w_ortho);
                    }

                    /* Legendre polynomials: P_2(mu) and P_4(mu) */
                    LegendreP2 = (3. * mu * mu - 1.) / 2.;
                    LegendreP4 = (35. * mu * mu * mu * mu - 30. * mu * mu + 3.) / 8.;

                    /* Pre-compute powers of mutrue for RSD mu-expansion */
                    double mu2t = mutrue * mutrue;
                    double mu4t = mu2t * mu2t;
                    double mu6t = mu4t * mu2t;
                    double mu8t = mu4t * mu4t;

                    /* IR resummation damping: Sigma_tot(k, mu) from Senatore & Zaldarriaga */
                    Sigmatot = SigmaBAO * (1. + f * mu2t * (2. + f)) + f * f * mu2t * (mu2t - 1.) * deltaSigmaBAO;
                    Exp = exp(-Sigmatot * ktrue * ktrue);

                    p_tree = (Pnw_ap_out + (1. + Sigmatot * ktrue * ktrue) * Pw_ap_out * Exp);

                    P13ratio = 1. + (Pw_ap_out / Pnw_ap_out) * Exp;

                    if (has_fNL) {
                        P12ratio = 1. + (Tw_ap_out / Tnw_ap_out) * Exp;
                    }

                    P1b1 = (Pnw[index_j] + Pw[index_j] * Exp) * pnlpt->gauss_w[index_gauss2];
                    P1 = (Pnw[index_j] + Pw[index_j] * Exp) * f * pnlpt->gauss_x[index_gauss2] * pnlpt->gauss_x[index_gauss2] * pnlpt->gauss_w[index_gauss2];

                    p_tree_vv = (Pnw_ap_out + (1. + Sigmatot * ktrue * ktrue) * Pw_ap_out * Exp) * f * f * mu4t * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    p_tree_vd = (Pnw_ap_out + (1. + Sigmatot * ktrue * ktrue) * Pw_ap_out * Exp) * 2. * f * mu2t * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    p_tree_dd = (Pnw_ap_out + (1. + Sigmatot * ktrue * ktrue) * Pw_ap_out * Exp) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                    Pctr0 = ktrue * ktrue * (Pnw_ap_out + Pw_ap_out * Exp) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    Pctr2 = (Pnw_ap_out + Pw_ap_out * Exp) * pnlpt->gauss_w[index_gauss2] * f * mu2t * ktrue * ktrue * hratio / Dratio / Dratio;
                    Pctr4 = ktrue * ktrue * (Pnw_ap_out + Pw_ap_out * Exp) * pnlpt->gauss_w[index_gauss2] * f * f * mu4t * hratio / Dratio / Dratio;

                    /* P_{1-loop,vv}: mu^4 + mu^6 + mu^8 terms (tree-level is in p_tree_vv) */
                    P1loopvv = ((P13_mu4_vv_ap_out * P13ratio + P22_mu4_vv_ap_out + (P22_mu4_vv_w_ap_out + P13_mu4_vv_w_ap_out) * Exp) * mu4t + (P13_mu6_ap_out * P13ratio + P22_mu6_vv_ap_out + (P22_mu6_vv_w_ap_out + P13_mu6_w_ap_out) * Exp) * mu6t + (P22_mu8_ap_out + P22_mu8_w_ap_out * Exp) * mu8t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                    if (has_fNL) {
                        P12vv = ((P12_mu4_vv_ap_out * P12ratio + P12_mu4_vv_w_ap_out * Exp) * mu4t + (P12_mu6_vv_ap_out * P12ratio + P12_mu6_vv_w_ap_out * Exp) * mu6t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                        P12vv_ortho = ((P12_mu4_vv_ortho_ap_out * P12ratio + P12_mu4_vv_w_ortho_ap_out * Exp) * mu4t + (P12_mu6_vv_ortho_ap_out * P12ratio + P12_mu6_vv_w_ortho_ap_out * Exp) * mu6t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    }

                    /* P_{1-loop,dd}: mu^0 + mu^2 + mu^4 terms (tree-level is in p_tree_dd) */
                    P1loopdd = ((P22_mu0_dd_ap_out + P13_mu0_dd_ap_out * P13ratio + (P13_mu0_dd_w_ap_out + P22_mu0_dd_w_ap_out) * Exp) + (P22_mu2_dd_ap_out + P13_mu2_dd_ap_out * P13ratio + (P22_mu2_dd_w_ap_out + P13_mu2_dd_w_ap_out) * Exp) * mu2t + (P22_mu4_dd_ap_out + P22_mu4_dd_w_ap_out * Exp) * mu4t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                    if (has_fNL) {
                        P12dd = ((P12_mu0_dd_ap_out * P12ratio + P12_mu0_dd_w_ap_out * Exp) + (P12_mu2_dd_ap_out * P12ratio + P12_mu2_dd_w_ap_out * Exp) * mu2t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                        P12dd_ortho = ((P12_mu0_dd_ortho_ap_out * P12ratio + P12_mu0_dd_w_ortho_ap_out * Exp) + (P12_mu2_dd_ortho_ap_out * P12ratio + P12_mu2_dd_w_ortho_ap_out * Exp) * mu2t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    }

                    /* P_{1-loop,vd}: mu^2 + mu^4 + mu^6 terms (tree-level is in p_tree_vd) */
                    P1loopvd = ((P13_mu2_vd_ap_out * P13ratio + P22_mu2_vd_ap_out + (P22_mu2_vd_w_ap_out + P13_mu2_vd_w_ap_out) * Exp) * mu2t + (P13_mu4_vd_ap_out * P13ratio + P22_mu4_vd_ap_out + (P22_mu4_vd_w_ap_out + P13_mu4_vd_w_ap_out) * Exp) * mu4t + (P22_mu6_vd_ap_out + P22_mu6_vd_w_ap_out * Exp) * mu6t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                    if (has_fNL) {
                        P12vd = ((P12_mu2_vd_ap_out * P12ratio + P12_mu2_vd_w_ap_out * Exp) * mu2t + (P12_mu4_vd_ap_out * P12ratio + P12_mu4_vd_w_ap_out * Exp) * mu4t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                        P12vd_ortho = ((P12_mu2_vd_ortho_ap_out * P12ratio + P12_mu2_vd_w_ortho_ap_out * Exp) * mu2t + (P12_mu4_vd_ortho_ap_out * P12ratio + P12_mu4_vd_w_ortho_ap_out * Exp) * mu4t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    }

                    P1loopdd_ap_ir = ((p_tree + P22_mu0_dd_ap_out + P13_mu0_dd_ap_out * P13ratio + (P13_mu0_dd_w_ap_out + P22_mu0_dd_w_ap_out) * Exp) + (P22_mu2_dd_ap_out + P13_mu2_dd_ap_out * P13ratio + (P22_mu2_dd_w_ap_out + P13_mu2_dd_w_ap_out) * Exp) * mu2t + (P22_mu4_dd_ap_out + P22_mu4_dd_w_ap_out * Exp) * mu4t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                    P1loopvd_ap_ir = ((p_tree * 2. * f + P13_mu2_vd_ap_out * P13ratio + P22_mu2_vd_ap_out + (P22_mu2_vd_w_ap_out + P13_mu2_vd_w_ap_out) * Exp) * mu2t + (P13_mu4_vd_ap_out * P13ratio + P22_mu4_vd_ap_out + (P22_mu4_vd_w_ap_out + P13_mu4_vd_w_ap_out) * Exp) * mu4t + (P22_mu6_vd_ap_out + P22_mu6_vd_w_ap_out * Exp) * mu6t) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                    LEGENDRE_PROJECT(P1loopvv, P1loop_0_vv, P1loop_2_vv, P1loop_4_vv);
                    P1loop_0_dd[index_j] += P1loopdd * LegendreP0 / 2.;
                    P1loop_2_dd[index_j] += P1loopdd_ap_ir * LegendreP2 * 2.5;
                    P1loop_4_dd[index_j] += P1loopdd_ap_ir * LegendreP4 * 4.5;
                    P1loop_0_vd[index_j] += P1loopvd * LegendreP0 / 2.;
                    P1loop_2_vd[index_j] += P1loopvd * LegendreP2 * 2.5;
                    P1loop_4_vd[index_j] += P1loopvd_ap_ir * LegendreP4 * 4.5;

                    if (has_fNL) {
                        LEGENDRE_PROJECT(P12vv, P12_0_vv, P12_2_vv, P12_4_vv);
                        LEGENDRE_PROJECT(P12dd, P12_0_dd, P12_2_dd, P12_4_dd);
                        LEGENDRE_PROJECT(P12vd, P12_0_vd, P12_2_vd, P12_4_vd);
                        LEGENDRE_PROJECT(P12vv_ortho, P12_0_vv_ortho, P12_2_vv_ortho, P12_4_vv_ortho);
                        LEGENDRE_PROJECT(P12dd_ortho, P12_0_dd_ortho, P12_2_dd_ortho, P12_4_dd_ortho);
                        LEGENDRE_PROJECT(P12vd_ortho, P12_0_vd_ortho, P12_2_vd_ortho, P12_4_vd_ortho);
                    }

                    P_CTR_0[index_j] += Pctr0 * LegendreP0 / 2.;
                    P_CTR_2[index_j] += Pctr2 * LegendreP2 * 2.5;
                    P_CTR_4[index_j] += Pctr4 * LegendreP4 * 4.5;

                    LEGENDRE_PROJECT(p_tree_vv, Ptree_0_vv, Ptree_2_vv, Ptree_4_vv);
                    Ptree_0_vd[index_j] += p_tree_vd * LegendreP0 / 2.;
                    Ptree_0_dd[index_j] += p_tree_dd * LegendreP0 / 2.;
                    Ptree_2_vd[index_j] += p_tree_vd * LegendreP2 * 2.5;

                    P10b1[index_j] += P1b1 * LegendreP0 / 2.;
                    P10[index_j] += P1 * LegendreP0 / 2.;
                    P12[index_j] += P1 * LegendreP2 * 2.5;
                }
            }
            TIMER_ADD(rsd_ap);

            { void *_f[] = {
                dd_P13_mu4_vv, dd_P22_mu4_vv, dd_P22_mu4_vv_w, dd_P13_mu4_vv_w,
                dd_P13_mu6, dd_P22_mu6_vv, dd_P22_mu6_vv_w, dd_P13_mu6_w,
                dd_P22_mu8, dd_P22_mu8_w, dd_P13_mu0_dd, dd_P22_mu0_dd,
                dd_P13_mu0_dd_w, dd_P22_mu0_dd_w, dd_P22_mu2_dd, dd_P13_mu2_dd,
                dd_P22_mu2_dd_w, dd_P13_mu2_dd_w, dd_P22_mu4_dd, dd_P22_mu4_dd_w,
                dd_P13_mu2_vd, dd_P22_mu2_vd, dd_P22_mu2_vd_w, dd_P13_mu2_vd_w,
                dd_P13_mu4_vd, dd_P22_mu4_vd, dd_P22_mu4_vd_w, dd_P13_mu4_vd_w,
                dd_P22_mu6_vd, dd_P22_mu6_vd_w
            };
            for (int fi_ = 0; fi_ < 30; fi_++) free(_f[fi_]); }

            { void *_f[] = {
                P13_mu0_dd_w, P13_mu0_dd, P13UV_mu0_dd, P22_mu0_dd, P22_mu0_dd_w,
                P22_mu8, P22_mu6_vd, P22_mu6_vv, P22_mu4_vv, P22_mu4_vd,
                P22_mu4_dd, P22_mu2_vd, P22_mu2_dd, P13_mu4_vd, P13_mu4_vv,
                P13_mu2_vd, P13_mu2_dd, P13_mu6, P13UV_mu4_vd, P13UV_mu4_vv,
                P13UV_mu2_vd, P13UV_mu2_dd, P13UV_mu6, P22_mu8_w, P22_mu6_vd_w,
                P22_mu6_vv_w, P22_mu4_vv_w, P22_mu4_vd_w, P22_mu4_dd_w, P22_mu2_vd_w,
                P22_mu2_dd_w, P13_mu4_vd_w, P13_mu4_vv_w, P13_mu2_vd_w, P13_mu2_dd_w,
                P13_mu6_w, cmsym_w, cmsym_nw,
            };
            for (int fi_ = 0; fi_ < 38; fi_++) free(_f[fi_]); }

            { void *_f[] = {
                dd_P12_mu6_vv, dd_P12_mu4_vd, dd_P12_mu4_vv, dd_P12_mu2_vd, dd_P12_mu2_dd,
                dd_P12_mu0_dd, dd_P12_mu6_vv_w, dd_P12_mu4_vd_w, dd_P12_mu4_vv_w, dd_P12_mu2_vd_w,
                dd_P12_mu2_dd_w, dd_P12_mu0_dd_w,
                P12_mu6_vv, P12_mu4_vd, P12_mu4_vv, P12_mu2_vd, P12_mu2_dd,
                P12_mu0_dd, P12_mu6_vv_w, P12_mu4_vd_w, P12_mu4_vv_w, P12_mu2_vd_w,
                P12_mu2_dd_w, P12_mu0_dd_w, cmsym_w_transfer, cmsym_nw_transfer,
            };
            for (int fi_ = 0; fi_ < 26; fi_++) free(_f[fi_]); }
            { void *_f[] = {
                dd_P12_mu6_vv_ortho, dd_P12_mu4_vd_ortho, dd_P12_mu4_vv_ortho,
                dd_P12_mu2_vd_ortho, dd_P12_mu2_dd_ortho, dd_P12_mu0_dd_ortho,
                dd_P12_mu6_vv_w_ortho, dd_P12_mu4_vd_w_ortho, dd_P12_mu4_vv_w_ortho,
                dd_P12_mu2_vd_w_ortho, dd_P12_mu2_dd_w_ortho, dd_P12_mu0_dd_w_ortho,
                P12_mu6_vv_ortho, P12_mu4_vd_ortho, P12_mu4_vv_ortho, P12_mu2_vd_ortho, P12_mu2_dd_ortho,
                P12_mu0_dd_ortho, P12_mu6_vv_w_ortho, P12_mu4_vd_w_ortho, P12_mu4_vv_w_ortho, P12_mu2_vd_w_ortho,
                P12_mu2_dd_w_ortho, P12_mu0_dd_w_ortho,
            };
            for (int fi_ = 0; fi_ < 24; fi_++) free(_f[fi_]); }

        } /* end of second IR resummation condition */

        /* Constructing the final output spectra */

        TIMER_START(spline_out);
        SPLINE_INTERP_OUTPUT(P1loop_0_vv, pk_l_0_vv, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             -1. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav * (f * f * (441. + 566. * f + 175. * f * f) / 1225.) + large_for_logs_big, 0);

        SPLINE_INTERP_OUTPUT(P1loop_0_vd, pk_l_0_vd, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             -1. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav * (2. * f * (625. + 558. * f + 315. * f * f) / 1575.) + large_for_logs_big, 0);

        SPLINE_INTERP_OUTPUT(P1loop_0_dd, pk_l_0_dd, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             -1. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav * ((61. - 2. * f + 35. * f * f) / 105.) + large_for_logs_big, 0);

        SPLINE_INTERP_OUTPUT(P1loop_2_vv, pk_l_2_vv, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             -1. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav * (2. * f * f * (54. + 74. * f + 25. * f * f) / 105.) + large_for_logs_big, 0);

        SPLINE_INTERP_OUTPUT(P1loop_2_vd, pk_l_2_vd, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             -1. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav * 4. * f * (175. + 180. * f + 126. * f * f) / 441. + large_for_logs_big, 0);

        SPLINE_INTERP_OUTPUT(P1loop_2_dd, pk_l_2_dd, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             -1. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav * (2. * f * (35. * f - 2.) / 105.) + large_for_logs_big, 0);

        SPLINE_INTERP_OUTPUT(P1loop_4_vv, pk_l_4_vv, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             -1. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav * (24. * f * f * (33. + 58. * f + 25. * f * f) / 1925.) + large_for_logs_big, 0);

        SPLINE_INTERP_OUTPUT(P1loop_4_vd, pk_l_4_vd, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             -1. * exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * sigmav * 16. * f * f * (22. + 35. * f) / 1225. + large_for_logs_big, 0);

        double *ddpk_nl_4_dd;
        class_alloc(ddpk_nl_4_dd, Nmax * sizeof(double), pnlpt->error_message);

        SPLINE_SETUP(kdisc, Nmax, P1loop_4_dd, ddpk_nl_4_dd, _SPLINE_NATURAL_);
        last_index = 0;
        pk_nl_out = 0;
        double ir4dd;
        SPLINE_EVAL(kdisc, Nmax, P1loop_4_dd, ddpk_nl_4_dd, kminnew, &last_index, &ir4dd);
        last_index = 0;
        pk_nl_out = 0;
        for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
            if (pnlpt->k[index_k] <= kmaxnew && pnlpt->k[index_k] >= kminnew) {
                SPLINE_EVAL(kdisc, Nmax, P1loop_4_dd, ddpk_nl_4_dd, pnlpt->k[index_k], &last_index, &pk_nl_out);

                pk_l_4_dd[index_k] = pk_nl_out * exp(-pow(pnlpt->k[index_k] / 3., 4.)) + large_for_logs_big;
            } else {
                pk_l_4_dd[index_k] = large_for_logs_big + ir4dd * pow((pnlpt->k[index_k] / kminnew), 4.) * exp(-pow(pnlpt->k[index_k] / 3., 4.));
            }
        }

        { /* fNL P12 RSD multipoles with exponential damping */
            double *_p12_in[] = {
                P12_0_dd, P12_0_dd_ortho, P12_0_vd, P12_0_vd_ortho,
                P12_0_vv, P12_0_vv_ortho, P12_2_dd, P12_2_dd_ortho,
                P12_2_vd, P12_2_vd_ortho, P12_2_vv, P12_2_vv_ortho,
                P12_4_dd, P12_4_dd_ortho, P12_4_vd, P12_4_vd_ortho,
                P12_4_vv, P12_4_vv_ortho};
            double *_p12_out[] = {
                pk_l_fNL_0_dd, pk_l_fNL_0_dd_ortho, pk_l_fNL_0_vd, pk_l_fNL_0_vd_ortho,
                pk_l_fNL_0_vv, pk_l_fNL_0_vv_ortho, pk_l_fNL_2_dd, pk_l_fNL_2_dd_ortho,
                pk_l_fNL_2_vd, pk_l_fNL_2_vd_ortho, pk_l_fNL_2_vv, pk_l_fNL_2_vv_ortho,
                pk_l_fNL_4_dd, pk_l_fNL_4_dd_ortho, pk_l_fNL_4_vd, pk_l_fNL_4_vd_ortho,
                pk_l_fNL_4_vv, pk_l_fNL_4_vv_ortho};
            SPLINE_INTERP_BATCH(_p12_in, _p12_out, 18, kminnew, kmaxnew,
                out_tmp_ * exp(-pow(pnlpt->k[index_k] / 3., 4.)) + large_for_logs_fNL,
                large_for_logs_fNL + epsilon_for_logs_fNL, 0);
        }

        SPLINE_INTERP_OUTPUT(P_CTR_0, pk_CTR_0, kminnew, kmaxnew,
                             (out_tmp_ <= 0 ? 1e-16 : out_tmp_),
                             exp(lnpk_l[index_k] + 2. * lnk_l[index_k]), 0);

        SPLINE_INTERP_OUTPUT(P_CTR_2, pk_CTR_2, kminnew, kmaxnew,
                             (out_tmp_ <= 0 ? 1e-16 : out_tmp_),
                             exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * f * 2. / 3., 0);

        SPLINE_INTERP_OUTPUT(P_CTR_4, pk_CTR_4, kminnew, kmaxnew,
                             (out_tmp_ <= 0 ? 1e-16 : out_tmp_),
                             exp(lnpk_l[index_k] + 2. * lnk_l[index_k]) * f * f * 8. / 35., 0);

        SPLINE_INTERP_OUTPUT(Ptree_0_vv, pk_Tree_0_vv, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             exp(lnpk_l[index_k]) * f * f / 5. + large_for_logs_big, 1);

        SPLINE_INTERP_OUTPUT(Ptree_0_vd, pk_Tree_0_vd, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             exp(lnpk_l[index_k]) * 2. * f / 3. + large_for_logs_big, 1);

        SPLINE_INTERP_OUTPUT(Ptree_0_dd, pk_Tree_0_dd, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             exp(lnpk_l[index_k]) + large_for_logs_big, 1);

        SPLINE_INTERP_OUTPUT(Ptree_2_vv, pk_Tree_2_vv, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             exp(lnpk_l[index_k]) * 4. * f * f / 7. + large_for_logs_big, 1);

        SPLINE_INTERP_OUTPUT(Ptree_2_vd, pk_Tree_2_vd, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             exp(lnpk_l[index_k]) * 4. * f / 3. + large_for_logs_big, 1);

        SPLINE_INTERP_OUTPUT(Ptree_4_vv, pk_Tree_4_vv, kminnew, kmaxnew,
                             out_tmp_ + large_for_logs_big,
                             exp(lnpk_l[index_k]) * 8. * f * f / 35. + large_for_logs_big, 1);
        TIMER_ADD(spline_out);
        free(Ptree_2_dd);
        free(Ptree_4_vd);

        { void *_f[] = {
            ddpk_nl_4_dd, P13UV_0_vv, P13_0_vv, P1loop_0_vv, P22_0_vv,
            P13UV_0_vd, P13_0_vd, P1loop_0_vd, P22_0_vd, P13UV_0_dd,
            P13_0_dd, P1loop_0_dd, P22_0_dd, P13UV_2_vv, P13_2_vv,
            P1loop_2_vv, P22_2_vv, P13UV_2_vd, P13_2_vd, P1loop_2_vd,
            P22_2_vd, P13UV_2_dd, P13_2_dd, P1loop_2_dd, P22_2_dd,
            P13UV_4_vv, P13_4_vv, P1loop_4_vv, P22_4_vv, P13UV_4_vd,
            P13_4_vd, P1loop_4_vd, P22_4_vd, P1loop_4_dd, P22_4_dd,
            P12_4_dd, P12_4_vd, P12_4_vv, P12_2_dd, P12_2_vd,
            P12_2_vv, P12_0_dd, P12_0_vd, P12_0_vv, P12_4_dd_ortho,
            P12_4_vd_ortho, P12_4_vv_ortho, P12_2_dd_ortho, P12_2_vd_ortho, P12_2_vv_ortho,
            P12_0_dd_ortho, P12_0_vd_ortho, P12_0_vv_ortho, P_CTR_0, P_CTR_2,
            P_CTR_4,
        };
        for (int fi_ = 0; fi_ < 56; fi_++) free(_f[fi_]); }

    }
    else if ((pnlpt->rsd == rsd_yes) && (qcdm_qscf_qs_active == _TRUE_)) {
        TIMER_START(qcdm_rsd);
        /* DW: exact-time c-c' two-fluid RSD branch. Unlike the legacy EdS
         * matrices, this path evolves the model kernels, applies AP in the
         * observed coordinates, and optionally evaluates equilateral and
         * orthogonal primordial-bispectrum P12 templates. */
        if (pnlpt->nonlinear_pt_verbose > 0) {
            printf("DW/QCDM-QSCF: computing exact-QS two-fluid tracer RSD multipoles with cb/AP/PNG support.\n");
        }

        double *TF_P0,*TF_P2,*TF_P4;
        double *TF_P0eq,*TF_P2eq,*TF_P4eq;
        double *TF_P0ortho,*TF_P2ortho,*TF_P4ortho;
        double **tf_ptrs[] = {&TF_P0,&TF_P2,&TF_P4,
                              &TF_P0eq,&TF_P2eq,&TF_P4eq,
                              &TF_P0ortho,&TF_P2ortho,&TF_P4ortho};
        for (int ai=0; ai<9; ai++)
            class_alloc(*tf_ptrs[ai],Nmax*sizeof(double),pnlpt->error_message);

        if (qcdm_ir_state.enabled==_TRUE_) {
            class_call(qcdm_tf_exact_rsd_multipoles_ir(pba,pnlpt,tau,Nmax,kdisc,Pnw,Pw,
                                                        &qcdm_ir_state,hratio,Dratio,cutoff,
                                                        TF_P0,TF_P2,TF_P4,
                                                        TF_P0eq,TF_P2eq,TF_P4eq,
                                                        TF_P0ortho,TF_P2ortho,TF_P4ortho),
                       pnlpt->error_message,pnlpt->error_message);
        }
        else {
            class_call(qcdm_tf_exact_rsd_multipoles(pba,pnlpt,tau,Nmax,kdisc,Pbin,
                                                     qcdm_Pphi,qcdm_Mphi,
                                                     hratio,Dratio,cutoff,
                                                     TF_P0,TF_P2,TF_P4,
                                                     TF_P0eq,TF_P2eq,TF_P4eq,
                                                     TF_P0ortho,TF_P2ortho,TF_P4ortho),
                       pnlpt->error_message,pnlpt->error_message);
        }

        /* DW: calculate the tree and EFT-only projections separately so the
         * diagnostic output can isolate loop/PNG effects and so k values
         * outside the explicitly integrated loop grid retain the correct AP
         * remapped tree+EFT prediction instead of a clamped endpoint. */
        double *tree_mu,*tree_w;
        int Nmu_tree=(qcdm_ir_state.enabled==_TRUE_)?40:MAX(pnlpt->qcdm_tf_Nmu,5);
        class_alloc(tree_mu,Nmu_tree*sizeof(double),pnlpt->error_message);
        class_alloc(tree_w,Nmu_tree*sizeof(double),pnlpt->error_message);
        class_call(qcdm_tf_gauss_legendre_grid(Nmu_tree,tree_mu,tree_w,pnlpt->error_message),
                   pnlpt->error_message,pnlpt->error_message);

        /* Standalone RSD decomposition: enabled by the existing
         * write_qcdm_multipoles option.  The original multipoles file is
         * unchanged.  Use the same LOS quadrature and AP map as the tree
         * contribution, so all components share exactly one convention. */
        FILE *rsd_decomp_fp=NULL;
        if (pnlpt->qcdm_write_multipoles==_TRUE_) {
            FileName rsd_decomp_name;
            char rsd_decomp_zsuffix[16];
            /* nonlinear_pt_loop receives tau, rather than the init loop's
             * i_z. Recover the exact requested redshift slot locally. */
            int rsd_decomp_iz=0;
            double rsd_decomp_dz=fabs(qcdm_timing_z-pnlpt->z_pk[0]);
            for (int iz=1;iz<pnlpt->z_pk_num;iz++) {
                double dz=fabs(qcdm_timing_z-pnlpt->z_pk[iz]);
                if (dz<rsd_decomp_dz) {
                    rsd_decomp_iz=iz;
                    rsd_decomp_dz=dz;
                }
            }
            class_test(rsd_decomp_dz>1.e-6*(1.0+fabs(qcdm_timing_z)),
                       pnlpt->error_message,
                       "Could not identify QCDM RSD output z=%.12g in z_pk",qcdm_timing_z);
            if (pnlpt->z_pk_num==1) rsd_decomp_zsuffix[0]='\0';
            else class_sprintf(rsd_decomp_zsuffix,"z%d_",rsd_decomp_iz+1);
            class_sprintf(rsd_decomp_name,"%s%sqcdm_pt_rsd_decomposition.dat",
                          pnlpt->qcdm_pt_output_root,rsd_decomp_zsuffix);
            rsd_decomp_fp=fopen(rsd_decomp_name,"w");
            class_test(rsd_decomp_fp==NULL,pnlpt->error_message,
                       "Cannot open QCDM RSD decomposition %s",rsd_decomp_name);
            fprintf(rsd_decomp_fp,"# Exact-QS redshift-space galaxy multipole decomposition; z=%.12g\n",
                    pnlpt->z_pk[rsd_decomp_iz]);
            fprintf(rsd_decomp_fp,"# ell=0,2,4; k in h/Mpc; P in (Mpc/h)^3; AP=%s; IR=%s\n",
                    pnlpt->AP_effect==AP_effect_yes?"yes":"no",
                    qcdm_ir_state.enabled==_TRUE_?"yes":"no");
            fprintf(rsd_decomp_fp,"# P_treeIR = P_linear + P_IR; P_total = P_linear + P_IR + P_1loop + P_ct + P_stoch + P_PNG\n");
            fprintf(rsd_decomp_fp,"# P_1loop is the numerical residual P_total-P_treeIR-P_ct-P_stoch-P_PNG (same total as the native RSD solver).\n");
            fprintf(rsd_decomp_fp,"# Fractions use P_total as denominator (NOT P_linear); undefined ratios near multipole zero crossings are NaN.\n");
            fprintf(rsd_decomp_fp,"# k");
            for (int elli=0;elli<3;elli++) {
                const int ell=2*elli;
                fprintf(rsd_decomp_fp,
                        " P%d_total P%d_linear P%d_treeIR P%d_IR P%d_1loop P%d_ct P%d_stoch P%d_PNG"
                        " frac%d_linear frac%d_treeIR frac%d_IR frac%d_1loop frac%d_ct frac%d_stoch frac%d_PNG",
                        ell,ell,ell,ell,ell,ell,ell,ell,
                        ell,ell,ell,ell,ell,ell,ell);
            }
            fprintf(rsd_decomp_fp,"\n");
        }

        for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
            double kobs=pnlpt->k[index_k];
            double P0=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P0,kobs);
            double P2=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P2,kobs);
            double P4=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P4,kobs);
            double P0eq=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P0eq,kobs);
            double P2eq=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P2eq,kobs);
            double P4eq=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P4eq,kobs);
            double P0ortho=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P0ortho,kobs);
            double P2ortho=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P2ortho,kobs);
            double P4ortho=qcdm_qscf_interp_linear(Nmax,kdisc,TF_P4ortho,kobs);

            double T0=0.0,T2=0.0,T4=0.0;
            double E0=0.0,E2=0.0,E4=0.0;
            /* Explicit LOS projections of linear, CT, and stochastic parts.
             * T0/2/4 are the IR-resummed tree spectra; linear is the
             * unresummed Z1^2 P_L at the SAME AP-mapped (k,mu). */
            double L0=0.0,L2=0.0,L4=0.0;
            double C0=0.0,C2=0.0,C4=0.0;
            double S0=0.0,S2=0.0,S4=0.0;
            for (int im=0; im<Nmu_tree; im++) {
                double muobs=tree_mu[im];
                double dmu=tree_w[im];
                double ktrue=kobs, mutrue=muobs, jac=1.0;
                if (pnlpt->AP_effect==AP_effect_yes) {
                    double invD2=1.0/(Dratio*Dratio);
                    double apfac=sqrt(invD2+(hratio*hratio-invD2)*muobs*muobs);
                    ktrue=kobs*apfac;
                    mutrue=muobs*hratio/apfac;
                    jac=hratio/(Dratio*Dratio);
                }
                qcdm_qscf_kernel1_result K1;
                class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,ktrue,&K1,pnlpt->error_message),
                           pnlpt->error_message,pnlpt->error_message);
                double Pcl=0.0,Ptrlin=0.0,Plin=0.0;
                if (qcdm_ir_state.enabled==_TRUE_) {
                    double Pnwt=qcdm_qscf_interp_linear(Nmax,kdisc,Pnw,ktrue);
                    double Pwt=qcdm_qscf_interp_linear(Nmax,kdisc,Pw,ktrue);
                    double S=qcdm_tf_ir_exponent(&qcdm_ir_state,ktrue,mutrue);
                    double Dir=exp(-S);
                    Pcl=Pnwt+Dir*Pwt;
                    Ptrlin=Pnwt+Dir*(1.0+S)*Pwt;
                    Plin=Pnwt+Pwt;
                }
                else {
                    Pcl=qcdm_qscf_interp_linear(Nmax,kdisc,Pbin,ktrue);
                    Ptrlin=Pcl;
                    Plin=Pcl;
                }
                double Z1=qcdm_tf_Z1(pnlpt,&K1,mutrue);
                double Ptr=jac*Z1*Z1*Ptrlin;
                double Pl=jac*Z1*Z1*Plin;
                double mu2=mutrue*mutrue,mu4=mu2*mu2;
                double Pmr=K1.F1_r_red*Pcl;
                double Pct=-(pnlpt->qcdm_tf_ct0+pnlpt->qcdm_tf_ct2*mu2+pnlpt->qcdm_tf_ct4*mu4)*ktrue*ktrue*Pcl
                            -(pnlpt->qcdm_tf_ct_r0+pnlpt->qcdm_tf_ct_r2*mu2+pnlpt->qcdm_tf_ct_r4*mu4)*ktrue*ktrue*Pmr;
                double Pstoch=pnlpt->qcdm_tf_stoch0+ktrue*ktrue*(pnlpt->qcdm_tf_stoch2+pnlpt->qcdm_tf_stoch2_mu2*mu2);
                Pct*=jac;
                Pstoch*=jac;
                /* Preserve the original EFT projection E_l exactly. */
                double Peft=Pct+Pstoch;
                double W0=0.5*dmu*qcdm_tf_legendre(0,muobs);
                double W2=2.5*dmu*qcdm_tf_legendre(2,muobs);
                double W4=4.5*dmu*qcdm_tf_legendre(4,muobs);
                T0+=W0*Ptr; T2+=W2*Ptr; T4+=W4*Ptr;
                E0+=W0*Peft; E2+=W2*Peft; E4+=W4*Peft;
                if (rsd_decomp_fp!=NULL) {
                    L0+=W0*Pl; L2+=W2*Pl; L4+=W4*Pl;
                    C0+=W0*Pct; C2+=W2*Pct; C4+=W4*Pct;
                    S0+=W0*Pstoch; S2+=W2*Pstoch; S4+=W4*Pstoch;
                }
            }

            if ((kobs<kdisc[0])||(kobs>kdisc[Nmax-1])) {
                P0=T0+E0; P2=T2+E2; P4=T4+E4;
                P0eq=P2eq=P4eq=0.0;
                P0ortho=P2ortho=P4ortho=0.0;
            }

            /* Output the exact decomposition at this observed k.  The loop
             * is determined from the solver's total after separately
             * projecting the linear/IR/CT/stochastic/PNG pieces.  The
             * resulting sum closes by construction, including the treatment
             * outside the explicitly integrated loop grid. */
            if (rsd_decomp_fp!=NULL) {
                const double ptotal[3]={P0,P2,P4};
                const double plinear[3]={L0,L2,L4};
                const double ptree[3]={T0,T2,T4};
                const double pct[3]={C0,C2,C4};
                const double pstoch[3]={S0,S2,S4};
                const double ppng[3]={
                    pnlpt->qcdm_fNL_equilateral*P0eq+pnlpt->qcdm_fNL_orthogonal*P0ortho,
                    pnlpt->qcdm_fNL_equilateral*P2eq+pnlpt->qcdm_fNL_orthogonal*P2ortho,
                    pnlpt->qcdm_fNL_equilateral*P4eq+pnlpt->qcdm_fNL_orthogonal*P4ortho
                };
                const double h3=pba->h*pba->h*pba->h;
                fprintf(rsd_decomp_fp,"%.12e",kobs/pba->h);
                for (int elli=0;elli<3;elli++) {
                    const double ir=ptree[elli]-plinear[elli];
                    const double loop=ptotal[elli]-ptree[elli]-pct[elli]-pstoch[elli]-ppng[elli];
                    const double scale=MAX(1.e-12,1.e-8*fabs(plinear[elli]));
                    const double denom=(fabs(ptotal[elli])>scale)?ptotal[elli]:NAN;
                    const double comps[8]={ptotal[elli],plinear[elli],ptree[elli],ir,
                                           loop,pct[elli],pstoch[elli],ppng[elli]};
                    const double fractions[7]={plinear[elli]/denom,ptree[elli]/denom,
                                               ir/denom,loop/denom,pct[elli]/denom,
                                               pstoch[elli]/denom,ppng[elli]/denom};
                    for (int ci=0;ci<8;ci++)
                        fprintf(rsd_decomp_fp," %.12e",comps[ci]*h3);
                    for (int ci=0;ci<7;ci++)
                        fprintf(rsd_decomp_fp," %.12e",fractions[ci]);
                }
                fprintf(rsd_decomp_fp,"\n");
            }

            pk_l_0_dd[index_k] = P0 + large_for_logs_big;
            pk_l_2_dd[index_k] = P2 + large_for_logs_big;
            pk_l_4_dd[index_k] = P4 + large_for_logs_big;
            pk_l_0_vv[index_k] = large_for_logs_big;
            pk_l_0_vd[index_k] = large_for_logs_big;
            pk_l_2_vv[index_k] = large_for_logs_big;
            pk_l_2_vd[index_k] = large_for_logs_big;
            pk_l_4_vv[index_k] = large_for_logs_big;
            pk_l_4_vd[index_k] = large_for_logs_big;

            pk_Tree_0_dd[index_k] = T0 + large_for_logs_big;
            pk_Tree_2_vd[index_k] = T2 + large_for_logs_big;
            pk_Tree_4_vv[index_k] = T4 + large_for_logs_big;
            pk_Tree_0_vv[index_k] = large_for_logs_big;
            pk_Tree_0_vd[index_k] = large_for_logs_big;
            pk_Tree_2_vv[index_k] = large_for_logs_big;

            pk_CTR_0[index_k] = kobs*kobs*pk_l[index_k];
            pk_CTR_2[index_k] = kobs*kobs*pk_l[index_k];
            pk_CTR_4[index_k] = kobs*kobs*pk_l[index_k];

            /* DW: store exact-time unit-f_NL multipole templates in the dd
             * slots used by the public CLASS-PT interpolation interface. */
            pk_l_fNL_0_dd[index_k] = P0eq + large_for_logs_fNL;
            pk_l_fNL_2_dd[index_k] = P2eq + large_for_logs_fNL;
            pk_l_fNL_4_dd[index_k] = P4eq + large_for_logs_fNL;
            pk_l_fNL_0_dd_ortho[index_k] = P0ortho + large_for_logs_fNL;
            pk_l_fNL_2_dd_ortho[index_k] = P2ortho + large_for_logs_fNL;
            pk_l_fNL_4_dd_ortho[index_k] = P4ortho + large_for_logs_fNL;
            pk_l_fNL_0_vv[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_0_vd[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_2_vv[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_2_vd[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_4_vv[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_4_vd[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_0_vv_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_0_vd_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_2_vv_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_2_vd_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_4_vv_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_l_fNL_4_vd_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
        }
        if (rsd_decomp_fp!=NULL) fclose(rsd_decomp_fp);
        free(tree_mu); free(tree_w);
        for (int ai=0; ai<9; ai++) free(*tf_ptrs[ai]);
        TIMER_ADD(qcdm_rsd);
    } /* end of RSD conditional expression */

    { void *_f[] = {
        f13, x, x_transfer, x_w, x_w_transfer,
        f22, f12, Pdisc, Ptotaldisc, PPRIMdisc, ddpk_nl,
        ddpk_nl_fNL, ddpk_nl_fNL_ortho, ddpk_CTR, ddpk_Tree, etam,
        etam_transfer, cmsym,
        P22, P13, P12_fNL, P12_fNL_ortho,
        P13UV, P_CTR, P1loop, Ptree, myddlnpk,
        myddlnpPRIMk,
        Xr, Xi, kb_cache, sc_cos, sc_sin,
        /* DW: release the exact PNG transfer buffers after matter/tracer/RSD use. */
        qcdm_Pphi, qcdm_Mphi,
    };
    for (size_t fi_ = 0; fi_ < sizeof(_f)/sizeof(_f[0]); fi_++) free(_f[fi_]); }

    /* ================================================================== */
    /* === BIASED TRACER POWER SPECTRA === */
    /* ================================================================== */
    /* Compute FFTLog coefficients with bias exponents (b=-1.6, b2=-1.25)
     * for the quadratic bias operator integrals. */

    if ((pnlpt->bias == bias_yes) && (qcdm_qscf_qs_active != _TRUE_)) {
        if (pnlpt->nonlinear_pt_verbose > 0)
            printf("Computing the spectra for biased tracers...\n");

        double b2 = -1.6000001;
        double b2_transfer = -1.25;

        double complex *etam2;
        double complex *etam2_transfer;
        double *kb2_cache, *kb2_transfer_cache;
        double *sc_cos2, *sc_sin2;
        double *sc_cos2_t, *sc_sin2_t;

        /* etam2 arrays are cheap to compute — allocate locally each call */
        class_alloc(etam2, (Nmax + 1) * sizeof(complex double), pnlpt->error_message);
        class_alloc(etam2_transfer, (Nmax + 1) * sizeof(complex double), pnlpt->error_message);
        for (int index_c2 = 0; index_c2 < Nmax + 1; index_c2++) {
            etam2[index_c2] = b2 + 2. * M_PI * _Complex_I * js[index_c2] / Nmaxd / Delta;
            etam2_transfer[index_c2] = b2_transfer + 2. * M_PI * _Complex_I * js[index_c2] / Nmaxd / Delta;
        }

        class_alloc(kb2_cache, Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(kb2_transfer_cache, Nmax * sizeof(double), pnlpt->error_message);
        for (int j = 0; j < Nmax; j++) {
            kb2_cache[j] = pow(kdisc[j], b2);
            kb2_transfer_cache[j] = pow(kdisc[j], b2_transfer);
        }
        class_alloc(sc_cos2, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(sc_sin2, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(sc_cos2_t, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(sc_sin2_t, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        precompute_sincos_table(Nmax, Np1, etam2, lnk_cache, sc_cos2, sc_sin2);
        precompute_sincos_table(Nmax, Np1, etam2_transfer, lnk_cache, sc_cos2_t, sc_sin2_t);

        /* FFTLog decomposition with bias exponents */
        TIMER_START(fft);
        double complex *cmsym2;
        FFT_TO_CMSYM(Pbin, b2, etam2, cmsym2);
        double complex *cmsym2_transfer;
        FFT_TO_CMSYM(Tbin, b2_transfer, etam2_transfer, cmsym2_transfer);
        TIMER_ADD(fft);

        /* ================================================================== */
        /* === BIAS OPERATOR LOOP INTEGRALS === */
        /* ================================================================== */
        /* Compute one-loop integrals of quadratic bias operators for the
         * galaxy power spectrum model:
         *
         *   P_gg(k) = b1^2 * P_mm(k) + b1*b2 * P_{b1b2}(k)
         *           + b2^2 * I[d2d2](k) + b1*bG2 * P_{b1bG2}(k)
         *           + b2*bG2 * I[d2G2](k) + bG2^2 * I[G2G2](k) + ...
         *
         * The bias operators are (Desjacques, Jeong & Schmidt 2018):
         *   d2 = delta^2 - <delta^2>     (local density squared)
         *   G2 = (nabla_ij Phi)^2 - (nabla^2 Phi)^2  (tidal field squared)
         *   FG2 = F_2[G2]                 (non-local gravity-tidal coupling)
         *
         * Uses a separate FFTLog basis with bias exponents b2=-1.6,
         * b2_transfer=-1.25 (different from the matter b=-0.3) for
         * better convergence of the bias integrals.
         *
         * For fNL (has_fNL==1), P12 corrections couple the primordial
         * transfer function T(k) to the bias operators.
         *
         * See Section 4 of arXiv:2004.10607 for the full model. */

        /* --- I[delta^2 delta^2] --- */

        double complex *f22_Id2d2;
        double *P_Id2d2;
        class_alloc(f22_Id2d2, Nmax * sizeof(complex double), pnlpt->error_message);
        class_alloc(P_Id2d2, Nmax * sizeof(double), pnlpt->error_message);

        double epsilon_for_logs = 1.e-6;

        int count2 = 0;

        /* Build X matrix for eta2 basis (used by Id2d2 and other bias operators) */
        double *Xr2_bias, *Xi2_bias;
        class_alloc(Xr2_bias, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(Xi2_bias, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        TIMER_START(buildx);
        build_X_from_tables(Nmax, Np1, cmsym2, kb2_cache, sc_cos2, sc_sin2, Xr2_bias, Xi2_bias);
        TIMER_ADD(buildx);

        /* Compute Id2d2 using batch quadratic form */
        {
            double *f22_Id2d2_real = malloc(Nmax * sizeof(double));
            TIMER_START(blas);
            batch_quadratic_form(Np1, Nmax, pnlpt->M22basic_oneline_complex, Xr2_bias, Xi2_bias, f22_Id2d2_real, Mr_ws, Mi_ws, Y1_ws, Y2_ws);
            TIMER_ADD(blas);
            for (index_j = 0; index_j < Nmax; index_j++) {
                f22_Id2d2[index_j] = 2. * f22_Id2d2_real[index_j];
                P_Id2d2[index_j] = fabs(kdisc[index_j] * kdisc[index_j] * kdisc[index_j] * f22_Id2d2[index_j] - kdisc[0] * kdisc[0] * kdisc[0] * 2. * f22_Id2d2_real[0] + epsilon_for_logs);
            }
            free(f22_Id2d2_real);
        }

        /*  Having eta2 we compute the rest of the PT matrices */

        TIMER_START(m22fill);
        {
            int _N = Nmax + 1;
            #pragma omp parallel for schedule(static)
            for (int _l = 0; _l < _N; _l++) {
                int _count_base = _l * _N - _l * (_l - 1) / 2;
                for (int _i = _l; _i < _N; _i++) {
                    int _c = _count_base + (_i - _l);
                    double complex _ei = etam2[_i];
                    double complex _el = etam2[_l];

                    pnlpt->M_IG2G2[_c] = pnlpt->M22basic_oneline_complex[_c] * ((3. + _ei + _el) * (1. + _ei + _el) / ((-0.5 * _ei) * (-0.5 * _el) * (1. - 0.5 * _ei) * (1. - 0.5 * _el)));

                    pnlpt->M_Id2[_c] = pnlpt->M22basic_oneline_complex[_c] * ((3. + _ei + _el) * (4. + 3.5 * (_ei + _el)) / (14. * (-0.5 * _el) * (-0.5 * _ei)));

                    pnlpt->M_Id2G2[_c] = pnlpt->M22basic_oneline_complex[_c] * ((3. + _ei + _el) / ((-0.5 * _ei) * (-0.5 * _el)));

                    pnlpt->M_IG2[_c] = pnlpt->M22basic_oneline_complex[_c] * (-1. * (3. + _ei + _el) * (1. + _ei + _el) * (6. - 3.5 * (_ei + _el)) / (28. * (1. - 0.5 * _ei) * (1. - 0.5 * _el) * (-0.5 * _ei) * (-0.5 * _el)));

                    if (has_fNL) {
                        pnlpt->M_fNLd2[_c] = pnlpt->M12_oneline_complex_bias_real_space_b2[_c];
                        pnlpt->M_fNLd2_ortho[_c] = pnlpt->M12_oneline_complex_bias_real_space_b2_ortho[_c];
                        pnlpt->M_fNLG2[_c] = pnlpt->M12_oneline_complex_bias_real_space_bG2[_c];
                        pnlpt->M_fNLG2_ortho[_c] = pnlpt->M12_oneline_complex_bias_real_space_bG2_ortho[_c];
                    }
                }
            }
        }
        TIMER_ADD(m22fill);

        /* --- fNL contributions: P12 integrals for delta^2 bias --- */
        double *P_fNLd2 = calloc(Nmax, sizeof(double));
        double *P_fNLd2_ortho = calloc(Nmax, sizeof(double));

        /* Build X matrix for transfer with eta2 basis */
        double *Xr2_transfer, *Xi2_transfer;
        class_alloc(Xr2_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(Xi2_transfer, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        if (has_fNL) {
            TIMER_START(buildx);
            build_X_from_tables(Nmax, Np1, cmsym2_transfer, kb2_transfer_cache, sc_cos2_t, sc_sin2_t, Xr2_transfer, Xi2_transfer);
            TIMER_ADD(buildx);
        }

        if (has_fNL) {
            TIMER_START(blas);
            COMPUTE_FNL_P12_PAIR(pnlpt->M_fNLd2, pnlpt->M_fNLd2_ortho,
                                 Xr2_transfer, Xi2_transfer, P_fNLd2, P_fNLd2_ortho);
            TIMER_ADD(blas);
        }

        /* fNL contribution from G2 bias */
        double *P_fNLG2 = calloc(Nmax, sizeof(double));
        double *P_fNLG2_ortho = calloc(Nmax, sizeof(double));

        if (has_fNL) {
            TIMER_START(blas);
            COMPUTE_FNL_P12_PAIR(pnlpt->M_fNLG2, pnlpt->M_fNLG2_ortho,
                                 Xr_transfer, Xi_transfer, P_fNLG2, P_fNLG2_ortho);
            TIMER_ADD(blas);
        }

        /* Tidal field P22 integrals: I[d2], I[G2] */
        double *P_Id2 = malloc(Nmax * sizeof(double));
        TIMER_START(blas);
        TIDAL_P22(pnlpt->M_Id2, Xr2_bias, Xi2_bias, P_Id2, IDENTITY);
        TIMER_ADD(blas);

        double *P_IG2 = malloc(Nmax * sizeof(double));
        TIMER_START(blas);
        TIDAL_P22(pnlpt->M_IG2, Xr2_bias, Xi2_bias, P_IG2, fabs);
        TIMER_ADD(blas);

        TIMER_START(spline_out);
        if (has_fNL) { /* fNL bias contributions */
            double *_fnl_in[] = {P_fNLd2, P_fNLd2_ortho, P_fNLG2, P_fNLG2_ortho};
            double *_fnl_out[] = {pk_fNLd2, pk_fNLd2_ortho, pk_fNLG2, pk_fNLG2_ortho};
            SPLINE_INTERP_BATCH(_fnl_in, _fnl_out, 4, kmin, kmax,
                out_tmp_ + large_for_logs_fNL, large_for_logs_fNL + epsilon_for_logs_fNL, 1);
        } else {
            double *_fnl_in[] = {P_fNLd2, P_fNLd2_ortho, P_fNLG2, P_fNLG2_ortho};
            double *_fnl_out[] = {pk_fNLd2, pk_fNLd2_ortho, pk_fNLG2, pk_fNLG2_ortho};
            for (int fi_ = 0; fi_ < 4; fi_++) {
                for (index_k = 0; index_k < pnlpt->k_size; index_k++)
                    _fnl_out[fi_][index_k] = large_for_logs_fNL;
                free(_fnl_in[fi_]);
            }
        }

        SPLINE_INTERP_OUTPUT(P_Id2, pk_Id2, kmin, kmax,
                             large_for_logs_small + out_tmp_,
                             large_for_logs_small, 1);

        SPLINE_INTERP_OUTPUT(P_IG2, pk_IG2, kmin, kmax,
                             out_tmp_ + large_for_logs_small,
                             epsilon_for_logs + large_for_logs_small, 1);
        TIMER_ADD(spline_out);

        /* Tidal field P22 integrals: I[d2 G2], I[G2 G2] */
        double *P_Id2G2 = malloc(Nmax * sizeof(double));
        TIMER_START(blas);
        TIDAL_P22(pnlpt->M_Id2G2, Xr2_bias, Xi2_bias, P_Id2G2, fabs);
        TIMER_ADD(blas);

        double *P_IG2G2 = malloc(Nmax * sizeof(double));
        TIMER_START(blas);
        TIDAL_P22(pnlpt->M_IG2G2, Xr2_bias, Xi2_bias, P_IG2G2, fabs);
        TIMER_ADD(blas);

        /* --- I[F2 G2] (non-local bias from gravity-tidal coupling) --- */

        double *P_IFG2 = malloc(Nmax * sizeof(double));

        /* Build X matrix for IFG2 (uses cmsym2, etam2) */
        double *Xr2, *Xi2;
        class_alloc(Xr2, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        class_alloc(Xi2, Np1 * Nmax * sizeof(double), pnlpt->error_message);
        TIMER_START(buildx);
        build_X_from_tables(Nmax, Np1, cmsym2, kb2_cache, sc_cos2, sc_sin2, Xr2, Xi2);
        TIMER_ADD(buildx);

        /* Compute P_IFG2 using batch operations */
        double *f13_IFG2_tmp = malloc(Nmax * sizeof(double));
        TIMER_START(blas);
        batch_dot_product(Np1, Nmax, Xr2, Xi2, pnlpt->IFG2_oneline_complex, f13_IFG2_tmp, dot_mr_ws, dot_mi_ws);
        TIMER_ADD(blas);
        for (int j = 0; j < Nmax; j++) {
            P_IFG2[j] = fabs(kdisc[j] * kdisc[j] * kdisc[j] * f13_IFG2_tmp[j] * Pbin[j]);
        }
        free(f13_IFG2_tmp);
        free(Xr2);
        free(Xi2);

        /* Note: P_IFG2 is NOT freed here (do_free=0) because the RSD+AP
         * bias section uses it later for AP spline setup via memcpy. */
        TIMER_START(spline_out);
        SPLINE_INTERP_OUTPUT(P_IFG2, pk_IFG2, kmin, kmax,
                             out_tmp_ + large_for_logs_small,
                             epsilon_for_logs + large_for_logs_small, 0);
        TIMER_ADD(spline_out);

        /* --- fNL bias operator integrals in RSD --- */
        if (pnlpt->rsd == rsd_yes) {
            /* Monopole fNL P12 bias contributions (b1b2, b2, b1bG2, bG2) */
            double *P12_0_b1b2 = malloc(Nmax * sizeof(double));
            double *P12_0_b1b2_ortho = malloc(Nmax * sizeof(double));
            FILL_M12_BIAS(pnlpt->M12_0_b1b2_oneline_complex, pnlpt->M12_0_b1b2_oneline_complex_ortho,
                3., pnlpt->M12_oneline_complex_bias_multipoles_b2_vv0_f1, pnlpt->M12_oneline_complex_bias_multipoles_b2_vv0_f1_ortho,
                P12_0_b1b2, P12_0_b1b2_ortho, cmsym2_transfer, kb2_transfer_cache, sc_cos2_t, sc_sin2_t);

            double *P12_0_b2 = malloc(Nmax * sizeof(double));
            double *P12_0_b2_ortho = malloc(Nmax * sizeof(double));
            FILL_M12_BIAS(pnlpt->M12_0_b2_oneline_complex, pnlpt->M12_0_b2_oneline_complex_ortho,
                f, pnlpt->M12_oneline_complex_bias_multipoles_b2_vv0_f1, pnlpt->M12_oneline_complex_bias_multipoles_b2_vv0_f1_ortho,
                P12_0_b2, P12_0_b2_ortho, cmsym2_transfer, kb2_transfer_cache, sc_cos2_t, sc_sin2_t);

            double *P12_0_b1bG2 = malloc(Nmax * sizeof(double));
            double *P12_0_b1bG2_ortho = malloc(Nmax * sizeof(double));
            FILL_M12_BIAS(pnlpt->M12_0_b1bG2_oneline_complex, pnlpt->M12_0_b1bG2_oneline_complex_ortho,
                3., pnlpt->M12_oneline_complex_bias_multipoles_bG2_vv0_f1, pnlpt->M12_oneline_complex_bias_multipoles_bG2_vv0_f1_ortho,
                P12_0_b1bG2, P12_0_b1bG2_ortho, cmsym_transfer, kb_transfer_cache, sc_cos_t, sc_sin_t);

            double *P12_0_bG2 = malloc(Nmax * sizeof(double));
            double *P12_0_bG2_ortho = malloc(Nmax * sizeof(double));
            FILL_M12_BIAS(pnlpt->M12_0_bG2_oneline_complex, pnlpt->M12_0_bG2_oneline_complex_ortho,
                f, pnlpt->M12_oneline_complex_bias_multipoles_bG2_vv0_f1, pnlpt->M12_oneline_complex_bias_multipoles_bG2_vv0_f1_ortho,
                P12_0_bG2, P12_0_bG2_ortho, cmsym_transfer, kb_transfer_cache, sc_cos_t, sc_sin_t);

            /* Computing bias P22 corrections in RSD */

            double *P_0_b1b2, *P_0_b2, *P_0_b1bG2, *P_0_bG2;
            double *P_2_b1b2, *P_2_b2, *P_2_b1bG2, *P_2_bG2;
            double *P_4_b2, *P_4_bG2;
            { /* Allocate all P22 bias arrays */
                double **_p22b[] = {
                    &P_0_b1b2, &P_0_b2, &P_0_b1bG2, &P_0_bG2,
                    &P_2_b1b2, &P_2_b2, &P_2_b1bG2, &P_2_bG2,
                    &P_4_b2, &P_4_bG2};
                for (int ai_ = 0; ai_ < 10; ai_++)
                    class_alloc(*_p22b[ai_], Nmax * sizeof(double), pnlpt->error_message);
            }

            double *Xr2, *Xi2;
            class_alloc(Xr2, Np1 * Nmax * sizeof(double), pnlpt->error_message);
            class_alloc(Xi2, Np1 * Nmax * sizeof(double), pnlpt->error_message);
            TIMER_START(buildx);
            build_X_from_tables(Nmax, Np1, cmsym2, kb2_cache, sc_cos2, sc_sin2, Xr2, Xi2);
            TIMER_ADD(buildx);
            double *f22_tmp = malloc(Nmax * sizeof(double));

            /* Monopole (ell=0) M22 bias corrections */
            TIMER_START(m22fill);
            FILL_M22_BIAS(pnlpt->M22_0_b1b2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * (-12. + 7. * (3. + f) * nu1 + 7. * (3. + f) * nu2) / (42. * nu1 * nu2), P_0_b1b2);
            FILL_M22_BIAS(pnlpt->M22_0_b2_oneline_complex,
                (7. * f * f * (12. + 6. * nu1 * nu1 - 17. * nu2 + 6. * nu2 * nu2 + nu1 * (-17. + 12. * nu2)) + 5. * f * (24. + 14. * nu1 * nu1 - 37. * nu2 + 14. * nu2 * nu2 + nu1 * (-37. + 28. * nu2))) / (210. * nu1 * nu2), P_0_b2);
            FILL_M22_BIAS(pnlpt->M22_0_b1bG2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * (-1. + 2. * nu1 + 2. * nu2) * (7. * f * (2. + nu1 + nu2) + 3. * (6. + 7. * nu1 + 7. * nu2)) / (42. * nu1 * (1. + nu1) * nu2 * (1. + nu2)), P_0_b1bG2);
            FILL_M22_BIAS(pnlpt->M22_0_bG2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * (-1. + 2. * nu1 + 2. * nu2) * (-10. * f + 7. * f * (5. * (nu1 + nu2) + f * (-2. + 3. * nu1 + 3. * nu2))) / (210. * nu1 * (1. + nu1) * nu2 * (1. + nu2)), P_0_bG2);
            TIMER_ADD(m22fill);

            /* Quadrupole (ell=2) fNL P12 bias contributions */
            double *P12_2_b2 = malloc(Nmax * sizeof(double));
            double *P12_2_b2_ortho = malloc(Nmax * sizeof(double));
            FILL_M12_BIAS(pnlpt->M12_2_b2_oneline_complex, pnlpt->M12_2_b2_oneline_complex_ortho,
                2. * f, pnlpt->M12_oneline_complex_bias_multipoles_b2_vv0_f1, pnlpt->M12_oneline_complex_bias_multipoles_b2_vv0_f1_ortho,
                P12_2_b2, P12_2_b2_ortho, cmsym2_transfer, kb2_transfer_cache, sc_cos2_t, sc_sin2_t);
            double *P12_2_bG2 = malloc(Nmax * sizeof(double));
            double *P12_2_bG2_ortho = malloc(Nmax * sizeof(double));
            FILL_M12_BIAS(pnlpt->M12_2_bG2_oneline_complex, pnlpt->M12_2_bG2_oneline_complex_ortho,
                2. * f, pnlpt->M12_oneline_complex_bias_multipoles_bG2_vv0_f1, pnlpt->M12_oneline_complex_bias_multipoles_bG2_vv0_f1_ortho,
                P12_2_bG2, P12_2_bG2_ortho, cmsym_transfer, kb_transfer_cache, sc_cos_t, sc_sin_t);

            /* Quadrupole (ell=2) M22 bias corrections */
            TIMER_START(m22fill);
            FILL_M22_BIAS(pnlpt->M22_2_b1b2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * f * (nu1 + nu2) / (3. * nu1 * nu2), P_2_b1b2);
            FILL_M22_BIAS(pnlpt->M22_2_b2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * f * (-16. + 14. * (nu1 + nu2) + f * (-13. + 12. * (nu1 + nu2))) / (42. * nu1 * nu2), P_2_b2);
            FILL_M22_BIAS(pnlpt->M22_2_b1bG2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * (-1. + 2. * nu1 + 2. * nu2) * f * (2. + nu1 + nu2) / (3. * nu1 * (1. + nu1) * nu2 * (1. + nu2)), P_2_b1bG2);

            FILL_M22_BIAS(pnlpt->M22_2_bG2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * (-1. + 2. * nu1 + 2. * nu2) * f * (-2. - f + 7. * (nu1 + nu2) + 6. * f * (nu1 + nu2)) / (21. * nu1 * (1. + nu1) * nu2 * (1. + nu2)), P_2_bG2);
            TIMER_ADD(m22fill);

            /* Hexadecapole (ell=4) M22 bias corrections */
            TIMER_START(m22fill);
            FILL_M22_BIAS(pnlpt->M22_4_b2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * (-1. + 2. * nu1 + 2. * nu2) * 2. * f * f / (35. * nu1 * nu2), P_4_b2);
            FILL_M22_BIAS(pnlpt->M22_4_bG2_oneline_complex,
                (-3. + 2. * nu1 + 2. * nu2) * (-1. + 2. * nu1 + 2. * nu2) * 4. * f * f * (1. + nu1 + nu2) / (35. * nu1 * (1. + nu1) * nu2 * (1. + nu2)), P_4_bG2);
            TIMER_ADD(m22fill);

            free(f22_tmp);
            free(Xr2);
            free(Xi2);

            /* Numerical integration for bias tracers */
            double *P_Id2d2_2, *P_Id2d2_4, *P_Id2G2_2, *P_Id2G2_4;
            double *P_IG2G2_2, *P_IG2G2_4, *P_4_b1b2, *P_4_b1bG2;
            double *P_IFG2_0b1_x, *P_IFG2_0, *P_IFG2_2;
            double *P_Id2d2_new, *P_Id2G2_new, *P_IG2G2_new, *P_IFG2_new;
            double *P_0_bG2_new, *P_2_bG2_new, *P_4_bG2_new;
            double *P_0_b1b2_new, *P_2_b1b2_new, *P_0_b1bG2_new, *P_2_b1bG2_new;
            double *P_0_b2_new, *P_2_b2_new, *P_4_b2_new;
            double *P12_0_b1b2_new, *P12_0_b1bG2_new, *P12_0_b2_new;
            double *P12_2_b2_new, *P12_0_bG2_new, *P12_2_bG2_new;
            double *P12_0_b1b2_new_ortho, *P12_0_b1bG2_new_ortho, *P12_0_b2_new_ortho;
            double *P12_2_b2_new_ortho, *P12_0_bG2_new_ortho, *P12_2_bG2_new_ortho;
            double *P12_2_b1b2, *P12_2_b1bG2, *P12_4_b1b2, *P12_4_b1bG2;
            double *P12_4_b2, *P12_4_bG2;
            double *P12_2_b1b2_ortho, *P12_2_b1bG2_ortho;
            double *P12_4_b1b2_ortho, *P12_4_b1bG2_ortho;
            double *P12_4_b2_ortho, *P12_4_bG2_ortho;

            { /* Allocate all bias AP arrays */
                double **_bap[] = {
                    &P_Id2d2_2, &P_Id2d2_4, &P_Id2G2_2, &P_Id2G2_4,
                    &P_IG2G2_2, &P_IG2G2_4, &P_4_b1b2, &P_4_b1bG2,
                    &P_IFG2_0b1_x, &P_IFG2_0, &P_IFG2_2,
                    &P_Id2d2_new, &P_Id2G2_new, &P_IG2G2_new, &P_IFG2_new,
                    &P_0_bG2_new, &P_2_bG2_new, &P_4_bG2_new,
                    &P_0_b1b2_new, &P_2_b1b2_new, &P_0_b1bG2_new, &P_2_b1bG2_new,
                    &P_0_b2_new, &P_2_b2_new, &P_4_b2_new,
                    &P12_0_b1b2_new, &P12_0_b1bG2_new, &P12_0_b2_new,
                    &P12_2_b2_new, &P12_0_bG2_new, &P12_2_bG2_new,
                    &P12_0_b1b2_new_ortho, &P12_0_b1bG2_new_ortho, &P12_0_b2_new_ortho,
                    &P12_2_b2_new_ortho, &P12_0_bG2_new_ortho, &P12_2_bG2_new_ortho,
                    &P12_2_b1b2, &P12_2_b1bG2, &P12_4_b1b2, &P12_4_b1bG2,
                    &P12_4_b2, &P12_4_bG2,
                    &P12_2_b1b2_ortho, &P12_2_b1bG2_ortho,
                    &P12_4_b1b2_ortho, &P12_4_b1bG2_ortho,
                    &P12_4_b2_ortho, &P12_4_bG2_ortho};
                for (int ai_ = 0; ai_ < 49; ai_++)
                    class_alloc(*_bap[ai_], Nmax * sizeof(double), pnlpt->error_message);
            }

            { /* Copy bias arrays to _new variants for AP spline setup */
                size_t sz = Nmax * sizeof(double);
                double *_src[] = {
                    P_Id2d2, P_Id2G2, P_IG2G2, P_0_b1b2, P_2_b1b2,
                    P_0_b1bG2, P_2_b1bG2, P_4_b2, P_2_b2, P_0_b2,
                    P_4_bG2, P_2_bG2, P_0_bG2, P_IFG2,
                    P12_0_b1b2, P12_0_b1bG2, P12_0_b2, P12_2_b2, P12_0_bG2, P12_2_bG2,
                    P12_0_b1b2_ortho, P12_0_b1bG2_ortho, P12_0_b2_ortho,
                    P12_2_b2_ortho, P12_0_bG2_ortho, P12_2_bG2_ortho};
                double *_dst[] = {
                    P_Id2d2_new, P_Id2G2_new, P_IG2G2_new, P_0_b1b2_new, P_2_b1b2_new,
                    P_0_b1bG2_new, P_2_b1bG2_new, P_4_b2_new, P_2_b2_new, P_0_b2_new,
                    P_4_bG2_new, P_2_bG2_new, P_0_bG2_new, P_IFG2_new,
                    P12_0_b1b2_new, P12_0_b1bG2_new, P12_0_b2_new, P12_2_b2_new, P12_0_bG2_new, P12_2_bG2_new,
                    P12_0_b1b2_new_ortho, P12_0_b1bG2_new_ortho, P12_0_b2_new_ortho,
                    P12_2_b2_new_ortho, P12_0_bG2_new_ortho, P12_2_bG2_new_ortho};
                for (int ci_ = 0; ci_ < 26; ci_++) memcpy(_dst[ci_], _src[ci_], sz);
            }

            TIMER_START(ap_spline);
            AP_SPLINE_SETUP(P_Id2d2);
            AP_SPLINE_SETUP(P_Id2G2);
            AP_SPLINE_SETUP(P_IG2G2);
            AP_SPLINE_SETUP_EX(P_0_b1b2, P_0_b1b2_new);
            AP_SPLINE_SETUP_EX(P_2_b1b2, P_2_b1b2_new);
            AP_SPLINE_SETUP(P_0_b1bG2);
            AP_SPLINE_SETUP(P_2_b1bG2);
            AP_SPLINE_SETUP_EX(P_0_b2, P_0_b2_new);
            AP_SPLINE_SETUP_EX(P_2_b2, P_2_b2_new);
            AP_SPLINE_SETUP_EX(P_4_b2, P_4_b2_new);
            AP_SPLINE_SETUP_EX(P_0_bG2, P_0_bG2_new);
            AP_SPLINE_SETUP_EX(P_2_bG2, P_2_bG2_new);
            AP_SPLINE_SETUP_EX(P_4_bG2, P_4_bG2_new);
            AP_SPLINE_SETUP_EX(P_IFG2, P_IFG2_new);
            AP_SPLINE_SETUP(Pbin);
            /* P12 bias AP splines + Tbin: only needed for fNL.
             * We declare dd_ and _ap_out variables manually so they exist
             * in function scope for the free block and AP_INTERP macros. */
            double *dd_P12_0_b2 = NULL, *dd_P12_0_b2_ortho = NULL;
            double *dd_P12_0_b1b2 = NULL, *dd_P12_0_b1b2_ortho = NULL;
            double *dd_P12_0_bG2 = NULL, *dd_P12_0_bG2_ortho = NULL;
            double *dd_P12_0_b1bG2 = NULL, *dd_P12_0_b1bG2_ortho = NULL;
            double *dd_P12_2_b2 = NULL, *dd_P12_2_b2_ortho = NULL;
            double *dd_P12_2_bG2 = NULL, *dd_P12_2_bG2_ortho = NULL;
            double *dd_Tbin = NULL;
            double P12_0_b2_ap_out = 0, P12_0_b2_ortho_ap_out = 0;
            double P12_0_b1b2_ap_out = 0, P12_0_b1b2_ortho_ap_out = 0;
            double P12_0_bG2_ap_out = 0, P12_0_bG2_ortho_ap_out = 0;
            double P12_0_b1bG2_ap_out = 0, P12_0_b1bG2_ortho_ap_out = 0;
            double P12_2_b2_ap_out = 0, P12_2_b2_ortho_ap_out = 0;
            double P12_2_bG2_ap_out = 0, P12_2_bG2_ortho_ap_out = 0;
            double Tbin_ap_out = 0;
            if (has_fNL) {
                /* Allocate and compute spline tables for P12 bias arrays */
                double **_dd_p12[] = {
                    &dd_P12_0_b2, &dd_P12_0_b2_ortho, &dd_P12_0_b1b2, &dd_P12_0_b1b2_ortho,
                    &dd_P12_0_bG2, &dd_P12_0_bG2_ortho, &dd_P12_0_b1bG2, &dd_P12_0_b1bG2_ortho,
                    &dd_P12_2_b2, &dd_P12_2_b2_ortho, &dd_P12_2_bG2, &dd_P12_2_bG2_ortho, &dd_Tbin};
                double *_p12_data[] = {
                    P12_0_b2_new, P12_0_b2_new_ortho, P12_0_b1b2_new, P12_0_b1b2_new_ortho,
                    P12_0_bG2_new, P12_0_bG2_new_ortho, P12_0_b1bG2_new, P12_0_b1bG2_new_ortho,
                    P12_2_b2_new, P12_2_b2_new_ortho, P12_2_bG2_new, P12_2_bG2_new_ortho, Tbin};
                for (int si_ = 0; si_ < 13; si_++) {
                    class_alloc(*_dd_p12[si_], sizeof(double) * Nmax, pnlpt->error_message);
                    class_call(array_spline_table_columns(kdisc, Nmax, _p12_data[si_], 1, *_dd_p12[si_],
                                                          _SPLINE_NATURAL_, pnlpt->error_message),
                               pnlpt->error_message, pnlpt->error_message);
                }
            }
            TIMER_ADD(ap_spline);

            double Pd2d2_in = 0., Pd2G2_in = 0., PG2G2_in = 0.;
            double Pb1b2_in = 0., Pb1bG2_in = 0., Pb2_in = 0., PbG2_in = 0.;
            double IFG2_in = 0.;
            double P12b1b2_in = 0., P12b1bG2_in = 0., P12b2_in = 0., P12bG2_in = 0.;
            double P12b1b2_in_ortho = 0., P12b1bG2_in_ortho = 0.;
            double P12b2_in_ortho = 0., P12bG2_in_ortho = 0.;
            double LegendreP0 = 1., LegendreP2 = 0., LegendreP4 = 0.;
            double LegendreP2true = 0., LegendreP4true = 0.;

            int index_gauss2 = 0;
            double mu = 0.;
            double mutrue = 0.;
            double ktrue = 0.;
            last_index = 0;
            double Sigmatot = 0.;
            double p_lo = 0.;
            double Exp = 0.;
            double t_lo = 0.;

            /* Zero all bias AP projection arrays */
            { void *_z[] = {
                P_IFG2_0b1_x, P_IFG2_0, P_IFG2_2, P_Id2d2_2, P_Id2d2_4,
                P_Id2G2_2, P_Id2G2_4, P_IG2G2_2, P_IG2G2_4,
                P_4_b1b2, P_4_b1bG2,
                P12_2_b1b2, P12_2_b1bG2, P12_4_b1b2, P12_4_b1bG2, P12_4_b2, P12_4_bG2,
                P12_2_b1b2_ortho, P12_2_b1bG2_ortho,
                P12_4_b1b2_ortho, P12_4_b1bG2_ortho, P12_4_b2_ortho, P12_4_bG2_ortho,
                P_Id2d2, P_Id2G2, P_IG2G2,
                P_0_b1b2, P_2_b1b2, P_0_b1bG2, P_2_b1bG2,
                P_0_b2, P_2_b2, P_4_b2, P_0_bG2, P_2_bG2, P_4_bG2,
                P12_0_b2, P12_2_b2, P12_0_bG2, P12_2_bG2, P12_0_b1b2, P12_0_b1bG2,
                P12_0_b2_ortho, P12_2_b2_ortho, P12_0_bG2_ortho, P12_2_bG2_ortho,
                P12_0_b1b2_ortho, P12_0_b1bG2_ortho
            };
            for (int zi_ = 0; zi_ < 48; zi_++) memset(_z[zi_], 0, sizeof(double) * Nmax); }

            TIMER_START(bias_ap);
            /* Precompute AP constants (reuse from RSD loop if already set) */
            double bias_ap_inv_Dr2_ = 1. / (Dratio * Dratio);
            double bias_ap_hr2_minus_inv_Dr2_ = hratio * hratio - bias_ap_inv_Dr2_;

            for (index_j = Nside; index_j < Nmax - Nside; index_j++) {
                Pnw_ap_out = 0.;
                Pw_ap_out = 0.;
                Tnw_ap_out = 0.;
                Tw_ap_out = 0.;

                for (index_gauss2 = 0; index_gauss2 < 40; index_gauss2++) {
                    mu = pnlpt->gauss_x[index_gauss2];

                    if (pnlpt->AP_effect == AP_effect_yes) {
                        double ap_fac_ = sqrt(bias_ap_inv_Dr2_ + bias_ap_hr2_minus_inv_Dr2_ * mu * mu);
                        mutrue = mu * hratio / ap_fac_;
                        ktrue = kdisc[index_j] * ap_fac_;
                    } else {
                        mutrue = mu;
                        ktrue = kdisc[index_j];
                    }

                    LegendreP2 = (3. * mu * mu - 1.) / 2.;
                    LegendreP4 = (35. * mu * mu * mu * mu - 30. * mu * mu + 3.) / 8.;
                    double mu2t = mutrue * mutrue;
                    double mu4t = mu2t * mu2t;
                    LegendreP2true = (3. * mu2t - 1.) / 2.;
                    LegendreP4true = (35. * mu4t - 30. * mu2t + 3.) / 8.;

                    /* Single binary search for all spline interpolations at this ktrue */
                    AP_BSEARCH_SETUP();

                    AP_INTERP_FAST(Pnw);
                    AP_INTERP_FAST(Pw);
                    AP_INTERP_FAST(Pbin);

                    if (has_fNL) {
                        AP_INTERP_FAST(Tnw);
                        AP_INTERP_FAST(Tw);
                        AP_INTERP_FAST(Tbin);
                        AP_INTERP_EX_FAST(P12_0_b1b2, P12_0_b1b2_new);
                        AP_INTERP_EX_FAST(P12_0_b1bG2, P12_0_b1bG2_new);
                        AP_INTERP_EX_FAST(P12_0_b2, P12_0_b2_new);
                        AP_INTERP_EX_FAST(P12_2_b2, P12_2_b2_new);
                        AP_INTERP_EX_FAST(P12_0_bG2, P12_0_bG2_new);
                        AP_INTERP_EX_FAST(P12_2_bG2, P12_2_bG2_new);
                        AP_INTERP_EX_FAST(P12_0_b1b2_ortho, P12_0_b1b2_new_ortho);
                        AP_INTERP_EX_FAST(P12_0_b1bG2_ortho, P12_0_b1bG2_new_ortho);
                        AP_INTERP_EX_FAST(P12_0_b2_ortho, P12_0_b2_new_ortho);
                        AP_INTERP_EX_FAST(P12_2_b2_ortho, P12_2_b2_new_ortho);
                        AP_INTERP_EX_FAST(P12_0_bG2_ortho, P12_0_bG2_new_ortho);
                        AP_INTERP_EX_FAST(P12_2_bG2_ortho, P12_2_bG2_new_ortho);
                    }

                    AP_INTERP_EX_FAST(P_IFG2, P_IFG2_new);
                    AP_INTERP_EX_FAST(P_Id2d2, P_Id2d2_new);
                    AP_INTERP_EX_FAST(P_Id2G2, P_Id2G2_new);
                    AP_INTERP_EX_FAST(P_IG2G2, P_IG2G2_new);
                    AP_INTERP_EX_FAST(P_0_b1b2, P_0_b1b2_new);
                    AP_INTERP_EX_FAST(P_2_b1b2, P_2_b1b2_new);
                    AP_INTERP_EX_FAST(P_0_b1bG2, P_0_b1bG2_new);
                    AP_INTERP_EX_FAST(P_2_b1bG2, P_2_b1bG2_new);
                    AP_INTERP_EX_FAST(P_0_b2, P_0_b2_new);
                    AP_INTERP_EX_FAST(P_2_b2, P_2_b2_new);
                    AP_INTERP_EX_FAST(P_4_b2, P_4_b2_new);
                    AP_INTERP_EX_FAST(P_0_bG2, P_0_bG2_new);
                    AP_INTERP_EX_FAST(P_2_bG2, P_2_bG2_new);
                    AP_INTERP_EX_FAST(P_4_bG2, P_4_bG2_new);

                    Sigmatot = SigmaBAO * (1. + f * mu2t * (2. + f)) + f * f * mu2t * (mu2t - 1.) * deltaSigmaBAO;
                    Exp = exp(-Sigmatot * ktrue * ktrue);
                    p_lo = (Pnw_ap_out + Pw_ap_out * Exp) / Pbin_ap_out;

                    IFG2_in = p_lo * P_IFG2_ap_out * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                    Pd2d2_in = P_Id2d2_ap_out * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    Pd2G2_in = P_Id2G2_ap_out * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    PG2G2_in = P_IG2G2_ap_out * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    Pb1b2_in = (P_0_b1b2_ap_out + LegendreP2true * P_2_b1b2_ap_out) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    Pb1bG2_in = (P_0_b1bG2_ap_out + LegendreP2true * P_2_b1bG2_ap_out) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    Pb2_in = (P_0_b2_ap_out + LegendreP2true * P_2_b2_ap_out + LegendreP4true * P_4_b2_ap_out) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    PbG2_in = (P_0_bG2_ap_out + LegendreP2true * P_2_bG2_ap_out + LegendreP4true * P_4_bG2_ap_out) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                    if (has_fNL) {
                        t_lo = (Tnw_ap_out + Tw_ap_out * Exp) / Tbin_ap_out;

                        P12b1b2_in = t_lo * (P12_0_b1b2_ap_out)*pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                        P12b1bG2_in = t_lo * (P12_0_b1bG2_ap_out)*pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                        P12b2_in = t_lo * (P12_0_b2_ap_out + LegendreP2true * P12_2_b2_ap_out) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                        P12bG2_in = t_lo * (P12_0_bG2_ap_out + LegendreP2true * P12_2_bG2_ap_out) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;

                        P12b1b2_in_ortho = t_lo * (P12_0_b1b2_ortho_ap_out)*pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                        P12b1bG2_in_ortho = t_lo * (P12_0_b1bG2_ortho_ap_out)*pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                        P12b2_in_ortho = t_lo * (P12_0_b2_ortho_ap_out + LegendreP2true * P12_2_b2_ortho_ap_out) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                        P12bG2_in_ortho = t_lo * (P12_0_bG2_ortho_ap_out + LegendreP2true * P12_2_bG2_ortho_ap_out) * pnlpt->gauss_w[index_gauss2] * hratio / Dratio / Dratio;
                    }

                    P_IFG2_0b1_x[index_j] += IFG2_in * LegendreP0 / 2.;
                    P_IFG2_0[index_j] += IFG2_in * f * mu2t * LegendreP0 / 2.;
                    P_IFG2_2[index_j] += IFG2_in * f * mu2t * LegendreP2 * 2.5;
                    LEGENDRE_PROJECT(Pd2d2_in, P_Id2d2, P_Id2d2_2, P_Id2d2_4);
                    LEGENDRE_PROJECT(Pd2G2_in, P_Id2G2, P_Id2G2_2, P_Id2G2_4);
                    LEGENDRE_PROJECT(PG2G2_in, P_IG2G2, P_IG2G2_2, P_IG2G2_4);
                    LEGENDRE_PROJECT(Pb1b2_in, P_0_b1b2, P_2_b1b2, P_4_b1b2);
                    LEGENDRE_PROJECT(Pb1bG2_in, P_0_b1bG2, P_2_b1bG2, P_4_b1bG2);
                    LEGENDRE_PROJECT(Pb2_in, P_0_b2, P_2_b2, P_4_b2);
                    LEGENDRE_PROJECT(PbG2_in, P_0_bG2, P_2_bG2, P_4_bG2);

                    if (has_fNL) {
                        LEGENDRE_PROJECT(P12b2_in, P12_0_b2, P12_2_b2, P12_4_b2);
                        LEGENDRE_PROJECT(P12bG2_in, P12_0_bG2, P12_2_bG2, P12_4_bG2);
                        LEGENDRE_PROJECT(P12b1b2_in, P12_0_b1b2, P12_2_b1b2, P12_4_b1b2);
                        LEGENDRE_PROJECT(P12b1bG2_in, P12_0_b1bG2, P12_2_b1bG2, P12_4_b1bG2);
                        LEGENDRE_PROJECT(P12b2_in_ortho, P12_0_b2_ortho, P12_2_b2_ortho, P12_4_b2_ortho);
                        LEGENDRE_PROJECT(P12bG2_in_ortho, P12_0_bG2_ortho, P12_2_bG2_ortho, P12_4_bG2_ortho);
                        LEGENDRE_PROJECT(P12b1b2_in_ortho, P12_0_b1b2_ortho, P12_2_b1b2_ortho, P12_4_b1b2_ortho);
                        LEGENDRE_PROJECT(P12b1bG2_in_ortho, P12_0_b1bG2_ortho, P12_2_b1bG2_ortho, P12_4_b1bG2_ortho);
                    }
                }
            }
            TIMER_ADD(bias_ap);

            { void *_f[] = {
                P_Id2d2_new, P_Id2G2_new, P_IG2G2_new, P_0_b1b2_new, P_2_b1b2_new,
                P_0_b1bG2_new, P_2_b1bG2_new, P_0_b2_new, P_2_b2_new, P_4_b2_new,
                P_0_bG2_new, P_2_bG2_new, P_4_bG2_new, dd_P_Id2d2, dd_P_Id2G2,
                dd_P_IG2G2, dd_P_0_b1b2, dd_P_2_b1b2, dd_P_0_b1bG2, dd_P_2_b1bG2,
                dd_P_0_b2, dd_P_2_b2, dd_P_4_b2, dd_P_0_bG2, dd_P_2_bG2,
                dd_P_4_bG2, dd_P_IFG2, dd_Pbin, dd_Tbin, P12_0_b1b2_new,
                dd_P12_0_b1b2, P12_0_b1bG2_new, dd_P12_0_b1bG2, P12_0_b2_new, dd_P12_0_b2,
                P12_2_b2_new, dd_P12_2_b2, P12_0_bG2_new, dd_P12_0_bG2, P12_2_bG2_new,
                dd_P12_2_bG2, P12_0_b1b2_new_ortho, dd_P12_0_b1b2_ortho, P12_0_b1bG2_new_ortho, dd_P12_0_b1bG2_ortho,
                P12_0_b2_new_ortho, dd_P12_0_b2_ortho, P12_2_b2_new_ortho, dd_P12_2_b2_ortho, P12_0_bG2_new_ortho,
                dd_P12_0_bG2_ortho, P12_2_bG2_new_ortho, dd_P12_2_bG2_ortho,
            };
            for (int fi_ = 0; fi_ < 53; fi_++) free(_f[fi_]); }

            TIMER_START(spline_out);
            if (has_fNL) { /* fNL P12 biased tracer multipoles */
                double *_p12b_in[] = {
                    P12_0_b1b2, P12_0_b1b2_ortho, P12_0_b1bG2, P12_0_b1bG2_ortho,
                    P12_0_b2, P12_0_b2_ortho, P12_0_bG2, P12_0_bG2_ortho,
                    P12_2_b2, P12_2_b2_ortho, P12_2_bG2, P12_2_bG2_ortho,
                    P12_4_b2, P12_4_b2_ortho, P12_4_bG2, P12_4_bG2_ortho,
                    P12_2_b1b2, P12_2_b1b2_ortho, P12_4_b1b2, P12_4_b1b2_ortho,
                    P12_2_b1bG2, P12_2_b1bG2_ortho, P12_4_b1bG2, P12_4_b1bG2_ortho};
                double *_p12b_out[] = {
                    pk12_l_0_b1b2, pk12_l_0_b1b2_ortho, pk12_l_0_b1bG2, pk12_l_0_b1bG2_ortho,
                    pk12_l_0_b2, pk12_l_0_b2_ortho, pk12_l_0_bG2, pk12_l_0_bG2_ortho,
                    pk12_l_2_b2, pk12_l_2_b2_ortho, pk12_l_2_bG2, pk12_l_2_bG2_ortho,
                    pk12_l_4_b2, pk12_l_4_b2_ortho, pk12_l_4_bG2, pk12_l_4_bG2_ortho,
                    pk12_l_2_b1b2, pk12_l_2_b1b2_ortho, pk12_l_4_b1b2, pk12_l_4_b1b2_ortho,
                    pk12_l_2_b1bG2, pk12_l_2_b1bG2_ortho, pk12_l_4_b1bG2, pk12_l_4_b1bG2_ortho};
                SPLINE_INTERP_BATCH(_p12b_in, _p12b_out, 24, kminnew, kmaxnew,
                    out_tmp_ + large_for_logs_fNL, large_for_logs_fNL + epsilon_for_logs_fNL, 1);
            } else { /* No fNL: fill with offset constant and free input arrays */
                double *_p12b_in[] = {
                    P12_0_b1b2, P12_0_b1b2_ortho, P12_0_b1bG2, P12_0_b1bG2_ortho,
                    P12_0_b2, P12_0_b2_ortho, P12_0_bG2, P12_0_bG2_ortho,
                    P12_2_b2, P12_2_b2_ortho, P12_2_bG2, P12_2_bG2_ortho,
                    P12_4_b2, P12_4_b2_ortho, P12_4_bG2, P12_4_bG2_ortho,
                    P12_2_b1b2, P12_2_b1b2_ortho, P12_4_b1b2, P12_4_b1b2_ortho,
                    P12_2_b1bG2, P12_2_b1bG2_ortho, P12_4_b1bG2, P12_4_b1bG2_ortho};
                double *_p12b_out[] = {
                    pk12_l_0_b1b2, pk12_l_0_b1b2_ortho, pk12_l_0_b1bG2, pk12_l_0_b1bG2_ortho,
                    pk12_l_0_b2, pk12_l_0_b2_ortho, pk12_l_0_bG2, pk12_l_0_bG2_ortho,
                    pk12_l_2_b2, pk12_l_2_b2_ortho, pk12_l_2_bG2, pk12_l_2_bG2_ortho,
                    pk12_l_4_b2, pk12_l_4_b2_ortho, pk12_l_4_bG2, pk12_l_4_bG2_ortho,
                    pk12_l_2_b1b2, pk12_l_2_b1b2_ortho, pk12_l_4_b1b2, pk12_l_4_b1b2_ortho,
                    pk12_l_2_b1bG2, pk12_l_2_b1bG2_ortho, pk12_l_4_b1bG2, pk12_l_4_b1bG2_ortho};
                for (int fi_ = 0; fi_ < 24; fi_++) {
                    for (index_k = 0; index_k < pnlpt->k_size; index_k++)
                        _p12b_out[fi_][index_k] = large_for_logs_fNL;
                    free(_p12b_in[fi_]); /* match do_free=1 behavior */
                }
            }
            { /* Id2d2/Id2G2/IG2G2 higher multipoles */
                double *_id_in[] = {P_Id2d2_2, P_Id2d2_4, P_Id2G2_2, P_Id2G2_4, P_IG2G2_2, P_IG2G2_4};
                double *_id_out[] = {pk_Id2d2_2, pk_Id2d2_4, pk_Id2G2_2, pk_Id2G2_4, pk_IG2G2_2, pk_IG2G2_4};
                SPLINE_INTERP_BATCH(_id_in, _id_out, 6, kminnew, kmaxnew,
                    out_tmp_ + large_for_logs_big, large_for_logs_big, 1);
            }
            { /* Biased tracer P22 multipoles */
                double *_bt_in[] = {
                    P_0_b1b2, P_0_b2, P_0_b1bG2, P_2_b1b2, P_4_b1b2, P_0_bG2,
                    P_2_b2, P_2_b1bG2, P_4_b1bG2, P_2_bG2, P_4_b2, P_4_bG2};
                double *_bt_out[] = {
                    pk_l_0_b1b2, pk_l_0_b2, pk_l_0_b1bG2, pk_l_2_b1b2, pk_l_4_b1b2, pk_l_0_bG2,
                    pk_l_2_b2, pk_l_2_b1bG2, pk_l_4_b1bG2, pk_l_2_bG2, pk_l_4_b2, pk_l_4_bG2};
                SPLINE_INTERP_BATCH(_bt_in, _bt_out, 12, kminnew, kmaxnew,
                    out_tmp_ + large_for_logs_big, epsilon_for_logs + large_for_logs_big, 1);
            }
            free(P_IFG2_new);
            { /* IFG2 multipoles */
                double *_ifg_in[] = {P_IFG2_0b1_x, P_IFG2_0, P_IFG2_2};
                double *_ifg_out[] = {pk_IFG2_0b1, pk_IFG2_0, pk_IFG2_2};
                SPLINE_INTERP_BATCH(_ifg_in, _ifg_out, 3, kminnew, kmaxnew,
                    out_tmp_ + large_for_logs_big, large_for_logs_big, 0);
            }
            TIMER_ADD(spline_out);
            free(P_IFG2_0b1_x);
            free(P_IFG2_0);
            free(P_IFG2_2);

            /* P22 bias coefficient arrays (P_0_b1b2, ..., P_4_bG2) are freed
             * by SPLINE_INTERP_BATCH above (do_free=1), so no explicit free here. */

        } /* end of RSD conditional expression */

        TIMER_START(spline_out);
        { /* Id2d2/Id2G2/IG2G2 monopole */
            double *_id0_in[] = {P_Id2d2, P_Id2G2, P_IG2G2};
            double *_id0_out[] = {pk_Id2d2, pk_Id2G2, pk_IG2G2};
            SPLINE_INTERP_BATCH(_id0_in, _id0_out, 3, kminnew, kmaxnew,
                out_tmp_ + large_for_logs_big, epsilon_for_logs + large_for_logs_big, 1);
        }
        TIMER_ADD(spline_out);
        free(Xr2_bias);
        free(Xi2_bias);
        free(Xr2_transfer);
        free(Xi2_transfer);
        free(cmsym2);
        free(cmsym2_transfer);
        free(P_IFG2);
        free(f22_Id2d2);
        free(etam2);
        free(etam2_transfer);
        free(kb2_cache);
        free(kb2_transfer_cache);
        free(sc_cos2);
        free(sc_sin2);
        free(sc_cos2_t);
        free(sc_sin2_t);

    }
    else if ((pnlpt->bias == bias_yes) && (qcdm_qscf_qs_active == _TRUE_)) {
        /* DW: QCDM-QSCF exact-QS two-fluid bias templates.
         * The final combined tracer multipoles are produced in the RSD branch
         * when RSD=Yes. Here we populate the real-space bias slots with useful
         * two-fluid building blocks rather than EdS single-fluid templates:
         *   Id2      -> P_{m r}^{tree}
         *   Id2d2    -> P_{r r}^{tree}
         *   IG2      -> P_{m theta_r}^{tree}
         *   IFG2     -> P_{theta_r theta_r}^{tree}
         * Higher single-fluid EdS loop bias slots are set to log-safe zero
         * because their meanings do not survive the two-fluid basis. */
        if (pnlpt->nonlinear_pt_verbose > 0) {
            printf("DW/QCDM-QSCF: writing exact-QS two-fluid real-space bias building blocks.\n");
        }
        for (index_k = 0; index_k < pnlpt->k_size; index_k++) {
            double kk=pnlpt->k[index_k];
            qcdm_qscf_kernel1_result K1;
            class_call(qcdm_qscf_qs_solve_F1_reduced(pba,tau,kk,&K1,pnlpt->error_message),pnlpt->error_message,pnlpt->error_message);
            double Pl=pk_l[index_k];
            pk_Id2[index_k] = K1.F1_r_red*Pl;
            pk_Id2d2[index_k] = K1.F1_r_red*K1.F1_r_red*Pl + large_for_logs_big;
            pk_IG2[index_k] = K1.G1_r_red*Pl;
            pk_IFG2[index_k] = K1.G1_r_red*K1.G1_r_red*Pl;
            pk_Id2d2_2[index_k] = large_for_logs_big;
            pk_Id2d2_4[index_k] = large_for_logs_big;
            pk_Id2G2[index_k] = large_for_logs_big;
            pk_Id2G2_2[index_k] = large_for_logs_big;
            pk_Id2G2_4[index_k] = large_for_logs_big;
            pk_IG2G2[index_k] = large_for_logs_big;
            pk_IG2G2_2[index_k] = large_for_logs_big;
            pk_IG2G2_4[index_k] = large_for_logs_big;
            pk_IFG2_0b1[index_k] = large_for_logs_big;
            pk_IFG2_0[index_k] = large_for_logs_big;
            pk_IFG2_2[index_k] = large_for_logs_big;

            pk_l_0_b1b2[index_k] = large_for_logs_big;
            pk_l_0_b2[index_k] = large_for_logs_big;
            pk_l_0_b1bG2[index_k] = large_for_logs_big;
            pk_l_0_bG2[index_k] = large_for_logs_big;
            pk_l_2_b1b2[index_k] = large_for_logs_big;
            pk_l_2_b2[index_k] = large_for_logs_big;
            pk_l_2_b1bG2[index_k] = large_for_logs_big;
            pk_l_2_bG2[index_k] = large_for_logs_big;
            pk_l_4_b1b2[index_k] = large_for_logs_big;
            pk_l_4_b2[index_k] = large_for_logs_big;
            pk_l_4_b1bG2[index_k] = large_for_logs_big;
            pk_l_4_bG2[index_k] = large_for_logs_big;

            pk_fNLd2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_fNLG2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_fNLd2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk_fNLG2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;

            pk12_l_0_b1b2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_0_b2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_0_b1bG2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_0_bG2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_2_b1b2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_2_b2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_2_b1bG2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_2_bG2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_4_b1b2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_4_b2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_4_b1bG2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_4_bG2[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_0_b1b2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_0_b2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_0_b1bG2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_0_bG2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_2_b1b2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_2_b2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_2_b1bG2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_2_bG2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_4_b1b2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_4_b2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_4_b1bG2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
            pk12_l_4_bG2_ortho[index_k] = large_for_logs_fNL + epsilon_for_logs_fNL;
        }
    } /* end of bias conditional expression */

    else
    {
        if (pnlpt->nonlinear_pt_verbose > 0)
            printf("No bias tracers requested.\n");

        double epsilon_for_logs = 1.e-6;
        double *_eps_arr[] = {pk_Id2d2, pk_Id2, pk_IG2, pk_Id2G2, pk_IG2G2, pk_IFG2};
        for (int ai_ = 0; ai_ < 6; ai_++)
            for (index_k = 0; index_k < pnlpt->k_size; index_k++)
                _eps_arr[ai_][index_k] = epsilon_for_logs;
    }
    /* --- Free arrays shared between main and bias sections --- */
    free(cmsym_transfer);
    free(Xr_transfer);
    free(Xi_transfer);
    free(lnk_cache);
    free(kb_transfer_cache);
    free(sc_cos_t);
    free(sc_sin_t);

    /* --- Free temporary arrays --- */
    { void *_f[] = {
        js, kdisc, Pbin, Tbin,
        Pw, Pnw, dd_Pnw, dd_Pw,
        Tw, Tnw, dd_Tnw, dd_Tw, P10b1,
        P10, P12,
    };
    for (int fi_ = 0; fi_ < 15; fi_++) free(_f[fi_]); }

    /* Free per-thread workspace and restore BLAS threads */
    for (int _t = 0; _t < _n_threads; _t++) {
        free(ws_Mr[_t]); free(ws_Mi[_t]);
        free(ws_Y1[_t]); free(ws_Y2[_t]);
        free(ws_dot_mr[_t]); free(ws_dot_mi[_t]);
    }
    free(ws_Mr); free(ws_Mi); free(ws_Y1); free(ws_Y2);
    free(ws_dot_mr); free(ws_dot_mi);
/* DW: write timing diagnostics beneath the user-selected CLASS output root. */
    classpt_blas_set_num_threads(_blas_threads_saved);

    clock_gettime(CLOCK_MONOTONIC, &_ts_total1);
    double _t_total = (_ts_total1.tv_sec - _ts_total0.tv_sec)
                    + 1e-9 * (_ts_total1.tv_nsec - _ts_total0.tv_nsec);
    {
/* DW: write timing diagnostics beneath the user-selected CLASS output root. */
        FileName _timing_name;
        class_sprintf(_timing_name,"%sqcdm_pt_timing.log",pnlpt->qcdm_pt_output_root);
        FILE *_tf = fopen(_timing_name, "a");
        if (_tf) {
            fprintf(_tf, "=== nonlinear_pt_loop timing z=%.9g (%.3fs total) ===\n", qcdm_timing_z, _t_total);
            fprintf(_tf, "  BLAS:          %.3fs (%.0f%%)\n", _t_blas, 100.*_t_blas/_t_total);
            fprintf(_tf, "  FFT:           %.3fs (%.0f%%)\n", _t_fft, 100.*_t_fft/_t_total);
            fprintf(_tf, "  build_X:       %.3fs (%.0f%%)\n", _t_buildx, 100.*_t_buildx/_t_total);
            fprintf(_tf, "  IR resum:      %.3fs (%.0f%%)\n", _t_ir, 100.*_t_ir/_t_total);
            fprintf(_tf, "  AP spline:     %.3fs (%.0f%%)\n", _t_ap_spline, 100.*_t_ap_spline/_t_total);
            fprintf(_tf, "  RSD AP loop:   %.3fs (%.0f%%)\n", _t_rsd_ap, 100.*_t_rsd_ap/_t_total);
            fprintf(_tf, "  Bias AP loop:  %.3fs (%.0f%%)\n", _t_bias_ap, 100.*_t_bias_ap/_t_total);
            fprintf(_tf, "  Spline out:    %.3fs (%.0f%%)\n", _t_spline_out, 100.*_t_spline_out/_t_total);
            fprintf(_tf, "  M22 fill:      %.3fs (%.0f%%)\n", _t_m22fill, 100.*_t_m22fill/_t_total);
            fprintf(_tf, "  QCDM real:     %.3fs (%.0f%%)\n", _t_qcdm_real, 100.*_t_qcdm_real/_t_total);
            fprintf(_tf, "  QCDM RSD:      %.3fs (%.0f%%)\n", _t_qcdm_rsd, 100.*_t_qcdm_rsd/_t_total);
            double _t_tagged = _t_blas + _t_fft + _t_buildx + _t_ir + _t_ap_spline
                             + _t_rsd_ap + _t_bias_ap + _t_spline_out + _t_m22fill
                             + _t_qcdm_real + _t_qcdm_rsd;
            double _t_other = _t_total - _t_tagged;
            fprintf(_tf, "  Other:         %.3fs (%.0f%%)\n", _t_other, 100.*_t_other/_t_total);
            fclose(_tf);
        }
    }

    if ((qcdm_qscf_qs_active == _TRUE_) && (pnlpt->nonlinear_pt_verbose > 0))
        qcdm_qscf_qs_cache_print_stats();

    return _SUCCESS_;
}


#define _NLPT_NUM_COMPONENTS_ 96

/**
 * Return pointer to the ln_pk_* array for component index c (0..95).
 * The returned pointer has layout [i_z * k_size + index_k].
 *
 * The ordering matches the parameter order of nonlinear_pt_bias_at_z_i()
 * and nonlinear_pt_pk_at_k_and_z(), i.e. component 0 = P_nl, 1 = Id2d2, etc.
 */
static double * _nlpt_component_array(struct nonlinear_pt *pnlpt, int c) {
    switch (c) {
    /* Gaussian real-space (0-14) */
    case  0: return pnlpt->ln_pk_nl;
    case  1: return pnlpt->ln_pk_Id2d2;
    case  2: return pnlpt->ln_pk_Id2;
    case  3: return pnlpt->ln_pk_IG2;
    case  4: return pnlpt->ln_pk_Id2G2;
    case  5: return pnlpt->ln_pk_IG2G2;
    case  6: return pnlpt->ln_pk_IFG2;
    case  7: return pnlpt->ln_pk_IFG2_0b1;
    case  8: return pnlpt->ln_pk_IFG2_0;
    case  9: return pnlpt->ln_pk_IFG2_2;
    case 10: return pnlpt->ln_pk_CTR;
    case 11: return pnlpt->ln_pk_CTR_0;
    case 12: return pnlpt->ln_pk_CTR_2;
    case 13: return pnlpt->ln_pk_CTR_4;
    case 14: return pnlpt->ln_pk_Tree;
    /* Tree-level RSD multipoles (15-20) */
    case 15: return pnlpt->ln_pk_Tree_0_vv;
    case 16: return pnlpt->ln_pk_Tree_0_vd;
    case 17: return pnlpt->ln_pk_Tree_0_dd;
    case 18: return pnlpt->ln_pk_Tree_2_vv;
    case 19: return pnlpt->ln_pk_Tree_2_vd;
    case 20: return pnlpt->ln_pk_Tree_4_vv;
    /* One-loop RSD multipoles (21-29) */
    case 21: return pnlpt->ln_pk_0_vv;
    case 22: return pnlpt->ln_pk_0_vd;
    case 23: return pnlpt->ln_pk_0_dd;
    case 24: return pnlpt->ln_pk_2_vv;
    case 25: return pnlpt->ln_pk_2_vd;
    case 26: return pnlpt->ln_pk_2_dd;
    case 27: return pnlpt->ln_pk_4_vv;
    case 28: return pnlpt->ln_pk_4_vd;
    case 29: return pnlpt->ln_pk_4_dd;
    /* Bias multipoles (30-47) */
    case 30: return pnlpt->ln_pk_0_b1b2;
    case 31: return pnlpt->ln_pk_0_b2;
    case 32: return pnlpt->ln_pk_0_b1bG2;
    case 33: return pnlpt->ln_pk_0_bG2;
    case 34: return pnlpt->ln_pk_2_b1b2;
    case 35: return pnlpt->ln_pk_2_b2;
    case 36: return pnlpt->ln_pk_2_b1bG2;
    case 37: return pnlpt->ln_pk_2_bG2;
    case 38: return pnlpt->ln_pk_4_b2;
    case 39: return pnlpt->ln_pk_4_bG2;
    case 40: return pnlpt->ln_pk_4_b1b2;
    case 41: return pnlpt->ln_pk_4_b1bG2;
    case 42: return pnlpt->ln_pk_Id2d2_2;
    case 43: return pnlpt->ln_pk_Id2G2_2;
    case 44: return pnlpt->ln_pk_IG2G2_2;
    case 45: return pnlpt->ln_pk_Id2d2_4;
    case 46: return pnlpt->ln_pk_Id2G2_4;
    case 47: return pnlpt->ln_pk_IG2G2_4;
    /* fNL equilateral (48-71) */
    case 48: return pnlpt->ln_pk_nl_fNL;
    case 49: return pnlpt->ln_pk_fNLd2;
    case 50: return pnlpt->ln_pk_fNLG2;
    case 51: return pnlpt->ln_pk_fNL_0_vv;
    case 52: return pnlpt->ln_pk_fNL_0_vd;
    case 53: return pnlpt->ln_pk_fNL_0_dd;
    case 54: return pnlpt->ln_pk_fNL_2_vv;
    case 55: return pnlpt->ln_pk_fNL_2_vd;
    case 56: return pnlpt->ln_pk_fNL_2_dd;
    case 57: return pnlpt->ln_pk_fNL_4_vv;
    case 58: return pnlpt->ln_pk_fNL_4_vd;
    case 59: return pnlpt->ln_pk_fNL_4_dd;
    case 60: return pnlpt->ln_pk12_0_b1b2;
    case 61: return pnlpt->ln_pk12_0_b2;
    case 62: return pnlpt->ln_pk12_0_b1bG2;
    case 63: return pnlpt->ln_pk12_0_bG2;
    case 64: return pnlpt->ln_pk12_2_b1b2;
    case 65: return pnlpt->ln_pk12_2_b2;
    case 66: return pnlpt->ln_pk12_2_b1bG2;
    case 67: return pnlpt->ln_pk12_2_bG2;
    case 68: return pnlpt->ln_pk12_4_b1b2;
    case 69: return pnlpt->ln_pk12_4_b2;
    case 70: return pnlpt->ln_pk12_4_b1bG2;
    case 71: return pnlpt->ln_pk12_4_bG2;
    /* fNL orthogonal (72-95) */
    case 72: return pnlpt->ln_pk_nl_fNL_ortho;
    case 73: return pnlpt->ln_pk_fNLd2_ortho;
    case 74: return pnlpt->ln_pk_fNLG2_ortho;
    case 75: return pnlpt->ln_pk_fNL_0_vv_ortho;
    case 76: return pnlpt->ln_pk_fNL_0_vd_ortho;
    case 77: return pnlpt->ln_pk_fNL_0_dd_ortho;
    case 78: return pnlpt->ln_pk_fNL_2_vv_ortho;
    case 79: return pnlpt->ln_pk_fNL_2_vd_ortho;
    case 80: return pnlpt->ln_pk_fNL_2_dd_ortho;
    case 81: return pnlpt->ln_pk_fNL_4_vv_ortho;
    case 82: return pnlpt->ln_pk_fNL_4_vd_ortho;
    case 83: return pnlpt->ln_pk_fNL_4_dd_ortho;
    case 84: return pnlpt->ln_pk12_0_b1b2_ortho;
    case 85: return pnlpt->ln_pk12_0_b2_ortho;
    case 86: return pnlpt->ln_pk12_0_b1bG2_ortho;
    case 87: return pnlpt->ln_pk12_0_bG2_ortho;
    case 88: return pnlpt->ln_pk12_2_b1b2_ortho;
    case 89: return pnlpt->ln_pk12_2_b2_ortho;
    case 90: return pnlpt->ln_pk12_2_b1bG2_ortho;
    case 91: return pnlpt->ln_pk12_2_bG2_ortho;
    case 92: return pnlpt->ln_pk12_4_b1b2_ortho;
    case 93: return pnlpt->ln_pk12_4_b2_ortho;
    case 94: return pnlpt->ln_pk12_4_b1bG2_ortho;
    case 95: return pnlpt->ln_pk12_4_bG2_ortho;
    default: return NULL;
    }
}

/**
 * Pack 96 output pointers into an array matching the component ordering
 * of _nlpt_component_array(). This macro is used in bias_at_z_i and
 * pk_at_k_and_z to avoid repeating the same 96-element initializer.
 *
 * Arguments: P = prefix of variable names (e.g. output_tot or pk_tot).
 * The caller must have local variables named P, P_Id2d2, P_Id2, etc.
 */
#define _NLPT_PACK_OUTPUTS_(P) {                                            \
    P, P##_Id2d2, P##_Id2, P##_IG2, P##_Id2G2, P##_IG2G2,                 \
    P##_IFG2, P##_IFG2_0b1, P##_IFG2_0, P##_IFG2_2,                      \
    P##_CTR, P##_CTR_0, P##_CTR_2, P##_CTR_4, P##_Tree,                   \
    P##_Tree_0_vv, P##_Tree_0_vd, P##_Tree_0_dd,                          \
    P##_Tree_2_vv, P##_Tree_2_vd, P##_Tree_4_vv,                          \
    P##_0_vv, P##_0_vd, P##_0_dd,                                         \
    P##_2_vv, P##_2_vd, P##_2_dd,                                         \
    P##_4_vv, P##_4_vd, P##_4_dd,                                         \
    P##_0_b1b2, P##_0_b2, P##_0_b1bG2, P##_0_bG2,                        \
    P##_2_b1b2, P##_2_b2, P##_2_b1bG2, P##_2_bG2,                        \
    P##_4_b2, P##_4_bG2, P##_4_b1b2, P##_4_b1bG2,                        \
    P##_2_b2b2, P##_2_b2bG2, P##_2_bG2bG2,                               \
    P##_4_b2b2, P##_4_b2bG2, P##_4_bG2bG2,                               \
    P##_pk_nl_fNL, P##_pk_fNLd2, P##_pk_fNLG2,                           \
    P##_pk_l_fNL_0_vv, P##_pk_l_fNL_0_vd, P##_pk_l_fNL_0_dd,            \
    P##_pk_l_fNL_2_vv, P##_pk_l_fNL_2_vd, P##_pk_l_fNL_2_dd,            \
    P##_pk_l_fNL_4_vv, P##_pk_l_fNL_4_vd, P##_pk_l_fNL_4_dd,            \
    P##_pk12_l_0_b1b2, P##_pk12_l_0_b2,                                   \
    P##_pk12_l_0_b1bG2, P##_pk12_l_0_bG2,                                 \
    P##_pk12_l_2_b1b2, P##_pk12_l_2_b2,                                   \
    P##_pk12_l_2_b1bG2, P##_pk12_l_2_bG2,                                 \
    P##_pk12_l_4_b1b2, P##_pk12_l_4_b2,                                   \
    P##_pk12_l_4_b1bG2, P##_pk12_l_4_bG2,                                 \
    P##_pk_nl_fNL_ortho, P##_pk_fNLd2_ortho, P##_pk_fNLG2_ortho,         \
    P##_pk_l_fNL_0_vv_ortho, P##_pk_l_fNL_0_vd_ortho,                    \
    P##_pk_l_fNL_0_dd_ortho,                                               \
    P##_pk_l_fNL_2_vv_ortho, P##_pk_l_fNL_2_vd_ortho,                    \
    P##_pk_l_fNL_2_dd_ortho,                                               \
    P##_pk_l_fNL_4_vv_ortho, P##_pk_l_fNL_4_vd_ortho,                    \
    P##_pk_l_fNL_4_dd_ortho,                                               \
    P##_pk12_l_0_b1b2_ortho, P##_pk12_l_0_b2_ortho,                       \
    P##_pk12_l_0_b1bG2_ortho, P##_pk12_l_0_bG2_ortho,                     \
    P##_pk12_l_2_b1b2_ortho, P##_pk12_l_2_b2_ortho,                       \
    P##_pk12_l_2_b1bG2_ortho, P##_pk12_l_2_bG2_ortho,                     \
    P##_pk12_l_4_b1b2_ortho, P##_pk12_l_4_b2_ortho,                       \
    P##_pk12_l_4_b1bG2_ortho, P##_pk12_l_4_bG2_ortho                      \
}


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
) {

    int index_k, c;
    int nk = pnlpt->ln_k_size;
    int offset = i_z * pnlpt->k_size;
    double * outputs[_NLPT_NUM_COMPONENTS_] = _NLPT_PACK_OUTPUTS_(output_tot);

    for (c = 0; c < _NLPT_NUM_COMPONENTS_; c++) {
        double * src = _nlpt_component_array(pnlpt, c);
        for (index_k = 0; index_k < nk; index_k++)
            outputs[c][index_k] = src[offset + index_k];
    }

    if (mode == linear) {
        for (c = 0; c < _NLPT_NUM_COMPONENTS_; c++)
            for (index_k = 0; index_k < nk; index_k++)
                outputs[c][index_k] = exp(outputs[c][index_k]);
    }

    return _SUCCESS_;
}


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
) {

    int c;
    double pk_buf[_NLPT_NUM_COMPONENTS_];

    class_test((k < exp(pnlpt->ln_k[0])) || (k > exp(pnlpt->ln_k[pnlpt->ln_k_size-1])),
               pnlpt->error_message,
               "k=%e out of bounds [%e:%e]",
               k, 0., exp(pnlpt->ln_k[pnlpt->ln_k_size-1]));

    class_call(nonlinear_pt_pk_mult_at_kvec_and_z(pnlpt, &k, 1, z, pk_buf),
               pnlpt->error_message, pnlpt->error_message);

    /* The macro cannot be used here because pk_at_k_and_z uses a different
     * naming convention for fNL parameters (pk_tot_fNL_* vs pk_tot_pk_*).
     * We must list explicitly. */
    double * outputs[_NLPT_NUM_COMPONENTS_] = {
        pk_tot, pk_tot_Id2d2, pk_tot_Id2, pk_tot_IG2,
        pk_tot_Id2G2, pk_tot_IG2G2, pk_tot_IFG2,
        pk_tot_IFG2_0b1, pk_tot_IFG2_0, pk_tot_IFG2_2,
        pk_tot_CTR, pk_tot_CTR_0, pk_tot_CTR_2, pk_tot_CTR_4,
        pk_tot_Tree,
        pk_tot_Tree_0_vv, pk_tot_Tree_0_vd, pk_tot_Tree_0_dd,
        pk_tot_Tree_2_vv, pk_tot_Tree_2_vd, pk_tot_Tree_4_vv,
        pk_tot_0_vv, pk_tot_0_vd, pk_tot_0_dd,
        pk_tot_2_vv, pk_tot_2_vd, pk_tot_2_dd,
        pk_tot_4_vv, pk_tot_4_vd, pk_tot_4_dd,
        pk_tot_0_b1b2, pk_tot_0_b2, pk_tot_0_b1bG2, pk_tot_0_bG2,
        pk_tot_2_b1b2, pk_tot_2_b2, pk_tot_2_b1bG2, pk_tot_2_bG2,
        pk_tot_4_b2, pk_tot_4_bG2, pk_tot_4_b1b2, pk_tot_4_b1bG2,
        pk_tot_2_b2b2, pk_tot_2_b2bG2, pk_tot_2_bG2bG2,
        pk_tot_4_b2b2, pk_tot_4_b2bG2, pk_tot_4_bG2bG2,
        pk_tot_fNL, pk_tot_fNLd2, pk_tot_fNLG2,
        pk_tot_fNL_0_vv, pk_tot_fNL_0_vd, pk_tot_fNL_0_dd,
        pk_tot_fNL_2_vv, pk_tot_fNL_2_vd, pk_tot_fNL_2_dd,
        pk_tot_fNL_4_vv, pk_tot_fNL_4_vd, pk_tot_fNL_4_dd,
        pk_tot_fNL_0_b1b2, pk_tot_fNL_0_b2,
        pk_tot_fNL_0_b1bG2, pk_tot_fNL_0_bG2,
        pk_tot_fNL_2_b1b2, pk_tot_fNL_2_b2,
        pk_tot_fNL_2_b1bG2, pk_tot_fNL_2_bG2,
        pk_tot_fNL_4_b1b2, pk_tot_fNL_4_b2,
        pk_tot_fNL_4_b1bG2, pk_tot_fNL_4_bG2,
        pk_tot_fNL_ortho, pk_tot_fNLd2_ortho, pk_tot_fNLG2_ortho,
        pk_tot_fNL_0_vv_ortho, pk_tot_fNL_0_vd_ortho,
        pk_tot_fNL_0_dd_ortho,
        pk_tot_fNL_2_vv_ortho, pk_tot_fNL_2_vd_ortho,
        pk_tot_fNL_2_dd_ortho,
        pk_tot_fNL_4_vv_ortho, pk_tot_fNL_4_vd_ortho,
        pk_tot_fNL_4_dd_ortho,
        pk_tot_fNL_0_b1b2_ortho, pk_tot_fNL_0_b2_ortho,
        pk_tot_fNL_0_b1bG2_ortho, pk_tot_fNL_0_bG2_ortho,
        pk_tot_fNL_2_b1b2_ortho, pk_tot_fNL_2_b2_ortho,
        pk_tot_fNL_2_b1bG2_ortho, pk_tot_fNL_2_bG2_ortho,
        pk_tot_fNL_4_b1b2_ortho, pk_tot_fNL_4_b2_ortho,
        pk_tot_fNL_4_b1bG2_ortho, pk_tot_fNL_4_bG2_ortho
    };

    for (c = 0; c < _NLPT_NUM_COMPONENTS_; c++)
        *outputs[c] = pk_buf[c];

    return _SUCCESS_;
}


int nonlinear_pt_pk_mult_at_kvec_and_z(
    struct nonlinear_pt * pnlpt,
    double * kvec,
    int n_k,
    double z,
    double * pk_mult
) {
    int i_z, c, ik, nk;
    int last_index;
    int offset;
    double val;
    double * data = NULL;
    double * spline_c = NULL;
    double * ln_kvec = NULL;

    nk = pnlpt->ln_k_size;

    for (ik = 0; ik < n_k; ik++) {
        class_test((kvec[ik] < exp(pnlpt->ln_k[0])) || (kvec[ik] > exp(pnlpt->ln_k[nk-1])),
                   pnlpt->error_message,
                   "nonlinear_pt_pk_mult_at_kvec_and_z: k=%e out of bounds [%e:%e]",
                   kvec[ik], 0., exp(pnlpt->ln_k[nk-1]));
    }

    for (i_z = 0; i_z < pnlpt->z_pk_num; i_z++)
        if (pnlpt->z_pk[i_z] == z) break;
    if (i_z == pnlpt->z_pk_num) {
        sprintf(pnlpt->error_message,
                "nonlinear_pt_pk_mult_at_kvec_and_z: z=%e not in z_pk list", z);
        return _FAILURE_;
    }

    offset = i_z * pnlpt->k_size;

    class_alloc(ln_kvec, n_k * sizeof(double), pnlpt->error_message);
    for (ik = 0; ik < n_k; ik++)
        ln_kvec[ik] = log(kvec[ik]);

    class_alloc(data, nk * sizeof(double), pnlpt->error_message);
    class_alloc(spline_c, nk * sizeof(double), pnlpt->error_message);

    for (c = 0; c < _NLPT_NUM_COMPONENTS_; c++) {
        double * src = _nlpt_component_array(pnlpt, c);
        for (ik = 0; ik < nk; ik++)
            data[ik] = src[offset + ik];

        class_call(array_spline_table_lines(
                       pnlpt->ln_k, nk, data, 1, spline_c,
                       _SPLINE_NATURAL_, pnlpt->error_message),
                   pnlpt->error_message, pnlpt->error_message);

        last_index = 0;
        for (ik = 0; ik < n_k; ik++) {
            class_call(array_interpolate_spline(
                           pnlpt->ln_k, nk, data, spline_c, 1,
                           ln_kvec[ik], &last_index, &val, 1,
                           pnlpt->error_message),
                       pnlpt->error_message, pnlpt->error_message);
            pk_mult[c * n_k + ik] = exp(val);
        }
    }

    free(data);
    free(spline_c);
    free(ln_kvec);

    return _SUCCESS_;
}
