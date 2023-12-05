#ifndef WARPX_INTERPOLATE_H_
#define WARPX_INTERPOLATE_H_

#include <AMReX_REAL.H>

#include <AMReX_BaseFwd.H>

#include <array>
#include <memory>

namespace Interpolate
{
    using namespace amrex;

    /** \brief Samples/Interpolates the coarse scalar multifab `F_cp` on the
     * fine grid associated with the fine multifab `F_fp`.
     */
    std::unique_ptr<amrex::MultiFab>
        getInterpolatedScalar(
            const amrex::MultiFab& F_cp, const amrex::MultiFab& F_fp,
            const amrex::DistributionMapping& dm, amrex::IntVect r_ratio,
            const Real* /*dx*/, amrex::IntVect ngrow );

    /** \brief Samples/Interpolates the coarse vector multifab `F*_cp` on the
     * fine grid associated with the fine multifab `F*_fp`.
     */
    std::array<std::unique_ptr<amrex::MultiFab>, 3>
        getInterpolatedVector(
            const amrex::MultiFab* Fx_cp,
            const amrex::MultiFab* Fy_cp,
            const amrex::MultiFab* Fz_cp,
            const amrex::MultiFab* Fx_fp,
            const amrex::MultiFab* Fy_fp,
            const amrex::MultiFab* Fz_fp,
            const amrex::DistributionMapping& dm, amrex::IntVect r_ratio,
            const Real* dx, amrex::IntVect ngrow );

}

#endif // WARPX_INTERPOLATE_H_
