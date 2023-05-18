#include "MLLeastSquaresSolver.H"

namespace Solver {

RT
MLLeastSquaresSolver::dotxy (const MFSet& r, const MFSet& z, bool local)
{
    BL_PROFILE_VAR_NS("MLLeastSquaresSolver::ParallelAllReduce", blp_par);
    if (!local) { 
        BL_PROFILE_VAR_START(blp_par);
    }
    RT result = Lp.xdoty(amrlev, mglev, r, z, local);
    if (!local) { 
        BL_PROFILE_VAR_STOP(blp_par); 
    }
    return result;
}

RT
MLLeastSquaresSolver::norm_inf (const MFSet& res, bool local)
{
    // int ncomp = res.nComp();
    RT result = res.norminf();
    for (const amrex::MultiFab& mf : res) {

    }

    if (!local) {
        BL_PROFILE("MLLeastSquaresSolver::ParallelAllReduce");
        amrex::ParallelAllReduce::Max(result, Lp.BottomCommunicator());
    }
    return result;
}

int
MLLeastSquaresSolver::solve_BiCGStab (MFSet& sol, const MFSet& rhs, RT eps_rel, RT eps_abs) {
    BL_PROFILE("MLLeastSquaresSolver::bicgstab");

    // amrex::MultiFab& sol_test;
    // const int ncomp_in = sol_test.nComp(); // Number of components for input
    // const amrex::BoxArray& ba_sol = sol_test.boxArray();
    // const amrex::DistributionMapping& dm_sol = sol_test.DistributionMap();
    // const auto& factory_sol = sol_test.Factory();

    // amrex::MultiFab(ba_sol, dm_sol, ncomp_in, nghost, amrex::MFInfo(), factory_sol); // Input size

    amrex::MFInfo mfinfo = amrex::MFInfo();

    MFSet ph(sol, mfinfo, nghost); // Input size
    MFSet sh(sol, mfinfo, nghost); // Input size

    ph.setVal(RT(0.0));
    sh.setVal(RT(0.0));

    MFSet sorig(sol, mfinfo, nghost); // Input size
    MFSet p    (rhs, mfinfo, nghost); // Output size
    MFSet r    (rhs, mfinfo, nghost); // Output size, residual
    MFSet s    (sol, mfinfo, nghost); //
    MFSet rh   (rhs, mfinfo, nghost); // Output size
    MFSet v    (rhs, mfinfo, nghost); // Output size
    MFSet t    (rhs, mfinfo, nghost); // Output size

    // This suggests that r and rhs are the same size, i.e., output size
    Lp.correctionResidual(amrlev, mglev, r, sol, rhs, BCMode::Homogeneous);

    // Then normalize
    Lp.normalize(amrlev, mglev, r);

    // This suggests sorig is the same size as sol, i.e., input size
    sorig.LocalCopy(sol,0,0,ncomp_in,nghost);
    // This suggests rh is the same size as r, i.e., output size
    rh.LocalCopy   (r  ,0,0,ncomp_in,nghost);

    sol.setVal(RT(0.0));

    RT rnorm = norm_inf(r);
    const RT rnorm0 = rnorm;

    if ( verbose > 0 )
    {
        amrex::Print() << "MLLeastSquaresSolver_BiCGStab: Initial error (error0) =        " << rnorm0 << '\n';
    }
    int ret = 0;
    iter = 1;
    RT rho_1 = 0, alpha = 0, omega = 0;

    if ( rnorm0 == 0 || rnorm0 < eps_abs )
    {
        if ( verbose > 0 )
        {
            amrex::Print() << "MLLeastSquaresSolver_BiCGStab: niter = 0,"
                        << ", rnorm = " << rnorm
                        << ", eps_abs = " << eps_abs << std::endl;
        }
        return ret;
    }

    for (; iter <= maxiter; ++iter)
    {
        const RT rho = dotxy(rh,r);
        if ( rho == 0 )
        {
            ret = 1; break;
        }
        if ( iter == 1 )
        {
            // This suggests p is the same size as r, i.e., output size
            p.LocalCopy(r,0,0,ncomp_in,nghost);
        }
        else
        {
            const RT beta = (rho/rho_1)*(alpha/omega);
            // This suggests v is the same size as p, i.e., output size
            MFSet::Saxpy(p, -omega, v, 0, 0, ncomp_in, nghost); // p += -omega*v
            MFSet::Xpay(p, beta, r, 0, 0, ncomp_in, nghost); // p = r + beta*p
        }
        ph.LocalCopy(p,0,0,ncomp_in,nghost);

        // This line indicates that v is output size and ph is input size
        Lp.apply(amrlev, mglev, v, ph, amrex::MLLinOpT<MFSet>::BCMode::Homogeneous, amrex::MLLinOpT<MFSet>::StateMode::Correction);
        Lp.normalize(amrlev, mglev, v);

        RT rhTv = dotxy(rh,v);
        if ( rhTv != RT(0.0) )
        {
            alpha = rho/rhTv;
        }
        else
        {
            ret = 2; break;
        }
        MFSet::Saxpy(sol, alpha, ph, 0, 0, ncomp_in, nghost); // sol += alpha * ph
        MFSet::LinComb(s, RT(1.0), r, 0, -alpha, v, 0, 0, ncomp_in, nghost); // s = r - alpha * v

        rnorm = norm_inf(s);

        if ( verbose > 2 && amrex::ParallelDescriptor::IOProcessor() )
        {
            amrex::Print() << "MLLeastSquaresSolver_BiCGStab: Half Iter "
                        << std::setw(11) << iter
                        << " rel. err. "
                        << rnorm/(rnorm0) << '\n';
        }

        if ( rnorm < eps_rel*rnorm0 || rnorm < eps_abs ) break;

        // This suggests s and sh are the same size
        sh.LocalCopy(s,0,0,ncomp_in,nghost);
        // This suggests that t is output size and sh is input size
        Lp.apply(amrlev, mglev, t, sh, MLLinOpT<MFSet>::BCMode::Homogeneous, MLLinOpT<MFSet>::StateMode::Correction);
        Lp.normalize(amrlev, mglev, t);
        //
        // This is a little funky.  I want to elide one of the reductions
        // in the following two dotxy()s.  We do that by calculating the "local"
        // values and then reducing the two local values at the same time.
        //
        RT tvals[2] = { dotxy(t,t,true), dotxy(t,s,true) };

        BL_PROFILE_VAR("MLLeastSquaresSolver::ParallelAllReduce", blp_par);
        amrex::ParallelAllReduce::Sum(tvals,2,Lp.BottomCommunicator());
        BL_PROFILE_VAR_STOP(blp_par);

        if ( tvals[0] != RT(0.0) )
        {
            omega = tvals[1]/tvals[0];
        }
        else
        {
            ret = 3; break;
        }
        MFSet::Saxpy(sol, omega, sh, 0, 0, ncomp_in, nghost); // sol += omega * sh
        // This suggests r, s, and t share the same size, i.e., output size
        MFSet::LinComb(r, RT(1.0), s, 0, -omega, t, 0, 0, ncomp_in, nghost); // r = s - omega * t

        rnorm = norm_inf(r);

        if ( verbose > 2 )
        {
            amrex::Print() << "MLLeastSquaresSolver_BiCGStab: Iteration "
                        << std::setw(11) << iter
                        << " rel. err. "
                        << rnorm/(rnorm0) << '\n';
        }

        if ( rnorm < eps_rel*rnorm0 || rnorm < eps_abs ) break;

        if ( omega == 0 )
        {
            ret = 4; break;
        }
        rho_1 = rho;
    }

    if ( verbose > 0 )
    {
        amrex::Print() << "MLLeastSquaresSolver_BiCGStab: Final: Iteration "
                    << std::setw(4) << iter
                    << " rel. err. "
                    << rnorm/(rnorm0) << '\n';
    }

    if ( ret == 0 && rnorm > eps_rel*rnorm0 && rnorm > eps_abs)
    {
        if ( verbose > 0 && amrex::ParallelDescriptor::IOProcessor() )
            amrex::Warning("MLLeastSquaresSolver_BiCGStab:: failed to converge!");
        ret = 8;
    }

    if ( ( ret == 0 || ret == 8 ) && (rnorm < rnorm0) )
    {
        sol.LocalAdd(sorig, 0, 0, ncomp_in, nghost);
    }
    else
    {
        sol.setVal(RT(0.0));
        sol.LocalAdd(sorig, 0, 0, ncomp_in, nghost);
    }

    return ret;
}

int
MLLeastSquaresSolver::solve_CGLS (MFSet& sol, const MFSet& rhs, RT eps_rel, RT eps_abs)
{
    BL_PROFILE("MLLeastSquaresSolver::cg");

    amrex::MFInfo mfinfo = amrex::MFInfo();

    MFSet p = MFSet(sol, mfinfo);

    MFSet sorig (sol, mfinfo, nghost);
    MFSet r     (sol, mfinfo, nghost);
    MFSet LTr   (sol, mfinfo, nghost);
    MFSet q     (sol, mfinfo, nghost);

    p.setVal(RT(0.0));
    sorig.LocalCopy(sol, nghost);
    Lp.correctionResidual(amrlev, mglev, r, sol, rhs, BCMode::Homogeneous);
    sol.setVal(RT(0.0));

    RT       rnorm    = norm_inf(r);
    const RT rnorm0   = rnorm;

    if ( verbose > 0 ) {
        amrex::Print() << "MLLeastSquaresSolver_CG: Initial error (error0) :        " << rnorm0 << '\n';
    }

    RT rho_1 = 0;
    int  ret = 0;
    iter = 1;

    if ( rnorm0 == 0 || rnorm0 < eps_abs )
    {
        if ( verbose > 0 ) {
            amrex::Print() << "MLLeastSquaresSolver_CG: niter = 0,"
                           << ", rnorm = " << rnorm
                           << ", eps_abs = " << eps_abs << std::endl;
        }
        return ret;
    }

    for (; iter <= maxiter; ++iter)
    {
        RT rho = dotxy(r,r);

        if ( rho == 0 )
        {
            ret = 1; break;
        }
        if (iter == 1)
        {
            p.LocalCopy(r, nghost);
        }
        else
        {
            RT beta = rho/rho_1;
            MFSet::Xpay(p, beta, p, 0, 0, ncomp, nghost); // p = z + beta * p
        }
        Lp.apply(amrlev, mglev, q, p, MLLinOpT<MFSet>::BCMode::Homogeneous, MLLinOpT<MFSet>::StateMode::Correction);

        RT alpha;
        RT pw = dotxy(p,q);
        if ( pw != RT(0.0))
        {
            alpha = rho/pw;
        }
        else
        {
            ret = 1; break;
        }

        if ( verbose > 2 )
        {
            amrex::Print() << "MLLeastSquaresSolver_cg:"
                           << " iter " << iter
                           << " rho " << rho
                           << " alpha " << alpha << '\n';
        }
        MFSet::Saxpy(sol, alpha, p, 0, 0, ncomp, nghost); // sol += alpha * p
        MFSet::Saxpy(r, -alpha, q, 0, 0, ncomp, nghost); // r += -alpha * q
        rnorm = norm_inf(r);

        if ( verbose > 2 )
        {
            amrex::Print() << "MLLeastSquaresSolver_cg:       Iteration"
                           << std::setw(4) << iter
                           << " rel. err. "
                           << rnorm/(rnorm0) << '\n';
        }

        if ( rnorm < eps_rel*rnorm0 || rnorm < eps_abs ) {
            break;
        }

        rho_1 = rho;
    }

    if ( verbose > 0 )
    {
        amrex::Print() << "MLLeastSquaresSolver_cg: Final Iteration"
                       << std::setw(4) << iter
                       << " rel. err. "
                       << rnorm/(rnorm0) << '\n';
    }

    if ( ret == 0 && rnorm > eps_rel*rnorm0 && rnorm > eps_abs ) {
        if ( verbose > 0 && amrex::ParallelDescriptor::IOProcessor() ) {
            amrex::Warning("MLLeastSquaresSolver_cg: failed to converge!");
        }
        ret = 8;
    }

    if ( ( ret == 0 || ret == 8 ) && (rnorm < rnorm0) ) {
        sol.LocalAdd(sorig, nghost);
    }
    else {
        sol.setVal(RT(0.0));
        sol.LocalAdd(sorig, nghost);
    }

    return ret;
}








}