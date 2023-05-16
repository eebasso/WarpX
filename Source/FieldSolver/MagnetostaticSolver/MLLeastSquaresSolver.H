#include <AMReX_Config.H>
#include <AMReX_MLLinOp.H>
#include "MLLinOpMFSet.H"

namespace Solver {

using MFSet = Solver::MultiFabSet;
using RT = amrex::Real;

class MLLeastSquaresSolver
{
    
public:

    /**
     * \brief Choose solver method for least squares problem.
     * 
     */
    enum struct SolverMethod { 
        BiCGStab /*Biconjugate gradient stabilized*/ ,
        CGLS /* Conjugate gradient least squares */  ,
        LSQR /* Least squares QR decomposition */     
    };

    MLLeastSquaresSolver (Solver::MLLinOpMFV& _lp, SolverMethod _typ = SolverMethod::BiCGStab) 
            : Lp(_lp), solver_type(_typ), amrlev(0), mglev(_lp.NMGLevels(0)-1) {}

    ~MLLeastSquaresSolver () {}

    MLLeastSquaresSolver (const MLLeastSquaresSolver& rhs) = delete;
    MLLeastSquaresSolver& operator= (const MLLeastSquaresSolver& rhs) = delete;

    void setSolver (SolverMethod _typ) noexcept { solver_type = _typ; }

    void setVerbose (int _verbose) { verbose = _verbose; }
    int getVerbose () const { return verbose; }

    void setMaxIter (int _maxiter) { maxiter = _maxiter; }
    int getMaxIter () const { return maxiter; }

    void setNGhost(int _nghost) {nghost = amrex::IntVect(_nghost);}
    int getNGhost() {return nghost[0];}

    RT dotxy (const MFSet r, const MFSet z, bool local = false);

    RT norm_inf (const MFSet res, bool local = false);

    /**
     * \brief Solve the system, Lp(sol)=rhs to relative err, tolerance
     * 
     * \param sol Solution at level
     * \param rhs The right hand side (RHS) a.k.a. the inhomogeneous part of the linear differential equation.
     * \param eps_rel Relative error
     * \param eps_abs Absolute tolerance
     * \return Returns an int indicating success or failure.
     * 0 means success.
     * 1 means failed for loss of precision.
     * 2 means iterations exceeded.
     */
    int solve (MFSet& sol, const MFSet& rhs, RT eps_rel, RT eps_abs) {
        switch (solver_type) {
            case SolverMethod::BiCGStab: return solve_BiCGStab(sol,rhs,eps_rel,eps_abs);
            case SolverMethod::CGLS:     return solve_CGLS(sol,rhs,eps_rel,eps_abs);
            case SolverMethod::LSQR:     return solve_LSQR(sol,rhs,eps_rel,eps_abs);
            default:                     return 0;
        }
    }

    /**
     * \brief Solve L(sol)=rhs with biconjugate gradient stabilized method
     * 
     * \return Returns an int indicating success or failure.
     * 0 means success.
     * 1 means failed for loss of precision.
     * 2 means iterations exceeded.
     */
    int solve_BiCGStab (MFSet& sol, const MFSet& rhs, RT eps_rel, RT eps_abs);

    /**
     * \brief Solve L(sol)=rhs with conjugate gradient least squares method.
     *  
     * \return Returns an int indicating success or failure.
     * 0 means success.
     * 1 means failed for loss of precision.
     * 2 means iterations exceeded.
     */
    int solve_CGLS (MFSet& sol, const MFSet& rhs, RT eps_rel, RT eps_abs);

    /**
     * \brief Solve L(sol)=rhs with least squares QR decomposition method.
     * 
    * \return Returns an int indicating success or failure.
     * 0 means success.
     * 1 means failed for loss of precision.
     * 2 means iterations exceeded.
     */
    int solve_LSQR (MFSet sol, const MFSet rhs, RT eps_rel, RT eps_abs);

    /**
     * \brief Get the number of iterations
     */
    int getNumIters () const noexcept { return iter; }

private:

    Solver::MLLinOpMFV& Lp;
    SolverMethod solver_type;
    const int amrlev; // AMR level that solver is being called at
    const int mglev; // Multi-grid that solver is being called at
    int verbose   = 0;
    int maxiter   = 100;
    amrex::IntVect nghost = amrex::IntVect(0);
    int iter = -1;

};


}