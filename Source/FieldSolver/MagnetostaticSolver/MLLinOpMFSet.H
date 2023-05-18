#include <AMReX_Config.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MLLinOp.H>
#include <MultiFabSet.H>
#include <AMReX_REAL.H>

namespace Solver {

class MLLinOpMFSet
{
public:

    using MF = typename amrex::MultiFab;
    using MFSet = MultiFabSet;
    using FAB = amrex::FArrayBox;
    using RT  = amrex::Real;
    using BCType = amrex::LinOpBCType;
    using BCMode    = amrex::LinOpEnumType::BCMode;
    using StateMode = amrex::LinOpEnumType::StateMode;
    using Location  = amrex::LinOpEnumType::Location;
    using BottomSolver = amrex::BottomSolver;


    friend class MLLeastSquaresSolver;
    friend class MLMGWarpX;

    MLLinOpMFSet () = default;
    virtual ~MLLinOpMFSet () = default;

    MLLinOpMFSet (const MLLinOpMFSet&) = delete;
    MLLinOpMFSet (MLLinOpMFSet&&) = delete;
    MLLinOpMFSet& operator= (const MLLinOpMFSet&) = delete;
    MLLinOpMFSet& operator= (MLLinOpMFSet&&) = delete;

    void define (const amrex::Vector<amrex::Geometry>& a_geom,
                 const amrex::Vector<amrex::BoxArray>& a_grids,
                 const amrex::Vector<amrex::DistributionMapping>& a_dmap,
                 const amrex::LPInfo& a_info,
                 const amrex::Vector<amrex::FabFactory<FAB> const*>& a_factory,
                 bool eb_limit_coarsening = true,
                 int a_nsolcomp = 1,
                 int a_nrhscomp = 1);

    [[nodiscard]] virtual std::string name () const { return std::string("Unspecified"); }

    /**
     * \brief Boundary of the whole domain.
     *
     * This functions must be called, and must be called before other bc
     * functions. This version is for single-component solve or when all the
     * compoents have the same BC types.
     *
     * \param lobc lower boundaries
     * \param hibc upper boundaries
     */
    void setDomainBC (const amrex::Array<BCType,AMREX_SPACEDIM>& lobc,
                      const amrex::Array<BCType,AMREX_SPACEDIM>& hibc) noexcept;

    /**
     * \brief Boundary of the whole domain.
     *
     * This functions must be called, and must be called before other bc
     * functions.
     *
     * \param lobc lower boundaries. The Vector index corresponds to each component of MFSet.
     * \param hibc upper boundaries. The Vector index corresponds to each component of MFSet.
     */
    void setDomainBC (const amrex::Vector<amrex::Array<BCType,AMREX_SPACEDIM> >& lobc,
                      const amrex::Vector<amrex::Array<BCType,AMREX_SPACEDIM> >& hibc) noexcept;

    /**
     * \brief Set location of domain boundaries.
     *
     * By default, domain BC is on the domain face.  If that's the
     * case, this function doesn't need to be called.  However, one
     * could use this function to set non-zero domain BC locations.
     * Note all values should be >= 0.  If this function is called,
     * it MUST be called before setLevelBC.
     */
    void setDomainBCLoc (const amrex::Array<amrex::Real,AMREX_SPACEDIM>& lo_bcloc,
                         const amrex::Array<amrex::Real,AMREX_SPACEDIM>& hi_bcloc) noexcept;

    /**
    * \brief Needs coarse data for bc?
    *
    * If the lowest level grids does not cover the entire domain, coarse
    * level data are needed for supplying Dirichlet bc at coarse/fine
    * boundary, even when the domain bc is not Dirichlet.
    */
    [[nodiscard]] bool needsCoarseDataForBC () const noexcept { return m_needs_coarse_data_for_bc; }

    /**
    * \brief Set coarse/fine boundary conditions. For cell-centered solves
    * only.
    *
    * If we want to do a linear solve where the boundary conditions on the
    * coarsest AMR level of the solve come from a coarser level (e.g. the
    * base AMR level of the solve is > 0 and does not cover the entire
    * domain), we must explicitly provide the coarser data.  Boundary
    * conditions from a coarser level are always Dirichlet.  The MultiFab
    * crse does not need to have ghost cells and is at a coarser resolution
    * than the coarsest AMR level of the solve; it is used to supply
    * (interpolated) boundary conditions for the solve.  NOTE: If this is
    * called, it must be called before `setLevelBC`.  If crse is nullptr,
    * then the bc values are assumed to be zero.
    *
    * \param crse       the coarse AMR level data
    * \param crse_ratio the coarsening ratio between fine and coarse AMR levels.
    */
    void setCoarseFineBC (const MFSet* crse, int crse_ratio) noexcept;

    template <typename AMF, std::enable_if_t<!std::is_same_v<MFSet,AMF>,int> = 0>
    void setCoarseFineBC (const AMF* crse, int crse_ratio) noexcept;

    /**
    * \brief Set boundary conditions for given level. For cell-centered
    * solves only.
    *
    * This must be called for each level.  Argument `levelbcdata` is used to
    * supply Dirichlet or Neumann bc at the physical domain; if those data
    * are homogeneous we can pass nullptr instead of levelbcdata.
    * Regardless, this function must be called.  If used, the MultiFab
    * levelbcdata must have one ghost cell.  Only the data outside the
    * physical domain will be used.  It is assumed that the data in those
    * ghost cells outside the domain live exactly on the face of the
    * physical domain.  Argument `amrlev` is relative level such that the
    * lowest to the solver is always 0.  The optional arguments
    * robinbc_[a|b|f] provide Robin boundary condition `a*phi + b*dphi/dn =
    * f`.  Note that `d./dn` is `d./dx` at the upper boundary and `-d./dx`
    * at the lower boundary, for Robin BC. However, for inhomogeneous
    * Neumann BC, the value in leveldata is assumed to be `d./dx`.
    */
    virtual void setLevelBC (int /*amrlev*/, const MFSet* /*levelbcdata*/,
                             const MFSet* /*robinbc_a*/ = nullptr,
                             const MFSet* /*robinbc_b*/ = nullptr,
                             const MFSet* /*robinbc_f*/ = nullptr) = 0;

    template <typename AMF, std::enable_if_t<!std::is_same_v<MFSet,AMF>,int> = 0>
    void setLevelBC (int amrlev, const AMF* levelbcdata,
                     const AMF* robinbc_a = nullptr,
                     const AMF* robinbc_b = nullptr,
                     const AMF* robinbc_f = nullptr);

    //! Set verbosity
    void setVerbose (int v) noexcept { verbose = v; }

    //! Set order of interpolation at coarse/fine boundary
    void setMaxOrder (int o) noexcept { maxorder = o; }
    //! Get order of interpolation at coarse/fine boundary
    [[nodiscard]] int getMaxOrder () const noexcept { return maxorder; }

    //! Set the flag for whether the solver should try to make singular
    //! problem solvable, which is on by default.
    void setEnforceSingularSolvable (bool o) noexcept { enforceSingularSolvable = o; }

    //! Get the flag for whether the solver should try to make singular
    //! problem solvable.
    [[nodiscard]] bool getEnforceSingularSolvable () const noexcept { return enforceSingularSolvable; }

    [[nodiscard]] virtual int getNGrow (int /*a_lev*/ = 0, int /*mg_lev*/ = 0) const { return 0; }

    //! Return the number of MultiFabs in the MultiFabSet solution. This is distinct from the number of components in any particular MultiFab, which is usually 1 in this context.
    int getNCompSol () const { return m_ncomp_sol; }

    //! Return the number of Multifabs in the MultiFabSet RHS. This is distinct from the number of components in any particular MultiFab, which is usually 1 in this context.
    int getNCompRHS () const { return m_ncomp_rhs; }

    //! Does it need update if it's reused?
    [[nodiscard]] virtual bool needsUpdate () const { return false; }

    //! Update for reuse.
    virtual void update () {}

    /**
     * \brief Restriction onto coarse MG level
     *
     * \param amrlev AMR level
     * \param cmglev coarse MG level
     * \param crse   coarse data. This is the output.
     * \param fine   fine data. This is the input. Some operators might need to fill ghost cells.
     *               This is why it's not a const reference.
     */
    virtual void restriction (int amrlev, int cmglev, MFSet& crse, MFSet& fine) const = 0;

    /**
     * \brief Add interpolated coarse MG level data to fine MG level data
     *
     * \param amrlev AMR level
     * \param fmglev fine MG level
     * \param crse   coarse data.
     * \param fine   fine data.
     */
    virtual void interpolation (int amrlev, int fmglev, MFSet& fine, const MFSet& crse) const = 0;

    /**
     * \brief Overwrite fine MG level data with interpolated coarse data.
     *
     * \param amrlev AMR level
     * \param fmglev fine MG level
     * \param fine   fine MG level data
     * \param crse   coarse MG level data
     */
    virtual void interpAssign (int amrlev, int fmglev, MFSet& fine, MFSet& crse) const = 0;

    /**
     * \brief Interpolation between AMR levels
     *
     * \param famrlev fine AMR level
     * \param fine    fine level data
     * \param crse    coarse level data
     * \parame nghost number of ghost cells
     */
    virtual void interpolationAmr (int famrlev, MFSet& fine, const MFSet& crse,
                                   amrex::IntVect const& nghost) const = 0;

    /**
     * \brief Average-down data from fine AMR level to coarse AMR level.
     *
     * \param camrlev  coarse AMR level
     * \param crse_sol solutoin on coarse AMR level
     * \param crse_rhs RHS on coarse AMR level
     * \param fine_sol solution on fine AMR level
     * \param fine_rhs RHS on fine AMR level
     */
    virtual void averageDownSolutionRHS (int camrlev, MFSet& crse_sol, MFSet& crse_rhs,
                                         const MFSet& fine_sol, const MFSet& fine_rhs) = 0;

    /**
     * \brief Apply the linear operator, b = L(x)
     *
     * \param amrlev  AMR level
     * \param mglev   MG level
     * \param b       output set, same size as rhs
     * \param x       input set, same size as solution
     * \param bc_mode Is the BC homogeneous or inhomogeneous?
     * \param s_mode  Are data data solution or correction?
     * \param bndry   object for handling coarse/fine and physical boundaries
     */
    virtual void apply (int amrlev, int mglev, MFSet& b, MFSet& x, BCMode bc_mode,
                        StateMode s_mode, const amrex::MLMGBndryT<MFSet>* bndry=nullptr) const = 0;

    /**
     * \brief Apply the transpose of the linear operator, x = L^T(b)
     *
     * \param amrlev  AMR level
     * \param mglev   MG level
     * \param x       output set, same size as solution
     * \param b       input set, same size as rhs
     * \param bc_mode Is the BC homogeneous or inhomogeneous?
     * \param s_mode  Are data data solution or correction?
     * \param bndry   object for handling coarse/fine and physical boundaries
     */
    virtual void applyTranspose (int amrlev, int mglev, MFSet& x, MFSet& b, BCMode bc_mode,
                                 StateMode s_mode, const amrex::MLMGBndryT<MFSet>* bndry=nullptr) const = 0;

    /**
     * \brief Smooth
     *
     * \param amrlev            AMR level
     * \param mglev             MG level
     * \param sol               unknowns
     * \param rhs               RHS
     * \param skip_fillboundary flag controlling whether ghost cell filling can be skipped.
     */
    virtual void smooth (int amrlev, int mglev, MFSet& sol, const MFSet& rhs,
                         bool skip_fillboundary=false) const = 0;

    //! Divide mf by the diagonal component of the operator. Used by bicgstab.
    virtual void normalize (int amrlev, int mglev, MFSet& mf) const {
        amrex::ignore_unused(amrlev, mglev, mf);
    }

    /**
     * \brief Compute residual for solution
     *
     * \param amrlev       AMR level
     * \param resid        residual
     * \param x            solution
     * \param b            RHS
     * \param crse_bc_data optional argument providing BC at coarse/fine boundary.
     * 
     */
    virtual void solutionResidual (int amrlev, MFSet& resid, MFSet& x, const MFSet& b,
                                   const MFSet* crse_bcdata=nullptr) = 0;

    /**
     * \brief Compute residual for the residual-correction form, resid = b - L(x)
     *
     * \param amrlev       AMR level
     * \param mglev        MG level
     * \param resid        residual
     * \param x            unknown in the residual-correction form
     * \param b            RHS in the residual-correction form
     * \param bc_mode      Is the BC homegeneous or inhomogeneous?
     * \param crse_bc_data optional argument providing BC at coarse/fine boundary.
     */
    virtual void correctionResidual (int amrlev, int mglev, MFSet& resid, MFSet& x, const MFSet& b,
                                     BCMode bc_mode, const MFSet* crse_bcdata=nullptr) = 0;

    /**
     * \brief Reflux at AMR coarse/fine boundary
     *
     * \param crse_amrlev coarse AMR level
     * \param res         coarse level residual
     * \param crse_sol    coarse level solution
     * \param crse_rhs    coarse level RHS
     * \param fine_res    fine level residual
     * \param fine_sol    fine level solution
     * \param fine_rhs    fine level RHS
     */
    virtual void reflux (int crse_amrlev,
                         MFSet& res, const MFSet& crse_sol, const MFSet& crse_rhs,
                         MFSet& fine_res, MFSet& fine_sol, const MFSet& fine_rhs) const = 0;

    /**
     * \brief Compute fluxes
     *
     * \param amrlev AMR level
     * \param fluxes fluxes
     * \param sol    solution
     * \param loc    location of the fluxes
     */
    virtual void compFlux (int amrlev, const amrex::Array<MFSet*,AMREX_SPACEDIM>& fluxes,
                           MFSet& sol, Location loc) const = 0;

    /**
     * \brief Compute gradients of the solution
     *
     * \param amrlev AMR level
     * \param grad   grad(sol)
     * \param sol    solution
     * \param loc    location of the gradients
     */
    virtual void compGrad (int amrlev, const amrex::Array<MFSet*,AMREX_SPACEDIM>& grad,
                           MFSet& sol, Location loc) const = 0;

    //! apply metric terms if there are any
    virtual void applyMetricTerm (int amrlev, int mglev, MFSet& rhs) const = 0;

    //! unapply metric terms if there are any
    virtual void unapplyMetricTerm (int amrlev, int mglev, MFSet& rhs) const = 0;

    //! This is needed for our nodal projection solver
    virtual void unimposeNeumannBC (int /*amrlev*/, MFSet& /*rhs*/) const {}

    //! Extra terms introduced when we treate inhomogeneous Nuemann BC as homogeneous.
    virtual void applyInhomogNeumannTerm (int /*amrlev*/, MFSet& /*rhs*/) const {}

    //! for overset solver only
    virtual void applyOverset (int /*amlev*/, MFSet& /*rhs*/) const {}

    //! scale RHS to fix solvability
    virtual void scaleRHS (int /*amrlev*/, MFSet& /*rhs*/) const {}

    //! get offset for fixing solvability
    virtual amrex::Vector<RT> getSolvabilityOffset (int /*amrlev*/, int /*mglev*/,
                                               MFSet const& /*rhs*/) const { return {}; }

    //! fix solvability by subtracting offset from RHS
    virtual void fixSolvabilityByOffset (int /*amrlev*/, int /*mglev*/, MFSet& /*rhs*/,
                                         amrex::Vector<RT> const& /*offset*/) const {}

    virtual void prepareForSolve () = 0;

    //! Is it singular on given AMR level?
    [[nodiscard]] virtual bool isSingular (int amrlev) const = 0;

    //! Is the bottom of MG singular?
    [[nodiscard]] virtual bool isBottomSingular () const = 0;

    //! x dot y, used by the bottom solver
    virtual amrex::Real xdoty (int amrlev, int mglev, const MFSet& x, const MFSet& y, bool local) const = 0;

    virtual std::unique_ptr<MLLinOpMFSet> makeNLinOp (int grid_size) const = 0;

    [[nodiscard]] virtual bool supportNSolve () const { return false; }

    virtual void copyNSolveSolution (MFSet&, MFSet const&) const {}

    virtual void postSolve (amrex::Vector<MFSet>& /*sol*/) const {}

    [[nodiscard]] virtual amrex::Real normInf (int amrlev, MFSet const& mf, bool local) const = 0;

    virtual void averageDownAndSync (amrex::Vector<MFSet>& sol) const = 0;

    virtual void avgDownResAmr (int clev, MFSet& cres, MFSet const& fres) const = 0;
    virtual void avgDownResMG (int clev, MFSet& cres, MFSet const& fres) const;

    [[nodiscard]] bool isMFIterSafe (int amrlev, int mglev1, int mglev2) const;

protected:

    struct CommContainer {
        MPI_Comm comm;
        CommContainer (MPI_Comm m) noexcept : comm(m) {}
        CommContainer (const CommContainer&) = delete;
        CommContainer (CommContainer&&) = delete;
        void operator= (const CommContainer&) = delete;
        void operator= (CommContainer&&) = delete;
        ~CommContainer () { // NOLINT(modernize-use-equals-default)
#ifdef BL_USE_MPI
            if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
#endif
        }
    };
    
    //! Return the number of AMR levels
    [[nodiscard]] int NAMRLevels () const noexcept { return m_num_amr_levels; }

    //! Return the number of MG levels at given AMR level
    [[nodiscard]] int NMGLevels (int amrlev) const noexcept { return m_num_mg_levels[amrlev]; }

    //! Return AMR refinement ratios
    [[nodiscard]] const amrex::Vector<int>& AMRRefRatio () const noexcept { return m_amr_ref_ratio; }

    //! Return AMR refinement ratio at given AMR level
    [[nodiscard]] int AMRRefRatio (int amr_lev) const noexcept { return m_amr_ref_ratio[amr_lev]; }

    [[nodiscard]] const amrex::Geometry& Geom (int amr_lev, int mglev=0) const noexcept { return m_geom[amr_lev][mglev]; }

    [[nodiscard]] amrex::FabFactory<FAB> const* Factory (int amr_lev, int mglev=0) const noexcept {
        return m_factory[amr_lev][mglev].get();
    }

    [[nodiscard]] amrex::GpuArray<BCType,AMREX_SPACEDIM> LoBC (int icomp = 0) const noexcept {
        return amrex::GpuArray<BCType,AMREX_SPACEDIM>{{AMREX_D_DECL(m_lobc[icomp][0],
                                                             m_lobc[icomp][1],
                                                             m_lobc[icomp][2])}};
    }
    [[nodiscard]] amrex::GpuArray<BCType,AMREX_SPACEDIM> HiBC (int icomp = 0) const noexcept {
        return amrex::GpuArray<BCType,AMREX_SPACEDIM>{{AMREX_D_DECL(m_hibc[icomp][0],
                                                             m_hibc[icomp][1],
                                                             m_hibc[icomp][2])}};
    }

    [[nodiscard]] bool hasBC (BCType bct) const noexcept;
    [[nodiscard]] bool hasInhomogNeumannBC () const noexcept { return hasBC(BCType::inhomogNeumann); };
    [[nodiscard]] bool hasRobinBC () const noexcept { return hasBC(BCType::Robin); };

    [[nodiscard]] virtual bool supportRobinBC () const noexcept { return false; }
    [[nodiscard]] virtual bool supportInhomogNeumannBC () const noexcept { return false; }

#ifdef BL_USE_MPI
    [[nodiscard]] bool isBottomActive () const noexcept { return m_bottom_comm != MPI_COMM_NULL; }
#else
    [[nodiscard]] bool isBottomActive () const noexcept { return true; }
#endif
    [[nodiscard]] MPI_Comm BottomCommunicator () const noexcept { return m_bottom_comm; }
    [[nodiscard]] MPI_Comm Communicator () const noexcept { return m_default_comm; }

    void setCoarseFineBCLocation (const amrex::RealVect& cloc) noexcept { m_coarse_bc_loc = cloc; }

    [[nodiscard]] bool doAgglomeration () const noexcept { return m_do_agglomeration; }
    [[nodiscard]] bool doConsolidation () const noexcept { return m_do_consolidation; }
    [[nodiscard]] bool doSemicoarsening () const noexcept { return m_do_semicoarsening; }

    virtual void make (amrex::Vector<amrex::Vector<MFSet> >& mfs, amrex::IntVect const& ng) const;
    [[nodiscard]] virtual MFSet make (int amrlev, int mglev, amrex::IntVect const& ng) const;

    // [[nodiscard]] virtual MFSet makeAlias (MFSet const& mfs) const;
    [[nodiscard]] virtual MFSet makeCoarseMG (int amrlev, int mglev, amrex::IntVect const& ng) const;
    [[nodiscard]] virtual MFSet makeCoarseAmr (int famrlev, amrex::IntVect const& ng) const;
    [[nodiscard]] virtual std::unique_ptr<amrex::FabFactory<FAB> > makeFactory (int /*amrlev*/, int /*mglev*/) const {
        return std::make_unique<amrex::DefaultFabFactory<FAB>>();
    }

    virtual void resizeMultiGrid (int new_size);

    [[nodiscard]] bool hasHiddenDimension () const noexcept { return info.hasHiddenDimension(); }
    [[nodiscard]] int hiddenDirection () const noexcept { return info.hidden_direction; }
    [[nodiscard]] amrex::Box compactify (amrex::Box const& b) const noexcept;

    template <typename T>
    [[nodiscard]] amrex::Array4<T> compactify (amrex::Array4<T> const& a) const noexcept {
        switch (info.hidden_direction) {
            case 0: return Array4<T>(a.dataPtr(), {a.begin.y,a.begin.z,0}, {a.end.y,a.end.z,1}, a.nComp());
            case 1: return Array4<T>(a.dataPtr(), {a.begin.x,a.begin.z,0}, {a.end.x,a.end.z,1}, a.nComp());
            case 2: return Array4<T>(a.dataPtr(), {a.begin.x,a.begin.y,0}, {a.end.x,a.end.y,1}, a.nComp());
            default: return a;
        }
    }

    template <typename T>
    [[nodiscard]] T get_d0 (T const& dx, T const& dy, T const&) const noexcept {
        return (info.hidden_direction == 0) ? dy : dx;
    };

    template <typename T>
    [[nodiscard]] T get_d1 (T const&, T const& dy, T const& dz) const noexcept {
        return (info.hidden_direction == 0 || info.hidden_direction == 1) ? dz : dy;
    }

private:

    void defineGrids (const amrex::Vector<amrex::Geometry>& a_geom,
                      const amrex::Vector<amrex::BoxArray>& a_grids,
                      const amrex::Vector<amrex::DistributionMapping>& a_dmap,
                      const amrex::Vector<amrex::FabFactory<FAB> const*>& a_factory);
    void defineBC ();
    static void makeAgglomeratedDMap (const amrex::Vector<amrex::BoxArray>& ba, amrex::Vector<amrex::DistributionMapping>& dm);
    static void makeConsolidatedDMap (const amrex::Vector<amrex::BoxArray>& ba, amrex::Vector<amrex::DistributionMapping>& dm,
                                      int ratio, int strategy);
    [[nodiscard]] MPI_Comm makeSubCommunicator (const amrex::DistributionMapping& dm);

    virtual void checkPoint (std::string const& /*file_name*/) const {
        amrex::Abort("MLLinOp:checkPoint: not implemented");
    }

protected:

    int m_ncomp_sol;
    int m_ncomp_rhs;

    amrex::Vector<amrex::IndexType> m_ixtype_set_sol; // Should have length of m_ncomp_sol
    amrex::Vector<amrex::IndexType> m_ixtype_set_rhs; // Should have length of m_ncomp_rhs

    static constexpr int mg_coarsen_ratio = 2;
    static constexpr int mg_box_min_width = 2;
#ifdef AMREX_USE_EB
    static constexpr int mg_domain_min_width = 4;
#else
    static constexpr int mg_domain_min_width = 2;
#endif
    amrex::LPInfo info;
    int verbose = 0;
    int maxorder = 3;
    bool enforceSingularSolvable = true;
    int m_num_amr_levels;
    amrex::Vector<int> m_amr_ref_ratio;
    amrex::Vector<int> m_num_mg_levels;
    const MLLinOpMFSet* m_parent = nullptr;

    // amrex::IntVect m_ixtype; // Controls CellType in each direction. Need to replace this with vectors specifying such info for both the input and the output MFSet

    bool m_do_agglomeration = false;
    bool m_do_consolidation = false;
    bool m_do_semicoarsening = false;
    amrex::Vector<amrex::IntVect> mg_coarsen_ratio_vec;
    //! first amrex::Vector is for amr level and second is mg level
    amrex::Vector<amrex::Vector<amrex::Geometry> >            m_geom;
    amrex::Vector<amrex::Vector<amrex::BoxArray> >            m_grids;
    amrex::Vector<amrex::Vector<amrex::DistributionMapping> > m_dmap;
    amrex::Vector<amrex::Vector<std::unique_ptr<amrex::FabFactory<FAB> > > > m_factory;
    amrex::Vector<int>                          m_domain_covered;
    MPI_Comm m_default_comm = MPI_COMM_NULL;
    MPI_Comm m_bottom_comm = MPI_COMM_NULL;

    // BC
    amrex::Vector<amrex::Array<BCType, AMREX_SPACEDIM> > m_lobc;
    amrex::Vector<amrex::Array<BCType, AMREX_SPACEDIM> > m_hibc;

    // Need to save the original copy because we change the BC type to
    // Neumann for inhomogeneous Neumann and Robin.
    amrex::Vector<amrex::Array<BCType, AMREX_SPACEDIM> > m_lobc_orig;
    amrex::Vector<amrex::Array<BCType, AMREX_SPACEDIM> > m_hibc_orig;

    // amrex::Array<amrex::Real, AMREX_SPACEDIM> m_domain_bloc_lo {{AMREX_D_DECL(0._rt,0._rt,0._rt)}};
    // amrex::Array<amrex::Real, AMREX_SPACEDIM> m_domain_bloc_hi {{AMREX_D_DECL(0._rt,0._rt,0._rt)}};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_domain_bloc_lo {{AMREX_D_DECL(RT(0),RT(0),RT(0))}};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> m_domain_bloc_hi {{AMREX_D_DECL(RT(0),RT(0),RT(0))}};

    bool m_needs_coarse_data_for_bc;
    int m_coarse_data_crse_ratio = -1;
    amrex::RealVect m_coarse_bc_loc;
    const MFSet* m_coarse_data_for_bc = nullptr;
    MFSet m_coarse_data_for_bc_raii;
    std::unique_ptr<CommContainer> m_raii_comm;

private:

    amrex::Vector<std::unique_ptr<MFSet>> levelbc_raii;
    amrex::Vector<std::unique_ptr<MFSet>> robin_a_raii;
    amrex::Vector<std::unique_ptr<MFSet>> robin_b_raii;
    amrex::Vector<std::unique_ptr<MFSet>> robin_f_raii;

    // [[nodiscard]] bool isCellCentered () const noexcept { return m_ixtype == 0; }
    // [[nodiscard]] virtual amrex::BottomSolver getDefaultBottomSolver () const { return amrex::BottomSolver::bicgstab; }
    //! Return number of components
    // [[nodiscard]] virtual int getNComp () const { return 1; }
};

}