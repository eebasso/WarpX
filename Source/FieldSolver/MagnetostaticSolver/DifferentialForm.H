#include <AMReX_Vector.H>
#include <AMReX_IndexType.H>

#if AMREX_SPACEDIM==1
// AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
#  define WARPX_D_EXPR(a,b,c) ((void)((c),0))
// AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
#  define WARPX_D_DECL(a,b,c) c
// AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
#  define WARPX_D_TERM(a,b,c) c
// AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
#  define WARPX_D_PICK(a,b,c) c
#elif AMREX_SPACEDIM==2
// AMReX convention: x = first dimension, y = missing dimension, z = second dimension
#  define WARPX_D_EXPR(a,b,c) ((void)((a),(b),0))
// AMReX convention: x = first dimension, y = missing dimension, z = second dimension
#  define WARPX_D_DECL(a,b,c) a,c
// AMReX convention: x = first dimension, y = missing dimension, z = second dimension
#  define WARPX_D_TERM(a,b,c) a c
// AMReX convention: x = first dimension, y = missing dimension, z = second dimension
#  define WARPX_D_PICK(a,b,c) c
#elif AMREX_SPACEDIM==3
#  define WARPX_D_EXPR(a,b,c) ((void)((a),(b),(c),0))
#  define WARPX_D_DECL(a,b,c) a,b,c
#  define WARPX_D_TERM(a,b,c) a b c
#  define WARPX_D_PICK(a,b,c) c
#endif

namespace Solver {

class MultiFabSet;

enum struct DiffFormType { ZeroForm, OneForm, TwoForm, ThreeForm, None };

class DifferentialForm {

    using IT = amrex::IndexType;
    using CI = amrex::IndexType::CellIndex;

public:

    void
    GetDiffFormTypeFromMFSet(Solver::MultiFabSet mfs, int start_comp) {
        amrex::MultiFab& test = mfs[start_comp];
    };

    void
    GetDiffFormTypeFromMFSet(amrex::Vector<amrex::MultiFab> mfs, int start_comp) {
        amrex::MultiFab& test = mfs[start_comp];
    };

    // 0-form fields (e.g. rho, phi) live on the nodes of the grid.
    std::array<IT, 1> m_ZeroFormIndexTypeList = {

    };

    // 1-form fields (e.g. E,A,J) live on the lines connecting neighboring nodes of the grid.
    static std::array<IT, 3> const m_OneFormIndexTypeList = { 
        IT(WARPX_D_DECL(CI::CELL, CI::NODE, CI::NODE)),
        IT(WARPX_D_DECL(CI::NODE, CI::CELL, CI::NODE)),
        IT(WARPX_D_DECL(CI::NODE, CI::NODE, CI::CELL))
    };

    // 2-form fields (e.g. B, H) live on the faces formed by the nodes of the grid. Components are ordered as yz, zx, xy. Taking the Hodge dual of the 2-form makes this correspond to the pseudovector with components ordered x, y, z.
    std::array<IT, 3> m_TwoFormIndexTypeList = { 
        IT(WARPX_D_DECL(CI::NODE, CI::CELL, CI::CELL)),
        IT(WARPX_D_DECL(CI::CELL, CI::NODE, CI::CELL)),
        IT(WARPX_D_DECL(CI::CELL, CI::CELL, CI::NODE))
    };

    //3-form 

    // Ex_nodal_flag = IntVect(1); IntVect(0,1); IntVect(0,1,1);
    // Ey_nodal_flag = IntVect(1); IntVect(1,1); IntVect(1,0,1);
    // Ez_nodal_flag = IntVect(0); IntVect(1,0); IntVect(1,1,0);
    // Bx_nodal_flag = IntVect(0); IntVect(1,0); IntVect(1,0,0);
    // By_nodal_flag = IntVect(0); IntVect(0,0); IntVect(0,1,0);
    // Bz_nodal_flag = IntVect(1); IntVect(0,1); IntVect(0,0,1);
    // jx_nodal_flag = IntVect(1); IntVect(0,1); IntVect(0,1,1);
    // jy_nodal_flag = IntVect(1); IntVect(1,1); IntVect(1,0,1);
    // jz_nodal_flag = IntVect(0); IntVect(1,0); IntVect(1,1,0);

// #elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
    // AMReX convention: x = first dimension, y = missing dimension, z = second dimension
    // Ex_nodal_flag = 
    // Ey_nodal_flag = 
    // Ez_nodal_flag =
    // Bx_nodal_flag =
    // By_nodal_flag =
    // Bz_nodal_flag =
    // jx_nodal_flag =
    // jy_nodal_flag =
    // jz_nodal_flag =
// #elif defined(WARPX_DIM_3D)
    // Ex_nodal_flag = 
    // Ey_nodal_flag =
    // Ez_nodal_flag =
    // Bx_nodal_flag =
    // By_nodal_flag =
    // Bz_nodal_flag =
    // jx_nodal_flag =
    // jy_nodal_flag =
    // jz_nodal_flag =
// #endif

    };



};