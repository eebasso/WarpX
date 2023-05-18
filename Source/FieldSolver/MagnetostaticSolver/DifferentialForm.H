#include <AMReX_Config.H>
#include <AMReX_Vector.H>
#include <AMReX_IndexType.H>
#include "MultiFabSet.H"

// #if AMREX_SPACEDIM==1
// // AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
// #  define WARPX_D_EXPR(a,b,c) ((void)((c),0))
// // AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
// #  define WARPX_D_DECL(a,b,c) c
// // AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
// #  define WARPX_D_TERM(a,b,c) c
// // AMReX convention: x = missing dimension, y = missing dimension, z = only dimension
// #  define WARPX_D_PICK(a,b,c) c
// #elif AMREX_SPACEDIM==2
// // AMReX convention: x = first dimension, y = missing dimension, z = second dimension
// #  define WARPX_D_EXPR(a,b,c) ((void)((a),(b),0))
// // AMReX convention: x = first dimension, y = missing dimension, z = second dimension
// #  define WARPX_D_DECL(a,b,c) a,c
// // AMReX convention: x = first dimension, y = missing dimension, z = second dimension
// #  define WARPX_D_TERM(a,b,c) a c
// // AMReX convention: x = first dimension, y = missing dimension, z = second dimension
// #  define WARPX_D_PICK(a,b,c) c
// #elif AMREX_SPACEDIM==3
// #  define WARPX_D_EXPR(a,b,c) ((void)((a),(b),(c),0))
// #  define WARPX_D_DECL(a,b,c) a,b,c
// #  define WARPX_D_TERM(a,b,c) a b c
// #  define WARPX_D_PICK(a,b,c) c
// #endif

namespace Solver {

// class MultiFabSet;

enum struct DiffFormType { ZeroForm, OneForm, TwoForm, ThreeForm, None };

class DifferentialForm {

    using IT = amrex::IndexType;
    using CI = amrex::IndexType::CellIndex;

public:

    DiffFormType GetDiffFormTypeFromMFSet(MultiFabSet& mfs, int start_comp, int num_comp) {
        if (num_comp == 1) {
            mfs[start_comp].
        }
        else if (num_comp == 3) {

        }
        return DiffFormType::None
    };








    /**
     * \brief The appropriate IndexType for the MultiFab of a 0-form.
     * 
     * 0-form fields live on the nodes of the grid. They are scalar fields, and examples include
     * the charge density rho, the scalar potential phi, and the auxillary field F used to help 
     * enforce divE = rho in EvolveE.
     */
    static constexpr IT ZeroFormIndexType = IT::TheNodeType();

    /**
     * \brief The appropriate IndexType for each MultiFab of a 1-form with ordering {x,y,z}.
     * 
     * 1-form fields live on the lines connecting neighboring nodes of the grid. They are true vector 
     * fields, with important examples being the electric field E, the vector potential A and current
     * density J. Note that the magnetic field B is not a 1-form, but rather a pseudovector that is the 
     * Hodge dual to a 2-form.
     */
    static constexpr std::array<IT, 3> OneFormIndexTypeList = { 
        IT(WARPX_D_DECL(CI::CELL, CI::NODE, CI::NODE)),
        IT(WARPX_D_DECL(CI::NODE, CI::CELL, CI::NODE)),
        IT(WARPX_D_DECL(CI::NODE, CI::NODE, CI::CELL))
    };

    /**
     * \brief The appropriate IndexType for each MultiFab of a 2-form with ordering {yz,zx,xy}.
     * 
     * 2-form fields live on the faces formed by the nodes of the grid. The B, H, and M fields are
     * important examples. A 2-form can be thought of as an anti-symmetric matrix (i.e. rank-2
     * tensor) field.  The Hodge dual of the 2-form makes this correspond to the pseudovector with 
     * components ordered {x,y,z}. These are not true vectors, but rather pseudovectors, and
     * this changes where they live on the grid.
     */
    static constexpr std::array<IT, 3> TwoFormIndexTypeList = { 
        IT(WARPX_D_DECL(CI::NODE, CI::CELL, CI::CELL)),
        IT(WARPX_D_DECL(CI::CELL, CI::NODE, CI::CELL)),
        IT(WARPX_D_DECL(CI::CELL, CI::CELL, CI::NODE))
    };

    /**
     * \brief The appropriate IndexType for the MultiFab of a 3-form.
     * 
     * 3-form fields live in the volumes formed by the nodes of the grid. Hence, they are completely
     * cell-centered. Examples include divB and the auxillary G field used to enforce divB=0 in
     * EvolveB. In 3 dimensions, 3-forms are dual to 0-forms and hence only have one non-trivial
     * component (xyz). These are not true scalars, but rather pseudoscalars, and this changes where 
     * they live on the grid.
     */
    static constexpr IT ThreeFormIndexType = IT::TheCellType();

    };



};