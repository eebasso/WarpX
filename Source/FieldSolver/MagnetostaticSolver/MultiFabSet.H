#include <AMReX_Config.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

namespace Solver {

/**
 * \brief A set of MultiFabs usually made for input/output into multi-component linear operators such as MLLinOpMFSet.
 * 
 * The following explanation could be in error due to the nascent experience of the author:
 * 
 * While the MultiFab class allows for multiple components, each component is required to live on exactly the same location (IndexType) on the grid. Specifically, each component in a MultiFab shares the same BoxArray and thus IndexType. Unfortunately, differential forms such as 1-forms (vector fields) and 2-forms (psuedovector fields) are ideally used by finite difference methods when each component has a distinct IndexType. In particular, 1-form components live on the lines between nodes (i.e., the edges of the grid cells), and 2-form components live on the faces of grid cells. This requires each component to have slightly different BoxArray's and associated Array4's.
 * 
 * There are two possible solutions to this. The first category involves modifying MultiFab or creating a similar alternative such that the distinct IndexType's of each component are tracked relative to some common BoxArray. Each FArrayBox would have slightly different Box's that are related to the common box by their respective IndexType. Unfortunately, this would be incredibly complicated to implement as it would require an extensive rewrite of AMReX.
 * 
 * The second possibility is creating a separate MultiFab for each individual component. The latter is currently used by WarpX when representing all multi-component fields such as the E and B fields. An example of the distinct IndexType's for each component can be found in WarpX.cpp in the definitions of Ex_nodal_flag, Ey_nodal_flag, etc. Note that the nodal flag flags for B are different because the magnetic field is a 2-form and not a 1-form. This is also often described as stating that B lives on a staggered grid from E. MultiFabSet is intended to be a natural generalization of this approach.
 * 
 * In summary, MultiFabSet is intended to be a natural generalization of how E and B have been represented in WarpX thus far, which is a collection of single component MultiFab's that live on slightly different FAB's with distinct IndexType's (nodal flags).
 * 
 * Inheritances from amrex::Vector<amrex::MultiFab>.
 */
class MultiFabSet : public amrex::Vector<amrex::MultiFab> {

public:

    using MF = typename amrex::MultiFab;
    using MFInfo = const typename amrex::MFInfo;
    using IntVect = const typename amrex::IntVect &;
    using IxTypes = const typename amrex::Vector<amrex::IndexType> &;
    using FABFactory = const typename amrex::FabFactory<amrex::FArrayBox> &;

    MultiFabSet (const MultiFabSet&, MFInfo);
    MultiFabSet (const MultiFabSet&, MFInfo, int);
    MultiFabSet (const MultiFabSet&, MFInfo, IntVect);
    MultiFabSet (const MultiFabSet&, MFInfo, IxTypes);
    MultiFabSet (const MultiFabSet&, MFInfo, FABFactory);
    

    MultiFabSet (const MultiFabSet&, MFInfo, int, IntVect);
    MultiFabSet (const MultiFabSet&, MFInfo, int, IxTypes);
    MultiFabSet (const MultiFabSet&, MFInfo, int, FABFactory);
    // MultiFabSet (const MultiFabSet&, MFInfo, IntVect, IxTypes);
    MultiFabSet (const MultiFabSet&, MFInfo, IntVect, FABFactory);
    // MultiFabSet (const MultiFabSet&, MFInfo, IntVect, IxTypes);
    
    MultiFabSet (const MultiFabSet&, MFInfo);
    
    MultiFabSet (const MultiFabSet&, MFInfo, FABFactory);
    MultiFabSet (const MultiFabSet&, MFInfo, IntVect, FABFactory);
    // MultiFabSet (const MultiFabSet&, MFInfo, int, IntVect, FABFactory);

    MultiFabSet (const MultiFabSet&, MFInfo, int, IntVect, IxTypes, FABFactory);


    MultiFabSet::MultiFabSet (amrex::Vector<MF> mfv);

    MultiFabSet (const amrex::BoxArray& bxs, const amrex::DistributionMapping dm, int ncomps_in_set, IntVect ngrow, 
                MFInfo info = MFInfo(), FABFactory factory = amrex::FArrayBoxFactory(), int ncomps_in_mf = 1);

    MultiFabSet (const amrex::BoxArray& bxs, const amrex::DistributionMapping dm, IxTypes ixtype_set, IntVect ngrow,
        const MFInfo& info = MFInfo(), const FABFactory factory = amrex::FArrayBoxFactory(), int ncomps_in_mf = 1);

    /**
     * \brief Set all MultiFabs to a constant value
     * 
     * \param val The value to be set
     */
    void
    setVal (amrex::Real val) { setVal(val, 0, this->size()); };

    /**
     * \brief Set constant value for selected Multifabs
     * 
     * \param val The value to be set
     * \param start_index The starting index in the MultiFabSet
     * \param num_of_MF The number of MultiFabs affected starting from `start_index`
     */
    void
    setVal (amrex::Real val, int start_index, int num_of_MF) {
        for (int i = start_index; i < start_index + num_of_MF; ++i) {
            // (*this)[i].setVal(val); // is this better?
            this->operator[](i).setVal(val); // is this better?
        };
    };

    /**
     * \brief Finds the largest norminf amongst the set of MultiFabs
     * 
     * \return The L_infinity norm, i.e. the supremum of the absolute value.
     */
    amrex::Real
    norminf () const {
        amrex::Real result = amrex::Real(0);
        for (const MF& mf : *this) {
            result = std::max(result, mf.norminf(0, mf.nComp(), amrex::IntVect(0), true));
        }
        return result;
    };

    /**
     * \brief Find the largest norminf amongst the selected MultiFabs
     * 
     * \param start_index The starting index in the MultiFabSet
     * \param num_of_MF The number of MultiFabs affected starting from `start_index`
     * 
     * \return The L_infinity norm, i.e. the supremum of the absolute value.
     */
    amrex::Real
    norminf (int start_index, int num_comp) const {
        amrex::Real result = amrex::Real(0);
        for (int i = start_index; i < start_index + num_comp; ++i) {
            // MF& mf = this->operator[](i); // Is this line better?
            const MF& mf = (*this)[i];       // Or is this line better?
            result = std::max(result, mf.norminf(0, mf.nComp(), amrex::IntVect(0), true));
        }
        return result;
    }

    /**
     * \brief Copies each MultiFab from the source MultiFabSet using nGrowVect
     * 
     * \param mfs Source MultiFabSet
     */
    void
    LocalCopy (const MultiFabSet& mfs) {
        // TODO: Add check for length
        for (int i = 0; i < this->size(); ++i) {
            (*this)[i].LocalCopy(mfs[i], 0, 0, mfs[i].nComp(), mfs[i].nGrowVect());
        }
    }
    
    /**
     * \brief Copies each MultiFab from the source MultiFabSet
     * 
     * \param mfs Source MultiFabSet
     * \param ngrow Manual grow parameter
     */
    void
    LocalCopy (const MultiFabSet& mfs, IntVect ngrow) {
        // TODO: Add check for length
        for (int i = 0; i < this->size(); ++i) {
            (*this)[i].LocalCopy(mfs[i], 0, 0, mfs[i].nComp(), ngrow);
        }
    }

    amrex::Vector<amrex::IndexType>
    GetIndexTypes (int start_index, int num_comp) {
        // TODO: Add check for out of bounds
        amrex::Vector<amrex::IndexType> result(num_comp);
        for (int i = start_index; i < start_index + num_comp; ++i){
            result[i] = (*this)[i].ixType();
        }
        return result;
    }

    void
    LocalAdd(const MultiFabSet& srcSet, IntVect nghost) {
        for (int i = 0; i < this->size(); ++i) {
            (*this)[i].LocalAdd(srcSet[i], 0, 0, srcSet[i].nComp(), nghost);
        }
    }

    // void
    // InitializeAndAddMultiFab(const MF& mf, IntVect ngrow, MFInfo mfinfo) {
    //     this->push_back(MF(mf.boxArray(), mf.DistributionMap(), mf.nComp(), ngrow, mfinfo, mf.Factory() ) );
    // }

    // void
    // InitializeAndAddMultiFab(const MF& mf, IntVect ngrow, MFInfo mfinfo) {
    //     this->push_back(MF(mf.boxArray(), mf.DistributionMap(), mf.nComp(), ngrow, mfinfo, mf.Factory() ) );
    // }

};

};