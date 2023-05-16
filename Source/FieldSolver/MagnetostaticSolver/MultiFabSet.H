#include <AMReX_Config.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <DifferentialForm.H>

namespace Solver {

class MultiFabSet : public amrex::Vector<amrex::MultiFab> {

public:

    MultiFabSet (amrex::BoxArray a_ba, 
                amrex::DistributionMapping a_dmap, 
                amrex::Vector<amrex::IntVect> a_ixtype_set, 
                amrex::IntVect const& n_grow) {
    }

    MultiFabSet (MultiFabSet mfs, bool use_nGrowVects_from_MFs, amrex::MFInfo mfinfo) {
        this->clear();
        for (amrex::MultiFab& mf : mfs) {
            amrex::IntVect ngrow = use_nGrowVects_from_MFs ? mf.nGrowVect() : amrex::IntVect();
            AddNewMultiFab(mf, ngrow, mfinfo);
        }
    }

    MultiFabSet (MultiFabSet mfs, amrex::IntVect ngrow, amrex::MFInfo mfinfo) {
        this->clear();
        for (amrex::MultiFab& mf : mfs) {
            AddNewMultiFab(mf, ngrow, mfinfo);
        }
    }

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
     * \param num_of_MF The number of entries changed starting from `start_index`
     */
    void
    setVal (amrex::Real val, int start_index, int num_of_MF) {
        for (int i = start_index; i < start_index + num_of_MF; ++i) {
            // (*this)[i].setVal(val); // is this better?
            this->operator[](i).setVal(val); // is this better?
        };
    };

    amrex::Real
    norminf () const {
        amrex::Real result = amrex::Real(0);
        for (const amrex::MultiFab& mf : *this) {
            result = std::max(result, mf.norminf(0, mf.nComp(), amrex::IntVect(0), true));
        }
        return result;
    }

    amrex::Real
    norminf (int start_comp, int num_comp) const {
        amrex::Real result = amrex::Real(0);
        for (int i = start_comp; i < start_comp + num_comp; ++i) {
            // amrex::MultiFab& mf = this->operator[](i); // Is this line better?
            const amrex::MultiFab& mf = (*this)[i];       // Or is this line better?
            result = std::max(result, mf.norminf(0, mf.nComp(), amrex::IntVect(0), true));
        }
        return result;
    }

    void
    LocalCopy (const MultiFabSet& mfs, amrex::IntVect ngrow = amrex::IntVect(0)) {
        // TODO: Add check for length
        for (int i = 0; i < this->size(); ++i) {
            amrex::MultiFab& mf_dst = this->operator[](i);
            const amrex::MultiFab& mf_src = mfs[i];
            mf_dst.LocalCopy(mf_src, 0, 0, mf_src.nComp(), ngrow);
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

    DiffFormType
    GetDiffFormType (int start_index, int num_comp) {
        amrex::MultiFab& mf0 = (*this)[start_index];
        amrex::IndexType type0 = mf0.ixType();
        if (type0 == amrex::IndexType::TheNodeType()) {

        }
        return Solver::DiffFormType::None;
    }

    void
    LocalAdd(const MultiFabSet& srcSet, amrex::IntVect nghost) {
        for (int i = 0; i < this->size(); ++i) {
            (*this)[i].LocalAdd(srcSet[i], 0, 0, srcSet[i].nComp(), nghost);
        }
    }

private:

    void 
    AddNewMultiFab(const amrex::MultiFab& mf, amrex::IntVect ngrow, amrex::MFInfo mfinfo) {
        this->push_back(amrex::MultiFab(mf.boxArray(), 
                                        mf.DistributionMap(), 
                                        mf.nComp(), 
                                        ngrow, 
                                        mfinfo, 
                                        mf.Factory() ) );
    }

};

};