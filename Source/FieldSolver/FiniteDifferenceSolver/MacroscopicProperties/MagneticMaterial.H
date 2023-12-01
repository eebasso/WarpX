#ifndef WARPX_MAGNETICMATERIAL_H_
#define WARPX_MAGNETICMATERIAL_H_

#include <AMReX_Config.H>
#include <AMReX.H>
#include <AMReX_REAL.H>

class MagneticMaterial
{
public:

    MagneticMaterial () = default;

    ~MagneticMaterial () = default;

    virtual void UpdateHandM (
              amrex::Real& H_x     ,       amrex::Real& H_y     ,       amrex::Real& H_z    ,
              amrex::Real& M_x     ,       amrex::Real& M_y     ,       amrex::Real& M_z    ,
        const amrex::Real& B_next_x, const amrex::Real& B_next_y, const amrex::Real& B_next_z) const
    {
        amrex::ignore_unused(H_x,H_y,H_z,M_x,M_y,M_z,B_next_x,B_next_y,B_next_z);
    };

};

#endif /* WARPX_MAGNETICMATERIAL_H_ */