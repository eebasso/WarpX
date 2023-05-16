#include <MLLinOpMFSet.H>

namespace Solver {

void
MLLinOpMFV::define (const amrex::Vector<amrex::Geometry>& a_geom,
                    const amrex::Vector<amrex::BoxArray>& a_grids,
                    const amrex::Vector<amrex::DistributionMapping>& a_dmap,
                    const amrex::LPInfo& a_info,
                    const amrex::Vector<amrex::FabFactory<FAB> const*>& a_factory,
                    [[maybe_unused]] bool eb_limit_coarsening)
{
    BL_PROFILE("MLLinOp::define()");

    info = a_info;
#ifdef AMREX_USE_GPU
    if (Gpu::notInLaunchRegion())
    {
        if (info.agg_grid_size <= 0) info.agg_grid_size = AMREX_D_PICK(32, 16, 8);
        if (info.con_grid_size <= 0) info.con_grid_size = AMREX_D_PICK(32, 16, 8);
    }
    else
#endif
    {
        if (info.agg_grid_size <= 0) info.agg_grid_size = amrex::LPInfo::getDefaultAgglomerationGridSize();
        if (info.con_grid_size <= 0) info.con_grid_size = amrex::LPInfo::getDefaultConsolidationGridSize();
    }

#ifdef AMREX_USE_EB
    if (!a_factory.empty() && eb_limit_coarsening) {
        const auto *f = dynamic_cast<amrex::EBFArrayBoxFactory const*>(a_factory[0]);
        if (f) {
            info.max_coarsening_level = std::min(info.max_coarsening_level,
                                                 f->maxCoarseningLevel());
        }
    }
#endif
    defineGrids(a_geom, a_grids, a_dmap, a_factory);
    defineBC();
}

void
MLLinOpMFV::defineGrids (const amrex::Vector<amrex::Geometry>& a_geom,
                           const amrex::Vector<amrex::BoxArray>& a_grids,
                           const amrex::Vector<amrex::DistributionMapping>& a_dmap,
                           const amrex::Vector<amrex::FabFactory<FAB> const*>& a_factory)
{
    BL_PROFILE("MLLinOp::defineGrids()");

    m_num_amr_levels = static_cast<int>(a_geom.size());

    m_amr_ref_ratio.resize(m_num_amr_levels);
    m_num_mg_levels.resize(m_num_amr_levels);

    m_geom.resize(m_num_amr_levels);
    m_grids.resize(m_num_amr_levels);
    m_dmap.resize(m_num_amr_levels);
    m_factory.resize(m_num_amr_levels);

    m_default_comm = amrex::ParallelContext::CommunicatorSub();

    const amrex::RealBox& rb = a_geom[0].ProbDomain();
    const int coord = a_geom[0].Coord();
    const amrex::Array<int,AMREX_SPACEDIM>& is_per = a_geom[0].isPeriodic();

    amrex::IntVect mg_coarsen_ratio_v(mg_coarsen_ratio);
    amrex::IntVect mg_box_min_width_v(mg_box_min_width);
    amrex::IntVect mg_domain_min_width_v(mg_domain_min_width);
    if (hasHiddenDimension()) {
        AMREX_ASSERT_WITH_MESSAGE(AMREX_SPACEDIM == 3 && m_num_amr_levels == 1,
                                  "Hidden direction only supported for 3d level solve");
        mg_coarsen_ratio_v[info.hidden_direction] = 1;
        mg_box_min_width_v[info.hidden_direction] = 0;
        mg_domain_min_width_v[info.hidden_direction] = 0;
    }

    // fine amr levels
    for (int amrlev = m_num_amr_levels-1; amrlev > 0; --amrlev)
    {
        m_num_mg_levels[amrlev] = 1;
        m_geom[amrlev].push_back(a_geom[amrlev]);
        m_grids[amrlev].push_back(a_grids[amrlev]);
        m_dmap[amrlev].push_back(a_dmap[amrlev]);
        if (amrlev < a_factory.size()) {
            m_factory[amrlev].emplace_back(a_factory[amrlev]->clone());
        } else {
            m_factory[amrlev].push_back(std::make_unique<amrex::DefaultFabFactory<FAB>>());
        }

        amrex::IntVect rr = mg_coarsen_ratio_v;
        const amrex::Box& dom = a_geom[amrlev].Domain();
        for (int i = 0; i < 2; ++i)
        {
            if (!dom.coarsenable(rr)) amrex::Abort("MLLinOp: Uncoarsenable domain");

            const amrex::Box& cdom = amrex::coarsen(dom,rr);
            if (cdom == a_geom[amrlev-1].Domain()) break;

            ++(m_num_mg_levels[amrlev]);

            m_geom[amrlev].emplace_back(cdom, rb, coord, is_per);

            m_grids[amrlev].push_back(a_grids[amrlev]);
            AMREX_ASSERT(m_grids[amrlev].back().coarsenable(rr));
            m_grids[amrlev].back().coarsen(rr);

            m_dmap[amrlev].push_back(a_dmap[amrlev]);

            rr *= mg_coarsen_ratio_v;
        }

        if (hasHiddenDimension()) {
            m_amr_ref_ratio[amrlev-1] = rr[AMREX_SPACEDIM-info.hidden_direction];
        } else {
            m_amr_ref_ratio[amrlev-1] = rr[0];
        }
    }

    // coarsest amr level
    m_num_mg_levels[0] = 1;
    m_geom[0].push_back(a_geom[0]);
    m_grids[0].push_back(a_grids[0]);
    m_dmap[0].push_back(a_dmap[0]);
    if (!a_factory.empty()) {
        m_factory[0].emplace_back(a_factory[0]->clone());
    } else {
        m_factory[0].push_back(std::make_unique<amrex::DefaultFabFactory<FAB>>());
    }

    m_domain_covered.resize(m_num_amr_levels, false);
    auto npts0 = m_grids[0][0].numPts();
    m_domain_covered[0] = (npts0 == compactify(m_geom[0][0].Domain()).numPts());
    for (int amrlev = 1; amrlev < m_num_amr_levels; ++amrlev)
    {
        if (!m_domain_covered[amrlev-1]) break;
        m_domain_covered[amrlev] = (m_grids[amrlev][0].numPts() ==
                                    compactify(m_geom[amrlev][0].Domain()).numPts());
    }

    amrex::Box aggbox;
    bool aggable = false;

    if (info.do_agglomeration)
    {
        if (m_domain_covered[0])
        {
            aggbox = m_geom[0][0].Domain();
            if (hasHiddenDimension()) {
                aggbox.makeSlab(hiddenDirection(), m_grids[0][0][0].smallEnd(hiddenDirection()));
            }
            aggable = true;
        }
        else
        {
            aggbox = m_grids[0][0].minimalBox();
            aggable = (aggbox.numPts() == npts0);
        }
    }

    bool agged = false;
    bool coned = false;
    int agg_lev = 0, con_lev = 0;

    AMREX_ALWAYS_ASSERT( ! (info.do_semicoarsening && info.hasHiddenDimension())
                         && info.semicoarsening_direction >= -1
                         && info.semicoarsening_direction < AMREX_SPACEDIM );

    if (info.do_agglomeration && aggable)
    {
        amrex::Box dbx = m_geom[0][0].Domain();
        amrex::Box bbx = aggbox;
        amrex::Real const nbxs = static_cast<amrex::Real>(m_grids[0][0].size());
        amrex::Real const threshold_npts = static_cast<amrex::Real>(AMREX_D_TERM(info.agg_grid_size,
                                                                  *info.agg_grid_size,
                                                                  *info.agg_grid_size));
        amrex::Vector<amrex::Box> domainboxes{dbx};
        amrex::Vector<amrex::Box> boundboxes{bbx};
        amrex::Vector<int> agg_flag{false};
        amrex::Vector<amrex::IntVect> accum_coarsen_ratio{amrex::IntVect(1)};
        int numsclevs = 0;

        for (int lev = 0; lev < info.max_coarsening_level; ++lev)
        {
            amrex::IntVect rr_level = mg_coarsen_ratio_v;
            bool const do_semicoarsening_level = info.do_semicoarsening
                && numsclevs < info.max_semicoarsening_level;
            if (do_semicoarsening_level
                && info.semicoarsening_direction != -1)
            {
                rr_level[info.semicoarsening_direction] = 1;
            }
            amrex::IntVect is_coarsenable;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                amrex::IntVect rr_dir(1);
                rr_dir[idim] = rr_level[idim];
                is_coarsenable[idim] = dbx.coarsenable(rr_dir, mg_domain_min_width_v)
                    && bbx.coarsenable(rr_dir, mg_box_min_width_v);
                if (!is_coarsenable[idim] && do_semicoarsening_level
                    && info.semicoarsening_direction == -1)
                {
                    is_coarsenable[idim] = true;
                    rr_level[idim] = 1;
                }
            }
            if (is_coarsenable != amrex::IntVect(1) || rr_level == amrex::IntVect(1)) {
                break;
            }
            if (do_semicoarsening_level && info.semicoarsening_direction == -1) {
                // make sure there is at most one direction that is not coarsened
                int n_ones = AMREX_D_TERM(  static_cast<int>(rr_level[0] == 1),
                                          + static_cast<int>(rr_level[1] == 1),
                                          + static_cast<int>(rr_level[2] == 1));
                if (n_ones > 1) { break; }
            }
            if (rr_level != mg_coarsen_ratio_v) {
                ++numsclevs;
            }

            accum_coarsen_ratio.push_back(accum_coarsen_ratio.back()*rr_level);
            domainboxes.push_back(dbx.coarsen(rr_level));
            boundboxes.push_back(bbx.coarsen(rr_level));
            bool to_agg = (bbx.d_numPts() / nbxs) < 0.999*threshold_npts;
            agg_flag.push_back(to_agg);
        }

        for (int lev = 1, nlevs = static_cast<int>(domainboxes.size()); lev < nlevs; ++lev) {
            if (!agged && !agg_flag[lev] &&
                a_grids[0].coarsenable(accum_coarsen_ratio[lev], mg_box_min_width_v))
            {
                m_grids[0].push_back(amrex::coarsen(a_grids[0], accum_coarsen_ratio[lev]));
                m_dmap[0].push_back(a_dmap[0]);
            } else {
                amrex::IntVect cr = domainboxes[lev-1].length() / domainboxes[lev].length();
                if (!m_grids[0].back().coarsenable(cr)) {
                    break; // average_down would fail if fine boxarray is not coarsenable.
                }
                m_grids[0].emplace_back(boundboxes[lev]);
                amrex::IntVect max_grid_size(info.agg_grid_size);
                if (info.do_semicoarsening && info.max_semicoarsening_level >= lev
                    && info.semicoarsening_direction != -1)
                {
                    amrex::IntVect blen = amrex::enclosedCells(boundboxes[lev]).size();
                    AMREX_D_TERM(int mgs_0 = (max_grid_size[0]+blen[0]-1) / blen[0];,
                                 int mgs_1 = (max_grid_size[1]+blen[1]-1) / blen[1];,
                                 int mgs_2 = (max_grid_size[2]+blen[2]-1) / blen[2]);
                    max_grid_size[info.semicoarsening_direction]
                        *= AMREX_D_TERM(mgs_0, *mgs_1, *mgs_2);
                }
                m_grids[0].back().maxSize(max_grid_size);
                m_dmap[0].push_back(amrex::DistributionMapping());
                if (!agged) {
                    agged = true;
                    agg_lev = lev;
                }
            }
            m_geom[0].emplace_back(domainboxes[lev],rb,coord,is_per);
        }
    }
    else
    {
        amrex::Long consolidation_threshold = 0;
        amrex::Real avg_npts = 0.0;
        if (info.do_consolidation) {
            avg_npts = static_cast<amrex::Real>(a_grids[0].d_numPts()) / static_cast<amrex::Real>(amrex::ParallelContext::NProcsSub());
            consolidation_threshold = AMREX_D_TERM(amrex::Long(info.con_grid_size),
                                                       *info.con_grid_size,
                                                       *info.con_grid_size);
        }

        amrex::Box const& dom0 = a_geom[0].Domain();
        amrex::IntVect rr_vec(1);
        int numsclevs = 0;
        for (int lev = 0; lev < info.max_coarsening_level; ++lev)
        {
            amrex::IntVect rr_level = mg_coarsen_ratio_v;
            bool do_semicoarsening_level = info.do_semicoarsening
                && numsclevs < info.max_semicoarsening_level;
            if (do_semicoarsening_level
                && info.semicoarsening_direction != -1)
            {
                rr_level[info.semicoarsening_direction] = 1;
            }
            amrex::IntVect is_coarsenable;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                amrex::IntVect rr_dir(1);
                rr_dir[idim] = rr_vec[idim] * rr_level[idim];
                is_coarsenable[idim] = dom0.coarsenable(rr_dir, mg_domain_min_width_v)
                    && a_grids[0].coarsenable(rr_dir, mg_box_min_width_v);
                if (!is_coarsenable[idim] && do_semicoarsening_level
                    && info.semicoarsening_direction == -1)
                {
                    is_coarsenable[idim] = true;
                    rr_level[idim] = 1;
                }
            }
            if (is_coarsenable != amrex::IntVect(1) || rr_level == amrex::IntVect(1)) {
                break;
            }
            if (do_semicoarsening_level && info.semicoarsening_direction == -1) {
                // make sure there is at most one direction that is not coarsened
                int n_ones = AMREX_D_TERM(  static_cast<int>(rr_level[0] == 1),
                                          + static_cast<int>(rr_level[1] == 1),
                                          + static_cast<int>(rr_level[2] == 1));
                if (n_ones > 1) { break; }
            }
            if (rr_level != mg_coarsen_ratio_v) {
                ++numsclevs;
            }
            rr_vec *= rr_level;

            m_geom[0].emplace_back(amrex::coarsen(dom0, rr_vec), rb, coord, is_per);
            m_grids[0].push_back(amrex::coarsen(a_grids[0], rr_vec));

            if (info.do_consolidation)
            {
                if (avg_npts/static_cast<amrex::Real>(AMREX_D_TERM(rr_vec[0], *rr_vec[1], *rr_vec[2]))
                    < amrex::Real(0.999)*static_cast<amrex::Real>(consolidation_threshold))
                {
                    coned = true;
                    con_lev = m_dmap[0].size();
                    m_dmap[0].push_back(amrex::DistributionMapping());
                }
                else
                {
                    m_dmap[0].push_back(m_dmap[0].back());
                }
            }
            else
            {
                m_dmap[0].push_back(a_dmap[0]);
            }
        }
    }

    m_num_mg_levels[0] = m_grids[0].size();

    for (int mglev = 0; mglev < m_num_mg_levels[0] - 1; mglev++){
        const amrex::Box& fine_domain = m_geom[0][mglev].Domain();
        const amrex::Box& crse_domain = m_geom[0][mglev+1].Domain();
        mg_coarsen_ratio_vec.push_back(fine_domain.length()/crse_domain.length());
    }

    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        if (AMRRefRatio(amrlev) == 4 && mg_coarsen_ratio_vec.empty()) {
            mg_coarsen_ratio_vec.push_back(amrex::IntVect(2));
        }
    }

    if (agged)
    {
        makeAgglomeratedDMap(m_grids[0], m_dmap[0]);
    }
    else if (coned)
    {
        makeConsolidatedDMap(m_grids[0], m_dmap[0], info.con_ratio, info.con_strategy);
    }

    if (agged || coned)
    {
        m_bottom_comm = makeSubCommunicator(m_dmap[0].back());
    }
    else
    {
        m_bottom_comm = m_default_comm;
    }

    m_do_agglomeration = agged;
    m_do_consolidation = coned;

    if (verbose > 1) {
        if (agged) {
            amrex::Print() << "MLLinOp::defineGrids(): agglomerated AMR level 0 starting at MG level "
                    << agg_lev << " of " << m_num_mg_levels[0] << "\n";
        } else if (coned) {
            amrex::Print() << "MLLinOp::defineGrids(): consolidated AMR level 0 starting at MG level "
                    << con_lev << " of " << m_num_mg_levels[0]
                    << " (ratio = " << info.con_ratio << ")" << "\n";
        } else {
            amrex::Print() << "MLLinOp::defineGrids(): no agglomeration or consolidation of AMR level 0\n";
        }
    }

    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev)
    {
        for (int mglev = 1; mglev < m_num_mg_levels[amrlev]; ++mglev)
        {
            m_factory[amrlev].emplace_back(makeFactory(amrlev,mglev));
        }
    }

    for (int amrlev = 1; amrlev < m_num_amr_levels; ++amrlev)
    {
        AMREX_ASSERT_WITH_MESSAGE(m_grids[amrlev][0].coarsenable(m_amr_ref_ratio[amrlev-1]),
                                  "MLLinOp: grids not coarsenable between AMR levels");
    }
}

void
MLLinOpMFV::defineBC ()
{
    m_needs_coarse_data_for_bc = !m_domain_covered[0];

    levelbc_raii.resize(m_num_amr_levels);
    robin_a_raii.resize(m_num_amr_levels);
    robin_b_raii.resize(m_num_amr_levels);
    robin_f_raii.resize(m_num_amr_levels);
}

void
MLLinOpMFV::setDomainBC (const amrex::Array<BCType,AMREX_SPACEDIM>& a_lobc,
                           const amrex::Array<BCType,AMREX_SPACEDIM>& a_hibc) noexcept
{
    const int ncomp = MLLinOpMFV::getNComp();
    setDomainBC(amrex::Vector<amrex::Array<BCType,AMREX_SPACEDIM> >(ncomp,a_lobc),
                amrex::Vector<amrex::Array<BCType,AMREX_SPACEDIM> >(ncomp,a_hibc));
}

void
MLLinOpMFV::setDomainBC (const amrex::Vector<amrex::Array<BCType,AMREX_SPACEDIM> >& a_lobc,
                           const amrex::Vector<amrex::Array<BCType,AMREX_SPACEDIM> >& a_hibc) noexcept
{
    const int ncomp = getNComp();
    AMREX_ASSERT_WITH_MESSAGE(ncomp == a_lobc.size() && ncomp == a_hibc.size(),
                              "MLLinOp::setDomainBC: wrong size");
    m_lobc = a_lobc;
    m_hibc = a_hibc;
    m_lobc_orig = m_lobc;
    m_hibc_orig = m_hibc;
    for (int icomp = 0; icomp < ncomp; ++icomp) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            if (m_geom[0][0].isPeriodic(idim)) {
                AMREX_ALWAYS_ASSERT(m_lobc[icomp][idim] == BCType::Periodic &&
                                    m_hibc[icomp][idim] == BCType::Periodic);
            } else {
                AMREX_ALWAYS_ASSERT(m_lobc[icomp][idim] != BCType::Periodic &&
                                    m_hibc[icomp][idim] != BCType::Periodic);
            }

            if (m_lobc[icomp][idim] == BCType::inhomogNeumann ||
                m_lobc[icomp][idim] == BCType::Robin)
            {
                m_lobc[icomp][idim] = BCType::Neumann;
            }

            if (m_hibc[icomp][idim] == BCType::inhomogNeumann ||
                m_hibc[icomp][idim] == BCType::Robin)
            {
                m_hibc[icomp][idim] = BCType::Neumann;
            }
        }
    }

    if (hasHiddenDimension()) {
        const int hd = hiddenDirection();
        for (int n = 0; n < ncomp; ++n) {
            m_lobc[n][hd] = BCType::Neumann;
            m_hibc[n][hd] = BCType::Neumann;
        }
    }

    if (hasInhomogNeumannBC() && !supportInhomogNeumannBC()) {
        amrex::Abort("Inhomogeneous Neumann BC not supported");
    }
    if (hasRobinBC() && !supportRobinBC()) {
        amrex::Abort("Robin BC not supported");
    }
}

bool
MLLinOpMFV::hasBC (BCType bct) const noexcept
{
    int ncomp = m_lobc_orig.size();
    for (int n = 0; n < ncomp; ++n) {
        for (int idim = 0; idim <AMREX_SPACEDIM; ++idim) {
            if (m_lobc_orig[n][idim] == bct || m_hibc_orig[n][idim] == bct) {
                return true;
            }
        }
    }
    return false;
}

bool
MLLinOpMFV::hasInhomogNeumannBC () const noexcept
{
    return hasBC(BCType::inhomogNeumann);
}

bool
MLLinOpMFV::hasRobinBC () const noexcept
{
    return hasBC(BCType::Robin);
}

amrex::Box
MLLinOpMFV::compactify (amrex::Box const& b) const noexcept
{
#if (AMREX_SPACEDIM == 3)
    if (info.hasHiddenDimension()) {
        const auto& lo = b.smallEnd();
        const auto& hi = b.bigEnd();
        if (info.hidden_direction == 0) {
            return amrex::Box(amrex::IntVect(lo[1],lo[2],0), amrex::IntVect(hi[1],hi[2],0), b.ixType());
        } else if (info.hidden_direction == 1) {
            return amrex::Box(amrex::IntVect(lo[0],lo[2],0), amrex::IntVect(hi[0],hi[2],0), b.ixType());
        } else {
            return amrex::Box(amrex::IntVect(lo[0],lo[1],0), amrex::IntVect(hi[0],hi[1],0), b.ixType());
        }
    } else
#endif
    {
        return b;
    }
}

void
MLLinOpMFV::makeAgglomeratedDMap (const amrex::Vector<amrex::BoxArray>& ba,
                                  amrex::Vector<amrex::DistributionMapping>& dm)
{
    BL_PROFILE("MLLinOp::makeAgglomeratedDMap");

    BL_ASSERT(!dm[0].empty());
    for (int i = 1, N=static_cast<int>(ba.size()); i < N; ++i)
    {
        if (dm[i].empty())
        {
            const std::vector< std::vector<int> >& sfc = amrex::DistributionMapping::makeSFC(ba[i]);

            const int nprocs = amrex::ParallelContext::NProcsSub();
            AMREX_ASSERT(static_cast<int>(sfc.size()) == nprocs);

            amrex::Vector<int> pmap(ba[i].size());
            for (int iproc = 0; iproc < nprocs; ++iproc) {
                int grank = amrex::ParallelContext::local_to_global_rank(iproc);
                for (int ibox : sfc[iproc]) {
                    pmap[ibox] = grank;
                }
            }
            dm[i].define(std::move(pmap));
        }
    }
}

void
MLLinOpMFV::makeConsolidatedDMap (const amrex::Vector<amrex::BoxArray>& ba,
                                  amrex::Vector<amrex::DistributionMapping>& dm,
                                  int ratio, int strategy)
{
    BL_PROFILE("MLLinOp::makeConsolidatedDMap()");

    int factor = 1;
    BL_ASSERT(!dm[0].empty());
    for (int i = 1, N=static_cast<int>(ba.size()); i < N; ++i)
    {
        if (dm[i].empty())
        {
            factor *= ratio;

            const int nprocs = amrex::ParallelContext::NProcsSub();
            const auto& pmap_fine = dm[i-1].ProcessorMap();
            amrex::Vector<int> pmap(pmap_fine.size());
            amrex::ParallelContext::global_to_local_rank(pmap.data(), pmap_fine.data(), static_cast<int>(pmap.size()));
            if (strategy == 1) {
                for (auto& x: pmap) {
                    x /= ratio;
                }
            } else if (strategy == 2) {
                int nprocs_con = static_cast<int>(std::ceil(static_cast<amrex::Real>(nprocs)
                                                            / static_cast<amrex::Real>(factor)));
                for (auto& x: pmap) {
                    auto d = std::div(x,nprocs_con);
                    x = d.rem;
                }
            } else if (strategy == 3) {
                if (factor == ratio) {
                    const std::vector< std::vector<int> >& sfc = amrex::DistributionMapping::makeSFC(ba[i]);
                    for (int iproc = 0; iproc < nprocs; ++iproc) {
                        for (int ibox : sfc[iproc]) {
                            pmap[ibox] = iproc;
                        }
                    }
                }
                for (auto& x: pmap) {
                    x /= ratio;
                }
            }

            if (amrex::ParallelContext::CommunicatorSub() == amrex::ParallelDescriptor::Communicator()) {
                dm[i].define(std::move(pmap));
            } else {
                amrex::Vector<int> pmap_g(pmap.size());
                amrex::ParallelContext::local_to_global_rank(pmap_g.data(), pmap.data(), static_cast<int>(pmap.size()));
                dm[i].define(std::move(pmap_g));
            }
        }
    }
}

MPI_Comm
MLLinOpMFV::makeSubCommunicator (const amrex::DistributionMapping& dm)
{
    BL_PROFILE("MLLinOp::makeSubCommunicator()");

#ifdef BL_USE_MPI

    amrex::Vector<int> newgrp_ranks = dm.ProcessorMap();
    std::sort(newgrp_ranks.begin(), newgrp_ranks.end());
    auto last = std::unique(newgrp_ranks.begin(), newgrp_ranks.end());
    newgrp_ranks.erase(last, newgrp_ranks.end());

    MPI_Comm newcomm;
    MPI_Group defgrp, newgrp;
    MPI_Comm_group(m_default_comm, &defgrp);
    if (amrex::ParallelContext::CommunicatorSub() == amrex::ParallelDescriptor::Communicator()) {
        MPI_Group_incl(defgrp, static_cast<int>(newgrp_ranks.size()), newgrp_ranks.data(), &newgrp);
    } else {
        amrex::Vector<int> local_newgrp_ranks(newgrp_ranks.size());
        amrex::ParallelContext::global_to_local_rank(local_newgrp_ranks.data(),
                                              newgrp_ranks.data(), static_cast<int>(newgrp_ranks.size()));
        MPI_Group_incl(defgrp, static_cast<int>(local_newgrp_ranks.size()), local_newgrp_ranks.data(), &newgrp);
    }

    MPI_Comm_create(m_default_comm, newgrp, &newcomm);

    m_raii_comm = std::make_unique<CommContainer>(newcomm);

    MPI_Group_free(&defgrp);
    MPI_Group_free(&newgrp);

    return newcomm;
#else
    amrex::ignore_unused(dm);
    return m_default_comm;
#endif
}

void
MLLinOpMFV::setDomainBCLoc (const amrex::Array<amrex::Real,AMREX_SPACEDIM>& lo_bcloc,
                              const amrex::Array<amrex::Real,AMREX_SPACEDIM>& hi_bcloc) noexcept
{
    m_domain_bloc_lo = lo_bcloc;
    m_domain_bloc_hi = hi_bcloc;
}

void
MLLinOpMFV::setCoarseFineBC (const MFSet* crse, int crse_ratio) noexcept
{
    m_coarse_data_for_bc = crse;
    m_coarse_data_crse_ratio = crse_ratio;
}

template <typename AMF, std::enable_if_t<!std::is_same_v<MFSet,AMF>,int>>
void
MLLinOpMFV::setCoarseFineBC (const AMF* crse, int crse_ratio) noexcept
{
    m_coarse_data_for_bc_raii = MFSet(crse->boxArray(), crse->DistributionMap(),
                                   crse->nComp(), crse->nGrowVect());
    m_coarse_data_for_bc_raii.LocalCopy(*crse, 0, 0, crse->nComp(),
                                        crse->nGrowVect());
    m_coarse_data_for_bc = &m_coarse_data_for_bc_raii;
    m_coarse_data_crse_ratio = crse_ratio;
}

void
MLLinOpMFV::make (amrex::Vector<amrex::Vector<MFSet> >& mf, amrex::IntVect const& ng) const
{
    mf.clear();
    mf.resize(m_num_amr_levels);
    for (int alev = 0; alev < m_num_amr_levels; ++alev) {
        mf[alev].resize(m_num_mg_levels[alev]);
        for (int mlev = 0; mlev < m_num_mg_levels[alev]; ++mlev) {
            mf[alev][mlev] = make(alev, mlev, ng);
        }
    }
}

MFSet
MLLinOpMFV::make (int amrlev, int mglev, amrex::IntVect const& ng) const
{
    return amrex::MultiFab(amrex::convert(m_grids[amrlev][mglev], m_ixtype),
              m_dmap[amrlev][mglev], getNComp(), ng, amrex::MFInfo(),
              *m_factory[amrlev][mglev]);
}

// MFSet
// MLLinOpMFV::makeAlias (MFSet const& mf) const
// {
//     return MFSet(mf, amrex::make_alias, 0, 1);//mf.nComp());
// }

MFSet
MLLinOpMFV::makeCoarseMG (int amrlev, int mglev, amrex::IntVect const& ng) const
{
    amrex::BoxArray cba = m_grids[amrlev][mglev];
    amrex::IntVect ratio = (amrlev > 0) ? amrex::IntVect(2) : mg_coarsen_ratio_vec[mglev];
    cba.coarsen(ratio);
    // cba.convert(m_ixtype);
    return MFSet(cba, m_dmap[amrlev][mglev], m_ixtype_set, ng);

}

MFSet
MLLinOpMFV::makeCoarseAmr (int famrlev, amrex::IntVect const& ng) const
{
    amrex::BoxArray cba = m_grids[famrlev][0];
    amrex::IntVect ratio(AMRRefRatio(famrlev-1));
    cba.coarsen(ratio);
    cba.convert(m_ixtype);
    return MFSet(cba, m_dmap[famrlev][0], getNComp(), ng);
}

void
MLLinOpMFV::resizeMultiGrid (int new_size)
{
    if (new_size <= 0 || new_size >= m_num_mg_levels[0]) { return; }

    m_num_mg_levels[0] = new_size;

    m_geom[0].resize(new_size);
    m_grids[0].resize(new_size);
    m_dmap[0].resize(new_size);
    m_factory[0].resize(new_size);

    if (m_bottom_comm != m_default_comm) {
        m_bottom_comm = makeSubCommunicator(m_dmap[0].back());
    }
}

void
MLLinOpMFV::avgDownResMG (int clev, MFSet& cres, MFSet const& fres) const
{
    const int ncomp = this->getNComp();
    if constexpr (amrex::IsFabArray<MFSet>::value) {
#ifdef AMREX_USE_EB
        if (!fres.isAllRegular()) {
            amrex::EB_average_down(fres, cres, 0, ncomp, mg_coarsen_ratio_vec[clev-1]);
        } else
#endif
        {
            amrex::average_down(fres, cres, 0, ncomp, mg_coarsen_ratio_vec[clev-1]);
        }
    } else {
        amrex::Abort("For non-FabArray, MLLinOpMFV::avgDownResMG should be overridden.");
    }
}

bool
MLLinOpMFV::isMFIterSafe (int amrlev, int mglev1, int mglev2) const
{
    return m_dmap[amrlev][mglev1] == m_dmap[amrlev][mglev2]
        && amrex::BoxArray::SameRefs(m_grids[amrlev][mglev1], m_grids[amrlev][mglev2]);
}

template <typename AMF, std::enable_if_t<!std::is_same_v<MFSet,AMF>,int>>
void
MLLinOpMFV::setLevelBC (int amrlev, const AMF* levelbcdata,
                          const AMF* robinbc_a, const AMF* robinbc_b,
                          const AMF* robinbc_f)
{
    const int ncomp = this->getNComp();
    if (levelbcdata) {
        levelbc_raii[amrlev] = std::make_unique<MFSet>(levelbcdata->boxArray(),
                                                    levelbcdata->DistributionMap(),
                                                    ncomp, levelbcdata->nGrowVect());
        levelbc_raii[amrlev]->LocalCopy(*levelbcdata, 0, 0, ncomp,
                                        levelbcdata->nGrowVect());
    } else {
        levelbc_raii[amrlev].reset();
    }

    if (robinbc_a) {
        robin_a_raii[amrlev] = std::make_unique<MFSet>(robinbc_a->boxArray(),
                                                    robinbc_a->DistributionMap(),
                                                    ncomp, robinbc_a->nGrowVect());
        robin_a_raii[amrlev]->LocalCopy(*robinbc_a, 0, 0, ncomp,
                                        robinbc_a->nGrowVect());
    } else {
        robin_a_raii[amrlev].reset();
    }

    if (robinbc_b) {
        robin_b_raii[amrlev] = std::make_unique<MFSet>(robinbc_b->boxArray(),
                                                    robinbc_b->DistributionMap(),
                                                    ncomp, robinbc_b->nGrowVect());
        robin_b_raii[amrlev]->LocalCopy(*robinbc_b, 0, 0, ncomp,
                                        robinbc_b->nGrowVect());
    } else {
        robin_b_raii[amrlev].reset();
    }

    if (robinbc_f) {
        robin_f_raii[amrlev] = std::make_unique<MFSet>(robinbc_f->boxArray(),
                                                    robinbc_f->DistributionMap(),
                                                    ncomp, robinbc_f->nGrowVect());
        robin_f_raii[amrlev]->LocalCopy(*robinbc_f, 0, 0, ncomp,
                                        robinbc_f->nGrowVect());
    } else {
        robin_f_raii[amrlev].reset();
    }

    this->setLevelBC(amrlev, levelbc_raii[amrlev].get(), robin_a_raii[amrlev].get(),
                     robin_b_raii[amrlev].get(), robin_f_raii[amrlev].get());
}


}