#ifndef TESTYANVERTEXVALIDATION_HPP_
#define TESTYANVERTEXVALIDATION_HPP_

#include <cxxtest/TestSuite.h>
#include "CheckpointArchiveTypes.hpp"
#include "AbstractCellBasedTestSuite.hpp"
#include "PetscSetupAndFinalize.hpp"
#include "SimulationTime.hpp"

#include "YanVertexSimulationSetup.hpp"
#include "YanVertexTissueConfig.hpp"
#include "YanEpithelialGeometrySummary.hpp"
#include "YanAnisotropicSpringForce.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

/**
 * Scientific validation for Yan/Ogita Model-B quench in Chaste.
 *
 * Constitutive (generative): T = λ0(1 + a_r cos 2θ) - b0 ℓ, λ0=1, A0=1.
 *
 * With λ0=1, free-boundary quench requires area elasticity k = O(1–10) so
 * contractility does not drive T2 cell removal (calibrated: N conserved for k≳2).
 *
 * Morphological gates reject needle/spaghetti tissues (physics checks).
 */
class TestYanVertexValidation : public AbstractCellBasedTestSuite
{
private:
    struct RunResult
    {
        YanEpithelialGeometrySummary summary;
        unsigned n_start;
        unsigned n_end;
        std::string output_dir;
        double median_edge;
        double max_edge;
        double min_area;
        double max_area;
        double bbox_aspect;
    };

    static void MeshMetrics(ModifiedVertexBasedCellPopulation<2>& rPopulation,
                            double& median_edge,
                            double& max_edge,
                            double& min_area,
                            double& max_area,
                            double& bbox_aspect)
    {
        MutableVertexMesh<2, 2>& r_mesh =
            static_cast<MutableVertexMesh<2, 2>&>(rPopulation.rGetMesh());

        std::vector<double> edges;
        std::vector<double> areas;
        double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;

        for (unsigned i = 0; i < r_mesh.GetNumNodes(); ++i)
        {
            const c_vector<double, 2>& p = r_mesh.GetNode(i)->rGetLocation();
            xmin = std::min(xmin, p[0]);
            xmax = std::max(xmax, p[0]);
            ymin = std::min(ymin, p[1]);
            ymax = std::max(ymax, p[1]);
        }
        const double w = std::max(1e-12, xmax - xmin);
        const double h = std::max(1e-12, ymax - ymin);
        bbox_aspect = std::max(w / h, h / w);

        for (unsigned e = 0; e < r_mesh.GetNumElements(); ++e)
        {
            VertexElement<2, 2>* p_elem = r_mesh.GetElement(e);
            areas.push_back(r_mesh.GetVolumeOfElement(e));
            const unsigned n = p_elem->GetNumNodes();
            for (unsigned j = 0; j < n; ++j)
            {
                c_vector<double, 2> a = p_elem->GetNode(j)->rGetLocation();
                c_vector<double, 2> b = p_elem->GetNode((j + 1) % n)->rGetLocation();
                edges.push_back(norm_2(b - a));
            }
        }

        std::sort(edges.begin(), edges.end());
        std::sort(areas.begin(), areas.end());
        median_edge = edges.empty() ? 0.0 : edges[edges.size() / 2];
        max_edge = edges.empty() ? 0.0 : edges.back();
        min_area = areas.empty() ? 0.0 : areas.front();
        max_area = areas.empty() ? 0.0 : areas.back();
    }

    RunResult RunScenario(const std::string& label,
                          double a_r,
                          double b_0,
                          double k,
                          double end_time = 3.0,
                          unsigned initial_size = 5u)
    {
        SimulationTime::Instance()->Destroy();
        SimulationTime::Instance()->SetStartTime(0.0);
        RandomNumberGenerator::Instance()->Reseed(17u);

        YanVertexTissueConfig config = YanVertexTissueConfig::ValidationProfile();
        config.end_time = end_time;
        config.initial_size = initial_size;
        config.disable_adaptive_dt = true;
        config.dt = 0.001;
        config.write_mesh_snapshot = true;

        YanMechanicalParameters mech;
        mech.a_r = a_r;
        mech.b_0 = b_0;
        mech.k = k;

        YanMeshHolder mesh_holder(config);
        std::vector<CellPtr> cells;
        YanCreateQuenchCells(mesh_holder.mpMesh, cells);

        ModifiedVertexBasedCellPopulation<2> cell_population(*mesh_holder.mpMesh, cells);
        cell_population.SetRestrictVertexMovementBoolean(config.restrict_vertex_movement);
        cell_population.SetWriteCellVtkResults(false);
        cell_population.SetWriteEdgeVtkResults(false);

        const unsigned n_start = cell_population.GetNumRealCells();
        const std::string output_dir = "YanVertexValidation/" + label;

        YanRunQuenchSimulation(cell_population, output_dir, mech, config, 17, label);

        RunResult result;
        result.summary = YanEpithelialGeometrySummary::Compute(cell_population);
        result.n_start = n_start;
        result.n_end = cell_population.GetNumRealCells();
        result.output_dir = output_dir;
        MeshMetrics(cell_population, result.median_edge, result.max_edge,
                    result.min_area, result.max_area, result.bbox_aspect);

        const bool quench_ok = (result.n_start == result.n_end);
        YanEpithelialGeometrySummary::WriteValidationReport(
            output_dir, label, result.summary, a_r, b_0, k,
            result.n_start, result.n_end, quench_ok);
        return result;
    }

    static bool AllFinite(const YanEpithelialGeometrySummary& s)
    {
        for (unsigned i = 0; i < YanEpithelialGeometrySummary::DIM_S; ++i)
        {
            if (!std::isfinite(s.s[i]))
            {
                return false;
            }
        }
        return true;
    }

    /** Reject needle / spaghetti meshes and T2 tissue collapse. */
    static void AssertPhysicalEpithelium(const RunResult& r)
    {
        TS_ASSERT_EQUALS(r.n_start, r.n_end);
        TS_ASSERT(AllFinite(r.summary));
        // Honeycomb init edge ~0.62 for A0=1; allow mild stretch only.
        TS_ASSERT(r.median_edge < 1.8);
        TS_ASSERT(r.max_edge < 4.0 * std::max(0.2, r.median_edge));
        TS_ASSERT(r.min_area > 0.25);
        TS_ASSERT(r.max_area < 4.0);
        TS_ASSERT(r.bbox_aspect < 3.0);
        TS_ASSERT(r.summary.n_internal_cells >= 1u);
    }

public:
    void TestForceTensionClosedForm()
    {
        EXIT_IF_PARALLEL;
        YanAnisotropicSpringForce<2> force;
        force.SetBaselineLineTension(1.0);
        force.SetAnisotropyMagnitude(0.25);
        force.SetLengthSpringStiffness(0.15);
        force.SetAnisotropyAngle(0.0);

        const double ell = 0.62;
        const double theta = 0.0;
        // T = 1*(1 + 0.25*cos0) - 0.15*0.62 = 1.25 - 0.093 = 1.157
        const double expected = 1.0 * (1.0 + 0.25 * std::cos(0.0)) - 0.15 * ell;
        TS_ASSERT_DELTA(force.ComputeEdgeTension(ell, theta), expected, 1e-12);
        // Must stay contractile on honeycomb vertical edge too.
        const double t_vert = force.ComputeEdgeTension(ell, 0.5 * M_PI);
        TS_ASSERT(t_vert > 0.2);
    }

    void TestV0bIsotropicBaselinePhysical()
    {
        EXIT_IF_PARALLEL;
        // a_r=0, b0=0 => T=λ0=1 isotropic; k large enough vs λ0 to conserve N.
        RunResult r = RunScenario("V0b_isotropic_baseline", 0.0, 0.0, 5.0, 3.0, 5u);
        AssertPhysicalEpithelium(r);
        TS_ASSERT(std::fabs(r.summary.s[2]) < 0.25);
    }

    void TestV0QuenchThetaStarPhysical()
    {
        EXIT_IF_PARALLEL;
        // Stage-A θ*: T>0 on honeycomb; k balances λ0=1 contractility.
        RunResult r = RunScenario("V0_smoke", 0.25, 0.15, 5.0, 3.0, 5u);
        AssertPhysicalEpithelium(r);
    }

    void TestV1IsotropicNematicNearZero()
    {
        EXIT_IF_PARALLEL;
        RunResult iso = RunScenario("V1_isotropic", 0.0, 0.15, 5.0, 3.0, 5u);
        AssertPhysicalEpithelium(iso);
        TS_ASSERT(std::fabs(iso.summary.s[2]) < 0.30);
    }

    void TestV2AnisotropyMovesNematicPhysically()
    {
        EXIT_IF_PARALLEL;
        RunResult iso = RunScenario("V2_iso", 0.0, 0.15, 5.0, 3.0, 5u);
        RunResult ani = RunScenario("V2_ani", 0.35, 0.15, 5.0, 3.0, 5u);

        AssertPhysicalEpithelium(iso);
        AssertPhysicalEpithelium(ani);

        // Empirically (and consistently with φa=0): raising a_r increases mean_c2
        // under this summary definition while staying morphologically epithelial.
        TS_ASSERT(ani.summary.s[2] > iso.summary.s[2] + 0.05);
        TS_ASSERT(std::fabs(ani.summary.s[2] - iso.summary.s[2]) > 0.05);
        TS_ASSERT(ani.summary.s[2] < 0.85);
        TS_ASSERT(ani.median_edge < 1.5);
    }

    void TestV3AreaStiffnessMovesMeanLogArea()
    {
        EXIT_IF_PARALLEL;
        RunResult soft = RunScenario("V3_soft_k", 0.25, 0.15, 2.5, 3.0, 5u);
        RunResult stiff = RunScenario("V3_stiff_k", 0.25, 0.15, 8.0, 3.0, 5u);

        AssertPhysicalEpithelium(soft);
        AssertPhysicalEpithelium(stiff);
        const double d_mean = std::fabs(soft.summary.s[5] - stiff.summary.s[5]);
        const double d_std = std::fabs(soft.summary.s[6] - stiff.summary.s[6]);
        TS_ASSERT(d_mean + d_std > 1e-4);
    }

    void TestV4LengthSpringMovesMeanLogEll()
    {
        EXIT_IF_PARALLEL;
        RunResult weak = RunScenario("V4_weak_b0", 0.25, 0.05, 5.0, 3.0, 5u);
        RunResult strong = RunScenario("V4_strong_b0", 0.25, 0.30, 5.0, 3.0, 5u);

        AssertPhysicalEpithelium(weak);
        AssertPhysicalEpithelium(strong);
        const double d_mean = std::fabs(weak.summary.s[0] - strong.summary.s[0]);
        const double d_std = std::fabs(weak.summary.s[1] - strong.summary.s[1]);
        TS_ASSERT(d_mean + d_std > 1e-4);
    }
};

#endif /*TESTYANVERTEXVALIDATION_HPP_*/
