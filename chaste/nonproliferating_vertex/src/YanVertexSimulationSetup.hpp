#ifndef YANVERTEXSIMULATIONSETUP_HPP_
#define YANVERTEXSIMULATIONSETUP_HPP_

#include "YanVertexTissueConfig.hpp"
#include "YanEpithelialGeometrySummary.hpp"
#include "YanAnisotropicSpringForce.hpp"
#include "YanProgressModifier.hpp"

#include "OffLatticeSimulation.hpp"
#include "NoCellCycleModel.hpp"
#include "DifferentiatedCellProliferativeType.hpp"
#include "WildTypeCellMutationState.hpp"
#include "ForwardEulerNumericalMethod.hpp"
#include "SmartPointers.hpp"
#include "ModifiedVertexBasedCellPopulation.hpp"
#include "ExtendedHoneycombVertexMeshGenerator.hpp"
#include "FasterMutableVertexMesh.hpp"
#include "MutableVertexMesh.hpp"
#include "VoronoiVertexMeshGenerator.hpp"
#include "RandomNumberGenerator.hpp"
#include <boost/shared_ptr.hpp>

#include <chrono>
#include <string>
#include <vector>

struct YanMechanicalParameters
{
    double a_r;
    double b_0;
    double k;
    double anisotropy_angle;

    YanMechanicalParameters()
        : a_r(0.25),
          b_0(0.15),
          k(5.0),
          anisotropy_angle(0.0)
    {
    }
};

/**
 * Displace every (non-deleted) node by U(-scale, scale) in x and y to break
 * lattice symmetry. Shared by honeycomb-jitter and voronoi init paths.
 */
inline void YanApplyVertexJitter(MutableVertexMesh<2, 2>* p_mesh, double amp)
{
    if (amp <= 0.0 || p_mesh == nullptr || p_mesh->GetNumNodes() == 0)
    {
        return;
    }
    // Honeycomb / voronoi with target area ~1 has edge length O(1); 0.5 scale.
    const double scale = 0.5 * amp;
    RandomNumberGenerator* p_gen = RandomNumberGenerator::Instance();
    for (unsigned i = 0; i < p_mesh->GetNumNodes(); ++i)
    {
        Node<2>* p_node = p_mesh->GetNode(i);
        if (p_node->IsDeleted())
        {
            continue;
        }
        c_vector<double, 2> loc = p_node->rGetLocation();
        loc[0] += scale * (2.0 * p_gen->ranf() - 1.0);
        loc[1] += scale * (2.0 * p_gen->ranf() - 1.0);
        p_node->rGetModifiableLocation() = loc;
    }
}

/**
 * Owns ExtendedHoneycombVertexMeshGenerator for the simulation lifetime.
 *
 * This is the LEGACY / default init path (MFNNKB_YAN_INIT_MODE unset or
 * "honeycomb"). Its behaviour is unchanged; the Voronoi option lives in the
 * separate YanVoronoiMeshHolder below so honeycomb runs are never affected.
 */
class YanMeshHolder
{
public:
    ExtendedHoneycombVertexMeshGenerator* mpGenerator;
    FasterMutableVertexMesh<2, 2>* mpMesh;

    YanMeshHolder(const YanVertexTissueConfig& rConfig)
        : mpGenerator(new ExtendedHoneycombVertexMeshGenerator(
              rConfig.initial_size,
              rConfig.initial_size,
              false,
              rConfig.t1_swap_threshold,
              rConfig.t2_swap_threshold,
              1.0)),
          mpMesh(nullptr)
    {
        mpMesh = mpGenerator->GetMesh();
        mpMesh->SetRandomizeT1SwapOrderBoolean(rConfig.randomize_t1_swap_order);
        mpMesh->SetCellRearrangementRatio(rConfig.new_edge_length_factor);
        mpMesh->SetCheckForInternalIntersections(rConfig.check_internal_intersections);
        // T3 aborts on concave boundary under strong Yan spring/anisotropy; keep T1/T2 only for Stage A.
        mpMesh->SetCheckForT3Swaps(false);

        // Break perfect hex symmetry (optional; MFNNKB_YAN_INIT_VERTEX_NOISE).
        YanApplyVertexJitter(mpMesh, rConfig.init_vertex_noise);
    }

    ~YanMeshHolder()
    {
        delete mpGenerator;
    }
};

/**
 * Owns a Lloyd-relaxed Poisson-Voronoi initial mesh (MFNNKB_YAN_INIT_MODE
 * = "voronoi"). Opt-in only; the honeycomb path above is untouched.
 *
 * Produces a disordered start (frac_hex well below the ~0.78 hex attractor,
 * elevated std_log_A) to test whether Yan dynamics preserve or erase disorder
 * across short quench times, and whether k moves std_log_A first.
 *
 * VoronoiVertexMeshGenerator yields a plain MutableVertexMesh<2,2>, which
 * ModifiedVertexBasedCellPopulation accepts directly (FasterMutableVertexMesh
 * is used only by the honeycomb path). Standard Chaste T1/T2 remeshing applies.
 */
class YanVoronoiMeshHolder
{
public:
    VoronoiVertexMeshGenerator* mpGenerator;
    boost::shared_ptr<MutableVertexMesh<2, 2> > mpMeshOwner;
    MutableVertexMesh<2, 2>* mpMesh;

    YanVoronoiMeshHolder(const YanVertexTissueConfig& rConfig)
        : mpGenerator(nullptr),
          mpMesh(nullptr)
    {
        // Target area 1.0 matches the honeycomb A0 scale so summaries are comparable.
        mpGenerator = new VoronoiVertexMeshGenerator(
            rConfig.initial_size,
            rConfig.initial_size,
            rConfig.voronoi_relaxation_steps,
            1.0);
        mpMeshOwner = mpGenerator->GetMeshAfterReMesh();
        mpMesh = mpMeshOwner.get();

        mpMesh->SetCellRearrangementThreshold(rConfig.t1_swap_threshold);
        mpMesh->SetT2Threshold(rConfig.t2_swap_threshold);
        mpMesh->SetCellRearrangementRatio(rConfig.new_edge_length_factor);
        mpMesh->SetCheckForInternalIntersections(rConfig.check_internal_intersections);
        mpMesh->SetCheckForT3Swaps(false);

        // Optional extra jitter on top of Voronoi disorder.
        YanApplyVertexJitter(mpMesh, rConfig.init_vertex_noise);
    }

    ~YanVoronoiMeshHolder()
    {
        delete mpGenerator;
    }
};

inline void YanCreateQuenchCells(
    MutableVertexMesh<2, 2>* p_mesh,
    std::vector<CellPtr>& rCells)
{
    MAKE_PTR(WildTypeCellMutationState, p_state);
    MAKE_PTR(DifferentiatedCellProliferativeType, p_diff_type);

    for (unsigned elem_index = 0; elem_index < p_mesh->GetNumElements(); ++elem_index)
    {
        NoCellCycleModel* p_cc_model = new NoCellCycleModel();
        p_cc_model->SetDimension(2);

        CellPtr p_cell(new Cell(p_state, p_cc_model));
        p_cell->SetCellProliferativeType(p_diff_type);
        p_cell->SetBirthTime(0.0);
        rCells.push_back(p_cell);
    }
}

inline void YanRunQuenchSimulation(
    ModifiedVertexBasedCellPopulation<2>& rPopulation,
    const std::string& output_dir,
    const YanMechanicalParameters& rMech,
    const YanVertexTissueConfig& rConfig,
    int random_seed,
    const std::string& progress_label = "yan")
{
    std::cout << "[" << progress_label << "] START  a_r=" << rMech.a_r
              << "  b_0=" << rMech.b_0
              << "  k=" << rMech.k
              << "  seed=" << random_seed
              << "  dt=" << rConfig.dt
              << "  end_time=" << rConfig.end_time
              << "  N0=" << rPopulation.GetNumRealCells()
              << std::endl;

    OffLatticeSimulation<2> simulator(rPopulation);
    simulator.SetOutputDirectory(output_dir);
    simulator.SetSamplingTimestepMultiple(rConfig.sampling_timestep_multiple);
    simulator.SetDt(rConfig.dt);
    simulator.SetEndTime(rConfig.end_time);

    MAKE_PTR(YanAnisotropicSpringForce<2>, p_force);
    p_force->SetYanModelBParameters(rMech.a_r, rMech.b_0, rMech.k, rMech.anisotropy_angle);
    simulator.AddForce(p_force);

    // Quench: no TargetAreaLinearGrowthModifier (Yan P=-kA has no A0 growth).

    boost::shared_ptr<YanProgressModifier<2> > p_progress(
        new YanProgressModifier<2>(progress_label, 500u));
    simulator.AddSimulationModifier(p_progress);

    boost::shared_ptr<AbstractNumericalMethod<2, 2> > p_method(new ForwardEulerNumericalMethod<2, 2>());
    p_method->SetUseAdaptiveTimestep(!rConfig.disable_adaptive_dt);
    simulator.SetNumericalMethod(p_method);

    if (rConfig.write_mesh_snapshot)
    {
        YanEpithelialGeometrySummary::WriteMeshSnapshot(
            output_dir, rPopulation, "mesh_init.json", "init",
            rMech.a_r, rMech.b_0, rMech.k);
    }

    const auto t0 = std::chrono::steady_clock::now();
    simulator.Solve();
    const auto t1 = std::chrono::steady_clock::now();

    const double wall = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    std::cout << "[" << progress_label << "] DONE  wall=" << wall << "s  N_end="
              << rPopulation.GetNumRealCells() << std::endl;

    YanEpithelialGeometrySummary::WriteSummaryJson(
        output_dir,
        rPopulation,
        rConfig,
        rMech.a_r,
        rMech.b_0,
        rMech.k,
        random_seed,
        t1 - t0);

    if (rConfig.write_mesh_snapshot)
    {
        YanEpithelialGeometrySummary::WriteMeshSnapshot(
            output_dir, rPopulation, "mesh_final.json", "final",
            rMech.a_r, rMech.b_0, rMech.k);
    }
}

#endif /*YANVERTEXSIMULATIONSETUP_HPP_*/
