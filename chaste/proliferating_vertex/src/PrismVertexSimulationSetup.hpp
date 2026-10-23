#ifndef PRISMVERTEXSIMULATIONSETUP_HPP_
#define PRISMVERTEXSIMULATIONSETUP_HPP_

#include "PrismVertexTissueConfig.hpp"
#include "PrismVertexSummaryWriter.hpp"
#include "TissueVertexProgressModifier.hpp"

#include "OffLatticeSimulation.hpp"
#include "TargetAreaLinearGrowthModifier.hpp"
#include "FarhadifarForce.hpp"
#include "FixedSequenceCellCycleModel.hpp"
#include "TransitCellProliferativeType.hpp"
#include "WildTypeCellMutationState.hpp"
#include "ForwardEulerNumericalMethod.hpp"
#include "SmartPointers.hpp"
#include "ModifiedVertexBasedCellPopulation.hpp"
#include "CellCycleTimesGenerator.hpp"
#include "ExtendedHoneycombVertexMeshGenerator.hpp"
#include "FasterMutableVertexMesh.hpp"

#include "VertexModelDataWriter.hpp"
#include "CellProliferativePhasesWriter.hpp"
#include "CellAgesWriter.hpp"
#include "CellEdgeCountWriter.hpp"
#include "CellPerimeterWriter.hpp"
#include "FarhadifarForceWriter.hpp"
#include "PolygonNumberCorrelationWriter.hpp"
#include "AreaCorrelationWriter.hpp"
#include "NeighbourNumberCorrelationWriter.hpp"
#include "VertexEdgeLengthWriter.hpp"

#include <chrono>
#include <ctime>
#include <vector>

/** Shared mechanical / mesh defaults (Kursawe Table 1). */
struct TissueVertexMechanicalParameters
{
    unsigned number_generations;
    double average_cell_cycle_time;
    double new_edge_length_factor;
    bool restrict_vertex_movement;
    bool randomise_t1_swap_order;
    double t1_swap_threshold;
    double t2_swap_threshold;
    unsigned initial_size;
    double line_tension;
    double perimeter_contractility;

    TissueVertexMechanicalParameters()
        : number_generations(7u),
          average_cell_cycle_time(20.0),
          new_edge_length_factor(1.5),
          restrict_vertex_movement(true),
          randomise_t1_swap_order(false),
          t1_swap_threshold(0.01),
          t2_swap_threshold(0.001),
          initial_size(2u),
          line_tension(0.0),
          perimeter_contractility(0.0)
    {
    }
};

inline void TissueVertexConfigureCellCycleTimes(int random_seed, double average_cell_cycle_time)
{
    // Destroy any existing singleton so rate/seed can be reset between runs.
    CellCycleTimesGenerator::Destroy();
    CellCycleTimesGenerator* p_gen = CellCycleTimesGenerator::Instance();
    if (random_seed == 0)
    {
        p_gen->SetRandomSeed(static_cast<unsigned>(time(NULL)));
    }
    else
    {
        p_gen->SetRandomSeed(static_cast<unsigned>(random_seed));
    }
    p_gen->SetRate(3.0 / (2.0 * average_cell_cycle_time));
    p_gen->GenerateCellCycleTimeSequence();
}

/**
 * Owns ExtendedHoneycombVertexMeshGenerator for the simulation lifetime.
 * (Returning only GetMesh() from a stack generator caused use-after-free / SEGV.)
 */
class TissueVertexMeshHolder
{
public:
    ExtendedHoneycombVertexMeshGenerator* mpGenerator;
    FasterMutableVertexMesh<2, 2>* mpMesh;

    TissueVertexMeshHolder(const TissueVertexMechanicalParameters& rMech, const PrismVertexTissueConfig& rConfig)
        : mpGenerator(new ExtendedHoneycombVertexMeshGenerator(
              rMech.initial_size,
              rMech.initial_size,
              false,
              rMech.t1_swap_threshold,
              rMech.t2_swap_threshold,
              1.0)),
          mpMesh(nullptr)
    {
        mpMesh = mpGenerator->GetMesh();
        mpMesh->SetRandomizeT1SwapOrderBoolean(rMech.randomise_t1_swap_order);
        mpMesh->SetCellRearrangementRatio(rMech.new_edge_length_factor);
        mpMesh->SetCheckForInternalIntersections(
            rConfig.GetCheckForInternalIntersections(rMech.line_tension));
        mpMesh->SetCheckForT3Swaps(true);
    }

    ~TissueVertexMeshHolder()
    {
        delete mpGenerator;
    }
};

inline void TissueVertexCreateCells(
    FasterMutableVertexMesh<2, 2>* p_mesh,
    std::vector<CellPtr>& rCells,
    const TissueVertexMechanicalParameters& rMech)
{
    MAKE_PTR(WildTypeCellMutationState, p_state);
    MAKE_PTR(TransitCellProliferativeType, p_diff_type);

    for (unsigned elem_index = 0; elem_index < p_mesh->GetNumElements(); ++elem_index)
    {
        FixedSequenceCellCycleModel* p_cc_model = new FixedSequenceCellCycleModel();
        p_cc_model->SetDimension(2);
        p_cc_model->SetG2Duration((1.0 / 3.0) * rMech.average_cell_cycle_time);
        p_cc_model->SetMDuration(1e-12);
        p_cc_model->SetSDuration(1e-12);
        p_cc_model->SetMaxTransitGenerations(rMech.number_generations);

        CellPtr p_cell(new Cell(p_state, p_cc_model));
        p_cell->SetCellProliferativeType(p_diff_type);
        p_cell->SetBirthTime(0.0);
        rCells.push_back(p_cell);
    }
}

inline void TissueVertexAddPopulationWriters(ModifiedVertexBasedCellPopulation<2>& rPopulation, bool minimal_io)
{
    if (minimal_io)
    {
        return;
    }

    rPopulation.AddCellWriter<VertexModelDataWriter>();
    rPopulation.AddCellWriter<CellProliferativePhasesWriter>();
    rPopulation.AddCellWriter<CellAgesWriter>();
    rPopulation.AddCellWriter<CellEdgeCountWriter>();
    rPopulation.AddCellWriter<CellPerimeterWriter>();
    rPopulation.AddCellPopulationCountWriter<FarhadifarForceWriter>();
    rPopulation.AddCellPopulationCountWriter<AreaCorrelationWriter>();
    rPopulation.AddCellPopulationCountWriter<PolygonNumberCorrelationWriter>();
    rPopulation.AddCellPopulationCountWriter<NeighbourNumberCorrelationWriter>();
    rPopulation.AddPopulationWriter<VertexEdgeLengthWriter>();
}

inline void TissueVertexSetTargetAreas(ModifiedVertexBasedCellPopulation<2>& rPopulation, double initial_target_area = 1.0)
{
    for (typename AbstractCellPopulation<2>::Iterator cell_iter = rPopulation.Begin();
         cell_iter != rPopulation.End();
         ++cell_iter)
    {
        cell_iter->GetCellData()->SetItem("target area", initial_target_area);
    }
}

inline void TissueVertexRunSimulation(
    ModifiedVertexBasedCellPopulation<2>& rPopulation,
    const std::string& output_dir,
    const TissueVertexMechanicalParameters& rMech,
    const PrismVertexTissueConfig& rConfig,
    int random_seed,
    const std::string& progress_label = "run")
{
    const double actual_end_time = rConfig.GetEndTime();

    std::cout << "[" << progress_label << "] START  Lambda=" << rMech.line_tension
              << "  Gamma=" << rMech.perimeter_contractility
              << "  seed=" << random_seed
              << "  dt=" << rConfig.GetDt()
              << "  sampling=" << rConfig.GetSamplingTimestepMultiple()
              << "  end_time=" << actual_end_time
              << "  paper_mode=" << (rConfig.paper_mode ? 1 : 0)
              << "  minimal_io=" << (rConfig.minimal_io ? 1 : 0)
              << "  N0=" << rPopulation.GetNumRealCells()
              << std::endl;

    OffLatticeSimulation<2> simulator(rPopulation);
    simulator.SetOutputDirectory(output_dir);
    simulator.SetSamplingTimestepMultiple(rConfig.GetSamplingTimestepMultiple());
    simulator.SetDt(rConfig.GetDt());
    simulator.SetEndTime(actual_end_time);

    MAKE_PTR(FarhadifarForce<2>, p_force);
    p_force->SetPerimeterContractilityParameter(rMech.perimeter_contractility);
    p_force->SetLineTensionParameter(rMech.line_tension);
    if (p_force->GetLineTensionParameter() < 0.0)
    {
        p_force->SetBoundaryLineTensionParameter(0.0);
    }
    else
    {
        p_force->SetBoundaryLineTensionParameter(rMech.line_tension);
    }
    simulator.AddForce(p_force);

    MAKE_PTR(TargetAreaLinearGrowthModifier<2>, p_growth_modifier);
    simulator.AddSimulationModifier(p_growth_modifier);

    // Progress: print every 500 dt-steps (≈ 5 sim-time units in paper mode → ~140 prints / 700-unit run).
    boost::shared_ptr<TissueVertexProgressModifier<2>> p_progress(new TissueVertexProgressModifier<2>(progress_label, 500u));
    simulator.AddSimulationModifier(p_progress);

    boost::shared_ptr<AbstractNumericalMethod<2, 2>> p_method(new ForwardEulerNumericalMethod<2, 2>());
    p_method->SetUseAdaptiveTimestep(!rConfig.disable_adaptive_dt);
    simulator.SetNumericalMethod(p_method);

    const auto t0 = std::chrono::steady_clock::now();
    simulator.Solve();
    const auto t1 = std::chrono::steady_clock::now();

    const double wall = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    std::cout << "[" << progress_label << "] DONE  wall=" << wall << "s  N_end="
              << rPopulation.GetNumRealCells() << std::endl;

    PrismVertexSummaryWriter::WriteSummaryJson(
        output_dir,
        rPopulation,
        rConfig,
        rMech.line_tension,
        rMech.perimeter_contractility,
        random_seed,
        actual_end_time,
        t1 - t0);
}

#endif /*PRISMVERTEXSIMULATIONSETUP_HPP_*/
