/*

Copyright (c) 2005-2021, University of Oxford.
All rights reserved.

*/

#ifndef TESTPAPERVERTEXSIMULATION_HPP_
#define TESTPAPERVERTEXSIMULATION_HPP_

#include <cxxtest/TestSuite.h>
#include "CheckpointArchiveTypes.hpp"
#include "AbstractCellBasedTestSuite.hpp"
#include "PetscSetupAndFinalize.hpp"
#include "CommandLineArguments.hpp"
#include "PrismVertexSimulationSetup.hpp"
#include "PrismVertexTissueConfig.hpp"

/**
 * Command-line vertex driver for the proliferating tissue model.
 * Requires -opt1..-opt5 (lambda, gamma, run id, sim id, RNG seed).
 */
class TestPaperCommandLineSpeedSimulation : public AbstractCellBasedTestSuite
{
public:
    void TestRunningProliferatingVertexFromCommandLine()
    {
        EXIT_IF_PARALLEL;

        const double lambda = CommandLineArguments::Instance()->GetDoubleCorrespondingToOption("-opt1");
        const double gamma = CommandLineArguments::Instance()->GetDoubleCorrespondingToOption("-opt2");
        // -opt5 is the per-replicate RNG seed from Python. Use it directly.
        // Do not multiply by sim/run ids: large products overflow 32-bit int and
        // collapse distinct replicates onto the same seed.
        const unsigned long opt5_seed = std::stoul(
            CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt5"));
        const int random_seed = static_cast<int>((opt5_seed % 1999999999UL) + 1UL);

        PrismVertexTissueConfig config = PrismVertexTissueConfig::FromEnvironment();

        TissueVertexMechanicalParameters mech;
        mech.line_tension = lambda;
        mech.perimeter_contractility = gamma;

        TissueVertexConfigureCellCycleTimes(random_seed, mech.average_cell_cycle_time);
        TissueVertexMeshHolder mesh_holder(mech, config);
        std::vector<CellPtr> cells;
        TissueVertexCreateCells(mesh_holder.mpMesh, cells, mech);

        ModifiedVertexBasedCellPopulation<2> cell_population(*mesh_holder.mpMesh, cells);
        cell_population.SetRestrictVertexMovementBoolean(mech.restrict_vertex_movement);
        TissueVertexAddPopulationWriters(cell_population, config.minimal_io);
        cell_population.SetWriteCellVtkResults(false);
        cell_population.SetWriteEdgeVtkResults(false);
        TissueVertexSetTargetAreas(cell_population);

        const std::string default_out = "TestBayesianCommandLineRun/_Sim_Number_"
            + CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt4")
            + "Lambda__" + CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt1")
            + "_Gamma_" + CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt2")
            + "_Run_" + CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt3");
        const std::string output_dir = PrismVertexTissueConfig::GetOutputSubdir(default_out);

        TissueVertexRunSimulation(cell_population, output_dir, mech, config, random_seed);
    }
};

#endif /*TESTPAPERVERTEXSIMULATION_HPP_*/
