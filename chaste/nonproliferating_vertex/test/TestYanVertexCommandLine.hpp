#ifndef TESTYANVERTEXCOMMANDLINE_HPP_
#define TESTYANVERTEXCOMMANDLINE_HPP_

#include <cxxtest/TestSuite.h>
#include "CheckpointArchiveTypes.hpp"
#include "AbstractCellBasedTestSuite.hpp"
#include "PetscSetupAndFinalize.hpp"
#include "CommandLineArguments.hpp"

#include "YanVertexSimulationSetup.hpp"
#include "YanVertexTissueConfig.hpp"

#include <string>

/**
 * CLI driver for Yan epithelial vertex quench.
 *
 *   -opt1 a_r
 *   -opt2 b_0
 *   -opt3 k
 *   -opt4 sim_id
 *   -opt5 RNG seed
 */
class TestYanVertexCommandLine : public AbstractCellBasedTestSuite
{
public:
    void TestRunYanQuenchFromCommandLine()
    {
        EXIT_IF_PARALLEL;

        const double a_r = CommandLineArguments::Instance()->GetDoubleCorrespondingToOption("-opt1");
        const double b_0 = CommandLineArguments::Instance()->GetDoubleCorrespondingToOption("-opt2");
        const double k = CommandLineArguments::Instance()->GetDoubleCorrespondingToOption("-opt3");
        const unsigned long opt5_seed = std::stoul(
            CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt5"));
        const int random_seed = static_cast<int>((opt5_seed % 1999999999UL) + 1UL);

        YanVertexTissueConfig config = YanVertexTissueConfig::FromEnvironment();
        RandomNumberGenerator::Instance()->Reseed(static_cast<unsigned>(random_seed));

        YanMechanicalParameters mech;
        mech.a_r = a_r;
        mech.b_0 = b_0;
        mech.k = k;

        const std::string default_out = "YanVertexQuench/_Sim_"
            + CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt4")
            + "_ar_" + CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt1")
            + "_b0_" + CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt2")
            + "_k_" + CommandLineArguments::Instance()->GetStringCorrespondingToOption("-opt3");
        const std::string output_dir = YanVertexTissueConfig::GetOutputSubdir(default_out);

        // Init mode: "voronoi" opts into the disordered start; anything else
        // (default) uses the legacy honeycomb path unchanged.
        if (config.init_mode == "voronoi")
        {
            YanVoronoiMeshHolder mesh_holder(config);
            std::vector<CellPtr> cells;
            YanCreateQuenchCells(mesh_holder.mpMesh, cells);

            ModifiedVertexBasedCellPopulation<2> cell_population(*mesh_holder.mpMesh, cells);
            cell_population.SetRestrictVertexMovementBoolean(config.restrict_vertex_movement);
            cell_population.SetWriteCellVtkResults(false);
            cell_population.SetWriteEdgeVtkResults(false);

            YanRunQuenchSimulation(cell_population, output_dir, mech, config, random_seed);
        }
        else
        {
            YanMeshHolder mesh_holder(config);
            std::vector<CellPtr> cells;
            YanCreateQuenchCells(mesh_holder.mpMesh, cells);

            ModifiedVertexBasedCellPopulation<2> cell_population(*mesh_holder.mpMesh, cells);
            cell_population.SetRestrictVertexMovementBoolean(config.restrict_vertex_movement);
            cell_population.SetWriteCellVtkResults(false);
            cell_population.SetWriteEdgeVtkResults(false);

            YanRunQuenchSimulation(cell_population, output_dir, mech, config, random_seed);
        }
    }
};

#endif /*TESTYANVERTEXCOMMANDLINE_HPP_*/
