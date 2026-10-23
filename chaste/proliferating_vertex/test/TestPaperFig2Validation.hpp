/*

Copyright (c) 2005-2021, University of Oxford.
All rights reserved.

*/

#ifndef TESTPAPERFIG2VALIDATION_HPP_
#define TESTPAPERFIG2VALIDATION_HPP_

#include <cxxtest/TestSuite.h>
#include "CheckpointArchiveTypes.hpp"
#include "AbstractCellBasedTestSuite.hpp"
#include "PetscSetupAndFinalize.hpp"
#include "PrismVertexSimulationSetup.hpp"
#include "PrismVertexTissueConfig.hpp"
#include "KursawePaperSummary.hpp"
#include "PrismVertexSummaryWriter.hpp"

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>

/**
 * Fig. 2 reference points (Kursawe / Farhadifar comparison).
 * Long-running: use ctest --timeout 0.
 *
 * Run a single point only (Chaste binary always registers 3 CxxTest methods):
 *   export PRISM_VERTEX_FIG2_ONLY=C
 *   ./TestPaperFig2Validation
 */
class TestPaperFig2Validation : public AbstractCellBasedTestSuite
{
private:
    static bool ShouldRunFig2Point(const char* label)
    {
        const char* only = std::getenv("PRISM_VERTEX_FIG2_ONLY");
        if (only == nullptr || only[0] == '\0')
        {
            return true;
        }
        const std::string want(only);
        return (want == label || want == std::string(1, label[0]));
    }

    void RunFig2Point(double lambda, double gamma, const std::string& label, int random_seed)
    {
        const std::string run_tag = "Fig2-" + label;
        std::cout << "\n========== " << run_tag << "  Lambda=" << lambda
                  << " Gamma=" << gamma << "  seed=" << random_seed
                  << " ==========" << std::endl;

        PrismVertexTissueConfig config = PrismVertexTissueConfig::Table1Profile(true);

        TissueVertexMechanicalParameters mech;
        mech.line_tension = lambda;
        mech.perimeter_contractility = gamma;

        std::cout << "[" << run_tag << "] configure cell-cycle times..." << std::endl;
        TissueVertexConfigureCellCycleTimes(random_seed, mech.average_cell_cycle_time);

        std::cout << "[" << run_tag << "] build mesh..." << std::endl;
        TissueVertexMeshHolder mesh_holder(mech, config);

        std::cout << "[" << run_tag << "] create cells..." << std::endl;
        std::vector<CellPtr> cells;
        TissueVertexCreateCells(mesh_holder.mpMesh, cells, mech);

        std::cout << "[" << run_tag << "] build cell population (N0="
                  << cells.size() << ")..." << std::endl;
        ModifiedVertexBasedCellPopulation<2> cell_population(*mesh_holder.mpMesh, cells);
        cell_population.SetRestrictVertexMovementBoolean(mech.restrict_vertex_movement);
        TissueVertexAddPopulationWriters(cell_population, true);
        cell_population.SetWriteCellVtkResults(false);
        cell_population.SetWriteEdgeVtkResults(false);
        TissueVertexSetTargetAreas(cell_population);

        std::ostringstream out_name;
        out_name << "Fig2Validation_" << label << "_L" << lambda << "_G" << gamma;
        const std::string output_dir = PrismVertexTissueConfig::GetOutputSubdir(out_name.str());

        TissueVertexRunSimulation(cell_population, output_dir, mech, config, random_seed, run_tag);

        PrismVertexSummaryWriter::WriteFig2MeshSnapshot(
            output_dir, cell_population, label, lambda, gamma);

        const KursawePaperSummary summary = KursawePaperSummary::Compute(cell_population);
        const double l2_error = -1.0;
        PrismVertexSummaryWriter::WriteFig2ValidationReport(output_dir, lambda, gamma, summary, l2_error);

        std::cout << "[" << run_tag << "] FINISHED  bulk_cells="
                  << summary.GetNumBulkCells()
                  << "  mean_elong=" << summary.GetMeanElongation()
                  << std::endl;

        // bulk_cells can be 0 in smoke tests with very small end_time; allow that.
        const char* end_time_env = std::getenv("PRISM_VERTEX_END_TIME");
        const double end_time_val = end_time_env ? std::stod(end_time_env) : 700.0;
        if (end_time_val >= 100.0)
        {
            TS_ASSERT(summary.GetNumBulkCells() > 0u);
        }
    }

public:
    void TestFig2PointReferenceA()
    {
        EXIT_IF_PARALLEL;
        if (!ShouldRunFig2Point("A"))
        {
            return;
        }
        RunFig2Point(0.12, 0.04, "A", 1);
    }

    void TestFig2PointReferenceB()
    {
        EXIT_IF_PARALLEL;
        if (!ShouldRunFig2Point("B"))
        {
            return;
        }
        RunFig2Point(0.0, 0.1, "B", 2);
    }

    void TestFig2PointReferenceC()
    {
        EXIT_IF_PARALLEL;
        if (!ShouldRunFig2Point("C"))
        {
            return;
        }
        RunFig2Point(-0.85, 0.1, "C", 3);
    }
};

#endif /*TESTPAPERFIG2VALIDATION_HPP_*/
