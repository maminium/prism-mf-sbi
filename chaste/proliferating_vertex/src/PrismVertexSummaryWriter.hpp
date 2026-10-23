#ifndef PRISM_VERTEX_SUMMARY_WRITER_HPP_
#define PRISM_VERTEX_SUMMARY_WRITER_HPP_

#include "KursawePaperSummary.hpp"
#include "PrismVertexTissueConfig.hpp"
#include "ModifiedVertexBasedCellPopulation.hpp"
#include "PolygonNumberCorrelationWriter.hpp"

#include <chrono>
#include <string>

/** Write prism_vertex_summary.json and optional fig2_validation_report.json. */
class PrismVertexSummaryWriter
{
public:
    static void WriteSummaryJson(
        const std::string& output_dir,
        ModifiedVertexBasedCellPopulation<2>& rPopulation,
        const PrismVertexTissueConfig& rConfig,
        double line_tension,
        double perimeter_contractility,
        int random_seed,
        double actual_end_time,
        std::chrono::steady_clock::duration wall_time);

    static void WriteFig2ValidationReport(
        const std::string& output_dir,
        double line_tension,
        double perimeter_contractility,
        const KursawePaperSummary& rSummary,
        double l2_error_area_ratios);

    /** Write final vertex mesh as JSON for Fig. 2 morphology panels (fig2_final_mesh.json). */
    static void WriteFig2MeshSnapshot(
        const std::string& output_dir,
        ModifiedVertexBasedCellPopulation<2>& rPopulation,
        const std::string& label,
        double line_tension,
        double perimeter_contractility);
};

#endif /*PRISM_VERTEX_SUMMARY_WRITER_HPP_*/
