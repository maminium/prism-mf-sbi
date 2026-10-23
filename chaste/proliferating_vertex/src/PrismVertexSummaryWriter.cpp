#include "PrismVertexSummaryWriter.hpp"

#include "OutputFileHandler.hpp"
#include "SimulationTime.hpp"
#include "MutableVertexMesh.hpp"

#include <iomanip>
#include <sstream>

void PrismVertexSummaryWriter::WriteSummaryJson(
    const std::string& output_dir,
    ModifiedVertexBasedCellPopulation<2>& rPopulation,
    const PrismVertexTissueConfig& rConfig,
    double line_tension,
    double perimeter_contractility,
    int random_seed,
    double actual_end_time,
    std::chrono::steady_clock::duration wall_time)
{
    const KursawePaperSummary paper_summary = KursawePaperSummary::Compute(rPopulation);
    const c_vector<double, 6> s = paper_summary.AsVector();

    c_vector<double, 2> poly_stats =
        PolygonNumberCorrelationWriter<2, 2>().GetMeanInternalPolygonNumberAndVariance(&rPopulation);

    const double wall_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(wall_time).count();

    OutputFileHandler handler(output_dir, false);
    out_stream out = handler.OpenOutputFile("prism_vertex_summary.json");

    (*out) << std::setprecision(10);
    (*out) << "{\n";
    (*out) << "  \"end_time\": " << actual_end_time << ",\n";
    (*out) << "  \"simulation_time\": " << SimulationTime::Instance()->GetTime() << ",\n";
    (*out) << "  \"n_cells\": " << rPopulation.GetNumRealCells() << ",\n";
    (*out) << "  \"n_bulk_cells\": " << paper_summary.GetNumBulkCells() << ",\n";
    (*out) << "  \"lambda\": " << line_tension << ",\n";
    (*out) << "  \"gamma\": " << perimeter_contractility << ",\n";
    (*out) << "  \"random_seed\": " << random_seed << ",\n";
    (*out) << "  \"dt\": " << rConfig.GetDt() << ",\n";
    (*out) << "  \"sampling_timestep_multiple\": " << rConfig.GetSamplingTimestepMultiple() << ",\n";
    (*out) << "  \"paper_mode\": " << (rConfig.paper_mode ? "true" : "false") << ",\n";
    (*out) << "  \"minimal_io\": " << (rConfig.minimal_io ? "true" : "false") << ",\n";
    (*out) << "  \"wall_seconds\": " << wall_seconds << ",\n";
    (*out) << "  \"mean_polygon_number\": " << poly_stats[0] << ",\n";
    (*out) << "  \"var_polygon_number\": " << poly_stats[1] << ",\n";
    (*out) << "  \"bulk_mean_area\": " << paper_summary.GetBulkMeanArea() << ",\n";
    (*out) << "  \"mean_elongation\": " << paper_summary.GetMeanElongation() << ",\n";
    for (unsigned n = 4u; n <= 8u; ++n)
    {
        auto it = paper_summary.area_ratio_by_polygon_class.find(n);
        const double ratio = (it != paper_summary.area_ratio_by_polygon_class.end()) ? it->second : 0.0;
        (*out) << "  \"area_ratio_n" << n << "\": " << ratio << ",\n";
    }
    (*out) << "  \"s\": [" << s[0] << ", " << s[1] << ", " << s[2] << ", " << s[3] << ", " << s[4] << ", " << s[5] << "]\n";
    (*out) << "}\n";
    out->close();
}

void PrismVertexSummaryWriter::WriteFig2ValidationReport(
    const std::string& output_dir,
    double line_tension,
    double perimeter_contractility,
    const KursawePaperSummary& rSummary,
    double l2_error_area_ratios)
{
    const c_vector<double, 6> s = rSummary.AsVector();

    OutputFileHandler handler(output_dir, false);
    out_stream out = handler.OpenOutputFile("fig2_validation_report.json");

    (*out) << std::setprecision(10);
    (*out) << "{\n";
    (*out) << "  \"lambda\": " << line_tension << ",\n";
    (*out) << "  \"gamma\": " << perimeter_contractility << ",\n";
    (*out) << "  \"l2_error_area_ratios\": " << l2_error_area_ratios << ",\n";
    (*out) << "  \"s\": [" << s[0] << ", " << s[1] << ", " << s[2] << ", " << s[3] << ", " << s[4] << ", " << s[5] << "]\n";
    (*out) << "}\n";
    out->close();
}

void PrismVertexSummaryWriter::WriteFig2MeshSnapshot(
    const std::string& output_dir,
    ModifiedVertexBasedCellPopulation<2>& rPopulation,
    const std::string& label,
    double line_tension,
    double perimeter_contractility)
{
    MutableVertexMesh<2, 2>& r_mesh =
        static_cast<MutableVertexMesh<2, 2>&>(rPopulation.rGetMesh());

    OutputFileHandler handler(output_dir, false);
    out_stream out = handler.OpenOutputFile("fig2_final_mesh.json");

    (*out) << std::setprecision(10);
    (*out) << "{\n";
    (*out) << "  \"label\": \"" << label << "\",\n";
    (*out) << "  \"lambda\": " << line_tension << ",\n";
    (*out) << "  \"gamma\": " << perimeter_contractility << ",\n";
    (*out) << "  \"simulation_time\": " << SimulationTime::Instance()->GetTime() << ",\n";
    (*out) << "  \"n_nodes\": " << r_mesh.GetNumNodes() << ",\n";
    (*out) << "  \"n_elements\": " << r_mesh.GetNumElements() << ",\n";
    (*out) << "  \"nodes\": [\n";
    for (unsigned i = 0u; i < r_mesh.GetNumNodes(); ++i)
    {
        const c_vector<double, 2>& loc = r_mesh.GetNode(i)->rGetLocation();
        (*out) << "    [" << loc[0] << ", " << loc[1] << "]";
        (*out) << ((i + 1u < r_mesh.GetNumNodes()) ? ",\n" : "\n");
    }
    (*out) << "  ],\n";
    (*out) << "  \"elements\": [\n";
    for (unsigned e = 0u; e < r_mesh.GetNumElements(); ++e)
    {
        VertexElement<2, 2>* p_elem = r_mesh.GetElement(e);
        (*out) << "    [";
        for (unsigned j = 0u; j < p_elem->GetNumNodes(); ++j)
        {
            (*out) << p_elem->GetNodeGlobalIndex(j);
            if (j + 1u < p_elem->GetNumNodes())
            {
                (*out) << ", ";
            }
        }
        (*out) << "]";
        (*out) << ((e + 1u < r_mesh.GetNumElements()) ? ",\n" : "\n");
    }
    (*out) << "  ]\n";
    (*out) << "}\n";
    out->close();
}
