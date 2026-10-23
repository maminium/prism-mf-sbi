#ifndef YANEPITHELIALGEOMETRYSUMMARY_HPP_
#define YANEPITHELIALGEOMETRYSUMMARY_HPP_

#include "ModifiedVertexBasedCellPopulation.hpp"
#include "YanVertexTissueConfig.hpp"

#include <array>
#include <chrono>
#include <string>
#include <vector>

/**
 * 8D Yan/VDat-compatible geometry summary after area nondimensionalisation.
 *
 * Indices / names:
 *  0 mean_log_ell, 1 std_log_ell, 2 mean_c2, 3 mean_s2,
 *  4 ell_c2_cov, 5 mean_log_A, 6 std_log_A, 7 frac_hex
 */
struct YanEpithelialGeometrySummary
{
    static constexpr unsigned DIM_S = 8u;

    std::array<double, DIM_S> s;
    unsigned n_cells;
    unsigned n_internal_cells;
    unsigned n_internal_edges;
    double median_area;
    double length_scale;

    YanEpithelialGeometrySummary();

    static const char* const S_NAMES[DIM_S];

    static YanEpithelialGeometrySummary Compute(ModifiedVertexBasedCellPopulation<2>& rPopulation);

    static void WriteSummaryJson(
        const std::string& output_dir,
        ModifiedVertexBasedCellPopulation<2>& rPopulation,
        const YanVertexTissueConfig& rConfig,
        double a_r,
        double b_0,
        double k,
        int random_seed,
        std::chrono::steady_clock::duration wall_time);

    static void WriteValidationReport(
        const std::string& output_dir,
        const std::string& scenario,
        const YanEpithelialGeometrySummary& rSummary,
        double a_r,
        double b_0,
        double k,
        unsigned n_cells_start,
        unsigned n_cells_end,
        bool quench_ok);

    /** Vertex mesh snapshot for validation figures (nodes + element connectivity). */
    static void WriteMeshSnapshot(
        const std::string& output_dir,
        ModifiedVertexBasedCellPopulation<2>& rPopulation,
        const std::string& filename,
        const std::string& label,
        double a_r,
        double b_0,
        double k);
};



#endif /*YANEPITHELIALGEOMETRYSUMMARY_HPP_*/
