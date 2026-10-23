#include "YanEpithelialGeometrySummary.hpp"

#include "OutputFileHandler.hpp"
#include "MutableVertexMesh.hpp"
#include "SimulationTime.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <utility>

const char* const YanEpithelialGeometrySummary::S_NAMES[YanEpithelialGeometrySummary::DIM_S] = {
    "mean_log_ell",
    "std_log_ell",
    "mean_c2",
    "mean_s2",
    "ell_c2_cov",
    "mean_log_A",
    "std_log_A",
    "frac_hex"
};

YanEpithelialGeometrySummary::YanEpithelialGeometrySummary()
    : n_cells(0u),
      n_internal_cells(0u),
      n_internal_edges(0u),
      median_area(0.0),
      length_scale(1.0)
{
    s.fill(0.0);
}

namespace
{
bool EdgeIsBoundary(Node<2>* pNodeA, Node<2>* pNodeB)
{
    std::set<unsigned> a = pNodeA->rGetContainingElementIndices();
    std::set<unsigned> b = pNodeB->rGetContainingElementIndices();
    std::set<unsigned> shared;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::inserter(shared, shared.begin()));
    return shared.size() <= 1u;
}

bool ElementIsInternal(VertexElement<2, 2>* pElement)
{
    const unsigned n = pElement->GetNumNodes();
    for (unsigned i = 0; i < n; ++i)
    {
        Node<2>* p_a = pElement->GetNode(i);
        Node<2>* p_b = pElement->GetNode((i + 1) % n);
        if (EdgeIsBoundary(p_a, p_b))
        {
            return false;
        }
    }
    return true;
}

double MedianOf(std::vector<double> values)
{
    if (values.empty())
    {
        return 1.0;
    }
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n % 2u == 1u)
    {
        return values[n / 2u];
    }
    return 0.5 * (values[n / 2u - 1u] + values[n / 2u]);
}

double StdSample(const std::vector<double>& values, double mean)
{
    if (values.size() < 2u)
    {
        return 0.0;
    }
    double acc = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
    {
        const double d = values[i] - mean;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(values.size() - 1u));
}
} // namespace

YanEpithelialGeometrySummary YanEpithelialGeometrySummary::Compute(
    ModifiedVertexBasedCellPopulation<2>& rPopulation)
{
    YanEpithelialGeometrySummary out;
    MutableVertexMesh<2, 2>& r_mesh =
        static_cast<MutableVertexMesh<2, 2>&>(rPopulation.rGetMesh());

    out.n_cells = rPopulation.GetNumRealCells();

    std::vector<double> areas;
    std::vector<unsigned> internal_elem_indices;
    unsigned n_hex = 0u;

    for (typename VertexMesh<2, 2>::VertexElementIterator elem_iter = r_mesh.GetElementIteratorBegin();
         elem_iter != r_mesh.GetElementIteratorEnd();
         ++elem_iter)
    {
        VertexElement<2, 2>* p_elem = &(*elem_iter);
        if (!ElementIsInternal(p_elem))
        {
            continue;
        }
        const unsigned idx = p_elem->GetIndex();
        const double area = r_mesh.GetVolumeOfElement(idx);
        if (!(area > 0.0) || !std::isfinite(area))
        {
            continue;
        }
        areas.push_back(area);
        internal_elem_indices.push_back(idx);
        if (p_elem->GetNumNodes() == 6u)
        {
            ++n_hex;
        }
    }

    out.n_internal_cells = static_cast<unsigned>(internal_elem_indices.size());
    out.median_area = MedianOf(areas);
    out.length_scale = std::sqrt(std::max(out.median_area, 1e-16));

    // Unique internal edges via sorted node-index pairs.
    std::map<std::pair<unsigned, unsigned>, std::pair<double, double> > edges; // -> (ell, theta)

    for (size_t e = 0; e < internal_elem_indices.size(); ++e)
    {
        VertexElement<2, 2>* p_elem = r_mesh.GetElement(internal_elem_indices[e]);
        const unsigned n = p_elem->GetNumNodes();
        for (unsigned i = 0; i < n; ++i)
        {
            Node<2>* p_a = p_elem->GetNode(i);
            Node<2>* p_b = p_elem->GetNode((i + 1) % n);
            if (EdgeIsBoundary(p_a, p_b))
            {
                continue;
            }
            unsigned ia = p_a->GetIndex();
            unsigned ib = p_b->GetIndex();
            if (ia > ib)
            {
                std::swap(ia, ib);
            }
            const std::pair<unsigned, unsigned> key(ia, ib);
            if (edges.find(key) != edges.end())
            {
                continue;
            }
            c_vector<double, 2> vec = p_b->rGetLocation() - p_a->rGetLocation();
            const double ell = norm_2(vec) / out.length_scale;
            const double theta = std::atan2(vec[1], vec[0]);
            edges[key] = std::make_pair(ell, theta);
        }
    }

    out.n_internal_edges = static_cast<unsigned>(edges.size());

    std::vector<double> log_ell;
    std::vector<double> c2;
    std::vector<double> s2;
    std::vector<double> ell_nd;
    log_ell.reserve(edges.size());
    c2.reserve(edges.size());
    s2.reserve(edges.size());
    ell_nd.reserve(edges.size());

    for (std::map<std::pair<unsigned, unsigned>, std::pair<double, double> >::const_iterator it = edges.begin();
         it != edges.end();
         ++it)
    {
        const double ell = it->second.first;
        const double theta = it->second.second;
        if (!(ell > 0.0) || !std::isfinite(ell))
        {
            continue;
        }
        ell_nd.push_back(ell);
        log_ell.push_back(std::log(ell));
        c2.push_back(std::cos(2.0 * theta));
        s2.push_back(std::sin(2.0 * theta));
    }

    std::vector<double> log_A;
    log_A.reserve(areas.size());
    for (size_t i = 0; i < areas.size(); ++i)
    {
        const double a_nd = areas[i] / std::max(out.median_area, 1e-16);
        log_A.push_back(std::log(std::max(a_nd, 1e-16)));
    }

    auto mean_of = [](const std::vector<double>& v) -> double {
        if (v.empty())
        {
            return 0.0;
        }
        double acc = 0.0;
        for (size_t i = 0; i < v.size(); ++i)
        {
            acc += v[i];
        }
        return acc / static_cast<double>(v.size());
    };

    const double mean_log_ell = mean_of(log_ell);
    const double mean_c2 = mean_of(c2);
    const double mean_s2 = mean_of(s2);
    const double mean_ell = mean_of(ell_nd);
    const double mean_log_A = mean_of(log_A);

    double cov = 0.0;
    if (ell_nd.size() >= 2u)
    {
        double acc = 0.0;
        for (size_t i = 0; i < ell_nd.size(); ++i)
        {
            acc += (ell_nd[i] - mean_ell) * (c2[i] - mean_c2);
        }
        cov = acc / static_cast<double>(ell_nd.size() - 1u);
    }

    const double frac_hex = (out.n_internal_cells == 0u)
        ? 0.0
        : static_cast<double>(n_hex) / static_cast<double>(out.n_internal_cells);

    out.s[0] = mean_log_ell;
    out.s[1] = StdSample(log_ell, mean_log_ell);
    out.s[2] = mean_c2;
    out.s[3] = mean_s2;
    out.s[4] = cov;
    out.s[5] = mean_log_A;
    out.s[6] = StdSample(log_A, mean_log_A);
    out.s[7] = frac_hex;
    return out;
}

void YanEpithelialGeometrySummary::WriteSummaryJson(
    const std::string& output_dir,
    ModifiedVertexBasedCellPopulation<2>& rPopulation,
    const YanVertexTissueConfig& rConfig,
    double a_r,
    double b_0,
    double k,
    int random_seed,
    std::chrono::steady_clock::duration wall_time)
{
    const YanEpithelialGeometrySummary summary = Compute(rPopulation);
    const double wall_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(wall_time).count();

    OutputFileHandler handler(output_dir, false);
    out_stream out = handler.OpenOutputFile("prism_vertex_summary.json");

    (*out) << std::setprecision(10);
    (*out) << "{\n";
    (*out) << "  \"model\": \"YanAnisotropicSpringForce\",\n";
    (*out) << "  \"end_time\": " << rConfig.end_time << ",\n";
    (*out) << "  \"simulation_time\": " << SimulationTime::Instance()->GetTime() << ",\n";
    (*out) << "  \"n_cells\": " << summary.n_cells << ",\n";
    (*out) << "  \"n_internal_cells\": " << summary.n_internal_cells << ",\n";
    (*out) << "  \"n_internal_edges\": " << summary.n_internal_edges << ",\n";
    (*out) << "  \"a_r\": " << a_r << ",\n";
    (*out) << "  \"b_0\": " << b_0 << ",\n";
    (*out) << "  \"k\": " << k << ",\n";
    (*out) << "  \"random_seed\": " << random_seed << ",\n";
    (*out) << "  \"dt\": " << rConfig.dt << ",\n";
    (*out) << "  \"sampling_timestep_multiple\": " << rConfig.sampling_timestep_multiple << ",\n";
    (*out) << "  \"minimal_io\": " << (rConfig.minimal_io ? "true" : "false") << ",\n";
    (*out) << "  \"wall_seconds\": " << wall_seconds << ",\n";
    (*out) << "  \"median_area\": " << summary.median_area << ",\n";
    (*out) << "  \"length_scale\": " << summary.length_scale << ",\n";
    (*out) << "  \"s_names\": [";
    for (unsigned i = 0; i < DIM_S; ++i)
    {
        (*out) << "\"" << S_NAMES[i] << "\"";
        if (i + 1u < DIM_S)
        {
            (*out) << ", ";
        }
    }
    (*out) << "],\n";
    (*out) << "  \"s\": [";
    for (unsigned i = 0; i < DIM_S; ++i)
    {
        (*out) << summary.s[i];
        if (i + 1u < DIM_S)
        {
            (*out) << ", ";
        }
    }
    (*out) << "]\n";
    (*out) << "}\n";
    out->close();
}

void YanEpithelialGeometrySummary::WriteValidationReport(
    const std::string& output_dir,
    const std::string& scenario,
    const YanEpithelialGeometrySummary& rSummary,
    double a_r,
    double b_0,
    double k,
    unsigned n_cells_start,
    unsigned n_cells_end,
    bool quench_ok)
{
    OutputFileHandler handler(output_dir, false);
    out_stream out = handler.OpenOutputFile("yan_validation_report.json");

    (*out) << std::setprecision(10);
    (*out) << "{\n";
    (*out) << "  \"scenario\": \"" << scenario << "\",\n";
    (*out) << "  \"a_r\": " << a_r << ",\n";
    (*out) << "  \"b_0\": " << b_0 << ",\n";
    (*out) << "  \"k\": " << k << ",\n";
    (*out) << "  \"n_cells_start\": " << n_cells_start << ",\n";
    (*out) << "  \"n_cells_end\": " << n_cells_end << ",\n";
    (*out) << "  \"quench_ok\": " << (quench_ok ? "true" : "false") << ",\n";
    (*out) << "  \"n_internal_cells\": " << rSummary.n_internal_cells << ",\n";
    (*out) << "  \"n_internal_edges\": " << rSummary.n_internal_edges << ",\n";
    (*out) << "  \"s_names\": [";
    for (unsigned i = 0; i < DIM_S; ++i)
    {
        (*out) << "\"" << S_NAMES[i] << "\"";
        if (i + 1u < DIM_S)
        {
            (*out) << ", ";
        }
    }
    (*out) << "],\n";
    (*out) << "  \"s\": [";
    for (unsigned i = 0; i < DIM_S; ++i)
    {
        (*out) << rSummary.s[i];
        if (i + 1u < DIM_S)
        {
            (*out) << ", ";
        }
    }
    (*out) << "]\n";
    (*out) << "}\n";
    out->close();
}


void YanEpithelialGeometrySummary::WriteMeshSnapshot(
    const std::string& output_dir,
    ModifiedVertexBasedCellPopulation<2>& rPopulation,
    const std::string& filename,
    const std::string& label,
    double a_r,
    double b_0,
    double k)
{
    MutableVertexMesh<2, 2>& r_mesh =
        static_cast<MutableVertexMesh<2, 2>&>(rPopulation.rGetMesh());

    OutputFileHandler handler(output_dir, false);
    out_stream out = handler.OpenOutputFile(filename);

    (*out) << std::setprecision(10);
    (*out) << "{\n";
    (*out) << "  \"label\": \"" << label << "\",\n";
    (*out) << "  \"a_r\": " << a_r << ",\n";
    (*out) << "  \"b_0\": " << b_0 << ",\n";
    (*out) << "  \"k\": " << k << ",\n";
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
