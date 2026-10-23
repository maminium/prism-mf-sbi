#include "KursawePaperSummary.hpp"

#include "VertexBasedCellPopulation.hpp"
#include "MutableVertexMesh.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>

using namespace boost::accumulators;

KursawePaperSummary::KursawePaperSummary()
    : m_mean_elongation(0.0),
      m_bulk_mean_area(0.0),
      m_n_bulk_cells(0u)
{
    for (unsigned n = 4u; n <= 8u; ++n)
    {
        area_ratio_by_polygon_class[n] = 0.0;
    }
}

double KursawePaperSummary::GetMeanElongation() const
{
    return m_mean_elongation;
}

double KursawePaperSummary::GetBulkMeanArea() const
{
    return m_bulk_mean_area;
}

unsigned KursawePaperSummary::GetNumBulkCells() const
{
    return m_n_bulk_cells;
}

c_vector<double, 6> KursawePaperSummary::AsVector() const
{
    c_vector<double, 6> s;
    for (unsigned i = 0u; i < 5u; ++i)
    {
        const unsigned polygon_class = 4u + i;
        auto it = area_ratio_by_polygon_class.find(polygon_class);
        s[i] = (it != area_ratio_by_polygon_class.end()) ? it->second : 0.0;
    }
    s[5] = m_mean_elongation;
    return s;
}

template<unsigned DIM>
KursawePaperSummary KursawePaperSummary::Compute(VertexBasedCellPopulation<DIM>& rPopulation)
{
    KursawePaperSummary summary;

    MutableVertexMesh<DIM, DIM>& r_mesh =
        static_cast<MutableVertexMesh<DIM, DIM>&>(rPopulation.rGetMesh());

    std::map<unsigned, accumulator_set<double, features<tag::mean>>> area_by_class;
    accumulator_set<double, features<tag::mean>> bulk_area_accumulator;
    accumulator_set<double, features<tag::mean>> elongation_accumulator;
    unsigned n_bulk = 0u;

    for (typename AbstractCellPopulation<DIM>::Iterator cell_iter = rPopulation.Begin();
         cell_iter != rPopulation.End();
         ++cell_iter)
    {
        const unsigned location_index = rPopulation.GetLocationIndexUsingCell(*cell_iter);
        if (r_mesh.GetElement(location_index)->IsElementOnBoundary())
        {
            continue;
        }

        ++n_bulk;
        const unsigned num_edges = r_mesh.GetElement(location_index)->GetNumNodes();
        const double cell_area = r_mesh.GetVolumeOfElement(location_index);
        const double elongation = r_mesh.GetElongationShapeFactorOfElement(location_index);

        bulk_area_accumulator(cell_area);
        elongation_accumulator(elongation);

        if (num_edges >= 4u && num_edges <= 8u)
        {
            area_by_class[num_edges](cell_area);
        }
    }

    summary.m_n_bulk_cells = n_bulk;

    if (n_bulk == 0u)
    {
        return summary;
    }

    summary.m_bulk_mean_area = mean(bulk_area_accumulator);
    summary.m_mean_elongation = mean(elongation_accumulator);

    for (unsigned n = 4u; n <= 8u; ++n)
    {
        auto it = area_by_class.find(n);
        if (it != area_by_class.end())
        {
            summary.area_ratio_by_polygon_class[n] = mean(it->second) / summary.m_bulk_mean_area;
        }
        else
        {
            summary.area_ratio_by_polygon_class[n] = 0.0;
        }
    }

    return summary;
}

template KursawePaperSummary KursawePaperSummary::Compute<2>(VertexBasedCellPopulation<2>&);
