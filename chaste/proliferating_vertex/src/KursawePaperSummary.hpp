#ifndef KURSAWEPAPERSUMMARY_HPP_
#define KURSAWEPAPERSUMMARY_HPP_

#include "UblasVectorInclude.hpp"
#include <map>

template<unsigned DIM>
class VertexBasedCellPopulation;

/**
 * Paper observation vector s (Kursawe et al. 2018 Table 2):
 * 5 area ratios (polygon classes 4-8, bulk cells) + mean elongation -> 6D.
 */
struct KursawePaperSummary
{
    /** Area ratio for polygon class n (mean area in class / bulk mean area). */
    std::map<unsigned, double> area_ratio_by_polygon_class;

  private:
    double m_mean_elongation;
    double m_bulk_mean_area;
    unsigned m_n_bulk_cells;

  public:
    KursawePaperSummary();

    double GetMeanElongation() const;
    double GetBulkMeanArea() const;
    unsigned GetNumBulkCells() const;

    /** Components in fixed order: ratios for n=4..8, then mean elongation. */
    c_vector<double, 6> AsVector() const;

    /**
     * Compute from final cell population state (bulk cells only).
     */
    template<unsigned DIM>
    static KursawePaperSummary Compute(VertexBasedCellPopulation<DIM>& rPopulation);
};

#endif /*KURSAWEPAPERSUMMARY_HPP_*/
