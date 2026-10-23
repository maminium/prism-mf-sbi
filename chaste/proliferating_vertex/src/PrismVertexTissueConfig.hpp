#ifndef PRISM_VERTEX_TISSUE_CONFIG_HPP_
#define PRISM_VERTEX_TISSUE_CONFIG_HPP_

#include <string>

/** Runtime configuration for the vertex tissue Chaste driver (env vars). */
struct PrismVertexTissueConfig
{
    bool paper_mode;
    bool minimal_io;
    bool disable_adaptive_dt;

    double end_time;
    double dt;
    unsigned sampling_timestep_multiple;

  private:
    bool m_check_intersections_explicit;
    bool m_check_intersections_value;

  public:
    PrismVertexTissueConfig();

    /** Load from environment; apply paper-mode defaults when paper_mode is true. */
    static PrismVertexTissueConfig FromEnvironment();

    /** Kursawe Table 1 profile (fixed test / Fig. 2 validation). */
    static PrismVertexTissueConfig Table1Profile(bool minimal_io = false);

    double GetDt() const;
    unsigned GetSamplingTimestepMultiple() const;
    double GetEndTime() const;

    /**
     * Internal intersection checking on the vertex mesh.
     * If PRISM_VERTEX_CHECK_INTERSECTIONS is set, use that value.
     * Otherwise: false when line_tension < 0 (region B), else true.
     */
    bool GetCheckForInternalIntersections(double line_tension) const;

    static std::string GetOutputSubdir(const std::string& default_subdir);
};

#endif /*PRISM_VERTEX_TISSUE_CONFIG_HPP_*/
