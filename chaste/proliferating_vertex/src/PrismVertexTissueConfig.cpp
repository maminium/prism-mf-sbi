#include "PrismVertexTissueConfig.hpp"

#include <cstdlib>
#include <string>

namespace
{
bool EnvBool(const char* name, bool default_val)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0')
    {
        return default_val;
    }
    std::string s(raw);
    return (s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "YES");
}

double EnvDouble(const char* name, double default_val)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0')
    {
        return default_val;
    }
    try
    {
        return std::stod(std::string(raw));
    }
    catch (...)
    {
        return default_val;
    }
}

unsigned EnvUnsigned(const char* name, unsigned default_val)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0')
    {
        return default_val;
    }
    try
    {
        return static_cast<unsigned>(std::stoul(std::string(raw)));
    }
    catch (...)
    {
        return default_val;
    }
}

bool EnvIsSet(const char* name)
{
    const char* raw = std::getenv(name);
    return raw != nullptr && raw[0] != '\0';
}
} // namespace

PrismVertexTissueConfig::PrismVertexTissueConfig()
    : paper_mode(false),
      minimal_io(false),
      disable_adaptive_dt(false),
      end_time(700.0),
      dt(0.005),
      sampling_timestep_multiple(200u),
      m_check_intersections_explicit(false),
      m_check_intersections_value(true)
{
}

PrismVertexTissueConfig PrismVertexTissueConfig::Table1Profile(bool minimal_io)
{
    PrismVertexTissueConfig cfg;
    cfg.paper_mode = true;
    cfg.minimal_io = minimal_io;
    cfg.disable_adaptive_dt = EnvBool("PRISM_VERTEX_DISABLE_ADAPTIVE_DT", false);
    cfg.end_time = EnvDouble("PRISM_VERTEX_END_TIME", 700.0);
    cfg.dt = 0.01;
    cfg.sampling_timestep_multiple = 100u;
    if (EnvIsSet("PRISM_VERTEX_DT"))
    {
        cfg.dt = EnvDouble("PRISM_VERTEX_DT", cfg.dt);
    }
    if (EnvIsSet("PRISM_VERTEX_SAMPLING_MULTIPLE"))
    {
        cfg.sampling_timestep_multiple = EnvUnsigned("PRISM_VERTEX_SAMPLING_MULTIPLE", cfg.sampling_timestep_multiple);
    }
    if (EnvIsSet("PRISM_VERTEX_CHECK_INTERSECTIONS"))
    {
        cfg.m_check_intersections_explicit = true;
        cfg.m_check_intersections_value = EnvBool("PRISM_VERTEX_CHECK_INTERSECTIONS", true);
    }
    return cfg;
}

PrismVertexTissueConfig PrismVertexTissueConfig::FromEnvironment()
{
    PrismVertexTissueConfig cfg;
    cfg.paper_mode = EnvBool("PRISM_VERTEX_PAPER_MODE", false);
    cfg.minimal_io = EnvBool("PRISM_VERTEX_MINIMAL_IO", cfg.paper_mode);
    cfg.disable_adaptive_dt = EnvBool("PRISM_VERTEX_DISABLE_ADAPTIVE_DT", false);
    cfg.end_time = EnvDouble("PRISM_VERTEX_END_TIME", 700.0);

    if (cfg.paper_mode)
    {
        cfg.dt = 0.01;
        cfg.sampling_timestep_multiple = 100u;
    }
    else
    {
        cfg.dt = 0.005;
        cfg.sampling_timestep_multiple = 200u;
    }

    if (EnvIsSet("PRISM_VERTEX_DT"))
    {
        cfg.dt = EnvDouble("PRISM_VERTEX_DT", cfg.dt);
    }
    if (EnvIsSet("PRISM_VERTEX_SAMPLING_MULTIPLE"))
    {
        cfg.sampling_timestep_multiple = EnvUnsigned("PRISM_VERTEX_SAMPLING_MULTIPLE", cfg.sampling_timestep_multiple);
    }

    if (EnvIsSet("PRISM_VERTEX_CHECK_INTERSECTIONS"))
    {
        cfg.m_check_intersections_explicit = true;
        cfg.m_check_intersections_value = EnvBool("PRISM_VERTEX_CHECK_INTERSECTIONS", true);
    }

    return cfg;
}

double PrismVertexTissueConfig::GetDt() const
{
    return dt;
}

unsigned PrismVertexTissueConfig::GetSamplingTimestepMultiple() const
{
    return sampling_timestep_multiple;
}

double PrismVertexTissueConfig::GetEndTime() const
{
    return end_time;
}

bool PrismVertexTissueConfig::GetCheckForInternalIntersections(double line_tension) const
{
    if (m_check_intersections_explicit)
    {
        return m_check_intersections_value;
    }
    if (line_tension < 0.0)
    {
        return false;
    }
    return true;
}

std::string PrismVertexTissueConfig::GetOutputSubdir(const std::string& default_subdir)
{
    const char* raw = std::getenv("PRISM_VERTEX_OUTPUT_SUBDIR");
    if (raw == nullptr || raw[0] == '\0')
    {
        return default_subdir;
    }
    return std::string(raw);
}
