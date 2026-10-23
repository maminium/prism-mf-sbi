#ifndef YANVERTEXTISSUECONFIG_HPP_
#define YANVERTEXTISSUECONFIG_HPP_

#include <cstdlib>
#include <string>

/**
 * Runtime configuration for Yan epithelial vertex quench (env: MFNNKB_YAN_*).
 *
 * Python inference wrappers map onto these names.
 */
struct YanVertexTissueConfig
{
    double end_time;
    double dt;
    unsigned sampling_timestep_multiple;
    unsigned initial_size;
    double t1_swap_threshold;
    double t2_swap_threshold;
    double new_edge_length_factor;
    bool restrict_vertex_movement;
    bool minimal_io;
    bool disable_adaptive_dt;
    bool check_internal_intersections;
    bool write_mesh_snapshot;
    /** Randomize T1 candidate order (stochastic remeshing). */
    bool randomize_t1_swap_order;
    /**
     * Optional honeycomb break: displace each vertex by U(-amp,amp)*typical_edge
     * in x and y after mesh build (0 = off). Helps frac_hex / stds vs ordered hex.
     */
    double init_vertex_noise;
    /**
     * Initial mesh generator: "honeycomb" (default, unchanged legacy path) or
     * "voronoi" (Lloyd-relaxed Poisson-Voronoi disordered start). Any other value
     * falls back to honeycomb so existing behaviour never changes implicitly.
     */
    std::string init_mode;
    /**
     * Lloyd's relaxation steps for the Voronoi initial mesh only. 0 = maximal
     * disorder (pure Poisson-Voronoi); larger values relax towards regular hex.
     */
    unsigned voronoi_relaxation_steps;

    YanVertexTissueConfig()
        : end_time(50.0),
          dt(0.001),
          sampling_timestep_multiple(50u),
          initial_size(6u),
          t1_swap_threshold(0.05),
          t2_swap_threshold(0.001),
          new_edge_length_factor(1.5),
          restrict_vertex_movement(true),
          minimal_io(true),
          disable_adaptive_dt(false),
          check_internal_intersections(false),
          write_mesh_snapshot(true),
          randomize_t1_swap_order(false),
          init_vertex_noise(0.0),
          init_mode("honeycomb"),
          voronoi_relaxation_steps(0u)
    {
    }

    static double EnvDouble(const char* name, double fallback)
    {
        const char* raw = std::getenv(name);
        return raw ? std::atof(raw) : fallback;
    }

    static unsigned EnvUnsigned(const char* name, unsigned fallback)
    {
        const char* raw = std::getenv(name);
        return raw ? static_cast<unsigned>(std::atoi(raw)) : fallback;
    }

    static bool EnvBool(const char* name, bool fallback)
    {
        const char* raw = std::getenv(name);
        if (!raw)
        {
            return fallback;
        }
        return !(std::string(raw) == "0" || std::string(raw) == "false" || std::string(raw) == "False");
    }

    static std::string EnvString(const char* name, const std::string& fallback)
    {
        const char* raw = std::getenv(name);
        return raw ? std::string(raw) : fallback;
    }

    static YanVertexTissueConfig FromEnvironment()
    {
        YanVertexTissueConfig cfg;
        cfg.end_time = EnvDouble("MFNNKB_YAN_END_TIME", cfg.end_time);
        cfg.dt = EnvDouble("MFNNKB_YAN_DT", cfg.dt);
        cfg.initial_size = EnvUnsigned("MFNNKB_YAN_INITIAL_SIZE", cfg.initial_size);
        cfg.t1_swap_threshold = EnvDouble("MFNNKB_YAN_T1_THRESHOLD", cfg.t1_swap_threshold);
        cfg.t2_swap_threshold = EnvDouble("MFNNKB_YAN_T2_THRESHOLD", cfg.t2_swap_threshold);
        cfg.new_edge_length_factor = EnvDouble("MFNNKB_YAN_NEW_EDGE_FACTOR", cfg.new_edge_length_factor);
        cfg.restrict_vertex_movement = EnvBool("MFNNKB_YAN_RESTRICT_VERTEX", cfg.restrict_vertex_movement);
        cfg.minimal_io = EnvBool("MFNNKB_YAN_MINIMAL_IO", cfg.minimal_io);
        cfg.disable_adaptive_dt = EnvBool("MFNNKB_YAN_DISABLE_ADAPTIVE_DT", cfg.disable_adaptive_dt);
        cfg.check_internal_intersections = EnvBool("MFNNKB_YAN_CHECK_INTERSECTIONS", cfg.check_internal_intersections);
        cfg.write_mesh_snapshot = EnvBool("MFNNKB_YAN_WRITE_MESH", cfg.write_mesh_snapshot);
        cfg.randomize_t1_swap_order = EnvBool("MFNNKB_YAN_RANDOMIZE_T1", cfg.randomize_t1_swap_order);
        cfg.init_vertex_noise = EnvDouble("MFNNKB_YAN_INIT_VERTEX_NOISE", cfg.init_vertex_noise);
        cfg.init_mode = EnvString("MFNNKB_YAN_INIT_MODE", cfg.init_mode);
        cfg.voronoi_relaxation_steps = EnvUnsigned("MFNNKB_YAN_VORONOI_RELAX", cfg.voronoi_relaxation_steps);
        cfg.sampling_timestep_multiple = EnvUnsigned(
            "MFNNKB_YAN_SAMPLING_MULTIPLE",
            cfg.minimal_io ? 100000u : cfg.sampling_timestep_multiple);
        return cfg;
    }

    /** Short profile for unit/validation tests. */
    static YanVertexTissueConfig ValidationProfile()
    {
        YanVertexTissueConfig cfg;
        cfg.end_time = 1.0;
        cfg.dt = 0.001;
        cfg.sampling_timestep_multiple = 50u;
        cfg.initial_size = 4u;
        cfg.minimal_io = true;
        cfg.disable_adaptive_dt = true;
        // Negative / anisotropic tensions tangle free meshes; same as the proliferating operational box.
        cfg.check_internal_intersections = false;
        cfg.restrict_vertex_movement = true;
        cfg.write_mesh_snapshot = true;
        return cfg;
    }

    static std::string GetOutputSubdir(const std::string& default_subdir)
    {
        const char* raw = std::getenv("MFNNKB_YAN_OUTPUT_SUBDIR");
        return raw ? std::string(raw) : default_subdir;
    }
};

#endif /*YANVERTEXTISSUECONFIG_HPP_*/
