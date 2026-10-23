#ifndef VERTEX_PROGRESS_MODIFIER_HPP_
#define VERTEX_PROGRESS_MODIFIER_HPP_

#include "AbstractCellBasedSimulationModifier.hpp"
#include "MutableVertexMesh.hpp"
#include "VertexBasedCellPopulation.hpp"
#include "SimulationTime.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

/**
 * Prints "<label> step <i> t=<sim_time> N=<n_cells> wall=<sec>s" to std::cout
 * at end of each sampled timestep (no file I/O). Used for terminal progress.
 */
template<unsigned DIM>
class TissueVertexProgressModifier : public AbstractCellBasedSimulationModifier<DIM, DIM>
{
private:
    std::string mLabel;
    unsigned mPrintEveryNSampledSteps;
    unsigned mCounter;
    std::chrono::steady_clock::time_point mT0;

public:
    TissueVertexProgressModifier(const std::string& label, unsigned print_every_n_sampled_steps = 1u)
        : mLabel(label),
          mPrintEveryNSampledSteps(std::max(1u, print_every_n_sampled_steps)),
          mCounter(0u),
          mT0(std::chrono::steady_clock::now())
    {
    }

    void SetupSolve(AbstractCellPopulation<DIM, DIM>& /*rCellPopulation*/, std::string /*outputDirectory*/) override
    {
        mT0 = std::chrono::steady_clock::now();
        std::cout << "[" << mLabel << "] SetupSolve begin" << std::endl;
    }

    void UpdateAtEndOfTimeStep(AbstractCellPopulation<DIM, DIM>& rCellPopulation) override
    {
        ++mCounter;
        if (mCounter % mPrintEveryNSampledSteps != 0u)
        {
            return;
        }
        const double sim_time = SimulationTime::Instance()->GetTime();
        const unsigned n_cells = rCellPopulation.GetNumRealCells();
        const double wall = std::chrono::duration_cast<std::chrono::duration<double>>(
                                std::chrono::steady_clock::now() - mT0)
                                .count();
        MutableVertexMesh<DIM, DIM>& r_mesh =
            static_cast<MutableVertexMesh<DIM, DIM>&>(rCellPopulation.rGetMesh());
        const unsigned n_t1 = r_mesh.GetOperationRecorder()->GetT1SwapsInfo().size();
        const unsigned n_t2 = r_mesh.GetOperationRecorder()->GetT2SwapsInfo().size();
        std::cout << "[" << mLabel << "] step " << mCounter
                  << "  t=" << std::fixed << std::setprecision(2) << sim_time
                  << "  N=" << n_cells
                  << "  n_t1=" << n_t1
                  << "  n_t2=" << n_t2
                  << "  wall=" << std::setprecision(1) << wall << "s"
                  << std::endl;
    }

    void OutputSimulationModifierParameters(out_stream& /*rParamsFile*/) override
    {
    }
};

#endif /*VERTEX_PROGRESS_MODIFIER_HPP_*/
