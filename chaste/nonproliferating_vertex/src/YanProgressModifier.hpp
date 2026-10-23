#ifndef YANPROGRESSMODIFIER_HPP_
#define YANPROGRESSMODIFIER_HPP_

#include "AbstractCellBasedSimulationModifier.hpp"
#include "SimulationTime.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

template<unsigned DIM>
class YanProgressModifier : public AbstractCellBasedSimulationModifier<DIM, DIM>
{
private:
    std::string mLabel;
    unsigned mPrintEveryNSampledSteps;
    unsigned mCounter;
    std::chrono::steady_clock::time_point mT0;

public:
    YanProgressModifier(const std::string& label, unsigned print_every_n_sampled_steps = 1u)
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
        std::cout << "[" << mLabel << "] step " << mCounter
                  << "  t=" << std::fixed << std::setprecision(2) << sim_time
                  << "  N=" << n_cells
                  << "  wall=" << std::setprecision(1) << wall << "s"
                  << std::endl;
    }

    void OutputSimulationModifierParameters(out_stream& /*rParamsFile*/) override
    {
    }
};

#endif /*YANPROGRESSMODIFIER_HPP_*/
