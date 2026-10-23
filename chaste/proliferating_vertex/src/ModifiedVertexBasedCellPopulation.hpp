/*

Copyright (c) 2005-2016, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef MODIFIEDVERTEXBASEDCELLPOPULATION_HPP_
#define MODIFIEDVERTEXBASEDCELLPOPULATION_HPP_

#include "AbstractOffLatticeCellPopulation.hpp"
#include "MutableVertexMesh.hpp"
#include "AbstractVertexBasedDivisionRule.hpp"
#include "VertexBasedCellPopulation.hpp"

#include "ChasteSerialization.hpp"
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>

#include <iostream>
#include <fstream>

template<unsigned DIM> class AbstractVertexBasedDivisionRule; // Circular definition thing.

/**
 * A facade class encapsulating a vertex-based cell population.
 *
 * Contains a group of cells and maintains the associations
 * between CellPtrs and elements in the MutableVertexMesh.
 *
 */
template<unsigned DIM>
class ModifiedVertexBasedCellPopulation : public VertexBasedCellPopulation<DIM>
{
private:

	bool mRestrictVertexMovement;

    /** Try DivideElementAlongGivenAxis; on failure rotate axis (2D only) and retry. */
    unsigned DivideElementWithRotatedAxes(VertexElement<DIM, DIM>* pElement,
                                          c_vector<double, DIM> axis);

    friend class boost::serialization::access;
    /**
     * Serialize the object and its member variables.
     *
     * Note that serialization of the mesh and cells is handled by load/save_construct_data.
     *
     * Note also that member data related to writers is not saved - output must
     * be set up again by the caller after a restart.
     *
     * @param archive the archive
     * @param version the current version of this class
     */
    template<class Archive>
    void serialize(Archive & archive, const unsigned int version)
    {
        archive & boost::serialization::base_object<VertexBasedCellPopulation<DIM> >(*this);
    }

public:

    /**
     * Create a new cell population facade from a mesh and collection of cells.
     *
     * There must be precisely one CellPtr for each VertexElement in
     * the mesh.
     *
     * @param rMesh reference to a
     * @param rCells reference to a vector of CellPtrs
     * @param deleteMesh set to true if you want the cell population to free the mesh memory on destruction
     * @param validate whether to validate the cell population when it is created (defaults to true)
     * @param locationIndices an optional vector of location indices that correspond to real cells
     */
    ModifiedVertexBasedCellPopulation(MutableVertexMesh<DIM, DIM>& rMesh,
                              std::vector<CellPtr>& rCells,
                              bool deleteMesh=false,
                              bool validate=true,
                              const std::vector<unsigned> locationIndices=std::vector<unsigned>());

    /**
     * Constructor for use by boost serialization ONLY!
     *
     * @param rMesh a vertex mesh.
     * @param rPopSrn population SRN remapping helper (must match Chaste VertexBasedCellPopulation API).
     */
    ModifiedVertexBasedCellPopulation(MutableVertexMesh<DIM, DIM>& rMesh,
                                      VertexBasedPopulationSrn<DIM>& rPopSrn);

    /**
     * Destructor, which frees any memory allocated by the constructor.
     */
    virtual ~ModifiedVertexBasedCellPopulation();

    /**
     * Overridden UpdateNodeLocations() method.
     *
     * @param dt the time step
     */
    void UpdateNodeLocations(double dt);

    /**
     * Checks whether a given node displacement violates the movement threshold
     * for this population. If so, a stepSizeException is generated that contains
     * a warning/error message and a suggested smaller dt that should avoid the problem.
     *
     * @param nodeIndex Index of the node in question (allows us to check whether this is a ghost or particle)
     * @param rDisplacement Movement vector of the node at this time step
     * @param dt Current time step size
     */
    virtual void CheckForStepSizeException(unsigned nodeIndex, c_vector<double,DIM>& rDisplacement, double dt);

    /**
      * Overridden OutputCellPopulationParameters() method.
      *
      * @param rParamsFile the file stream to which the parameters are output
      */
    virtual void OutputCellPopulationParameters(out_stream& rParamsFile);

    /**
     * Set whether to restrict vertex movement (True) or to
     * not do that (False)
     *
     * @param restrictMovement
     */
    void SetRestrictVertexMovementBoolean(bool restrictVertexMovement);

    /**
     * Check whether vertex movment is restricted (True) or not (False)
     */
    bool GetRestrictVertexMovementBoolean();

    /**
     * Overridden AddCell: if short-axis division fails (degenerate polygon under
     * strong negative line tension), retry with rotated division axes.
     */
    CellPtr AddCell(CellPtr pNewCell, CellPtr pParentCell) override;
};

#include "SerializationExportWrapper.hpp"
EXPORT_TEMPLATE_CLASS_SAME_DIMS(ModifiedVertexBasedCellPopulation)

namespace boost
{
namespace serialization
{
/**
 * Serialize information required to construct a VertexBasedCellPopulation.
 */
template<class Archive, unsigned DIM>
inline void save_construct_data(
    Archive & ar, const ModifiedVertexBasedCellPopulation<DIM> * t, const unsigned int file_version)
{
    // Save data required to construct instance
    const MutableVertexMesh<DIM,DIM>* p_mesh = &(t->rGetMesh());
    ar & p_mesh;

    const VertexBasedPopulationSrn<DIM>& pop_srn = t->rGetVertexBasedPopulationSrn();
    ar & pop_srn;
}

/**
 * De-serialize constructor parameters and initialise a VertexBasedCellPopulation.
 * Loads the mesh from separate files.
 */
template<class Archive, unsigned DIM>
inline void load_construct_data(
    Archive & ar, ModifiedVertexBasedCellPopulation<DIM> * t, const unsigned int file_version)
{
    // Retrieve data from archive required to construct new instance
    MutableVertexMesh<DIM,DIM>* p_mesh;
    ar >> p_mesh;

    VertexBasedPopulationSrn<DIM> pop_srn;
    ar & pop_srn;
    // Invoke inplace constructor to initialise instance
    ::new(t)ModifiedVertexBasedCellPopulation<DIM>(*p_mesh, pop_srn);
}
}
}

#endif /*MODIFIEDVERTEXBASEDCELLPOPULATION_HPP_*/
