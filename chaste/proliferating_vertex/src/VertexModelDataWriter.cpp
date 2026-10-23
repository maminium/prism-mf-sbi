/*

Copyright (c) 2005-2013, University of Oxford.
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

#include "VertexModelDataWriter.hpp"
#include "AbstractCellPopulation.hpp"
#include "Cell.hpp"
#include "CellLabel.hpp"
#include "MutableVertexMesh.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
using namespace boost::accumulators;

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
VertexModelDataWriter<ELEMENT_DIM, SPACE_DIM>::VertexModelDataWriter()
        : AbstractCellWriter<ELEMENT_DIM, SPACE_DIM>("VertexData.txt")
{
	this->mVtkCellDataName = "VertexDataDummy";
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double VertexModelDataWriter<ELEMENT_DIM, SPACE_DIM>::GetCellDataForVtkOutput(CellPtr pCell, AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
    	return 2.0;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double VertexModelDataWriter<ELEMENT_DIM, SPACE_DIM>::GetAverageCellAreaOfNeighbours(CellPtr pCell, AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
	unsigned location_index = pCellPopulation->GetLocationIndexUsingCell(pCell);

	MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>* p_mesh = static_cast<MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>* >(&(pCellPopulation->rGetMesh()));
	std::set<unsigned> indices_of_neighbour_elements = p_mesh->GetNeighbouringElementIndices(location_index);

	accumulator_set< double, features<tag::mean > > area_accumulator;

	for( std::set<unsigned>::iterator this_iter = indices_of_neighbour_elements.begin();
	        this_iter != indices_of_neighbour_elements.end();
	        this_iter++)
	{
	    double this_area = p_mesh->GetVolumeOfElement(*this_iter);
	    area_accumulator(this_area);
	}

    return mean(area_accumulator);
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double VertexModelDataWriter<ELEMENT_DIM, SPACE_DIM>::GetAverageNeighbourNumberOfNeighbours(CellPtr pCell, AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
	unsigned location_index = pCellPopulation->GetLocationIndexUsingCell(pCell);

	MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>* p_mesh = static_cast<MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>* >(&(pCellPopulation->rGetMesh()));
	std::set<unsigned> indices_of_neighbour_elements = p_mesh->GetNeighbouringElementIndices(location_index);

	accumulator_set< double, features<tag::mean> > neighbour_number_accumulator;

	for( std::set<unsigned>::iterator this_iter = indices_of_neighbour_elements.begin();
	        this_iter != indices_of_neighbour_elements.end();
	        this_iter++)
	{
	    double this_neighbour_number = p_mesh->GetElement(*this_iter)->GetNumNodes();
	    neighbour_number_accumulator(this_neighbour_number);
	}

    return mean(neighbour_number_accumulator);
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
bool VertexModelDataWriter<ELEMENT_DIM, SPACE_DIM>::IsCellOnInnerBoundary(CellPtr pCell, AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
	unsigned location_index = pCellPopulation->GetLocationIndexUsingCell(pCell);

	MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>* p_mesh = static_cast<MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>* >(&(pCellPopulation->rGetMesh()));
	std::set<unsigned> indices_of_neighbour_elements = p_mesh->GetNeighbouringElementIndices(location_index);

	bool is_on_inner_boundary = false;

	for( std::set<unsigned>::iterator this_iter = indices_of_neighbour_elements.begin();
	        this_iter != indices_of_neighbour_elements.end();
	        this_iter++)
	{
	    if( p_mesh->GetElement(*this_iter)->IsElementOnBoundary() )
	    {
	        is_on_inner_boundary = true;
	        break;
	    }
	}

    return is_on_inner_boundary;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexModelDataWriter<ELEMENT_DIM, SPACE_DIM>::VisitCell(CellPtr pCell, AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
	unsigned location_index = pCellPopulation->GetLocationIndexUsingCell(pCell);
	unsigned cell_id = pCell->GetCellId();

	unsigned cell_type = 0; // get mutation state
	//        if (pCell->GetMutationState()->IsType<ProgenitorTypeOneCellMutationState>())
	//        {
	//            cell_type = 1;
	//        }
	//        else if (pCell->GetMutationState()->IsType<ProgenitorTypeTwoCellMutationState>())
	//        {
	//            cell_type = 2;
	//        }

	// Note that the line below will only return something sensible if using a VertexBasedCellPopulation
	MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>* p_mesh = static_cast<MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>* >(&(pCellPopulation->rGetMesh()));

	unsigned num_edges = p_mesh->GetElement(location_index)->GetNumNodes();
	double cell_area = p_mesh->GetVolumeOfElement(location_index);
	*this->mpOutStream << SimulationTime::Instance()->GetTime() << " " << location_index << " " << cell_id << " " << cell_type << " " << num_edges << " " << cell_area << " ";

	c_vector<double, SPACE_DIM> centre_location = pCellPopulation->GetLocationOfCellCentre(pCell);
	for (unsigned i=0; i<SPACE_DIM; i++)
	{
		*this->mpOutStream << " " << centre_location[i];
	}

	if (pCell->HasCellProperty<CellLabel>())
	{
		*this->mpOutStream << " " << 1; //labelled true
	}
	else
	{
		*this->mpOutStream << " " << 0; //labelled false
	}

	*this->mpOutStream << " " << p_mesh->GetElement(location_index)->IsElementOnBoundary();
	*this->mpOutStream << " " << p_mesh->GetSurfaceAreaOfElement(location_index);
	*this->mpOutStream << " " << p_mesh->GetElongationShapeFactorOfElement(location_index);
	*this->mpOutStream << " " << this->IsCellOnInnerBoundary( pCell, pCellPopulation);
	*this->mpOutStream << " " << this->GetAverageNeighbourNumberOfNeighbours( pCell, pCellPopulation);
	*this->mpOutStream << " " << this->GetAverageCellAreaOfNeighbours( pCell, pCellPopulation);
	*this->mpOutStream << "\n";
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexModelDataWriter<ELEMENT_DIM, SPACE_DIM>::WriteTimeStamp()
{
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexModelDataWriter<ELEMENT_DIM, SPACE_DIM>::WriteNewline()
{
}

// Explicit instantiation
template class VertexModelDataWriter<1,1>;
template class VertexModelDataWriter<1,2>;
template class VertexModelDataWriter<2,2>;
template class VertexModelDataWriter<1,3>;
template class VertexModelDataWriter<2,3>;
template class VertexModelDataWriter<3,3>;

#include "SerializationExportWrapperForCpp.hpp"
// Declare identifier for the serializer
EXPORT_TEMPLATE_CLASS_ALL_DIMS(VertexModelDataWriter)
