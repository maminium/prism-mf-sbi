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

#include "NeighbourNumberCorrelationWriter.hpp"

#include "AbstractCellPopulation.hpp"
#include "MeshBasedCellPopulation.hpp"
#include "CaBasedCellPopulation.hpp"
#include "NodeBasedCellPopulation.hpp"
#include "PottsBasedCellPopulation.hpp"
#include "VertexBasedCellPopulation.hpp"
#include "ImmersedBoundaryCellPopulation.hpp"
#include "SimulationTime.hpp"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/ref.hpp>
#include <boost/bind.hpp>

using namespace boost::accumulators;

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::NeighbourNumberCorrelationWriter()
    : AbstractCellPopulationCountWriter<ELEMENT_DIM, SPACE_DIM>("NeighbourNumberCorrelations.dat")
{
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::WriteHeader(AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
    if (PetscTools::AmMaster())
    {

        *this->mpOutStream << "Time NeighbourNumber_Correlation";

        this->WriteNewline();
    }
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::Visit(MeshBasedCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::Visit(CaBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::Visit(NodeBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::Visit(PottsBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::Visit(ImmersedBoundaryCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
c_vector<double, 2> NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::GetMeanInternalNeighbourNumberAndVariance(VertexBasedCellPopulation<SPACE_DIM>* pCellPopulation )
{
    if (SPACE_DIM == 2 && ELEMENT_DIM == 2){
	accumulator_set< double, features<tag::mean, tag::variance> > neighbour_number_accumulator;

    for (typename AbstractCellPopulation<SPACE_DIM>::Iterator cell_iter = pCellPopulation->Begin();
         cell_iter != pCellPopulation->End();
         ++cell_iter)
    {
        if( !pCellPopulation->GetElementCorrespondingToCell(*cell_iter)->IsElementOnBoundary() )
        {
	        double this_cell_neighbour_number = pCellPopulation->GetElementCorrespondingToCell(*cell_iter)
	                ->GetNumNodes();

	        neighbour_number_accumulator(this_cell_neighbour_number);
        }
    }
    c_vector<double, 2> neighbour_number_statistics;
    neighbour_number_statistics[0] = mean(neighbour_number_accumulator);
    neighbour_number_statistics[1] = variance(neighbour_number_accumulator);

    return neighbour_number_statistics;
    } else {
        EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only of 2 Spatial and Element dimensions.");
    }
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
std::vector< c_vector<unsigned,2> > NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::GetAllInternalCellNeighbourIndexPairs(VertexBasedCellPopulation<SPACE_DIM>* pCellPopulation )
{
    if (SPACE_DIM == 2 && ELEMENT_DIM == 2){
    std::vector< c_vector<unsigned,2> >internal_cell_index_pairs;

    for (typename AbstractCellPopulation<SPACE_DIM>::Iterator cell_iter = pCellPopulation->Begin();
            cell_iter != pCellPopulation->End();
            ++cell_iter)
    {
        auto p_this_element =
                pCellPopulation->GetElementCorrespondingToCell(*cell_iter);

        if (!p_this_element->IsElementOnBoundary())
        {
            unsigned this_element_index = p_this_element->GetIndex();

            std::set<unsigned> indices_of_neighbour_elements =
                    pCellPopulation->rGetMesh().GetNeighbouringElementIndices(this_element_index);

            for( std::set<unsigned>::iterator this_iter = indices_of_neighbour_elements.begin();
                    this_iter != indices_of_neighbour_elements.end();
                    this_iter++)
            {
                if( !pCellPopulation->GetElement(*this_iter)->IsElementOnBoundary() )
                {
                    c_vector<unsigned, 2> this_pair;
                    this_pair[0] = std::min(this_element_index, *this_iter);
                    this_pair[1] = std::max(this_element_index, *this_iter);

                    bool pair_is_already_found = false;
                    for (std::vector< c_vector<unsigned,2> >::iterator this_iterator =
                            internal_cell_index_pairs.begin();
                            this_iterator != internal_cell_index_pairs.end(); this_iterator++)
                    {
                        if ( (*this_iterator)[0] == this_pair[0] and
                             (*this_iterator)[1] == this_pair[1] )
                        {
                            pair_is_already_found = true;
                            break;
                        }
                    }

                    if( !pair_is_already_found )
                    {
                        internal_cell_index_pairs.push_back(this_pair);
                    }
                 }
            }
        }
    }
    return internal_cell_index_pairs;
    } else {
        EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only of 2 Spatial and Element dimensions.");
    }
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void NeighbourNumberCorrelationWriter<ELEMENT_DIM, SPACE_DIM>::Visit(VertexBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
    if (SPACE_DIM == 2 && ELEMENT_DIM == 2){
    std::vector< c_vector<unsigned,2> > internal_cell_pairs =
            GetAllInternalCellNeighbourIndexPairs(pCellPopulation);

    c_vector<double, 2> internal_neighbour_statistics = GetMeanInternalNeighbourNumberAndVariance(pCellPopulation);

    double mean_internal_neighbour_number_squared = internal_neighbour_statistics[0]*internal_neighbour_statistics[0];
    double internal_neighbour_number_variance = internal_neighbour_statistics[1];

	accumulator_set< double, features<tag::mean> > correlations_accumulator;

	for( std::vector< c_vector<unsigned,2> >::iterator this_pair = internal_cell_pairs.begin();
	        this_pair != internal_cell_pairs.end();
	        this_pair++)
	{
	    double first_cell_neighbour_number = pCellPopulation->GetElement((*this_pair)[0])->GetNumNodes();
	    double second_cell_neighbour_number = pCellPopulation->GetElement((*this_pair)[1])->GetNumNodes();

	    double this_correlation = (first_cell_neighbour_number*second_cell_neighbour_number -
	            mean_internal_neighbour_number_squared)/
	            internal_neighbour_number_variance;

	    correlations_accumulator(this_correlation);
	}

    *this->mpOutStream << mean(correlations_accumulator);
    } else {
        EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only of 2 Spatial and Element dimensions.");
    }
}

template class NeighbourNumberCorrelationWriter<1,1>;
template class NeighbourNumberCorrelationWriter<1,2>;
template class NeighbourNumberCorrelationWriter<2,2>;
template class NeighbourNumberCorrelationWriter<1,3>;
template class NeighbourNumberCorrelationWriter<2,3>;
template class NeighbourNumberCorrelationWriter<3,3>;

#include "SerializationExportWrapperForCpp.hpp"
// Declare identifier for the serializer
EXPORT_TEMPLATE_CLASS_ALL_DIMS(NeighbourNumberCorrelationWriter)
