/*

Copyright (c) 2005-2015, University of Oxford.
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

#include "VertexEdgeLengthWriter.hpp"
#include "AbstractCellPopulation.hpp"
#include "MeshBasedCellPopulation.hpp"
#include "CaBasedCellPopulation.hpp"
#include "NodeBasedCellPopulation.hpp"
#include "PottsBasedCellPopulation.hpp"
#include "VertexBasedCellPopulation.hpp"
#include "ImmersedBoundaryCellPopulation.hpp"
#include "VertexElement.hpp"
#include "UblasIncludes.hpp"

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
VertexEdgeLengthWriter<ELEMENT_DIM, SPACE_DIM>::VertexEdgeLengthWriter()
    : AbstractCellPopulationWriter<ELEMENT_DIM, SPACE_DIM>("EdgeLengths.dat")
{
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexEdgeLengthWriter<ELEMENT_DIM, SPACE_DIM>::Visit(MeshBasedCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
	EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexEdgeLengthWriter<ELEMENT_DIM, SPACE_DIM>::Visit(CaBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
	EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexEdgeLengthWriter<ELEMENT_DIM, SPACE_DIM>::Visit(NodeBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
	EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexEdgeLengthWriter<ELEMENT_DIM, SPACE_DIM>::Visit(PottsBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
	EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexEdgeLengthWriter<ELEMENT_DIM, SPACE_DIM>::Visit(ImmersedBoundaryCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void VertexEdgeLengthWriter<ELEMENT_DIM, SPACE_DIM>::Visit(VertexBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
	// First, we make a vector with all the edge lengths in it problem: we need to make sure we visit each edge only once

	std::set<int> visited_elements;

	std::vector<double> edge_lengths;

    MutableVertexMesh<SPACE_DIM,SPACE_DIM>& r_mesh = pCellPopulation->rGetMesh();

	// Set up cell cycle and initial target areas
	for (typename AbstractCellPopulation<SPACE_DIM>::Iterator cell_iter = pCellPopulation->Begin();
			cell_iter != pCellPopulation->End();
			++cell_iter)
	{
		unsigned location_index = pCellPopulation->GetLocationIndexUsingCell(*cell_iter);
		VertexElement<SPACE_DIM,SPACE_DIM>* p_element = r_mesh.GetElement(location_index);
		unsigned num_nodes = p_element->GetNumNodes();
	    for (unsigned vertex_index = 0; vertex_index < num_nodes; vertex_index++)
	    {
            Node<SPACE_DIM>* p_this_node = p_element->GetNode(vertex_index);
            Node<SPACE_DIM>* p_next_node = p_element->GetNode((vertex_index+1)%num_nodes);

            // Find the indices of the elements owned by each node
            std::set<unsigned> elements_containing_nodeA = p_this_node->rGetContainingElementIndices();
            std::set<unsigned> elements_containing_nodeB = p_next_node->rGetContainingElementIndices();

            // Find common elements
            std::set<unsigned> shared_elements;
            std::set_intersection(elements_containing_nodeA.begin(),
                                  elements_containing_nodeA.end(),
                                  elements_containing_nodeB.begin(),
                                  elements_containing_nodeB.end(),
                                  std::inserter(shared_elements, shared_elements.begin()));

            shared_elements.erase(p_element->GetIndex());
            if ( shared_elements.size() > 0 )
            {
            	int other_element_index = *shared_elements.begin();
            	if(visited_elements.find(other_element_index) == visited_elements.end())
            	{
            		edge_lengths.push_back( norm_2(  r_mesh.GetVectorFromAtoB(p_this_node->rGetLocation(), p_next_node->rGetLocation()) ) );
            	}
            }
	    }
        visited_elements.insert(location_index);
	}

    *this->mpOutStream << edge_lengths.size() << "\t";

    for (unsigned index = 0;  index < edge_lengths.size(); index++)
    {
        *this->mpOutStream << edge_lengths[index] << "\t";
    }
}

// Explicit instantiation
template class VertexEdgeLengthWriter<2,2>;

#include "SerializationExportWrapperForCpp.hpp"
// Declare identifier for the serializer
EXPORT_TEMPLATE_CLASS_ALL_DIMS(VertexEdgeLengthWriter)
