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

#include "FasterMutableVertexMesh.hpp"
#include "UblasCustomFunctions.hpp"
#include "RandomNumberGenerator.hpp"
#include "MutableVertexMesh.hpp"
#include "Warnings.hpp"
#include "LogFile.hpp"

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
FasterMutableVertexMesh<ELEMENT_DIM, SPACE_DIM>::FasterMutableVertexMesh(std::vector<Node<SPACE_DIM>*> nodes,
                                                             std::vector<VertexElement<ELEMENT_DIM,SPACE_DIM>*> vertexElements,
                                                             double cellRearrangementThreshold,
                                                             double t2Threshold)
        : MutableVertexMesh<ELEMENT_DIM,SPACE_DIM>(nodes,
        		vertexElements,
				cellRearrangementThreshold,
				t2Threshold),
	    	mRandomizeT1SwapOrder(false)

{
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
FasterMutableVertexMesh<ELEMENT_DIM, SPACE_DIM>::FasterMutableVertexMesh()
    : MutableVertexMesh<ELEMENT_DIM,SPACE_DIM>()
{
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
FasterMutableVertexMesh<ELEMENT_DIM, SPACE_DIM>::~FasterMutableVertexMesh()
{
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void FasterMutableVertexMesh<ELEMENT_DIM, SPACE_DIM>::SetRandomizeT1SwapOrderBoolean(bool randomizeT1SwapOrder)
{
    mRandomizeT1SwapOrder = randomizeT1SwapOrder;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
bool FasterMutableVertexMesh<ELEMENT_DIM, SPACE_DIM>::GetRandomizeT1SwapOrderBoolean()
{
	return mRandomizeT1SwapOrder;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
std::vector< c_vector<unsigned,2> > FasterMutableVertexMesh<ELEMENT_DIM, SPACE_DIM>::GatherAllEdges()
{
	std::vector< c_vector<unsigned,2> > all_edges_vector;

    for (typename VertexMesh<ELEMENT_DIM, SPACE_DIM>::VertexElementIterator elem_iter = this->GetElementIteratorBegin();
         elem_iter != this->GetElementIteratorEnd();
         ++elem_iter)
	{
	    for( unsigned local_index = 0; local_index < elem_iter->GetNumNodes(); local_index++)
	    {
            Node<SPACE_DIM>* this_node = elem_iter->GetNode(local_index);
            Node<SPACE_DIM>* next_node = elem_iter->GetNode((local_index + 1)%(elem_iter->GetNumNodes()));
            std::set<unsigned> elems_containing_this_node = this_node->rGetContainingElementIndices();
            std::set<unsigned> elems_containing_next_node = next_node->rGetContainingElementIndices();

           	unsigned this_global_id = this_node->GetIndex();
           	unsigned next_global_id = next_node->GetIndex();

           	c_vector<unsigned, 2> this_edge;
		    this_edge[0] = std::min(this_global_id, next_global_id);
           	this_edge[1] = std::max(this_global_id, next_global_id);

            bool edge_is_already_found = false;

            for( std::vector< c_vector<unsigned,2> >::iterator this_iterator = all_edges_vector.begin();
            		this_iterator != all_edges_vector.end(); this_iterator++)
            {
            	if ( (*this_iterator)[0] == this_edge[0] and (*this_iterator)[1] == this_edge[1] )
            	{
                    edge_is_already_found = true;
                    break;
              	}
            }

           	if( !edge_is_already_found )
            {
                all_edges_vector.push_back(this_edge);
            }
		}
	}

	return all_edges_vector;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
bool FasterMutableVertexMesh<ELEMENT_DIM, SPACE_DIM>::CheckForSwapsFromShortEdges()
{
    std::vector< c_vector<unsigned,2> > all_edges_vector = GatherAllEdges();

    c_vector<unsigned,2> zero_vector;
    zero_vector[0] = 0u;
    zero_vector[1] = 0u;
  	unsigned number_edges = all_edges_vector.size();
    std::vector< c_vector<unsigned,2> > all_edges_vector_resorted(  number_edges, zero_vector );

    if (mRandomizeT1SwapOrder)
    {
    	std::vector<unsigned> index_permutation;
    	RandomNumberGenerator::Instance()->Shuffle( all_edges_vector.size(), index_permutation);

    	unsigned new_index = 0u;
    	for (std::vector<unsigned>::iterator index_iterator = index_permutation.begin();
    			index_iterator != index_permutation.end(); ++index_iterator)
    	{
    	    all_edges_vector_resorted[new_index] = all_edges_vector[*index_iterator];
    	    new_index++;
    	}
    }
    else
    {
    	all_edges_vector_resorted = all_edges_vector;
    }

    // Loop over all edges to check for T1 swaps

  	for (std::vector< c_vector<unsigned,2> >::iterator edge_iterator = all_edges_vector_resorted.begin();
  			edge_iterator != all_edges_vector_resorted.end(); ++edge_iterator)
  	{
  		// Old: Find locations of the current node and anticlockwise node
  		// New: find locations of the two nodes
//        Node<SPACE_DIM>* p_current_node = elem_iter->GetNode(local_index);
        Node<SPACE_DIM>* p_first_node = this->GetNode((*edge_iterator)[0]);
//        unsigned local_index_plus_one = (local_index+1)%num_nodes;    ///\todo Use iterators to tidy this up (see #2401)
        Node<SPACE_DIM>* p_second_node =  this->GetNode((*edge_iterator)[1]);

        // Find distance between nodes
        double distance_between_nodes = this->GetDistanceBetweenNodes(p_first_node->GetIndex(), p_second_node->GetIndex());

        // If the nodes are too close together...
        if (distance_between_nodes < this->mCellRearrangementThreshold)
        {
            // ...then check if any triangular elements are shared by these nodes...
            std::set<unsigned> elements_of_node_a = p_first_node->rGetContainingElementIndices();
            std::set<unsigned> elements_of_node_b = p_second_node->rGetContainingElementIndices();

            std::set<unsigned> shared_elements;
            std::set_intersection(elements_of_node_a.begin(), elements_of_node_a.end(),
                           elements_of_node_b.begin(), elements_of_node_b.end(),
                           std::inserter(shared_elements, shared_elements.begin()));

            bool both_nodes_share_triangular_element = false;
            for (std::set<unsigned>::const_iterator it = shared_elements.begin();
                 it != shared_elements.end();
                 ++it)
            {
                if (this->GetElement(*it)->GetNumNodes() <= 3)
                {
                    both_nodes_share_triangular_element = true;
                    break;
                }
            }

            // ...and if none are, then perform the required type of swap and halt the search, returning true
            if (!both_nodes_share_triangular_element)
            {
                this->IdentifySwapType(p_first_node, p_second_node);
                return true;
            }
        }
    }
    return false;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
bool FasterMutableVertexMesh<ELEMENT_DIM, SPACE_DIM>::CheckForIntersections()
{
	   // If checking for internal intersections as well as on the boundary, then check that no nodes have overlapped any elements...
	    if (this->mCheckForInternalIntersections)
	    {
	        ///\todo Change to only loop over neighbouring elements (see #2401)
	        for (typename AbstractMesh<ELEMENT_DIM,SPACE_DIM>::NodeIterator node_iter = this->GetNodeIteratorBegin();
	             node_iter != this->GetNodeIteratorEnd();
	             ++node_iter)
	        {
	            assert(!(node_iter->IsDeleted()));

	            for (typename VertexMesh<ELEMENT_DIM, SPACE_DIM>::VertexElementIterator elem_iter = this->GetElementIteratorBegin();
	                 elem_iter != this->GetElementIteratorEnd();
	                 ++elem_iter)
	            {
	                unsigned elem_index = elem_iter->GetIndex();

	                // Check that the node is not part of this element
	                if (node_iter->rGetContainingElementIndices().count(elem_index) == 0)
	                {
	                    if (this->ElementIncludesPoint(node_iter->rGetLocation(), elem_index))
	                    {
	                        this->PerformIntersectionSwap(&(*node_iter), elem_index);
	                        return true;
	                    }
	                }
	            }
	        }
	    }
	    else
	    {
	        // ...otherwise, just check that no boundary nodes have overlapped any boundary elements
	        std::vector<unsigned> boundary_element_indices;
	        std::vector< c_vector<double, SPACE_DIM> > boundary_element_centroids;
	        for (typename VertexMesh<ELEMENT_DIM, SPACE_DIM>::VertexElementIterator elem_iter = this->GetElementIteratorBegin();
	             elem_iter != this->GetElementIteratorEnd();
	             ++elem_iter)
	        {
	            if (elem_iter->IsElementOnBoundary())
	            {
	            	unsigned element_index = elem_iter->GetIndex();
	                boundary_element_indices.push_back(element_index);
	                // should be a map but I am too lazy to look up the syntax
	                boundary_element_centroids.push_back(this->GetCentroidOfElement(element_index));
	            }
	        }

	        for (typename AbstractMesh<ELEMENT_DIM,SPACE_DIM>::NodeIterator node_iter = this->GetNodeIteratorBegin();
	             node_iter != this->GetNodeIteratorEnd();
	             ++node_iter)
	        {
	            if (node_iter->IsBoundaryNode())
	            {
	                assert(!(node_iter->IsDeleted()));

	                // an index makes for a lazy map
	                unsigned index_of_element_index = 0;
	                for (std::vector<unsigned>::iterator elem_iter = boundary_element_indices.begin();
	                     elem_iter != boundary_element_indices.end();
	                     ++elem_iter)
	                {
	                    // Check that the node is not part of this element
	                    if (node_iter->rGetContainingElementIndices().count(*elem_iter) == 0)
	                    {
	                    	c_vector<double, SPACE_DIM> node_location = node_iter->rGetLocation();
	                    	c_vector<double, SPACE_DIM> element_centroid = boundary_element_centroids[index_of_element_index];
	                    	double node_element_distance = norm_2(this->GetVectorFromAtoB(node_location, element_centroid));

	                    	if ( node_element_distance < 5.0 )
	                    	{
	                            if (this->ElementIncludesPoint(node_iter->rGetLocation(), *elem_iter))
	                            {
	                                this->PerformT3Swap(&(*node_iter), *elem_iter);
	                                return true;
	                            }
	                        }
	                    }
	                    // mappy mapping increments
	                    index_of_element_index +=1u;
	                }
	            }
	        }
	    }

	    return false;
}

// Explicit instantiation
template class FasterMutableVertexMesh<1,1>;
template class FasterMutableVertexMesh<1,2>;
template class FasterMutableVertexMesh<1,3>;
template class FasterMutableVertexMesh<2,2>;
template class FasterMutableVertexMesh<2,3>;
template class FasterMutableVertexMesh<3,3>;

// Serialization for Boost >= 1.36
#include "SerializationExportWrapperForCpp.hpp"
EXPORT_TEMPLATE_CLASS_ALL_DIMS(FasterMutableVertexMesh)
