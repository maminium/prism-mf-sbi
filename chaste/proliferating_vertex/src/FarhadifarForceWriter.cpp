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

#include "FarhadifarForceWriter.hpp"

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

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::FarhadifarForceWriter()
    : AbstractCellPopulationCountWriter<ELEMENT_DIM, SPACE_DIM>("FarhadifarForces.dat")
{
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::WriteHeader(AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
    if (PetscTools::AmMaster())
    {

        *this->mpOutStream << "Time Areaforce stdAreaForce LineTensionforce stdLineTensionForce Permiterforce stdperimeterforce";

        this->WriteNewline();
    }
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::Visit(MeshBasedCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::Visit(CaBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::Visit(NodeBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::Visit(PottsBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::Visit(ImmersedBoundaryCellPopulation<SPACE_DIM>* pCellPopulation)
{
    EXCEPTION("This writer is supposed to be used with a VertexBasedCellPopulation only.");
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::Visit(VertexBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
	boost::accumulators::accumulator_set< double, boost::accumulators::features<boost::accumulators::tag::mean, boost::accumulators::tag::variance> > area_accumulator;
	boost::accumulators::accumulator_set< double, boost::accumulators::features<boost::accumulators::tag::mean, boost::accumulators::tag::variance> > line_tension_accumulator;
	boost::accumulators::accumulator_set< double, boost::accumulators::features<boost::accumulators::tag::mean, boost::accumulators::tag::variance> > perimeter_accumulator;

    std::map<std::string, std::vector< double > > force_table = this->CalculateForces(pCellPopulation);
    std::vector< double > area_forces = force_table["area_forces"];
    std::vector< double > line_tension_forces = force_table["line_tension_forces"];
    std::vector< double > perimeter_forces = force_table["perimeter_forces"];

	std::for_each( area_forces.begin(), area_forces.end(), boost::bind<void>(boost::ref(area_accumulator), _1) );
	std::for_each( line_tension_forces.begin(), line_tension_forces.end(), boost::bind<void>(boost::ref(line_tension_accumulator), _1) );
	std::for_each( perimeter_forces.begin(), perimeter_forces.end(), boost::bind<void>(boost::ref(perimeter_accumulator), _1) );

    *this->mpOutStream <<
            boost::accumulators::mean(area_accumulator) << " " << sqrt( boost::accumulators::variance(area_accumulator) ) << " " <<
            boost::accumulators::mean(line_tension_accumulator) << " " << sqrt( boost::accumulators::variance(line_tension_accumulator) ) << " "<<
            boost::accumulators::mean(perimeter_accumulator) << " " << sqrt( boost::accumulators::variance(perimeter_accumulator) );
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
std::map<std::string, std::vector<double> > FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::CalculateForces(VertexBasedCellPopulation<SPACE_DIM>* pCellPopulation)
{
    if (SPACE_DIM != 2)
    {
        EXCEPTION("Cell Forces Writer is not yet implemented for Vertex simulations in 1D or 3D");
    }
    
    if constexpr ((SPACE_DIM == 2) && (ELEMENT_DIM == 2)){
    unsigned num_nodes = pCellPopulation->GetNumNodes();
    unsigned num_elements = pCellPopulation->GetNumElements();

    // Begin by computing the area and perimeter of each element in the mesh, to avoid having to do this multiple times
    std::vector<double> element_areas(num_elements);
    std::vector<double> element_perimeters(num_elements);
    std::vector<double> target_areas(num_elements);
    for (typename VertexMesh<2, 2>::VertexElementIterator elem_iter = pCellPopulation->rGetMesh().GetElementIteratorBegin();
         elem_iter != pCellPopulation->rGetMesh().GetElementIteratorEnd();
         ++elem_iter)
    {
        unsigned elem_index = elem_iter->GetIndex();
        element_areas[elem_index] = pCellPopulation->rGetMesh().GetVolumeOfElement(elem_index);
        element_perimeters[elem_index] = pCellPopulation->rGetMesh().GetSurfaceAreaOfElement(elem_index);
        try
        {
            // If we haven't specified a growth modifier, there won't be any target areas in the CellData array and CellData
            // will throw an exception that it doesn't have "target area" entries.  We add this piece of code to give a more
            // understandable message. There is a slight chance that the exception is thrown although the error is not about the
            // target areas.
            target_areas[elem_index] = pCellPopulation->GetCellUsingLocationIndex(elem_index)->GetCellData()->GetItem("target area");
        }
        catch (Exception&)
        {
            EXCEPTION("You need to add an AbstractTargetAreaModifier to the simulation in order to use a FarhadifarForceWriter");
        }
    }

    std::vector< double > area_forces(num_nodes);
    std::vector< double > line_tension_forces(num_nodes);
    std::vector< double > perimeter_forces(num_nodes);
    // Iterate over vertices in the cell population
    for (unsigned node_index=0; node_index<num_nodes; node_index++)
    {
        Node<SPACE_DIM>* p_this_node = pCellPopulation->GetNode(node_index);

        /*
         * The force on this Node is given by the gradient of the total free
         * energy of the CellPopulation, evaluated at the position of the vertex. This
         * free energy is the sum of the free energies of all CellPtrs in
         * the cell population. The free energy of each CellPtr is comprised of three
         * terms - an area deformation energy, a perimeter deformation energy
         * and line tension energy.
         *
         * Note that since the movement of this Node only affects the free energy
         * of the CellPtrs containing it, we can just consider the contributions
         * to the free energy gradient from each of these CellPtrs.
         */
        c_vector<double, 2> area_elasticity_contribution = zero_vector<double>(2);
        c_vector<double, 2> perimeter_contractility_contribution = zero_vector<double>(2);
        c_vector<double, 2> line_tension_contribution = zero_vector<double>(2);

        // Find the indices of the elements owned by this node
        std::set<unsigned> containing_elem_indices = pCellPopulation->GetNode(node_index)->rGetContainingElementIndices();

        // Iterate over these elements
        for (std::set<unsigned>::iterator iter = containing_elem_indices.begin();
             iter != containing_elem_indices.end();
             ++iter)
        {
            // Get this element, its index and its number of nodes
            auto p_element = pCellPopulation->GetElement(*iter);
            unsigned elem_index = p_element->GetIndex();
            unsigned num_nodes_elem = p_element->GetNumNodes();

            // Find the local index of this node in this element
            unsigned local_index = p_element->GetNodeLocalIndex(node_index);

            // Add the force contribution from this cell's area elasticity (note the minus sign)
            c_vector<double, 2> element_area_gradient =
                    pCellPopulation->rGetMesh().GetAreaGradientOfElementAtNode(p_element, local_index);
            area_elasticity_contribution -= GetAreaElasticityParameter()*(element_areas[elem_index] -
                    target_areas[elem_index])*element_area_gradient;

            // Get the previous and next nodes in this element
            unsigned previous_node_local_index = (num_nodes_elem+local_index-1)%num_nodes_elem;
            Node<SPACE_DIM>* p_previous_node = p_element->GetNode(previous_node_local_index);

            unsigned next_node_local_index = (local_index+1)%num_nodes_elem;
            Node<SPACE_DIM>* p_next_node = p_element->GetNode(next_node_local_index);

            // Compute the line tension parameter for each of these edges - be aware that this is half of the actual
            // value for internal edges since we are looping over each of the internal edges twice
            double previous_edge_line_tension_parameter = GetLineTensionParameter(p_previous_node, p_this_node, *pCellPopulation);
            double next_edge_line_tension_parameter = GetLineTensionParameter(p_this_node, p_next_node, *pCellPopulation);

            // Compute the gradient of each these edges, computed at the present node
            c_vector<double, 2> previous_edge_gradient =
                    -pCellPopulation->rGetMesh().GetNextEdgeGradientOfElementAtNode(p_element, previous_node_local_index);
            c_vector<double, 2> next_edge_gradient = pCellPopulation->rGetMesh().GetNextEdgeGradientOfElementAtNode(p_element, local_index);

            // Add the force contribution from cell-cell and cell-boundary line tension (note the minus sign)
            line_tension_contribution -= previous_edge_line_tension_parameter*previous_edge_gradient +
                    next_edge_line_tension_parameter*next_edge_gradient;

            // Add the force contribution from this cell's perimeter contractility (note the minus sign)
            c_vector<double, 2> element_perimeter_gradient = previous_edge_gradient + next_edge_gradient;
            perimeter_contractility_contribution -= GetPerimeterContractilityParameter()* element_perimeters[elem_index]*
                                                                                                     element_perimeter_gradient;
        }

        area_forces[node_index] = norm_2(area_elasticity_contribution);
        line_tension_forces[node_index] = norm_2(line_tension_contribution);
        perimeter_forces[node_index] = norm_2(perimeter_contractility_contribution);

    }

    std::map<std::string, std::vector< double > > force_table;
    force_table["area_forces"] = area_forces;
    force_table["line_tension_forces"] = line_tension_forces;
    force_table["perimeter_forces"] = perimeter_forces;
    return force_table;
    } else {
       auto a = pCellPopulation->GetNode(0);
       EXCEPTION("Cell Forces Writer is not yet implemented for Vertex simulations in 1D or 3D"); 
       // This is purely here as filler and should be removed in future TODO;
       std::cout << "Filler" << a << "\n";
    }
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::GetLineTensionParameter(Node<SPACE_DIM>* pNodeA, Node<SPACE_DIM>* pNodeB, VertexBasedCellPopulation<SPACE_DIM>& rVertexCellPopulation)
{
    // Find the indices of the elements owned by each node
    std::set<unsigned> elements_containing_nodeA = pNodeA->rGetContainingElementIndices();
    std::set<unsigned> elements_containing_nodeB = pNodeB->rGetContainingElementIndices();

    // Find common elements
    std::set<unsigned> shared_elements;
    std::set_intersection(elements_containing_nodeA.begin(),
                          elements_containing_nodeA.end(),
                          elements_containing_nodeB.begin(),
                          elements_containing_nodeB.end(),
                          std::inserter(shared_elements, shared_elements.begin()));

    // Check that the nodes have a common edge
    assert(!shared_elements.empty());

    // Since each internal edge is visited twice in the loop above, we have to use half the line tension parameter
    // for each visit.
    double line_tension_parameter_in_calculation = GetLineTensionParameter()/2.0;

    // If the edge corresponds to a single element, then the cell is on the boundary
    if (shared_elements.size() == 1)
    {
        line_tension_parameter_in_calculation = GetBoundaryLineTensionParameter();
    }

    return line_tension_parameter_in_calculation;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::GetAreaElasticityParameter()
{
    return 1.0;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::GetPerimeterContractilityParameter()
{
    return 0.04;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::GetLineTensionParameter()
{
    return 0.12;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double FarhadifarForceWriter<ELEMENT_DIM, SPACE_DIM>::GetBoundaryLineTensionParameter()
{
    return 0.12;
}

template class FarhadifarForceWriter<2,2>;

#include "SerializationExportWrapperForCpp.hpp"
// Declare identifier for the serializer
EXPORT_TEMPLATE_CLASS_ALL_DIMS(FarhadifarForceWriter)
