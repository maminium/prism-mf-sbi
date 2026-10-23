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

#include "CellForcesWriter.hpp"

#include "AbstractCellPopulation.hpp"

CellForcesWriter::CellForcesWriter()
    : AbstractCellWriter<2, 2>("CellForces.dat")
{
    this->mVtkCellDataName = "AreaForceDummy";
}

double CellForcesWriter::GetCellDataForVtkOutput(CellPtr pCell, AbstractCellPopulation<2, 2>* pCellPopulation)
{
    return 42.0;
}

void CellForcesWriter::VisitCell(CellPtr pCell, AbstractCellPopulation<2, 2>* pCellPopulation)
{
    // Get VertexBasedCellPopulation
    if (dynamic_cast<VertexBasedCellPopulation<2>*>(pCellPopulation) == NULL)
    {
        EXCEPTION("CellForcesWriter is to be used with a VertexBasedCellPopulation only");
    }

    // Define some helper variables
    VertexBasedCellPopulation<2>* p_cell_population = static_cast<VertexBasedCellPopulation<2>*>(pCellPopulation);;

    *this->mpOutStream << SimulationTime::Instance()->GetTime() << " ";

    //Get Cell Id
    *this->mpOutStream << pCell->GetCellId() << " ";

    //Write Area force
    double cell_area_contribution = GetAreaForceContribution(pCell, p_cell_population);
    *this->mpOutStream << cell_area_contribution << " ";
    pCell->GetCellData()->SetItem("area force", cell_area_contribution);

    double cell_line_tension_contribution = GetLineTensionForceContribution(pCell, p_cell_population);
    *this->mpOutStream << cell_line_tension_contribution << " ";
    pCell->GetCellData()->SetItem("line tension force", cell_line_tension_contribution);

    double cell_perimeter_contribution = GetPerimeterForceContribution(pCell, p_cell_population);
    *this->mpOutStream << cell_perimeter_contribution << "\n";
    pCell->GetCellData()->SetItem("perimeter force", cell_perimeter_contribution);
}

double CellForcesWriter::GetAreaForceContribution(CellPtr pCell, VertexBasedCellPopulation<2>* pCellPopulation)
{
    double target_area;

    unsigned location_index = pCellPopulation->GetLocationIndexUsingCell(pCell);
    VertexElement<2, 2>* p_element = pCellPopulation->GetElement(location_index);

    try
    {
        target_area = pCellPopulation->GetCellUsingLocationIndex(location_index)->GetCellData()->GetItem("target area");
    }
    catch (Exception&)
    {
        EXCEPTION("You need to add an AbstractTargetAreaModifier to the simulation in order to use a CellForcesWriter");
    }

    // Find the local index of this node in this element

    std::vector<double> area_forces(p_element->GetNumNodes());
    // Add the force contribution from this cell's area elasticity (note the minus sign)
    for (unsigned local_index = 0; local_index < p_element->GetNumNodes(); local_index++)
    {
        c_vector<double, 2> area_elasticity_contribution = zero_vector<double>(2);
        c_vector<double, 2> element_area_gradient =
                pCellPopulation->rGetMesh().GetAreaGradientOfElementAtNode(p_element, local_index);
        area_elasticity_contribution -= GetAreaElasticityParameter()*( pCellPopulation->
                rGetMesh().GetVolumeOfElement(p_element->GetIndex()) -
                target_area)*element_area_gradient;
        area_forces[local_index] = norm_2(area_elasticity_contribution);
    }

	boost::accumulators::accumulator_set< double, boost::accumulators::features<boost::accumulators::tag::mean> > area_accumulator;
	std::for_each( area_forces.begin(), area_forces.end(), boost::bind<void>(boost::ref(area_accumulator), _1) );

    return boost::accumulators::mean(area_accumulator);
}

double CellForcesWriter::GetLineTensionForceContribution(CellPtr pCell, VertexBasedCellPopulation<2>* pCellPopulation)
{

    unsigned location_index = pCellPopulation->GetLocationIndexUsingCell(pCell);
    VertexElement<2, 2>* p_element = pCellPopulation->GetElement(location_index);
    unsigned num_nodes = p_element->GetNumNodes();

    std::vector< double > line_tension_forces(num_nodes);

    for (unsigned local_index = 0; local_index < p_element->GetNumNodes(); local_index++)
    {
        Node<2>* p_this_node = p_element->GetNode(local_index);
        c_vector<double, 2> line_tension_contribution = zero_vector<double>(2);
        // Get the previous and next nodes in this element
        unsigned previous_node_local_index = (num_nodes+local_index-1)%num_nodes;
        Node<2>* p_previous_node = p_element->GetNode(previous_node_local_index);

        unsigned next_node_local_index = (local_index+1)%num_nodes;
        Node<2>* p_next_node = p_element->GetNode(next_node_local_index);

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

        line_tension_forces[local_index] = norm_2(line_tension_contribution);
    }
    boost::accumulators::accumulator_set< double, boost::accumulators::features<boost::accumulators::tag::mean> > line_tension_accumulator;
    std::for_each( line_tension_forces.begin(), line_tension_forces.end(), boost::bind<void>(boost::ref(line_tension_accumulator), _1) );

    return boost::accumulators::mean(line_tension_accumulator);
}

double CellForcesWriter::GetPerimeterForceContribution(CellPtr pCell, VertexBasedCellPopulation<2>* pCellPopulation)
{
    unsigned location_index = pCellPopulation->GetLocationIndexUsingCell(pCell);
    VertexElement<2, 2>* p_element = pCellPopulation->GetElement(location_index);
    unsigned num_nodes = p_element->GetNumNodes();

    std::vector< double > perimeter_forces(num_nodes);

    for (unsigned local_index = 0; local_index < p_element->GetNumNodes(); local_index++)
    {
        c_vector<double, 2> perimeter_contractility_contribution = zero_vector<double>(2);
        // Get the previous and next nodes in this element
        unsigned previous_node_local_index = (num_nodes+local_index-1)%num_nodes;

        // Compute the gradient of each these edges, computed at the present node
        c_vector<double, 2> previous_edge_gradient =
                -pCellPopulation->rGetMesh().GetNextEdgeGradientOfElementAtNode(p_element, previous_node_local_index);
        c_vector<double, 2> next_edge_gradient = pCellPopulation->rGetMesh().GetNextEdgeGradientOfElementAtNode(p_element, local_index);

        // Add the force contribution from this cell's perimeter contractility (note the minus sign)
        c_vector<double, 2> element_perimeter_gradient = previous_edge_gradient + next_edge_gradient;
        perimeter_contractility_contribution -= GetPerimeterContractilityParameter()*
                pCellPopulation->rGetMesh().GetSurfaceAreaOfElement(location_index)*
                element_perimeter_gradient;
        perimeter_forces[local_index] = norm_2(perimeter_contractility_contribution);
    }
    boost::accumulators::accumulator_set< double, boost::accumulators::features<boost::accumulators::tag::mean> > perimeter_accumulator;
    std::for_each( perimeter_forces.begin(), perimeter_forces.end(), boost::bind<void>(boost::ref(perimeter_accumulator), _1) );

    return boost::accumulators::mean(perimeter_accumulator);
}

double CellForcesWriter::GetLineTensionParameter(Node<2>* pNodeA, Node<2>* pNodeB, VertexBasedCellPopulation<2>& rVertexCellPopulation)
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

double CellForcesWriter::GetAreaElasticityParameter()
{
    return 1.0;
}

double CellForcesWriter::GetPerimeterContractilityParameter()
{
    return 0.04;
}

double CellForcesWriter::GetLineTensionParameter()
{
    return 0.12;
}

double CellForcesWriter::GetBoundaryLineTensionParameter()
{
    return 0.12;
}

void CellForcesWriter::WriteTimeStamp()
{
}

void CellForcesWriter::WriteNewline()
{
}

#include "SerializationExportWrapperForCpp.hpp"
// Declare identifier for the serializer
CHASTE_CLASS_EXPORT(CellForcesWriter)
