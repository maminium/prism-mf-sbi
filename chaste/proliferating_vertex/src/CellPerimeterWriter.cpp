/*

Copyright (c) 2005-2019, University of Oxford.
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

cellbased_src_writers_populationwriters

*/
#include "CellPerimeterWriter.hpp"
#include "AbstractCellPopulation.hpp"
#include "VertexBasedCellPopulation.hpp"
#include "UblasVectorInclude.hpp"
#include <cmath>

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
CellPerimeterWriter<ELEMENT_DIM, SPACE_DIM>::CellPerimeterWriter()
    : AbstractCellWriter<ELEMENT_DIM, SPACE_DIM>("cellperimeter.dat")
{
    this->mVtkCellDataName = "Cell perimeter";
    this->mOutputScalarData = true;
    this->mOutputVectorData = false;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
double CellPerimeterWriter<ELEMENT_DIM, SPACE_DIM>::GetCellDataForVtkOutput(CellPtr pCell, AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{
    // We first need to test that the cell population is vertex-based
    VertexBasedCellPopulation<SPACE_DIM>* p_vbcp = dynamic_cast<VertexBasedCellPopulation<SPACE_DIM>*>(pCellPopulation);
    double number_of_nodes;
    double cell_perimeter = 0;

    if (p_vbcp == nullptr)
    {
        EXCEPTION("Cell Perimeter Writer is only associated with vertex-based cell populations currently");
    }
    
    // We must also check that our simulations is in 2D
    if (SPACE_DIM != 2)
    {
        EXCEPTION("Cell Perimeter Writer is not yet implemented for Vertex simulations in 1D or 3D");
    }
    if(SPACE_DIM == 2){ 
    auto cellLocationIndex = pCellPopulation->GetLocationIndexUsingCell(pCell);
    c_vector<double, SPACE_DIM> cell_location = pCellPopulation->GetLocationOfCellCentre(pCell);
    auto p_element = p_vbcp->GetElement(cellLocationIndex);
    
    number_of_nodes = p_element->GetNumNodes();
    
    // We need to loop over the edges abd add teir length to the perimeter value.
    for (unsigned vertex_index = 0; vertex_index < number_of_nodes; vertex_index++){

         double edge_length = p_element->GetEdge(vertex_index)->rGetLength();
         cell_perimeter += edge_length;
    }

    } 
    return cell_perimeter;
}

template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
void CellPerimeterWriter<ELEMENT_DIM, SPACE_DIM>::VisitCell(CellPtr pCell, AbstractCellPopulation<ELEMENT_DIM, SPACE_DIM>* pCellPopulation)
{ 

    double cell_perimeter = GetCellDataForVtkOutput(pCell, pCellPopulation);
    *this->mpOutStream << cell_perimeter <<" ";
}

// Explicit instantiation
template class CellPerimeterWriter<1,1>;
template class CellPerimeterWriter<1,2>;
template class CellPerimeterWriter<2,2>;
template class CellPerimeterWriter<1,3>;
template class CellPerimeterWriter<2,3>;
template class CellPerimeterWriter<3,3>;

#include "SerializationExportWrapperForCpp.hpp"
// Declare identifier for the serializer
EXPORT_TEMPLATE_CLASS_ALL_DIMS(CellPerimeterWriter)