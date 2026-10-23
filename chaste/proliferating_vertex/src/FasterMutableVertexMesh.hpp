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
#ifndef FASTERMUTABLEVERTEXMESH_HPP_
#define FASTERMUTABLEVERTEXMESH_HPP_

#include <iostream>
#include <map>
#include <algorithm>

#include "ChasteSerialization.hpp"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/split_member.hpp>

#include "MutableVertexMesh.hpp"

/**
 * An accelerated version of the MuableVertexMesh
 */
template<unsigned ELEMENT_DIM, unsigned SPACE_DIM>
class FasterMutableVertexMesh : public MutableVertexMesh<ELEMENT_DIM, SPACE_DIM>
{
protected:

	/**
	 * Whether to randomize the T1 swap order
	 */
    bool mRandomizeT1SwapOrder;

    /**
     * Needed for testing
     */
    friend class TestFasterMutableVertexMeshReMesh;

    /** Needed for serialization. */
    friend class boost::serialization::access;

    /**
     * Serialize the mesh.
     *
     * Note that if you are calling this method (from subclasses) you should archive your
     * member variables FIRST. So that this method can call a ReMesh
     * (to convert from TrianglesMeshReader input format into your native format).
     *
     * @param archive the archive
     * @param version the current version of this class
     */
    template<class Archive>
    void serialize(Archive & archive, const unsigned int version)
    {
        archive & mRandomizeT1SwapOrder;
        archive & boost::serialization::base_object<MutableVertexMesh<ELEMENT_DIM, SPACE_DIM> >(*this);
    }

public:

    /**
     * Default constructor.
     *
     * @param nodes vector of pointers to nodes
     * @param vertexElements vector of pointers to VertexElements
     * @param cellRearrangementThreshold the minimum threshold distance for element rearrangement (defaults to 0.01)
     * @param t2Threshold the maximum threshold distance for Type 2 swaps (defaults to 0.001)
     * @param cellRearrangementRatio ratio between the minimum threshold distance for element
     *                                rearrangement node separation after remeshing (defaults to 1.5)
     * @param protorosetteFormationProbability the probability of a protorosette formation event happening instead of
     *                                a T1 swap
     * @param protorosetteResolutionProbabilityPerTimestep the probability that, in a given timestep, a protorosette
     *                                will resolve (similar to the completion of a T1 swap)
     * @param rosetteResolutionProbabilityPerTimestep the probability that, in a given timestep, a rosette will
     *                                resolve (reduce the number of cells sharing a common vertex by 1)
     */
    FasterMutableVertexMesh(std::vector<Node<SPACE_DIM>*> nodes,
                      std::vector<VertexElement<ELEMENT_DIM, SPACE_DIM>*> vertexElements,
                      double cellRearrangementThreshold=0.01,
                      double t2Threshold=0.001);

    /**
     * Default constructor for use by serializer.
     */
    FasterMutableVertexMesh();

    /**
     * Destructor.
     */
    virtual ~FasterMutableVertexMesh();

    /**
     * Set whether to randomize the t1 swap order at each time step (True) or
     * not do that (False)
     *
     * @param restrictMovement
     */
    void SetRandomizeT1SwapOrderBoolean(bool randomizeT1SwapOrder);

    /**
     * Check whether the T1 swap order is randomized (True) or not (False)
     */
    bool GetRandomizeT1SwapOrderBoolean();

    /**
     * Updated version of the same function in VertexMesh. It's attempted to be faster.
     *
     * @param rTestPoint the point to test
     * @param elementIndex global index of the element in the mesh
     *
     * @return if the point is included in the element.
     */
    virtual bool CheckForIntersections();

    /**
     * Helper method for ReMesh().
     *
     * Check if any neighbouring nodes in an element are closer than the mCellRearrangementThreshold
     * and are not contained in any triangular elements. If any such pair of nodes are found, then
     * call IdentifySwapType(), which in turn implements the appropriate local remeshing operation
     * (a T1 swap, void removal, or node merge).
     *
     * This method is overridden from MutableVertexMesh. In contrast to the MutableVertexMesh this
     * implementation will randomize the t1 swap order if demanded by mRandomizeT1SwapOrder.
     *
     * @return whether we need to check for, and implement, any further local remeshing operations
     *                   (true if any swaps are performed).
     */
    virtual bool CheckForSwapsFromShortEdges();

    /**
     * Collects all edges in the Mesh
     *
     * @returns all_edges. A standard vector of edges. Each edge is represented by two unsigned integers
     *                 that are the global indices of the nodes sharing that edge.
     *                 In the current implementation,the first global index is allways smaller
     *                 than the second global index.
     */
    std::vector< c_vector<unsigned,2> > GatherAllEdges();
};

#include "SerializationExportWrapper.hpp"
EXPORT_TEMPLATE_CLASS_ALL_DIMS(FasterMutableVertexMesh)

#endif /*FASTERMUTABLEVERTEXMESH_HPP_*/
