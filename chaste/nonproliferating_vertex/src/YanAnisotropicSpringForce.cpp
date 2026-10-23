#include "YanAnisotropicSpringForce.hpp"

#include <algorithm>
#include <cmath>

template<unsigned DIM>
YanAnisotropicSpringForce<DIM>::YanAnisotropicSpringForce()
    : FarhadifarForce<DIM>(),
      mBaselineLineTension(1.0),
      mAnisotropyMagnitude(0.25),
      mLengthSpringStiffness(0.15),
      mAnisotropyAngle(0.0)
{
    this->mAreaElasticityParameter = 5.0;
    this->mTargetAreaParameter = 1.0;
    this->mPerimeterContractilityParameter = 0.0;
    this->mLineTensionParameter = 0.0;
    this->mBoundaryLineTensionParameter = 0.0;
}

template<unsigned DIM>
YanAnisotropicSpringForce<DIM>::~YanAnisotropicSpringForce()
{
}

template<unsigned DIM>
double YanAnisotropicSpringForce<DIM>::ComputeEdgeTension(double edge_length, double edge_angle) const
{
    // T = λ0 (1 + a_r cos 2(θ-φa)) - b0 ℓ   with λ0 frozen at Stage A.
    return mBaselineLineTension
           * (1.0 + mAnisotropyMagnitude * std::cos(2.0 * (edge_angle - mAnisotropyAngle)))
           - mLengthSpringStiffness * edge_length;
}

template<unsigned DIM>
double YanAnisotropicSpringForce<DIM>::GetLineTensionParameter(
    Node<DIM>* pNodeA,
    Node<DIM>* pNodeB,
    VertexBasedCellPopulation<DIM>& /*rVertexCellPopulation*/)
{
    std::set<unsigned> elements_containing_nodeA = pNodeA->rGetContainingElementIndices();
    std::set<unsigned> elements_containing_nodeB = pNodeB->rGetContainingElementIndices();

    std::set<unsigned> shared_elements;
    std::set_intersection(elements_containing_nodeA.begin(),
                          elements_containing_nodeA.end(),
                          elements_containing_nodeB.begin(),
                          elements_containing_nodeB.end(),
                          std::inserter(shared_elements, shared_elements.begin()));

    assert(!shared_elements.empty());

    c_vector<double, DIM> vector = pNodeB->rGetLocation() - pNodeA->rGetLocation();
    const double edge_length = norm_2(vector);
    const double edge_angle = std::atan2(vector[1], vector[0]);
    const double tension = ComputeEdgeTension(edge_length, edge_angle);

    if (shared_elements.size() == 1)
    {
        return tension;
    }
    return 0.5 * tension;
}

template<unsigned DIM>
double YanAnisotropicSpringForce<DIM>::GetBaselineLineTension() const
{
    return mBaselineLineTension;
}

template<unsigned DIM>
double YanAnisotropicSpringForce<DIM>::GetAnisotropyMagnitude() const
{
    return mAnisotropyMagnitude;
}

template<unsigned DIM>
double YanAnisotropicSpringForce<DIM>::GetLengthSpringStiffness() const
{
    return mLengthSpringStiffness;
}

template<unsigned DIM>
double YanAnisotropicSpringForce<DIM>::GetAnisotropyAngle() const
{
    return mAnisotropyAngle;
}

template<unsigned DIM>
double YanAnisotropicSpringForce<DIM>::GetAreaElasticity() const
{
    return this->mAreaElasticityParameter;
}

template<unsigned DIM>
void YanAnisotropicSpringForce<DIM>::SetBaselineLineTension(double baselineLineTension)
{
    mBaselineLineTension = baselineLineTension;
}

template<unsigned DIM>
void YanAnisotropicSpringForce<DIM>::SetAnisotropyMagnitude(double anisotropyMagnitude)
{
    mAnisotropyMagnitude = anisotropyMagnitude;
}

template<unsigned DIM>
void YanAnisotropicSpringForce<DIM>::SetLengthSpringStiffness(double lengthSpringStiffness)
{
    mLengthSpringStiffness = lengthSpringStiffness;
}

template<unsigned DIM>
void YanAnisotropicSpringForce<DIM>::SetAnisotropyAngle(double anisotropyAngle)
{
    mAnisotropyAngle = anisotropyAngle;
}

template<unsigned DIM>
void YanAnisotropicSpringForce<DIM>::SetAreaElasticity(double areaElasticity)
{
    this->SetAreaElasticityParameter(areaElasticity);
    this->SetTargetAreaParameter(1.0);
    this->SetPerimeterContractilityParameter(0.0);
}

template<unsigned DIM>
void YanAnisotropicSpringForce<DIM>::SetYanModelBParameters(double a_r, double b_0, double k, double phi_a)
{
    SetBaselineLineTension(1.0);
    SetAnisotropyMagnitude(a_r);
    SetLengthSpringStiffness(b_0);
    SetAreaElasticity(k);
    SetAnisotropyAngle(phi_a);
}

template<unsigned DIM>
void YanAnisotropicSpringForce<DIM>::OutputForceParameters(out_stream& rParamsFile)
{
    *rParamsFile << "\t\t\t<BaselineLineTension>" << mBaselineLineTension << "</BaselineLineTension>\n";
    *rParamsFile << "\t\t\t<AnisotropyMagnitude>" << mAnisotropyMagnitude << "</AnisotropyMagnitude>\n";
    *rParamsFile << "\t\t\t<LengthSpringStiffness>" << mLengthSpringStiffness << "</LengthSpringStiffness>\n";
    *rParamsFile << "\t\t\t<AnisotropyAngle>" << mAnisotropyAngle << "</AnisotropyAngle>\n";
    FarhadifarForce<DIM>::OutputForceParameters(rParamsFile);
}

template class YanAnisotropicSpringForce<1>;
template class YanAnisotropicSpringForce<2>;
template class YanAnisotropicSpringForce<3>;

#include "SerializationExportWrapperForCpp.hpp"
EXPORT_TEMPLATE_CLASS_SAME_DIMS(YanAnisotropicSpringForce)
