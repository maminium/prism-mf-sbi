#ifndef YANANISOTROPICSPRINGFORCE_HPP_
#define YANANISOTROPICSPRINGFORCE_HPP_

#include "ChasteSerialization.hpp"
#include <boost/serialization/base_object.hpp>

#include "FarhadifarForce.hpp"

/**
 * Ogita/Yan anisotropic-spring Model B as a FarhadifarForce subclass
 * (same pattern as Chaste PlanarPolarisedFarhadifarForce).
 *
 * Generative tension (paper scale λ0 = 1):
 *
 *   T(ell, theta) = λ0 * (1 + a_r * cos(2*(theta - phi_a))) - b_0 * ell
 *
 * with λ0 frozen at 1.  Static Bayes Model-B code often writes only
 *   a_r cos(...) - b_0 ell
 * because force-balance has a tension gauge; omitting λ0 in *dynamics*
 * makes long edges strongly expansive and the tissue collapses into needles.
 *
 * Area: Farhadifar with A0=1  =>  F = -k (A-1) grad A
 * (equiv. P = -k(A-A0) after nondim). Gamma = 0.
 *
 * AddForceContribution is inherited; only GetLineTensionParameter is overridden.
 */
template<unsigned DIM>
class YanAnisotropicSpringForce : public FarhadifarForce<DIM>
{
friend class TestYanVertexValidation;

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & archive, const unsigned int version)
    {
        archive & boost::serialization::base_object<FarhadifarForce<DIM> >(*this);
        archive & mBaselineLineTension;
        archive & mAnisotropyMagnitude;
        archive & mLengthSpringStiffness;
        archive & mAnisotropyAngle;
    }

protected:
    /** λ0 : isotropic baseline line tension (Yan scale; Stage A freeze = 1). */
    double mBaselineLineTension;

    /** a_r : relative anisotropy amplitude (μ0 when λ0=1). */
    double mAnisotropyMagnitude;

    /** b_0 : isotropic length–tension coefficient (λ1 when μ1=0). */
    double mLengthSpringStiffness;

    /** phi_a : preferred anisotropy axis (radians); Stage A freeze = 0. */
    double mAnisotropyAngle;

public:
    YanAnisotropicSpringForce();

    virtual ~YanAnisotropicSpringForce();

    /**
     * Yan Model-B tension at current edge length and angle (no half-counting).
     */
    double ComputeEdgeTension(double edge_length, double edge_angle) const;

    /**
     * Overridden Farhadifar hook: T(ell,theta) with half-counting for internal edges.
     */
    virtual double GetLineTensionParameter(Node<DIM>* pNodeA,
                                           Node<DIM>* pNodeB,
                                           VertexBasedCellPopulation<DIM>& rVertexCellPopulation);

    double GetBaselineLineTension() const;
    double GetAnisotropyMagnitude() const;
    double GetLengthSpringStiffness() const;
    double GetAnisotropyAngle() const;
    double GetAreaElasticity() const;

    void SetBaselineLineTension(double baselineLineTension);
    void SetAnisotropyMagnitude(double anisotropyMagnitude);
    void SetLengthSpringStiffness(double lengthSpringStiffness);
    void SetAnisotropyAngle(double anisotropyAngle);
    void SetAreaElasticity(double areaElasticity);

    /** Set (a_r, b_0, k), freeze λ0=1 and φa, zero perimeter contractility. */
    void SetYanModelBParameters(double a_r, double b_0, double k, double phi_a = 0.0);

    void OutputForceParameters(out_stream& rParamsFile);
};

#include "SerializationExportWrapper.hpp"
EXPORT_TEMPLATE_CLASS_SAME_DIMS(YanAnisotropicSpringForce)

#endif /*YANANISOTROPICSPRINGFORCE_HPP_*/
