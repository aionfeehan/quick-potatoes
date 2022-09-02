//
//  torch_polynomials.hpp
//  quick-potatoes
//
//  Created by Aion Feehan on 4/20/22.
//

#ifndef torch_polynomials_hpp
#define torch_polynomials_hpp

#include <stdio.h>
#include <torch/script.h>

/**
 * 
 * @brief libtorch - compatible representation of \f$ f(x) = \sum_{k=0}^n a_kX^k \f$
 * 
 * Representation of \f$ f(x) = \sum_{k=0}^n a_kX^k \f$ as a tensor of coefficients \f$ a_k \f$ We support standard operations between polynomials
 * via their algebraic definition, and funciton application to double input \f$ x \f$
 */
class TorchPolynomial{

    public:

        TorchPolynomial(torch::Tensor in_coefficients, bool in_requires_grad=true);
        TorchPolynomial(double in_coefficient, bool in_requires_grad=true);

        TorchPolynomial operator+(const TorchPolynomial& other) const;
        TorchPolynomial operator+(const double other) const;
        TorchPolynomial operator-(const TorchPolynomial& other) const;
        TorchPolynomial operator-(const double other) const;
        TorchPolynomial operator*(const TorchPolynomial& other) const;
        TorchPolynomial operator*(const double other) const;

        torch::Tensor operator()(const torch::Tensor t) const;
        torch::Tensor operator()(const double t) const;

        bool operator==(const TorchPolynomial& other) const;
        bool operator!=(const TorchPolynomial& other) const;

        torch::Tensor operator[](int index) const;

        torch::Tensor coefficients() const;
        size_t degree() const;

        /**
         *  Some nice documentation here
         * 
         *  
         */
        TorchPolynomial derivative() const;
        TorchPolynomial antiderivative() const;
        TorchPolynomial clone() const;

    private:
        torch::Tensor coefficient_tensor;
        bool requires_grad;
        static torch::Tensor clean_trailing_zeros(torch::Tensor in_tensor);
};

#endif /* torch_polynomials_hpp */
