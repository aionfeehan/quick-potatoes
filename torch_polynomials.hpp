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


class TorchPolynomial{

    torch::Tensor coefficient_tensor;
    bool requires_grad;

    public:

        TorchPolynomial(torch::Tensor in_coefficients, bool in_requires_grad=true);
        TorchPolynomial(double in_coefficient, bool in_requires_grad=true);

        TorchPolynomial operator+(const TorchPolynomial& other);
        TorchPolynomial operator-(const TorchPolynomial& other);
        TorchPolynomial operator*(const TorchPolynomial& other);

        bool operator==(TorchPolynomial& other);
        auto operator[](int index);

        torch::Tensor coefficients();
        size_t degree() const;

        TorchPolynomial derivative();
        TorchPolynomial antiderivative();
};

#endif /* torch_polynomials_hpp */
