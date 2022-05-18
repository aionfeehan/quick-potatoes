//
//  torch_polynomials.cpp
//  quick-potatoes
//
//  Created by Aion Feehan on 4/20/22.
//

#include <torch/csrc/api/include/torch/nn/functional.h>
#include <ATen/ATen.h>
#include "torch_polynomials.hpp"

TorchPolynomial::TorchPolynomial(torch::Tensor in_coefficients, bool in_requires_grad){
    coefficient_tensor = in_coefficients;
    requires_grad = in_requires_grad;
    coefficient_tensor.set_requires_grad(requires_grad);
}

TorchPolynomial::TorchPolynomial(double in_coefficient, bool in_requires_grad): 
    coefficient_tensor(torch::tensor(in_coefficient)), 
    requires_grad(in_requires_grad){
        coefficient_tensor.set_requires_grad(requires_grad);
}


torch::Tensor TorchPolynomial::coefficients(){
     return coefficient_tensor;
}

size_t TorchPolynomial::degree() const {
    return coefficient_tensor.size(0) - 1;
}

namespace F = torch::nn::functional;

TorchPolynomial TorchPolynomial::operator+(const TorchPolynomial& other){
    const int this_degree = degree();
    const int other_degree = other.degree();
    const int new_degree = std::max(this_degree, other_degree);
    bool new_requires_grad = requires_grad || other.requires_grad;

    torch::Tensor this_padded_coefs = F::pad(coefficient_tensor, F::PadFuncOptions({0, new_degree - this_degree}));
    torch::Tensor other_padded_coefs = F::pad(other.coefficient_tensor, F::PadFuncOptions({0, new_degree - other_degree}));
    return this_padded_coefs + other_padded_coefs;
}

TorchPolynomial TorchPolynomial::operator*(const TorchPolynomial& other){
    const int this_degree = degree();
    const int other_degree = other.degree();
    const int new_degree = this_degree + other_degree;
    bool new_requires_grad = requires_grad || other.requires_grad;
    torch::Tensor new_coefficients = torch::zeros({new_degree + 1, new_degree + 1});

    for (int i = 0; i <= this_degree; ++i){
        for (int j = 0; j <= other_degree; ++j){
            new_coefficients[i + j][j] = coefficient_tensor[i] * other.coefficient_tensor[j];
        }
    }
    return TorchPolynomial(torch::sum(new_coefficients, 1));
}

TorchPolynomial TorchPolynomial::operator-(const TorchPolynomial& other){
    TorchPolynomial minus_one = TorchPolynomial(torch::tensor((double) -1));
    return operator+(minus_one.operator*(other));
}


bool TorchPolynomial::operator==(TorchPolynomial& other){
    const int this_degree = degree();
    const int other_degree = other.degree();
    if (this_degree != other_degree) {
        return false;
    }
    else {
        bool are_equal = true;
        for (int k = 0; k <= this_degree; ++k){
            are_equal &= (coefficient_tensor[k].item<double>() == other.coefficient_tensor[k].item<double>());
            if (not are_equal){
                break;
            }
        }
        return are_equal;
    }
}

auto TorchPolynomial::operator[](int index){
    return coefficient_tensor[index]; 
}


TorchPolynomial TorchPolynomial::derivative(){
    torch::Tensor new_coefficients = torch::zeros(degree());
    new_coefficients.set_requires_grad(requires_grad);
    for (int k = 1; k <= degree(); ++k){
        new_coefficients[k-1] += k * coefficient_tensor[k];
    }
    return TorchPolynomial(new_coefficients, requires_grad);
}

TorchPolynomial TorchPolynomial::antiderivative(){
    torch::Tensor new_coefficients = torch::zeros(degree() + 1);
    new_coefficients.set_requires_grad(requires_grad);
    for (int k = 0; k < degree(); ++k){
        new_coefficients[k + 1] += coefficient_tensor[k] / (float) k + 1;
    }
    return TorchPolynomial(new_coefficients, requires_grad);
}