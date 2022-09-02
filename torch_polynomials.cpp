//
//  torch_polynomials.cpp
//  quick-potatoes
//
//  Created by Aion Feehan on 4/20/22.
//

#include <ATen/ATen.h>
#include <torch/csrc/api/include/torch/nn/functional.h>
#include "torch_polynomials.hpp"

TorchPolynomial::TorchPolynomial(torch::Tensor in_coefficients, bool in_requires_grad){
    coefficient_tensor = clean_trailing_zeros(in_coefficients);
    requires_grad = in_requires_grad;
    coefficient_tensor.set_requires_grad(requires_grad);
}

TorchPolynomial::TorchPolynomial(double in_coefficient, bool in_requires_grad): 
    coefficient_tensor(clean_trailing_zeros(in_coefficient * torch::ones(1))), 
    requires_grad(in_requires_grad){
        coefficient_tensor.set_requires_grad(requires_grad);
}

static torch::Tensor clean_trailing_zeros(torch::Tensor in_tensor){
    int n_zeros = 0;
    int tensor_size = in_tensor.size(0);
    for (int i = tensor_size; i > 0; --i){
        if (in_tensor[i].item<double>() == 0){
            n_zeros++;
        }
        else {
            break;
        }
    }
    torch::Tensor out_tensor = in_tensor.index({torch::arange(0, tensor_size - n_zeros)});
    return out_tensor;
}


torch::Tensor TorchPolynomial::coefficients() const {
     return coefficient_tensor;
}

size_t TorchPolynomial::degree() const {
    return coefficient_tensor.size(0) - 1;
}

namespace F = torch::nn::functional;

TorchPolynomial TorchPolynomial::operator+(const TorchPolynomial& other) const {
    const int this_degree = degree();
    const int other_degree = other.degree();
    const int new_degree = std::max(this_degree, other_degree);
    bool new_requires_grad = requires_grad || other.requires_grad;

    torch::Tensor this_padded_coefs = F::pad(coefficient_tensor, F::PadFuncOptions({0, new_degree - this_degree}));
    torch::Tensor other_padded_coefs = F::pad(other.coefficient_tensor, F::PadFuncOptions({0, new_degree - other_degree}));
    return this_padded_coefs + other_padded_coefs;
}

TorchPolynomial TorchPolynomial::operator+(const double other) const {
    return operator+(TorchPolynomial(other));
}

TorchPolynomial TorchPolynomial::operator*(const TorchPolynomial& other) const {
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

TorchPolynomial TorchPolynomial::operator*(const double other) const {
    return operator*(TorchPolynomial(other));
}

TorchPolynomial TorchPolynomial::operator-(const TorchPolynomial& other) const {
    TorchPolynomial minus_one = TorchPolynomial(torch::tensor((double) -1));
    return operator+(minus_one.operator*(other));
}

TorchPolynomial TorchPolynomial::operator-(const double other) const {
    return operator-(TorchPolynomial(other));
}


bool TorchPolynomial::operator==(const TorchPolynomial& other) const {
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

bool TorchPolynomial::operator!=(const TorchPolynomial& other) const {
    return not operator==(other);
}

/**
 * 
 * \fn TorchPolynomial TorchPolynomial::derivative()
 * @brief Computes the algebraic derivative of the function.
 * 
 *  \f$ f'(X) = \sum_{k=0}^n k * a_k * X^{k-1}
 * 
 * @return TorchPolynomial 
 */
TorchPolynomial TorchPolynomial::derivative() const {
    std::vector<torch::Tensor> new_coefficients;
    if (degree() == 0){
        return TorchPolynomial(torch::zeros(1));
    }
    for (int k = 1; k <= degree(); ++k){
        new_coefficients.push_back(k * coefficient_tensor[k]);
    }
    return TorchPolynomial(torch::stack(torch::TensorList(new_coefficients)), requires_grad);
}

TorchPolynomial TorchPolynomial::antiderivative() const {
    std::vector<torch::Tensor> new_coefficients({torch::zeros(1)});
    for (int k = 0; k < degree(); ++k){
        new_coefficients.push_back(coefficient_tensor[k] / (float) k + 1);
    }
    return TorchPolynomial(torch::stack(torch::TensorList(new_coefficients)), requires_grad);
}

torch::Tensor TorchPolynomial::operator()(const double t) const {
    torch::Tensor torch_t = torch::tensor(t);
    return TorchPolynomial::operator()(torch_t);
}

torch::Tensor TorchPolynomial::operator()(const torch::Tensor t) const {
    const size_t this_degree = degree();
    std::vector<torch::Tensor> powers;
    for (int k = 0; k <= this_degree; ++k){
        powers.push_back(t.pow(k));
    }
    torch::Tensor stacked_powers = torch::stack(powers);
    return torch::dot(stacked_powers, coefficient_tensor);
}

TorchPolynomial TorchPolynomial::clone() const {
    torch::Tensor cloned_values = coefficient_tensor.clone();
    return TorchPolynomial(cloned_values, requires_grad);
}

torch::Tensor TorchPolynomial::operator[](const int index) const {
    return coefficient_tensor[index];
}