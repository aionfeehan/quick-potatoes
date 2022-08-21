//
//  segment_functions.cpp
//  quick-potatoes
//
//  Created by Aion Feehan on 5/23/22.
//

#include <stdio.h>
#include <torch/csrc/api/include/torch/all.h>
#include "segment_functions.hpp"

SegmentFunction::SegmentFunction(torch::Tensor in_exp_coefs, std::vector<TorchPolynomial> in_polynomials):
    exp_coefs(in_exp_coefs),
    polynomials(in_polynomials){
    _align_by_exp_coef();
}


SegmentFunction::SegmentFunction(
    std::vector<TorchPolynomial> in_polynomials
){
    int nb_polynomials = in_polynomials.size();
    exp_coefs = torch::zeros(nb_polynomials);
    polynomials = in_polynomials;
    _align_by_exp_coef();
}

SegmentFunction::SegmentFunction(TorchPolynomial in_polynomial):
    polynomials(std::vector<TorchPolynomial>{in_polynomial}){}

SegmentFunction::SegmentFunction(double in_constant):
    polynomials(std::vector<TorchPolynomial>{TorchPolynomial(in_constant)}){}

torch::Tensor SegmentFunction::get_exp_coefs() const {
    return exp_coefs;
}

std::vector<TorchPolynomial> SegmentFunction::get_polynomials() const {
    return polynomials;
}

SegmentFunction SegmentFunction::operator+(const SegmentFunction& other) const {
    torch::Tensor to_cat[2] = {exp_coefs, other.exp_coefs};
    torch::Tensor in_exp_coefs = torch::cat(to_cat);
    std::vector<TorchPolynomial> new_polynomials;
    new_polynomials.insert(new_polynomials.end(), polynomials.begin(), polynomials.end());
    new_polynomials.insert(new_polynomials.end(), other.polynomials.begin(), other.polynomials.end());
    SegmentFunction new_segment_function = SegmentFunction(
        in_exp_coefs,
        new_polynomials
    );
    return new_segment_function;
}

SegmentFunction SegmentFunction::operator+(const double other) const {
    return operator+(SegmentFunction(other));
}

SegmentFunction SegmentFunction::operator-(const SegmentFunction& other) const {
    std::vector<TorchPolynomial> new_other_polynomials;
    int n_other_polynomials = other.polynomials.size();
    for (int i = 0; i < n_other_polynomials; ++i){
        TorchPolynomial minus_one = TorchPolynomial(static_cast<double>(-1));
        TorchPolynomial new_polynomial = minus_one * other.polynomials[i];
        new_other_polynomials.push_back(new_polynomial);
    }
    SegmentFunction new_other = SegmentFunction(
        other.exp_coefs,
        new_other_polynomials
    );
    return operator+(new_other);
}

SegmentFunction SegmentFunction::operator-(const double other) const {
    return operator-(SegmentFunction(other));
}

SegmentFunction SegmentFunction::pow(const int power) const {
    if (power == 1){
        return *this;
    }
    else if (power == 0){
        return SegmentFunction(
            TorchPolynomial(torch::ones(1))
        );
    }
    else {
        return pow(power / 2) * pow(power - power / 2);
    }
}

SegmentFunction SegmentFunction::operator*(const SegmentFunction& other) const {
    std::vector<TorchPolynomial> new_polynomials;
    std::vector<torch::Tensor> new_exp_coefs;
    int n_this_polynomials = polynomials.size();
    int n_other_polynomials = other.polynomials.size();
    for (int i = 0; i < n_this_polynomials; ++i){
        for (int j = 0; j < n_other_polynomials; ++j){
            new_polynomials.push_back(polynomials[i] * other.polynomials[j]);
            new_exp_coefs.push_back(exp_coefs[i] + other.exp_coefs[j]);
        }
    }
    return SegmentFunction(
        torch::stack(new_exp_coefs),
        new_polynomials
    );
}

SegmentFunction SegmentFunction::operator*(const double other) const {
    return this->operator*(SegmentFunction(other));
}

bool SegmentFunction::operator==(const SegmentFunction& other) const {
    if (exp_coefs.size(0) != other.exp_coefs.size(0)){
        return false;
    }
    for (int i = 0; i < exp_coefs.size(0); ++i){
        // We assume that the exp_coefs for each SegmentFunction are already sorted
        if (exp_coefs[i].item<double>() != other.exp_coefs[i].item<double>()){
            return false;
        }
        if (polynomials[i] != other.polynomials[i]){
            return false;
        }
    }
    return true;
}

bool SegmentFunction::operator!=(const SegmentFunction& other) const {
    return not operator==(other);
}

void SegmentFunction::_align_by_exp_coef(){
    std::tuple<at::Tensor, at::Tensor, at::Tensor> values_invindex_counts = at::_unique2(exp_coefs, false, false, true);
    at::Tensor values = std::get<0>(values_invindex_counts);
    at::Tensor inverse_index = std::get<1>(values_invindex_counts);
    at::Tensor counts = std::get<2>(values_invindex_counts);
    std::tuple<at::Tensor, at::Tensor> sorted_argsort = torch::sort(values);

    values = std::get<0>(sorted_argsort);
    at::Tensor sorted_index = std::get<1>(sorted_argsort);
    counts = counts.index({sorted_index});
    const int num_coefs = exp_coefs.size(0);
    const int num_unique_coefs = values.size(0);
    if (num_unique_coefs == num_coefs) {
        // if we do not need to factor out any polynomials, we make sure the polynomials are sorted by exp coef

        // sorted_index is NOT the order for exp_coefs, but for values... 
        // we need to make sure that gradients will be properly passed through...

        exp_coefs = exp_coefs.index({inverse_index});
        std::vector<TorchPolynomial> new_polynomials;
        for (int i = 0; i < polynomials.size(); ++i){
            new_polynomials.push_back(polynomials[inverse_index[i].item<int>()]);
        }
        polynomials = new_polynomials;
        return;
    }
    else {
        std::vector<int> idx_to_keep;
        for (int k = 0; k < num_unique_coefs; ++k) {
            if (counts[k].item<double>() > 1){
                std::vector<int> is_value_idx;
                double value_to_match = values[k].item<double>();
                for (int i = 0; i < num_coefs; ++i){
                    if (exp_coefs[i].item<double>() == value_to_match){
                        is_value_idx.push_back(i);
                    }
                }
                idx_to_keep.push_back(is_value_idx[0]);
                int num_duplicates = is_value_idx.size();
                for (int i = 1; i < num_duplicates; ++i){
                    int idx_to_drop = is_value_idx[i];
                    TorchPolynomial old_polynomial = polynomials[is_value_idx[0]].clone();
                    polynomials[is_value_idx[0]] = old_polynomial + polynomials[idx_to_drop];
                }
            }
        }
        torch::Tensor new_exp_coefs = torch::zeros(idx_to_keep.size());
        std::vector<TorchPolynomial> new_polynomials;
        for (int k = 0; k < num_unique_coefs; ++k){
            new_exp_coefs[k] += exp_coefs[idx_to_keep[k]].item<double>();
            new_polynomials.push_back(polynomials[idx_to_keep[k]]);
        }
        exp_coefs = new_exp_coefs;
        polynomials = new_polynomials;
    }
}

SegmentFunction SegmentFunction::derivative() const {
    torch::Tensor new_exp_coefs = torch::stack({exp_coefs.clone(), exp_coefs.clone()});
    const int n_polynomials = exp_coefs.size(0);
    std::vector<TorchPolynomial> new_polynomials;
    for (int i = 0; i < n_polynomials; ++i) {
        new_polynomials.push_back(polynomials[i].derivative());
    }
    for (int i = 0; i < n_polynomials; ++i) {
        new_polynomials.push_back(polynomials[i] * exp_coefs[i].item<double>());
    }
    return SegmentFunction(
        new_exp_coefs,
        new_polynomials
    );
}

TorchPolynomial SegmentFunction::_single_antiderivative(const TorchPolynomial& in_polynomial, double in_exp_coef) const {
    if (in_exp_coef == 0){
        return in_polynomial.antiderivative();
    }
    else if (in_polynomial.degree() > 0) {
        TorchPolynomial derived_polynomial = in_polynomial.derivative() * (1 / in_exp_coef);
        return in_polynomial * (1 / in_exp_coef) - _single_antiderivative(derived_polynomial, in_exp_coef);
    }
    else {
        return in_polynomial * (1 / in_exp_coef);
    }
}

SegmentFunction SegmentFunction::antiderivative() const {
    std::vector<TorchPolynomial> new_polynomials;
    const int n_polynomials = polynomials.size();
    for (int i = 0; i < n_polynomials; ++i) {
        new_polynomials.push_back(_single_antiderivative(polynomials[i], exp_coefs[i].item<double>()));
    }
    return SegmentFunction(
        exp_coefs.clone(),
        new_polynomials
    );
}

SegmentFunction SegmentFunction::get_exponential() const {
    assert(degree() <= 1);
    bool exp_coefs_are_zero = true;
    for (int i = 0; i < exp_coefs.size(0); ++i){
       if (exp_coefs_are_zero) exp_coefs_are_zero = (exp_coefs[i].item<double>() == static_cast<double>(0));
    }
    assert(exp_coefs_are_zero);
    std::vector<TorchPolynomial> constants;
    torch::Tensor new_exp_coefs = torch::zeros(exp_coefs.size(0));
    for (int i = 0; i < polynomials.size(); ++i) {
        TorchPolynomial polynomials_i = polynomials[i];
        torch::Tensor exp_coef_i = polynomials_i[0];
        constants.push_back(TorchPolynomial(torch::exp(exp_coef_i)));
        if (polynomials_i.degree() == 1){
            double additional_exp_coef = polynomials_i[1].item<double>();
            new_exp_coefs[i] += additional_exp_coef;
        }
    }
    return SegmentFunction(
        new_exp_coefs,
        constants
    );
}

size_t SegmentFunction::degree() const {
    size_t greatest_degree = 0;
    for (int i = 0; i < polynomials.size(); ++i){
        if (polynomials[i].degree() > greatest_degree){
            greatest_degree = polynomials[i].degree();
        }
    }
    return greatest_degree;
}

void SegmentFunction::print() const {
    for (int i = 0; i < polynomials.size(); ++i){
            std::cout << "Exp " << exp_coefs[i].item<double>() << " * ";
            TorchPolynomial polynomial_i = polynomials[i];
            for (int j = 0; j < polynomial_i.coefficients().size(0); ++j){
                std::cout << polynomial_i.coefficients()[j].item<double>() << " ";
            }
            std::cout << std::endl;
        }
}