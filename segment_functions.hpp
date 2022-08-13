//
//  segment_functions.hpp
//  quick-potatoes
//
//  Created by Aion Feehan on 5/23/22.
//

#ifndef segment_functions_hpp
#define segment_functions_hpp

#include <stdio.h>
#include <torch/script.h>

#include "torch_polynomials.hpp"

class SegmentFunction{

    public:

        SegmentFunction(torch::Tensor in_exp_coefs, std::vector<TorchPolynomial> in_polynomials);
        SegmentFunction(std::vector<TorchPolynomial> in_polynomials);
        SegmentFunction(TorchPolynomial in_polynomial);
        SegmentFunction(double in_constant);

        torch::Tensor get_exp_coefs() const;
        std::vector<TorchPolynomial> get_polynomials() const;

        SegmentFunction operator+(const SegmentFunction& other) const;
        SegmentFunction operator+(const double other) const;
        SegmentFunction operator-(const SegmentFunction& other) const;
        SegmentFunction operator-(const double other) const;
        SegmentFunction operator*(const SegmentFunction& other) const;
        SegmentFunction operator*(const double other) const;
        bool operator==(const SegmentFunction& other) const;
        bool operator!=(const SegmentFunction& other) const;
        
        SegmentFunction pow(const int power) const;

        SegmentFunction derivative() const;
        SegmentFunction antiderivative() const;
        SegmentFunction get_exponential() const;

        size_t degree() const;
        void print() const;

    private:
        torch::Tensor exp_coefs;
        std::vector<TorchPolynomial> polynomials;

        void _align_by_exp_coef();
        TorchPolynomial _single_antiderivative(const TorchPolynomial& in_polynomial, double in_exp_coef) const;

};

#endif /* segment_functions_hpp */