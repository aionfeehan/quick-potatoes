/* 
    segment_function_tests.cpp
    quick-potatoes

    Created by Aion Feehan on 8/7/22
 */

#include <iostream>
#include <string>
#include "segment_functions.hpp"
#include "segment_function_tests.hpp"

SegmentFunction segment_function_tests::build_test_segment_function(){
    int degree_1 = 2, degree_2 = 0;
    torch::Tensor test_tensor_1 = torch::ones(degree_1 + 1);
    torch::Tensor test_tensor_2 = torch::ones(degree_2 + 1);
    TorchPolynomial test_poly_1 = TorchPolynomial(test_tensor_1);
    TorchPolynomial test_poly_2 = TorchPolynomial(test_tensor_2);
    torch::Tensor exp_coefs = torch::tensor({1, 0});
    std::vector<TorchPolynomial> input_vector{test_poly_1, test_poly_2};
    SegmentFunction test_segf = SegmentFunction(exp_coefs, input_vector);
    return test_segf;
}

int segment_function_tests::test_degree(){
    SegmentFunction test_segf = build_test_segment_function();
    bool is_correct = (test_segf.degree() == 2);
    std::string output_message = is_correct ? "Degree passed " : "Degree FAILED";
    std::cout << output_message << std::endl;
    if (not is_correct) {
        std::cout << "Degree expected: " << "2     " << "Degree received: " << test_segf.degree() << std::endl; 
    }
    int num_errors = static_cast<int>(not is_correct);
    return num_errors;
}

int segment_function_tests::test_addition(){
    SegmentFunction test_segf1 = build_test_segment_function();
    SegmentFunction test_segf2 = build_test_segment_function();
    torch::Tensor target_coeffs_1 = 2 * torch::ones(3);
    torch::Tensor target_coefs_2 = 2 * torch::ones(1);
    torch::Tensor target_exp_coefs = torch::tensor({1, 0});
    std::vector<TorchPolynomial> target_polynomials{TorchPolynomial(target_coeffs_1), TorchPolynomial(target_coefs_2)};

    SegmentFunction target_segf = SegmentFunction(target_exp_coefs, target_polynomials);
    SegmentFunction test_segf = test_segf1 + test_segf2;

    bool is_correct = (target_segf == test_segf);
    std::string output_message = is_correct ? "Addition passed " : "Addition FAILED";
    std::cout << output_message << std::endl;
    if (not is_correct){
        std::cout << "Results:" << std::endl;
        std::cout << "Target function: " << std::endl;
        target_segf.print();

        std::cout << "Received function: " << std::endl;
        test_segf.print();
    }
    int num_errors = static_cast<int>(not is_correct);
    return num_errors;
}

int segment_function_tests::test_subtraction(){
    SegmentFunction test_segf1 = build_test_segment_function();
    SegmentFunction test_segf2 = build_test_segment_function();
    torch::Tensor target_coeffs_1 = 0 * torch::ones(3);
    torch::Tensor target_coefs_2 = 0 * torch::ones(1);
    torch::Tensor target_exp_coefs = torch::tensor({1, 0});
    std::vector<TorchPolynomial> target_polynomials{TorchPolynomial(target_coeffs_1), TorchPolynomial(target_coefs_2)};

    SegmentFunction target_segf = SegmentFunction(target_exp_coefs, target_polynomials);
    SegmentFunction test_segf = test_segf1 - test_segf2;

    bool is_correct = (target_segf == test_segf);
    std::string output_message = is_correct ? "Subtraction passed " : "Subtraction FAILED";
    std::cout << output_message << std::endl;

    if (not is_correct) {
        std::cout << "Target function: " << std::endl;
        target_segf.print();

        std::cout << "Received function: " << std::endl;
        test_segf.print();
    }
    return static_cast<int>(not is_correct);
}

int segment_function_tests::test_derivative(){
    std::vector<TorchPolynomial> test_polynomial_1({TorchPolynomial(torch::ones(3))});
    SegmentFunction test_segf_1 = SegmentFunction(test_polynomial_1);
    torch::Tensor target_coefs_1 = torch::zeros(2);
    target_coefs_1[0] += 1;
    target_coefs_1[1] += 2;
    std::vector<TorchPolynomial> in_polynomials_1({target_coefs_1});
    SegmentFunction target_segf_1(in_polynomials_1);

    SegmentFunction test_derivative_1 = test_segf_1.derivative();
    bool test_1_is_correct = (target_segf_1 == test_derivative_1);
    std::string output_message = test_1_is_correct ? "First derivative test passed " : "First derivative test FAILED";
    std::cout << output_message << std::endl;

    if (not test_1_is_correct){

        std::cout << "Target function: " << std::endl;
        target_segf_1.print();

        std::cout << "Received function: " << std::endl;
        test_derivative_1.print();
    }


    std::vector<TorchPolynomial> test_polynomial_2({TorchPolynomial(torch::ones(1))});
    torch::Tensor exp_coefs_2 = torch::ones(1);
    SegmentFunction test_segf_2(exp_coefs_2, test_polynomial_2);
    SegmentFunction target_segf_2 = test_segf_2;

    SegmentFunction test_derivative_2 = test_segf_2.derivative();
    bool test_2_is_correct = (target_segf_2 == test_derivative_2);

    output_message = test_2_is_correct ? "Second derivative test passed " : "Second derivative test FAILED";
    std::cout << output_message << std::endl;

    if (not test_2_is_correct){
        std::cout << "Target function: " << std::endl;
        test_derivative_2.print();

        std::cout << "Received function: " << std::endl;
        target_segf_2.print();
    }

    int n_errors = static_cast<int>(not test_1_is_correct) + static_cast<int>(not test_2_is_correct);
    return n_errors;

}