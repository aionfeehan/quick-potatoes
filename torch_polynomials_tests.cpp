
//
//  torch_polynomaisl_tests.cpp
//  quick-potatoes
//
//  Created by Aion Feehan on 8/6/22.
//


#include <iostream>
#include <string>
#include "torch_polynomials.hpp"
#include "torch_polynomials_tests.hpp"

int torch_polynomials_tests::test_degree(){
    int degree = 2;
    torch::Tensor my_tensor = torch::ones(degree + 1);
    TorchPolynomial my_polynomial = TorchPolynomial(my_tensor);
    bool is_correct = (my_polynomial.degree() == degree);
    std::string output_message = is_correct ? "Degree passed " : "Degree FAILED: ";
    std::cout << output_message << "Degree = " << my_polynomial.degree() << "\n";
    int num_errors = (int) not is_correct;
    return num_errors;
}

int torch_polynomials_tests::test_addition(){
    torch::Tensor a_tensor = torch::ones(3);
    torch::Tensor b_tensor = torch::ones(4);
    TorchPolynomial a_polynomial = TorchPolynomial(a_tensor);
    TorchPolynomial b_polynomial = TorchPolynomial(b_tensor);

    TorchPolynomial result_polynomial = a_polynomial + b_polynomial;
    std::vector<float> f_target = {2, 2, 2, 1};
    torch::Tensor target_tensor = torch::tensor(f_target);
    TorchPolynomial target_polynomial = TorchPolynomial(target_tensor);
    bool is_correct = (target_polynomial == result_polynomial);
    std::string output_message = is_correct ? "Addition passed" : "Addition FALIED";
    std::cout << output_message << "\n";
    int num_errors = (int) not is_correct;
    return num_errors;
}

int torch_polynomials_tests::test_multiplication(){
    torch::Tensor a_tensor = torch::ones(2);
    torch::Tensor b_tensor = torch::ones(2);
    TorchPolynomial a_polynomial = TorchPolynomial(a_tensor);
    TorchPolynomial b_polynomial = TorchPolynomial(b_tensor);

    TorchPolynomial result_polynomial = a_polynomial * b_polynomial;
    std::vector<float> f_target = {1, 2, 1};
    TorchPolynomial target_polynomial = TorchPolynomial(torch::tensor(f_target));
    bool is_correct = (target_polynomial == result_polynomial);
    std::string output_message = is_correct ? "Multiplication passed" : "Multiplication FALIED";
    std::cout << output_message << "\n";
    int num_errors = (int) not is_correct;

    torch::Tensor result_tensor = result_polynomial.coefficients();
    torch::sum(result_tensor).backward();
    torch::Tensor a_grad = a_tensor.grad();

    std::cout << "Multiplication grad: " << a_grad[0] << "\n";
    return num_errors;
}