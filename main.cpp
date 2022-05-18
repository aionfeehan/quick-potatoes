//
//  main.cpp
//  quick-potatoes
//
//  Created by Aion Feehan on 4/20/22.
//

#include <iostream>
#include <string>
#include "torch_polynomials.hpp"

int test_degree(){
    int degree = 2;
    torch::Tensor my_tensor = torch::ones(degree + 1);
    TorchPolynomial my_polynomial = TorchPolynomial(my_tensor);
    bool is_correct = (my_polynomial.degree() == degree);
    std::string output_message = is_correct ? "Degree passed: " : "Degree FAILED: ";
    std::cout << output_message << "Degree = " << my_polynomial.degree() << std::endl;
    int num_errors = (int) not is_correct;
    return num_errors;
}

int test_addition(){
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
    std::cout << output_message << std::endl;
    int num_errors = (int) not is_correct;
    return num_errors;
}

int test_multiplication(){
    torch::Tensor a_tensor = torch::ones(2);
    torch::Tensor b_tensor = torch::ones(2);
    TorchPolynomial a_polynomial = TorchPolynomial(a_tensor);
    TorchPolynomial b_polynomial = TorchPolynomial(b_tensor);

    TorchPolynomial result_polynomial = a_polynomial * b_polynomial;
    std::vector<float> f_target = {1, 2, 1};
    TorchPolynomial target_polynomial = TorchPolynomial(torch::tensor(f_target));
    bool is_correct = (target_polynomial == result_polynomial);
    std::string output_message = is_correct ? "Multiplication passed" : "Multiplication FALIED";
    std::cout << output_message << std::endl;
    int num_errors = (int) not is_correct;
    return num_errors;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    int num_errors = 0;
    num_errors += test_degree();
    num_errors += test_addition();
    num_errors += test_multiplication();
    std::cout << "Found " << num_errors << " errors" << std::endl;
    return 0;
}  
