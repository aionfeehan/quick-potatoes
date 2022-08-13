//
//  main.cpp
//  quick-potatoes
//
//  Created by Aion Feehan on 4/20/22.
//

#include <stdio.h>
#include "torch_polynomials_tests.hpp"
#include "segment_function_tests.hpp"

int main(int argc, const char * argv[]) {
    // insert code here...
    int num_errors = 0;
    std::cout << "Testing TorchPolynomials" << std::endl;
    num_errors += torch_polynomials_tests::test_degree();
    num_errors += torch_polynomials_tests::test_addition();
    num_errors += torch_polynomials_tests::test_multiplication();


    std::cout << "Testing SegmentFunction" << std::endl;
    num_errors += segment_function_tests::test_degree();
    num_errors += segment_function_tests::test_addition();
    num_errors += segment_function_tests::test_subtraction();
    num_errors += segment_function_tests::test_derivative();
    std::cout << "Found " << num_errors << " errors" << std::endl;
    return 0;
}  
