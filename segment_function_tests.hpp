/* 
    segment_functions_tests.hpp
    quick-potatoes

    Created by Aion Feehan on 8/7/22
 */

#pragma once

#include "segment_functions.hpp"

namespace segment_function_tests {
    SegmentFunction build_test_segment_function();
    int test_degree();
    int test_addition();
    int test_subtraction();
    int test_multiplication();
    int test_derivative();
    int test_antiderivative();
}