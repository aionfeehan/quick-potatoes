# quick-potatoes

## Quick potatoes is a library of scripts designed to support native automatic differentiation of term structure models via a libtorch backend


#### Currently a work in progress

This is a C++ implementation of the closely related callable-potatoes repository, where I built a proof of concept of the model in Python. Language constraints being what they are, and most of the libtorch API being accessible in C++ anyway, I figured it was better to continue the work in C++.

Current status: TorchPolynomial base class (use for term structures) passes simple unit tests, working on building out the next layer of the term structure, the SegmentFunction. Code is written, currently in testing phase.