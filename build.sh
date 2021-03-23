#!/bin/bash
#export PKG_CXXFLAGS=`Rscript -e "Rcpp:::CxxFlags()"`' '`Rscript -e "RcppArmadillo:::CxxFlags()"`
R CMD SHLIB -o lib/qr_simplex.so src/*.cpp
