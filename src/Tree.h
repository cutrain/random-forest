#ifndef Tree_h
#define Tree_h

#include <vector>
#include <random>
//#include <iostream>
#include <armadillo>

// #include "Data.h"

typedef unsigned int uint;

class Tree {
public:


  Tree(arma::uvec&& lc,
       arma::uvec&& rc,
       arma::vec&& svars,
       arma::vec&& svals,
       arma::uvec&& isl){
    left_childs = lc;
    right_childs = rc;
    split_vars = svars;
    split_values = svals;
    isLeaf = isl;

  };

  // Tree();

  // Based on the data preparation step, the covariate values are integers
  //arma::uvec get_split_vars() const;
  const arma::vec& get_split_values() const;
  const arma::uvec& get_left_childs() const;
  const arma::uvec& get_right_childs() const;
  const arma::uvec& get_isLeaf() const;
  const arma::vec& get_split_vars() const;




private:
  arma::uvec left_childs;
  arma::uvec right_childs;
  arma::vec split_vars;
  arma::vec split_values;
  arma::uvec isLeaf;

};

#endif /* Tree_h */

