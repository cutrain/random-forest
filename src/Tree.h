#ifndef Tree_h
#define Tree_h

#include <vector>
#include <random>
#include <memory>
//#include <iostream>
#include <armadillo>

// #include "Data.h"

typedef unsigned int uint;
using std::shared_ptr;
using std::make_shared;
using arma::vec;
using arma::uvec;

class Tree {
public:

  shared_ptr<uvec> left_childs;
  shared_ptr<uvec> right_childs;
  shared_ptr<vec> split_vars;
  shared_ptr<vec> split_values;
  shared_ptr<uvec> isLeaf;

  Tree(uvec lc,
       uvec rc,
       vec svars,
       vec svals,
       uvec isl){
    left_childs = make_shared<uvec>(lc);
    right_childs = make_shared<uvec>(rc);
    split_vars = make_shared<vec>(svars);
    split_values = make_shared<vec>(svals);
    isLeaf = make_shared<uvec>(isl);
  };

  // Tree();

  // Based on the data preparation step, the covariate values are integers
  shared_ptr<uvec> get_left_childs() const;
  shared_ptr<uvec> get_right_childs() const;
  shared_ptr<vec> get_split_vars() const;
  shared_ptr<vec> get_split_values() const;
  shared_ptr<uvec> get_isLeaf() const;

};

#endif /* Tree_h */

