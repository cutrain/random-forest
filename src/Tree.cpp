#include "Tree.h"
#include "globals.h"


const arma::vec& Tree::get_split_vars() const
{
  return split_vars;
}

const arma::vec& Tree::get_split_values() const
{
  return split_values;
}

const arma::uvec& Tree::get_left_childs() const
{
  return left_childs;
}

const arma::uvec& Tree::get_right_childs() const
{
  return right_childs;
}

const arma::uvec& Tree::get_isLeaf() const
{
  return isLeaf;
}


