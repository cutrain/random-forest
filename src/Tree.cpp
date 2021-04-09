#include "Tree.h"
#include "globals.h"
#include <memory>


std::shared_ptr<arma::uvec> Tree::get_left_childs() const
{
  return left_childs;
}

std::shared_ptr<arma::uvec> Tree::get_right_childs() const
{
  return right_childs;
}

std::shared_ptr<arma::vec> Tree::get_split_vars() const
{
  return split_vars;
}

std::shared_ptr<arma::vec> Tree::get_split_values() const
{
  return split_values;
}

std::shared_ptr<arma::uvec> Tree::get_isLeaf() const
{
  return isLeaf;
}
