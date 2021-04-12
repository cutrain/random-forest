#ifndef ForestPrediction_h
#define ForestPrediction_h

#include "Tree.h"
#include "Forest.h"
#include "common.h"

class ForestPrediction {
public:

  ForestPrediction(const arma::mat& matX,
                   const arma::umat& ids,
                   const std::vector<std::shared_ptr<Tree> >& trees);



  const arma::field<arma::uvec>& get_nodeSize() const
  {
    return nodeSizeB;
  };
  const arma::umat& get_nodeLabel() const
  {
    return nodeLabelB;
  };
  const arma::field<arma::uvec>& get_nodeMap() const
  {
    return tndB;
  };
private:
  arma::field<arma::uvec> nodeSizeB;

  arma::umat nodeLabelB; // Each column is for one tree

  // vector that defines a map between node number and its location in matrix nodeSize
  arma::field<arma::uvec> tndB;

};

#endif /* ForestPrediction_h */
