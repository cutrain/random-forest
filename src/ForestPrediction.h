#ifndef ForestPrediction_h
#define ForestPrediction_h

#include "Tree.h"
#include "Forest.h"
#include "common.h"

class ForestPrediction {
public:

  ForestPrediction(const arma::mat& matX,
                   const arma::mat& matX0,
                   const arma::umat& ids,
                   const std::vector<std::shared_ptr<Tree> >& trees) {
    uint n_X = matX.n_rows;
    uint n_X0 = matX0.n_rows;
    uint NUM_TREE = trees.size();
    arma::umat ndy = arma::zeros<arma::umat>(NUM_TREE, n_X);
    arma::field<arma::uvec> ndsz(NUM_TREE);
    arma::field<arma::uvec> tnd3B(NUM_TREE);
    std::vector<std::shared_ptr<Tree> >::const_iterator it;
    int i = 0;
    // std::cout<<"DONE1"<<std::endl;
    for(it = trees.begin(); it != trees.end(); it++, i++) {
      // take the i th tree
      arma::uvec idi = ids.col(i);
      std::shared_ptr<Tree> tt = *it;
      arma::vec vars =  *tt->get_split_vars();
      arma::uvec lcs = *tt->get_left_childs();
      arma::uvec rcs = *tt->get_right_childs();
      arma::vec values = *tt->get_split_values();
      arma::uvec il = *tt->get_isLeaf();
      int nNd = arma::accu(il);
      arma::uvec ndszi = arma::zeros<arma::uvec>(nNd);

      // tnd: leaf number;
      arma::uvec tnd = arma::zeros<arma::uvec>(il.n_elem);
      arma::uvec tnd2 = (arma::find(il == 1));
      tnd.elem( tnd2 ) = arma::regspace<arma::uvec>(0, nNd-1);
      // std::cout<<"DONE2"<<std::endl;
      for(size_t j = 0; j != n_X; j++) {
        arma::rowvec matXj = matX.row( j );
        int isl = 0;
        int varsp = 0;
        double cutsp = 0;
        size_t k = 0;
        while(isl == 0) {
          varsp = vars(k);
          // std::cout<<"varsp is "<<varsp<<std::endl;
          cutsp = values(k);
          // std::cout<<"cutsp is "<<cutsp<<std::endl;
          if(matXj(varsp) > cutsp) {
            k = rcs(k);
          } else {
            k = lcs(k);
          }
          isl = il(k);
        }
        // std::cout<<"DONE3"<<std::endl;
        ndy(i,j) = k;
        ndszi(tnd(k))++;
      }

      ndsz(i) = ndszi;
      tnd3B(i) = tnd;

    }


    arma::mat rfweights = arma::zeros<arma::mat>(n_X,n_X0);
    for(uint it = 0;it<NUM_TREE;it++){
      // std::cout<<"it is "<<it<<std::endl;
      arma::uvec ndsz_temp = ndsz(it);
      // std::cout<<"DONE1"<<std::endl;
      arma::urowvec nodelabel = ndy.row(it);
      arma::uvec nodemap = tnd3B(it);

      for(uint i = 0;i<n_X;i++){
        // std::cout<<"i is "<<i<<std::endl;
        uint nl = nodelabel(i);
        uint nd = nodemap(nl);
        // std::cout<<"nd is "<<nd<<std::endl;
        arma::uvec id_temp = ids.col(it);
        for(uint l:id_temp){
          if(nodelabel(l)==nl){
            rfweights(l*n_X+i) = rfweights(l*n_X+i)+1/(double) ndsz_temp(nd);
          }

        }
      }
    }


    this->nodeLabelB = ndy;
    this->nodeSizeB = ndsz;
    this->tndB = tnd3B;
    this->rfweights = rfweights;
  }


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
  const arma::mat& get_weights() const
  {
    return rfweights;
  };
private:
  arma::field<arma::uvec> nodeSizeB;

  arma::umat nodeLabelB; // Each column is for one tree

  // vector that defines a map between node number and its location in matrix nodeSize
  arma::field<arma::uvec> tndB;

  arma::mat rfweights;

};

#endif /* ForestPrediction_h */
