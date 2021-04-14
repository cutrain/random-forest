#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]


#include "Tree.h"
#include "Forest.h"
#include "ForestPrediction.h"
#include "common.h"
#include <memory>


#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#endif

using std::shared_ptr;
using std::cout;
using std::endl;


//' @noRd
// [[Rcpp::export]]
SEXP CenQRForest_C(const arma::mat& matZ0,
                 const arma::mat& matX0,
                 const arma::vec& matY0,
                 const arma::vec& delta0,
                 const double& tau,
                 const arma::vec& weight_rf0,
                 const arma::mat& rfsrc_time_surv0,
                 const arma::vec& time_interest,
                 const arma::vec& quantile_level,
                 int numTree,
                 int minSplit1,
                 int maxNode,
                 int mtry) {
  print_prefix("CenQRForest:");
  int n = matZ0.n_rows;
  int n1 = ceil(n*0.8);
  std::vector<std::shared_ptr<Tree> > trees;
  print(1);
  trees.reserve(numTree);
  print(2);
  print("numtree=");
  print(numTree);
  Forest ff(numTree,maxNode,minSplit1,mtry);
  print(3);
  //arma::umat ids(n, numTree);
  arma::umat ids0(n1, numTree);
  ff.sampleWithoutReplacementSplit(n, n1, ids0);
  print(4);
  // bootstrap
  // ff.sampleWithReplacementSplit(n, n, ids0);

  arma::umat id1 = ids0;//ids.rows( arma::regspace<arma::uvec>(0, s-1)  );
  print(5);
  ff.trainRF(trees, matZ0,matX0,matY0,delta0,tau,weight_rf0,rfsrc_time_surv0,
             time_interest,quantile_level,ids0);
  print(6);
  arma::field<arma::umat> treeList(numTree);
  arma::field<arma::mat> tree_split(numTree);
  std::vector<std::shared_ptr<Tree> >::const_iterator it;
  int i = 0;

  print(7);
  print_enter("loop:");
  for(it = trees.begin(); it != trees.end(); it++, i++) {
    print(i);
    std::shared_ptr<Tree> tt = *it;
    shared_ptr<arma::vec> vars0 = tt->get_split_vars();
    std::cout << "vars0=" << vars0->n_elem << std::endl;
    arma::umat treeMat(vars0->n_elem, 4);
    print("xxx");
    treeMat.col(0) = *tt->get_left_childs();
    std::cout<<"DONE0"<<std::endl;
    treeMat.col(1) = *tt->get_right_childs();
    std::cout<<"DONE1"<<std::endl;
    treeMat.col(2) = *tt->get_isLeaf();
    std::cout<<"DONE2"<<std::endl;
    treeList(i) = treeMat;
    arma::mat treeMat_split(vars0->n_elem,2);
    treeMat_split.col(0) = *tt->get_split_vars();
    treeMat_split.col(1) = *tt->get_split_values();
    tree_split(i) = treeMat_split;
  }
  print_leave();
  print(8);
  // use subsampling observations
  arma::umat id2 = ids0;
  //ids.rows( arma::regspace<arma::uvec>(0, n-1)  );

  // ForestPrediction fp(zy0, zt0, id2, trees, n);
  return Rcpp::List::create(Rcpp::Named("trees") = treeList,
                            Rcpp::Named("split_values") = tree_split,
                            Rcpp::Named("subsample.id") = id2);
                            // Rcpp::Named("nodeLabel") = fp.get_nodeLabel(),
                            // Rcpp::Named("nodeSize") = fp.get_nodeSize(),
                            // Rcpp::Named("nodeMap") = fp.get_nodeMap(),
}

//' @noRd
// [[Rcpp::export]]
SEXP QPRForest_C(const arma::mat& matZ0,
                   const arma::mat& matX0,
                   const arma::vec& matY0,
                   const arma::vec& taurange,
                   const arma::vec& quantile_level,
                   uint max_num_tau,
                   int numTree,
                   int minSplit1,
                   int maxNode,
                   int mtry) {
  print_prefix("QPRForest:");
  int n = matZ0.n_rows;
  int n1 = ceil(n*0.8);
  std::vector<std::shared_ptr<Tree> > trees;
  print(1);
  trees.reserve(numTree);
  print(2);
  print("numtree=");
  print(numTree);
  Forest ff(numTree,maxNode,minSplit1,mtry);
  print(3);
  //arma::umat ids(n, numTree);
  arma::umat ids0(n1, numTree);
  ff.sampleWithoutReplacementSplit(n, n1, ids0);
  print(4);
  // bootstrap
  // ff.sampleWithReplacementSplit(n, n, ids0);

  arma::umat id1 = ids0;//ids.rows( arma::regspace<arma::uvec>(0, s-1)  );
  print(5);
  ff.trainRF(trees, matZ0,matX0,matY0,taurange,quantile_level,max_num_tau,
             ids0);
  print(6);
  arma::field<arma::umat> treeList(numTree);
  arma::field<arma::mat> tree_split(numTree);
  std::vector<std::shared_ptr<Tree> >::const_iterator it;
  int i = 0;

  print(7);
  print_enter("loop:");
  for(it = trees.begin(); it != trees.end(); it++, i++) {
    print(i);
    std::shared_ptr<Tree> tt = *it;
    shared_ptr<arma::vec> vars0 = tt->get_split_vars();
    // std::cout << "vars0=" << vars0->n_elem << std::endl;
    arma::umat treeMat(vars0->n_elem, 4);
    // print("xxx");
    treeMat.col(0) = *tt->get_left_childs();
    // std::cout<<"DONE0"<<std::endl;
    treeMat.col(1) = *tt->get_right_childs();
    // std::cout<<"DONE1"<<std::endl;
    treeMat.col(2) = *tt->get_isLeaf();
    // std::cout<<"DONE2"<<std::endl;
    treeList(i) = treeMat;
    arma::mat treeMat_split(vars0->n_elem,2);
    treeMat_split.col(0) = *tt->get_split_vars();
    treeMat_split.col(1) = *tt->get_split_values();
    tree_split(i) = treeMat_split;
  }
  print_leave();
  print(8);
  // use subsampling observations
  arma::umat id2 = ids0;
  //ids.rows( arma::regspace<arma::uvec>(0, n-1)  );

  ForestPrediction fp(matX0,matX0, id2, trees);
  return Rcpp::List::create(Rcpp::Named("trees") = treeList,
                            Rcpp::Named("split_values") = tree_split,
                            Rcpp::Named("subsample.id") = id2,
                            Rcpp::Named("nodeLabel") = fp.get_nodeLabel(),
                            Rcpp::Named("nodeSize") = fp.get_nodeSize(),
                            Rcpp::Named("nodeMap") = fp.get_nodeMap());
}


// [[Rcpp::export]]
Rcpp::List CenQRForest_WW_C(const arma::mat& matZ0,
                            const arma::mat& matX0,
                            const arma::vec& matY0,
                            const arma::uvec& delta,
                            const double& tau,
                            const arma::vec& weight_rf,
                            const arma::rowvec& weight_censor,
                            const arma::vec& quantile_level,
                            int numTree,
                            int minSplit1,
                            int maxNode,
                            int mtry) {
  print_prefix("QPRForest_WW:");
  int n_obs = matZ0.n_rows;
  int N = delta.n_elem;
  int n1 = ceil(N*0.8);
  std::vector<std::shared_ptr<Tree> > trees;
  print(1);
  trees.reserve(numTree);
  print(2);
  print("numtree=");
  print(numTree);
  Forest ff(numTree,maxNode,minSplit1,mtry);
  print(3);
  //arma::umat ids(n, numTree);
  arma::umat ids0 = arma::zeros<arma::umat>(n1, numTree);
  ff.sampleWithoutReplacementSplit(N, n1, ids0);
  print(4);
  // bootstrap
  // ff.sampleWithReplacementSplit(n, n, ids0);

  print(5);
  ff.trainRF(trees,matZ0,matX0,matY0,delta,tau,weight_rf,weight_censor,
             quantile_level,ids0);
  print(6);
  arma::field<arma::umat> treeList(numTree);
  arma::field<arma::mat> tree_split(numTree);
  std::vector<std::shared_ptr<Tree> >::const_iterator it;
  int i = 0;

  print(7);
  print_enter("loop:");
  for(it = trees.begin(); it != trees.end(); it++, i++) {
    std::cout<<"tree "<<i<<std::endl;
    std::shared_ptr<Tree> tt = *it;
    shared_ptr<arma::vec> vars0 = tt->get_split_vars();
    // std::cout << "vars0=" << vars0->n_elem << std::endl;
    arma::umat treeMat(vars0->n_elem, 3);
    // print("xxx");
    treeMat.col(0) = *tt->get_left_childs();
    // std::cout<<"DONE0"<<std::endl;
    treeMat.col(1) = *tt->get_right_childs();
    // std::cout<<"DONE1"<<std::endl;
    treeMat.col(2) = *tt->get_isLeaf();
    // std::cout<<"DONE2"<<std::endl;
    treeList(i) = treeMat;
    arma::mat treeMat_split(vars0->n_elem,2);
    treeMat_split.col(0) = *tt->get_split_vars();
    treeMat_split.col(1) = *tt->get_split_values();
    tree_split(i) = treeMat_split;
  }
  print_leave();
  print(8);
  // use subsampling observations
  arma::umat id2 = ids0;
  //ids.rows( arma::regspace<arma::uvec>(0, n-1)  );

  // arma::mat weights = getWeights(matX0.rows(arma::span(0,(N-1))),
  //                                matX0.rows(arma::span(0,(N-1))),id2,
  //                                trees);
  print(9);
  ForestPrediction fp(matX0.rows(arma::span(0,(N-1))),
                      matX0.rows(arma::span(0,(N-1))),id2,trees);
  print(10);
  return Rcpp::List::create(Rcpp::Named("trees") = treeList,
                            Rcpp::Named("split_values") = tree_split,
                            Rcpp::Named("subsample.id") = id2,
                            Rcpp::Named("nodeLabel") = fp.get_nodeLabel(),
                            Rcpp::Named("nodeSize") = fp.get_nodeSize(),
                            Rcpp::Named("nodeMap") = fp.get_nodeMap(),
                            Rcpp::Named("weights") = fp.get_weights());

}
