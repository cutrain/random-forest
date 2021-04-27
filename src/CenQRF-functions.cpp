#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]


#include "omp.h"
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
  trees.reserve(numTree);
  Forest ff(numTree,maxNode,minSplit1,mtry);
  arma::umat ids0(n1, numTree);
  ff.sampleWithoutReplacementSplit(n, n1, ids0);
  // bootstrap
  // ff.sampleWithReplacementSplit(n, n, ids0);

  arma::umat id1 = ids0;//ids.rows( arma::regspace<arma::uvec>(0, s-1)  );
  ff.trainRF(trees, matZ0,matX0,matY0,delta0,tau,weight_rf0,rfsrc_time_surv0,
             time_interest,quantile_level,ids0);
  arma::field<arma::umat> treeList(numTree);
  arma::field<arma::mat> tree_split(numTree);

#pragma omp parallel for
  for (int i = 0;i < trees.size(); ++i) {
    std::shared_ptr<Tree> tt = trees[i];
    shared_ptr<arma::vec> vars0 = tt->get_split_vars();
    arma::umat treeMat(vars0->n_elem, 4);
    treeMat.col(0) = *tt->get_left_childs();
    treeMat.col(1) = *tt->get_right_childs();
    treeMat.col(2) = *tt->get_isLeaf();
    treeList(i) = treeMat;
    arma::mat treeMat_split(vars0->n_elem,2);
    treeMat_split.col(0) = *tt->get_split_vars();
    treeMat_split.col(1) = *tt->get_split_values();
    tree_split(i) = treeMat_split;
  }
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
                 const arma::mat& matXnew,
                 const arma::vec& taurange,
                 const arma::vec& quantile_level,
                 uint max_num_tau,
                 int numTree,
                 int minSplit1,
                 int maxNode,
                 int mtry) {
  print_prefix("QPRForest:");
  int n_X0 = matX0.n_rows;
  int n1 = ceil(n_X0*0.8);
  std::vector<std::shared_ptr<Tree> > trees;
  trees.reserve(numTree);
  Forest ff(numTree,maxNode,minSplit1,mtry);
  //arma::umat ids(n, numTree);
  arma::umat ids0(n1, numTree);
  ff.sampleWithoutReplacementSplit(n_X0, n1, ids0);
  // bootstrap
  // ff.sampleWithReplacementSplit(n, n, ids0);

  arma::umat id1 = ids0;//ids.rows( arma::regspace<arma::uvec>(0, s-1)  );
  ff.trainRF(trees, matZ0,matX0,matY0,taurange,quantile_level,max_num_tau,
             ids0);
  arma::field<arma::umat> treeList(numTree);
  arma::field<arma::mat> tree_split(numTree);

#pragma omp parallel for
  for(int i=0; i < trees.size(); ++i) {
    shared_ptr<Tree> tt = trees[i];
    shared_ptr<arma::vec> vars0 = tt->get_split_vars();
    arma::umat treeMat(vars0->n_elem, 4);
    treeMat.col(0) = *tt->get_left_childs();
    treeMat.col(1) = *tt->get_right_childs();
    treeMat.col(2) = *tt->get_isLeaf();
    treeList(i) = treeMat;
    arma::mat treeMat_split(vars0->n_elem,2);
    treeMat_split.col(0) = *tt->get_split_vars();
    treeMat_split.col(1) = *tt->get_split_values();
    tree_split(i) = treeMat_split;
  }
  // use subsampling observations
  arma::umat id2 = ids0;
  //ids.rows( arma::regspace<arma::uvec>(0, n-1)  );

  ForestPrediction fp(matXnew,
                      matX0, matY0,matZ0,id2,
                      taurange, max_num_tau,trees);
  return Rcpp::List::create(Rcpp::Named("trees") = treeList,
                            Rcpp::Named("split_values") = tree_split,
                            Rcpp::Named("subsample.id") = id2,
                            Rcpp::Named("nodeLabel") = fp.get_nodeLabel(),
                            Rcpp::Named("nodeLabel_test") = fp.get_nodeLabel_test(),
                            Rcpp::Named("nodeSize") = fp.get_nodeSize(),
                            Rcpp::Named("nodeMap") = fp.get_nodeMap(),
                            Rcpp::Named("weights") = fp.get_weights(),
                            Rcpp::Named("estimate") = fp.get_estimate_process());
}

//' @noRd
// [[Rcpp::export]]
SEXP MQPRForest_C(const arma::mat& matX0,
                  const arma::vec& matY0,
                  const arma::mat& matXnew,
                  const arma::vec& taurange,
                  const arma::vec& quantile_level,
                  uint max_num_tau,
                  int numTree,
                  int minSplit1,
                  int maxNode,
                  int mtry) {
  print_prefix("QPRForest:");
  int n_X0 = matX0.n_rows;
  int n1 = ceil(n_X0*0.8);
  std::vector<std::shared_ptr<Tree> > trees;
  trees.reserve(numTree);
  Forest ff(numTree,maxNode,minSplit1,mtry);
  arma::umat ids0(n1, numTree);
  ff.sampleWithoutReplacementSplit(n_X0, n1, ids0);
  // bootstrap
  // ff.sampleWithReplacementSplit(n, n, ids0);
  arma::umat id1 = ids0;//ids.rows( arma::regspace<arma::uvec>(0, s-1)  );
  ff.trainRF(trees, matX0,matY0,taurange,quantile_level,max_num_tau,
             ids0);
  arma::field<arma::umat> treeList(numTree);
  arma::field<arma::mat> tree_split(numTree);

#pragma omp parallel for
  for (int i = 0;i < trees.size(); ++i) {
    std::shared_ptr<Tree> tt = trees[i];
    shared_ptr<arma::vec> vars0 = tt->get_split_vars();
    arma::umat treeMat(vars0->n_elem, 4);
    treeMat.col(0) = *tt->get_left_childs();
    treeMat.col(1) = *tt->get_right_childs();
    treeMat.col(2) = *tt->get_isLeaf();
    treeList(i) = treeMat;
    arma::mat treeMat_split(vars0->n_elem,2);
    treeMat_split.col(0) = *tt->get_split_vars();
    treeMat_split.col(1) = *tt->get_split_values();
    tree_split(i) = treeMat_split;
  }
  // use subsampling observations
  arma::umat id2 = ids0;
  //ids.rows( arma::regspace<arma::uvec>(0, n-1)  );

  ForestPrediction fp(matXnew,
                      matX0, matY0,id2,
                      taurange, max_num_tau,trees);
  return Rcpp::List::create(Rcpp::Named("trees") = treeList,
                            Rcpp::Named("split_values") = tree_split,
                            Rcpp::Named("subsample.id") = id2,
                            Rcpp::Named("nodeLabel") = fp.get_nodeLabel(),
                            Rcpp::Named("nodeLabel_test") = fp.get_nodeLabel_test(),
                            Rcpp::Named("nodeSize") = fp.get_nodeSize(),
                            Rcpp::Named("nodeMap") = fp.get_nodeMap(),
                            Rcpp::Named("weights") = fp.get_weights());
}


// [[Rcpp::export]]
SEXP CenQRForest_WW_C(const arma::mat& matZ0,
                      const arma::mat& matX0,
                      const arma::vec& matY0,
                      const arma::mat& matXnew,
                      const arma::uvec& delta,
                      const double& tau,
                      const arma::vec& weight_rf,
                      const arma::rowvec& weight_censor,
                      const arma::vec& quantile_level,
                      int numTree,
                      int minSplit1,
                      int maxNode,
                      int mtry) {
  print_prefix("CensQRForest_WW:");
  int n_obs = matZ0.n_rows;
  int N = delta.n_elem;
  int n1 = ceil(N*0.8);
  std::vector<std::shared_ptr<Tree> > trees;
  trees.reserve(numTree);
  Forest ff(numTree,maxNode,minSplit1,mtry);
  //arma::umat ids(n, numTree);
  arma::umat ids0 = arma::zeros<arma::umat>(n1, numTree);
  ff.sampleWithoutReplacementSplit(N, n1, ids0);
  // bootstrap
  // ff.sampleWithReplacementSplit(n, n, ids0);

  ff.trainRF(trees,matZ0,matX0,matY0,delta,tau,weight_rf,weight_censor,
             quantile_level,ids0);
  arma::field<arma::umat> treeList(numTree);
  arma::field<arma::mat> tree_split(numTree);

#pragma omp parallel for
  for (int i = 0;i < trees.size(); ++i) {
    std::shared_ptr<Tree> tt = trees[i];
    shared_ptr<arma::vec> vars0 = tt->get_split_vars();
    arma::umat treeMat(vars0->n_elem, 3);
    treeMat.col(0) = *tt->get_left_childs();
    treeMat.col(1) = *tt->get_right_childs();
    treeMat.col(2) = *tt->get_isLeaf();
    treeList(i) = treeMat;
    arma::mat treeMat_split(vars0->n_elem,2);
    treeMat_split.col(0) = *tt->get_split_vars();
    treeMat_split.col(1) = *tt->get_split_values();
    tree_split(i) = treeMat_split;
  }
  // use subsampling observations
  arma::umat id2 = ids0;
  //ids.rows( arma::regspace<arma::uvec>(0, n-1)  );

  // arma::mat weights = getWeights(matX0.rows(arma::span(0,(N-1))),
  //                                matX0.rows(arma::span(0,(N-1))),id2,
  //                                trees);
  ForestPrediction fp(matXnew,
                      matX0.rows(arma::span(0,(N-1))),
                      matY0.rows(arma::span(0,(N-1))),
                      matZ0.rows(arma::span(0,(N-1))),
                      id2,
                      tau,trees);
  return Rcpp::List::create(Rcpp::Named("trees") = treeList,
                            Rcpp::Named("split_values") = tree_split,
                            Rcpp::Named("subsample.id") = id2,
                            Rcpp::Named("nodeLabel") = fp.get_nodeLabel(),
                            Rcpp::Named("nodeLabel_test") = fp.get_nodeLabel_test(),
                            Rcpp::Named("nodeSize") = fp.get_nodeSize(),
                            Rcpp::Named("nodeMap") = fp.get_nodeMap(),
                            Rcpp::Named("weights") = fp.get_weights(),
                            Rcpp::Named("estimate") = fp.get_estimate());
}
