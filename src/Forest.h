#ifndef Forest_h
#define Forest_h

#include "RcppArmadillo.h"
#include "math.h"
#include "iostream"
#include "Tree.h"
#include <memory>
#include "quantreg.h"
// typedef unsigned int uint;

// using namespace Rcpp;
// using namespace arma;
// using namespace std;

typedef struct{
  int status;
  uint varsp;
  double cutsp;
}split_info;

class Forest{
public:
  Forest(uint nt,
         uint mn,
         uint minsp1,
         uint mt) : NUM_TREE(nt), MAX_NODE(mn),
         MIN_SPLIT1(minsp1), MTRY(mt) {};

  //subsampling
  void sampleWithoutReplacementSplit(arma::uword n, arma::uword n1,
                                     arma::umat& ids) {
    arma::uvec s0 = arma::regspace<arma::uvec>(0,n-1);
    for(size_t col_it = 0 ; col_it != NUM_TREE; ++col_it) {
      arma::uvec s = arma::shuffle(s0);
      ids.col(col_it) = s.subvec(0, (n1-1));
    }
  };

  //bootstrap
  void sampleWithReplacementSplit(arma::uword n, arma::uword n1, arma::umat& ids) {
    for(size_t col_it = 0 ; col_it != NUM_TREE; ++col_it) {
      ids.col(col_it) = arma::sort(arma::randi<arma::uvec>(n1, arma::distr_param(0,n1-1)));
    }
  };

  // GROW A FOREST
  int trainRF(std::vector<std::shared_ptr<Tree> >& trees,
              const arma::mat& matZ,
              const arma::mat& matX,
              const arma::mat& matY,
              const arma::vec& delta,
              const double& tau,
              const arma::vec& weight_rf,
              const arma::mat& rfsrc_time_surv,
              const arma::vec& time_interest,
              const arma::vec& quantile_level,
              const arma::umat& ids);

  int trainRF(std::vector<std::shared_ptr<Tree> >& trees,
              const arma::mat& matZ,
              const arma::mat& matX,
              const arma::mat& matY,
              const arma::vec& taurange,
              const arma::vec& quantile_level,
              uint max_num_tau,
              const arma::umat& ids);

  // Grow A Tree in the forest
  std::shared_ptr<Tree> train_tree(const arma::mat& matZ,
                                   const arma::mat& matX,
                                   const arma::mat& matY,
                                   const arma::vec& delta,
                                   const double& tau,
                                   const arma::vec& weight_rf,
                                   const arma::mat& rfsrc_time_surv,
                                   const arma::vec& time_interest,
                                   const arma::vec& quantile_level) const;

  std::shared_ptr<Tree> train_tree(const arma::mat& matZ,
                                   const arma::mat& matX,
                                   const arma::mat& matY,
                                   const arma::vec& taurange,
                                   const arma::vec& quantile_level,
                                   uint max_num_tau) const;

  uint split_generalized_MM(const arma::mat& matZ,
                            const arma::mat& matX,
                            const arma::mat& matY,
                            const arma::vec& delta,
                            const double& tau,
                            const arma::vec& weight_rf,
                            const arma::mat& rfsrc_time_surv,
                            const arma::vec& time_interest,
                            arma::field<arma::uvec>& nodeSample,
                            arma::uvec& isLeaf,
                            arma::vec& split_vars,
                            arma::vec& split_values,
                            arma::uvec& left_childs,
                            arma::uvec& right_childs,
                            uint& countsp,
                            uint& ndcount,
                            const arma::vec& quantile_level) const;

  arma::vec find_split_generalized_MM(uint nd,
                                      const arma::mat& matZ,
                                      const arma::mat& matX,
                                      const arma::mat& matY,
                                      const arma::vec& delta,
                                      const double& tau,
                                      const arma::vec& weight_rf,
                                      const arma::mat& rfsrc_time_surv,
                                      const arma::vec& time_interest,
                                      const arma::field<arma::uvec>& nodeSample,
                                      const arma::vec& quantile_level) const;

  double G_hat_rfsrc_cpp(double time,
                         const arma::mat& rfsrc_time_surv,
                         const arma::vec& time_interest,
                         arma::uword i) const;

  arma::mat ADiag(arma::mat& A,
                  arma::vec& diag_entries) const;

  void MMLQR_cens_rfsrc_cpp(const arma::mat& nodematZ,
                            const arma::mat& nodematY,
                            const arma::vec& nodedelta,
                            const double& tau,
                            const arma::vec& nodeweight_rf,
                            const arma::mat& noderfsrc_time_surv,
                            const arma::vec& nodetime_interest,
                            arma::vec beta_init,
                            arma::vec& G_mat,
                            arma::vec& beta,
                            arma::vec& r_vec,
                            uint& iteration,
                            double toler,
                            uint maxit) const;

  arma::vec rep_cpp(double num,uint times) const;

  split_info split_rankscore(const arma::mat& matZ,
                             const arma::mat& matX,
                             const arma::mat& matY,
                             const arma::vec& taurange,
                             arma::field<arma::uvec>& nodeSample,
                             arma::uvec& isLeaf,
                             arma::vec& split_vars,
                             arma::vec& split_values,
                             arma::uvec& left_childs,
                             arma::uvec& right_childs,
                             uint& countsp,
                             uint& ndcount,
                             const arma::vec& quantile_level,
                             uint max_num_tau) const;

  arma::vec find_split_rankscore(arma::uword nd,
                                 const arma::mat& matZ,
                                 const arma::mat& matX,
                                 const arma::mat& matY,
                                 const arma::vec& taurange,
                                 const arma::field<arma::uvec>& nodeSample,
                                 const arma::vec& quantile_level,
                                 uint max_num_tau) const;




private:



  uint NUM_TREE;
  uint MAX_NODE;
  uint MIN_SPLIT1;
  uint MTRY;


};

#endif /* Forest_h */
