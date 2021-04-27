#ifndef quantreg_h
#define quantreg_h

#include "RcppArmadillo.h"
#include "math.h"
#include "iostream"
#include <memory>
#include <armadillo>
#include <vector>
#include <random>
#include<cmath>
/*new*/
#include <utility>
#include <set>
#include <cfloat>
#ifdef DEBUG
#include <cstdio>
#include <fstream>
#include <sstream>
#include <gperftools/profiler.h>
#endif
// typedef unsigned int uint;

using namespace Rcpp;
// using namespace arma;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

typedef unsigned int uint;

class quantreg{
public:

  quantreg(double tau):TAU(tau) { };

  quantreg(arma::vec tr):taurange(tr) { };

  double G_func(uint& order, arma::vec& matY,arma::vec& delta) const;

  double KM_weight(uint& order, arma::vec& matY, arma::vec& delta) const;

  double KM_fun(const double& time, const arma::vec& matY, const arma::vec& delta) const;

  void qr_tau_para_diff_fix_cpp(const arma::mat& x,
                               const arma::colvec& y,
                               const arma::rowvec& weights,
                               const double& tau,
                               arma::vec& est_beta,
                               arma::vec& residual,
                               double tol,
                               uint maxit) const;

  void qr_tau_para_diff_cpp(const arma::mat& x,
                            const arma::colvec& y,
                            const arma::rowvec& weights,
                            const arma::colvec& taurange,
                            arma::mat& est_beta,
                            arma::sp_mat& dual_sol,
                            vector<double>& tau_list,
                            const double tau_min,
                            const double tol,
                            const uint maxit,
                            const uint max_num_tau,
                            const bool use_residual = true) const;

  arma::vec ranks_cpp(const arma::vec& matY,
                 const arma::mat& matZ,
                 const arma::rowvec& weights,
                 const arma::vec& taurange,
                 uint max_num_tau) const;
  
  arma::vec ranks_cpp_marginal(const arma::vec& matY) const;
  
  double rankscore_cpp(const arma::mat& matX,
                       const arma::mat& matZ,
                       const arma::rowvec& weights,
                       const arma::vec& taurange,
                       const arma::vec& ranks,
                       uint max_num_tau) const;
  
  double rankscore_cpp_marginal(const arma::mat& matX,
                                const arma::vec& ranks) const;

  double TAU;
  arma::vec taurange;

private:

};

#endif /* quantreg_h */
