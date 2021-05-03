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

  arma::rowvec rep_cpp(double num,uint times) const;

  arma::vec in_cpp(const arma::vec &a, const arma::uvec &b) const;

  void in(uint us,
          uint ue,
          uint vs,
          uint ve,
          const arma::uvec &IB,
          arma::vec &u,
          arma::vec &v) const;

  arma::vec get_dh(const arma::mat& gammaxb,
                   const arma::vec& u,
                   const arma::vec& v,
                   const double tau,
                   const double tau_min,
                   const arma::rowvec& weights) const;
  double tau_inc(uint n, uint nvar, 
                 const arma::rowvec& c0, 
                 const arma::uvec& IB, 
                 const arma::mat& gammax, 
                 const arma::rowvec& weights, 
                 const arma::uvec& r1) const;
  
  void inner_loop(int iter, uint n, uint nvar, int tsep, int t_rr, double t,
                  arma::mat& gammax,
                  arma::colvec& b,
                  arma::uvec& free_var_basic,
                  arma::uvec& r1,
                  arma::uvec& r2,
                  arma::uvec& IB,
                  const arma::rowvec& weights) const;
  
  void update_dual_sol_(arma::sp_mat& dual_sol, 
                        arma::vec&& x, 
                        int p) const;
  
  void update_dual_sol(int tau_t, double tau, 
                       double tau_min, bool use_residual,
                       arma::vec& pre, arma::vec& now,
                       arma::vec& u, arma::vec& v,
                       arma::vec& dh, arma::sp_mat& dual_sol) const ;

  void qr_tau_para_diff_fix_cpp(const arma::mat& x,
                                const arma::colvec& y,
                                const arma::rowvec& weights,
                                const double& tau,
                                arma::vec& est_beta,
                                arma::mat& dual_sol,
                                arma::vec& residual,
                                double tol,
                                uint maxit,
                                const bool use_residual) const;
  arma::vec ranks_tau_cpp(const arma::vec& matY,
                          const arma::mat& matZ,
                          const arma::rowvec& weights,
                          const double tau) const;
  
  double rankscore_tau_cpp(const arma::mat& matX,
                           const arma::mat& matZ,
                           const arma::rowvec& weights,
                           const double tau,
                           const arma::vec& ranks) const;

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
                                const arma::rowvec& weights,
                                const arma::vec& ranks) const;

  double TAU;
  arma::vec taurange;

private:

};

#endif /* quantreg_h */
