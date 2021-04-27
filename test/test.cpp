#ifdef DEBUG
#include "RcppArmadillo.h"
#include "math.h"
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <RInside.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <gperftools/profiler.h>
typedef unsigned int uint;
using namespace Rcpp;
using namespace std;

extern void qr_simplex_speedtest(const arma::mat x, const arma::colvec y, const arma::rowvec weights,
                          const arma::colvec taurange, const double tau_min = 1e-10,
                          const double tol = 1e-14,
                          const uint maxit = 100000, const uint max_num_tau = 1000,
                          const bool use_residual = true);

extern List qr_tau_para_diff_cpp(const arma::mat x, const arma::colvec y, const arma::rowvec weights,
                          const arma::colvec taurange, const double tau_min = 1e-10,
                          const double tol = 1e-14,
                          const uint maxit = 100000, const uint max_num_tau = 1000,
                          const bool use_residual = true);

extern SEXP CenQRForest_C(const arma::mat& matZ0,
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
                 int mtry);

extern SEXP CenQRForest_WW_C(const arma::mat& matZ0,
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
                      int mtry);

extern SEXP QPRForest_C(const arma::mat& matZ0,
                 const arma::mat& matX0,
                 const arma::vec& matY0,
                 const arma::mat& matXnew,
                 const arma::vec& taurange,
                 const arma::vec& quantile_level,
                 uint max_num_tau,
                 int numTree,
                 int minSplit1,
                 int maxNode,
                 int mtry);

void split(const std::string& s,
    std::vector<std::string>& sv,
    const char delim = ' ') {
  sv.clear();
  std::istringstream iss(s);
  std::string temp;
  while (std::getline(iss, temp, delim)) {
    sv.emplace_back(std::move(temp));
  }
}

void read_mat(const std::string& filename, int col, arma::mat& X) {
  string line;
  vector<string> input;
  ifstream f(filename);

  if (!f.is_open())
  {
    cout << "not open " << filename << endl;
    return;
  }
  int cnt = 0;
  while (getline(f, line))
  {
    if (cnt == 0)
    {
      cnt += 1;
      continue;
    }
    split(line, input, ',');
    for (int i = 1;i <= col; ++i)
      X(cnt-1, i-1) = stod(input[i]);
    cnt += 1;
  }
  f.close();
  cout << "read " << filename << ": " << cnt << endl;
}

template<class T>
void read_vec(const std::string& filename, T& y) {
  string line;
  vector<string> input;
  ifstream f(filename);

  if (!f.is_open())
  {
    cout << "not open " << filename << endl;
    return;
  }
  int cnt = 0;
  while (getline(f, line))
  {
    if (cnt == 0)
    {
      cnt += 1;
      continue;
    }
    split(line, input, ',');
    y(cnt-1) = stod(input[1]);
    cnt += 1;
  }
  f.close();
  cout << "read " << filename << ": " << cnt << endl;
}

template<>
void read_vec(const std::string& filename, arma::uvec& y) {
  string line;
  vector<string> input;
  ifstream f(filename);

  if (!f.is_open())
  {
    cout << "not open " << filename << endl;
    return;
  }
  int cnt = 0;
  while (getline(f, line))
  {
    if (cnt == 0)
    {
      cnt += 1;
      continue;
    }
    split(line, input, ',');
    y(cnt-1) = stoi(input[1]);
    cnt += 1;
  }
  f.close();
  cout << "read " << filename << ": " << cnt << endl;
}

void test_qr_simplex()
{
  int n = 1000;
  arma::mat X(n, 2);
  arma::colvec y(n);
  arma::rowvec weights(n);
  arma::colvec taurage(2);
  double tau_min = 1e-10;
  double tol = 1e-14;
  uint maxit = 100000L;
  uint max_num_tau = 2000L;
  bool use_residual = true;

  read_mat("data/simplex/x.txt", 2, X);
  read_vec("data/simplex/y.txt", y);

  for (int i = 0;i < n; ++i)
    weights(i) = 1.;
  taurage(0) = 0;
  taurage(1) = 1;

  // speedtest
  // qr_simplex_speedtest(X, y, weights, taurage, tau_min, tol, maxit, max_num_tau, use_residual);
  qr_tau_para_diff_cpp(X, y, weights, taurage, tau_min, tol, maxit, max_num_tau, use_residual);

}

void test_cqr()
{
  int n = 500;
  arma::mat matZ0(n, 2);
  arma::mat matX0(n, 5);
  arma::vec matY0(n);
  arma::vec delta0(n);
  double tau = 0.5;
  arma::vec weight_rf0(n);
  arma::mat rfsrc_time_surv0(n, 83);
  arma::vec time_interest(83);
  arma::vec quantile_level(48);
  int numTree = 1;
  int minSplit1 = 15;
  int maxNode = 500;
  int mtry = 3;

  read_mat("data/cqr/cqr_z.txt", 2, matZ0);
  read_mat("data/cqr/cqr_x.txt", 5, matX0);
  read_vec("data/cqr/cqr_y.txt", matY0);
  read_vec("data/cqr/cqr_delta.txt", delta0);
  read_vec("data/cqr/cqr_weight.txt", weight_rf0);
  read_mat("data/cqr/cqr_rfsrc.txt", 83, rfsrc_time_surv0);
  read_vec("data/cqr/cqr_time.txt", time_interest);
  read_vec("data/cqr/cqr_quant.txt", quantile_level);

  CenQRForest_C(
      matZ0, matX0, matY0, delta0,
      tau, weight_rf0, rfsrc_time_surv0,
      time_interest, quantile_level,
      numTree, minSplit1, maxNode, mtry);
}

void test_cqr_ww() {
  int n = 1161;
  arma::mat matZ0(n, 1);
  arma::mat matX0(n, 2);
  arma::vec matY0(n);
  arma::mat matXnew(196, 2);
  arma::uvec delta(1000);
  double tau = 0.5;
  arma::vec weight_rf(n);
  arma::rowvec weight_censor(n);
  arma::vec quantile_level(48);
  int numTree = 500;
  int minSplit1 = 8;
  int maxNode = 500;
  int mtry = 1;

  read_mat("data/cqr_ww/z.txt", 1, matZ0);
  read_mat("data/cqr_ww/x.txt", 2, matX0);
  read_mat("data/cqr_ww/x_new.txt", 2, matXnew);
  read_vec("data/cqr_ww/y.txt", matY0);
  read_vec("data/cqr_ww/delta.txt", delta);
  read_vec("data/cqr_ww/weight_rf.txt", weight_rf);
  read_vec("data/cqr_ww/weight_censor.txt", weight_censor);
  read_vec("data/cqr_ww/quant_level.txt", quantile_level);

  CenQRForest_WW_C(matZ0, matX0, matY0, matXnew, delta,
                      tau, weight_rf, weight_censor, quantile_level,
                      numTree, minSplit1, maxNode, mtry);
}

void test_qpr() {
  int n = 1000;

  arma::mat matZ0(n, 1);
  arma::mat matX0(n, 5);
  arma::vec matY0(n, 1);
  arma::mat matXnew(196, 5);
  arma::vec taurange(2);
  arma::vec quantile_level(48);
  uint max_num_tau = 2000;
  int numTree = 500;
  int minSplit1 = 8;
  int maxNode = 500;
  int mtry = 4;

  read_mat("data/qpr/z.txt", 1, matZ0);
  read_mat("data/qpr/x.txt", 5, matX0);
  read_mat("data/qpr/x_test.txt", 5, matXnew);
  read_vec("data/qpr/y.txt", matY0);
  read_vec("data/qpr/taurange.txt", taurange);
  read_vec("data/qpr/quantile_level.txt", quantile_level);

  QPRForest_C(matZ0, matX0, matY0, matXnew,
      taurange, quantile_level, max_num_tau,
      numTree, minSplit1, maxNode, mtry);
}

int main(int argc, char *argv[]) {
  // init
  RInside R(argc, argv);
  R.parseEvalQ("library('Matrix');");
  // ProfilerStart("rf.prof");

  // test_qr_simplex();
  // test_cqr();
  // test_cqr_ww();
  test_qpr();

  // ProfilerStop();
  return 0;
}
#endif
