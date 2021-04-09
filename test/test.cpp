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

void test_qr_simplex()
{
  int n = 1000;
  int d = 2;
  vector<double> X_c[d];
  vector<double> y_c;
  vector<double> w_c;
  string line;

  // read X
  ifstream f("data/simplex/x.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      cnt += 1;
      if (cnt == 1)
        continue;
      int row;
      double a,b;
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%lf,%lf", &row, &a, &b);
      X_c[0].push_back(a);
      X_c[1].push_back(b);
    }
    f.close();
    cout << "read X:" << cnt << endl;
  }
  else
    cout << "not open X" << endl;

  // read Y
  f = ifstream("data/simplex/y.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      cnt += 1;
      if (cnt == 1)
        continue;
      int row;
      double a;
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%lf", &row, &a);
      y_c.push_back(a);
    }
    f.close();
    cout << "read Y:" << cnt << endl;
  }
  else
    cout << "not open Y" << endl;

  // making input
  arma::mat X(n, d);
  arma::colvec y(n);
  arma::rowvec weights(n);
  arma::colvec taurage(2);
  double tau_min = 1e-10;
  double tol = 1e-14;
  uint maxit = 100000L;
  uint max_num_tau = 2000L;
  bool use_residual = true;
  for (int i = 0;i < n; ++i)
  {
    X(i, 0) = X_c[0][i];
    X(i, 1) = X_c[1][i];
    y(i) = y_c[i];
    weights(i) = 1.;
  }
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

  string line;
  vector<string> input;

  // read z
  ifstream f("data/cqr/cqr_z.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      if (cnt == 0)
        continue;
      int row, a[2];
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%d,%d", &row, a, a+1);
      matZ0(cnt, 0) = a[0];
      matZ0(cnt, 1) = a[1];
      cnt += 1;
    }
    f.close();
    cout << "read z:" << cnt << endl;
  }
  else
    cout << "not open z" << endl;

  // read x
  f = ifstream("data/cqr/cqr_x.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      if (cnt == 0)
        continue;
      int row, a[5];
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%d,%d,%d,%d,%d", &row, a, a+1,a+2,a+3,a+4);
      for (int i = 0;i < 5; ++i)
        matX0(cnt, i) = a[i];
      cnt += 1;
    }
    f.close();
    cout << "read x:" << cnt << endl;
  }
  else
    cout << "not open x" << endl;

  // read y
  f = ifstream("data/cqr/cqr_y.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      if (cnt == 0)
        continue;
      int row, a;
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%d", &row, &a);
      matY0[cnt] = a;
      cnt += 1;
    }
    f.close();
    cout << "read y:" << cnt << endl;
  }
  else
    cout << "not open y" << endl;

  // read delta
  f = ifstream("data/cqr/cqr_delta.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      if (cnt == 0)
        continue;
      int row, a;
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%d", &row, &a);
      delta0[cnt] = a;
      cnt += 1;
    }
    f.close();
    cout << "read delta:" << cnt << endl;
  }
  else
    cout << "not open delta" << endl;

  // read weight
  f = ifstream("data/cqr/cqr_weight.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      if (cnt == 0)
        continue;
      int row, a;
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%d", &row, &a);
      weight_rf0[cnt] = a;
      cnt += 1;
    }
    f.close();
    cout << "read weight:" << cnt << endl;
  }
  else
    cout << "not open weight" << endl;

  // read rfsrc
  f = ifstream("data/cqr/cqr_rfsrc.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      if (cnt == 0)
        continue;
      split(line, input, ',');
      for (int i = 1;i <= 83; ++i)
        rfsrc_time_surv0(cnt, i-1) = stod(input[i]);
      cnt += 1;
    }
    f.close();
    cout << "read rfsrc:" << cnt << endl;
  }
  else
    cout << "not open rfsrc" << endl;

  // read time_interest
  f = ifstream("data/cqr/cqr_time.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      if (cnt == 0)
        continue;
      int row, a;
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%d", &row, &a);
      time_interest[cnt] = a;
      cnt += 1;
    }
    f.close();
    cout << "read time:" << cnt << endl;
  }
  else
    cout << "not open time" << endl;

  // read quant
  f = ifstream("data/cqr/cqr_quant.txt");
  if (f.is_open())
  {
    int cnt = 0;
    while (getline(f, line))
    {
      if (cnt == 0)
        continue;
      int row, a;
      char *p=(char*)line.data();
      sscanf(p, "\"%d\",%d", &row, &a);
      quantile_level[cnt] = a;
      cnt += 1;
    }
    f.close();
    cout << "read quantile level:" << cnt << endl;
  }
  else
    cout << "not open quantile level" << endl;

  CenQRForest_C(
      matZ0, matX0, matY0, delta0,
      tau, weight_rf0, rfsrc_time_surv0,
      time_interest, quantile_level,
      numTree, minSplit1, maxNode, mtry);
}

int main(int argc, char *argv[]) {
  // init
  RInside R(argc, argv);
  R.parseEvalQ("library('Matrix');");

  // test_qr_simplex();
  test_cqr();

  return 0;
}
#endif
