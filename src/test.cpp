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

void test_qr_simplex() {
  int n = 1000;
  int d = 2;
  vector<double> X_c[d];
  vector<double> y_c;
  vector<double> w_c;
  string line;

  // read X
  ifstream f("x.txt");
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
  f = ifstream("y.txt");
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

int main(int argc, char *argv[]) {
  // init
  RInside R(argc, argv);
  R.parseEvalQ("library('Matrix');");

  test_qr_simplex();


  return 0;
}
#endif
