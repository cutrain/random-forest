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

extern arma::rowvec rep_cpp(double num,uint times);

extern arma::vec in_cpp(const arma::vec &a, const arma::uvec &b);

extern List qr_tau_para_diff_cpp(const arma::mat x, const arma::colvec y, const arma::rowvec weights,
                          const arma::colvec taurange, const double tau_min = 1e-10,
                          const double tol = 1e-14,
                          const uint maxit = 100000, const uint max_num_tau = 1000,
                          const bool use_residual = true);

void test_find(const arma::rowvec &theta, const arma::rowvec &r1j, int iters) {
  double theta_min = 0;
  for (int i = 0;i < iters; ++i) {
    theta_min = arma::as_scalar(theta.cols(find(r1j>0)).min());
  }
}

void test_nofind(const arma::rowvec &theta, const arma::rowvec &r1j, int nvar, int iters) {
  for (int i = 0;i < iters; ++i) {
    bool choose = false;
    double theta_min = 0;
    for (int i = 0;i < (1+nvar)*2; i++)
      if (r1j[i] > 0 && (!choose || theta_min > theta[i]))
      {
        choose = true;
        theta_min = theta[i];
      }
  }
}

void qr_simplex_speedtest(const arma::mat x, const arma::colvec y, const arma::rowvec weights,
                          const arma::colvec taurange, const double tau_min = 1e-10,
                          const double tol = 1e-14,
                          const uint maxit = 100000, const uint max_num_tau = 1000,
                          const bool use_residual = true){
  // build data
  uint n = x.n_rows;

  uint nvar = x.n_cols;
  double tau = tau_min+taurange(0);
  arma::rowvec cc(nvar+1+2*n,arma::fill::zeros);
  for(uint j = nvar+1;j<nvar+n+1;j++){
    cc[j] = tau*weights[j];
    cc[j+n] = (1-tau)*weights[j];
  }

  arma::colvec col_one(n);
  col_one.fill(1.0);
  arma::mat gammax_org = join_rows(col_one,x);
  arma::mat gammaxb = gammax_org.t();
  arma::colvec col_zero(1);
  col_zero.fill(0.0);
  arma::colvec b = join_cols(y,col_zero);
  {
    auto nega_y = find(y<0);
    gammax_org.rows(nega_y) = -gammax_org.rows(nega_y);
    b.rows(nega_y) = -b.rows(nega_y);
  }

  //IB: index of variables in the basic set;
  arma::uvec IB(n,arma::fill::zeros);
  for (uint j = 0;j < n; ++j) {
    if (y[j] >= 0)
      IB[j] = nvar+j+1;
    else
      IB[j] = nvar+n+j+1;
  }

  //transformation of the LP tableau to initialize optimization;
  arma::rowvec cc_trans = -cc.cols(IB);
  arma::mat gammax = join_cols(gammax_org,cc_trans*gammax_org);

  //once beta is pivoted to basic set, it cannot be pivoted out;
  arma::uvec free_var_basic(n+1,arma::fill::zeros);
  free_var_basic[n] = 1;

  //r1,r2: index of positive or negative beta in the basic set;
  arma::uvec r1 = arma::regspace<arma::uvec>(0,nvar);
  arma::uvec r2(nvar+1,arma::fill::zeros);
  arma::mat rr(2, 1+nvar,arma::fill::zeros);

  //Initialize estimation output matrix;
  //est_beta: estimation matrix;
  arma::mat est_beta(nvar+1,1,arma::fill::zeros);
  est_beta = arma::regspace(1,nvar+1);
  //dual_sol: dual solution matrix a sparse matrix
  arma::sp_mat dual_sol(n,max_num_tau);
  //tau_list:: a list of tau, automatically generated in alg;
  vector<double> tau_list;

  //c0: a vector helps to generate the next tau level;
  arma::rowvec c0(1+nvar+2*n,arma::fill::zeros);
  c0.subvec(nvar+1,nvar+n) = weights;
  c0.subvec(nvar+n+1,nvar+2*n) = -weights;


  //terminate loop indicator;
  bool last_flag = false;

  // variables used in the while loop;
  uint j = 0;
  arma::vec yy(gammax.n_rows,arma::fill::zeros);
  arma::vec ee(gammax.n_rows,arma::fill::zeros);
  uint t_rr = 0;
  uint tsep = 0;
  arma::uvec nokeep(gammax.n_rows,arma::fill::zeros);
  double t = 0;
  arma::vec k(gammax.n_rows,arma::fill::zeros);
  uint k_index = 0;
  arma::uvec tmp;
  arma::vec estimate(nvar+1,arma::fill::zeros);
  uint tau_t = 0;
  arma::rowvec r1j_temp(1+nvar,arma::fill::zeros);
  arma::rowvec r1j(2*(1+nvar),arma::fill::zeros);
  arma::rowvec r0j(2*(1+nvar),arma::fill::zeros);
  arma::rowvec theta(2*(1+nvar),arma::fill::zeros);
  arma::vec u(n,arma::fill::zeros);
  arma::vec u_temp = arma::regspace<arma::vec>(1+nvar,nvar+n);
  arma::vec v(n,arma::fill::zeros);
  arma::vec v_temp = arma::regspace<arma::vec>(1+nvar+n,nvar+2*n);
  arma::rowvec dbarh(n,arma::fill::zeros);
  arma::vec temp1(n,arma::fill::zeros);
  arma::vec temp(n,arma::fill::zeros);

  if (true){

    j = 0;
    while(j<maxit){
      //cout << "j is "<< j << endl;

      if(tau_t==0&&j<nvar+1){
        rr.row(0) = gammax.row(n);
        for(uint i = 0;i<r2.n_elem;i++){
          if(r2(i)==0){
            rr(1,i) = 0;
            rr(0,i) = -abs(rr(0,i));
          }else{
            rr(1,i) = weights(r2(i)-(1+nvar+n))-rr(0,i);
          }
        }

        t_rr = j;
        tsep = 0;
        t = arma::as_scalar(r1(t_rr));
      }else{
        //cout << rr.n_cols<< endl;
        //cout << gammax.n_cols <<endl;

        rr.row(0) = gammax.row(n);

        for(uint i = 0;i<r2.n_elem;i++){
          if(r2(i)==0){
            rr(1,i) = 0;
            rr(0,i) = -abs(rr(0,i));
          }else{
            rr(1,i) = weights(r2(i)-(1+nvar+n))-rr(0,i);
          }
        }

        if(rr.min()>-tol)
          break;
        //cout <<"rr_min is "<<rr.min()<<endl;
        tsep = rr.index_min();
        //cout << "tsep is "<< tsep << endl;
        t_rr = floor(tsep/2);
        //cout << "t_rr is "<< t_rr << endl;
        tsep = tsep-floor(tsep/2)*2;
        // cout << "tsep is "<< tsep << endl;

        if(tsep==0){
          t = r1(t_rr);
        }else{
          t = r2(t_rr);
        }
      }

      if (r2(t_rr)!=0){
        if(tsep==0){
          yy = gammax.col(t_rr);
        }else{
          yy = -gammax.col(t_rr);
        }

        k = b/yy;
        nokeep = arma::find(yy<=0 || free_var_basic==1);
        k(nokeep) = rep_cpp(INFINITY,nokeep.n_elem).t();
        k_index = k.index_min();

        if(tsep!=0){
          yy(n) = yy(n)+weights(r2[t_rr]-(1+nvar+n));
        }

      }else{
        //cout << t_rr<< endl;
        yy = gammax.col(t_rr);
        if (abs(yy(n)) < 1e-10) {
          cout << "eps found!!!" << endl;
        }
        if(yy(n)<0){

          k = b/yy;
          nokeep = arma::find(yy<=0 || free_var_basic==1);
          k(nokeep) = rep_cpp(INFINITY,nokeep.n_elem).t();
          k_index = k.index_min();
        }else{
          k = -b/yy;
          nokeep = arma::find(yy>=0 || free_var_basic==1);
          k(nokeep) = rep_cpp(INFINITY,nokeep.n_elem).t();
          k_index = k.index_min();
        }

        free_var_basic(k_index) = 1;
      }

      ee = yy/yy(k_index);
      ee(k_index) = 1-1/yy(k_index);

      // cout <<"IB(k_index) is "<<IB(k_index)<<endl;
      // cout <<"k_index is "<<k_index<<endl;
      if(IB(k_index)<=nvar+n){
        gammax.col(t_rr) = rep_cpp(0,gammax.n_rows).t();
        gammax(k_index,t_rr) = 1;
        r1(t_rr) = IB(k_index);
        r2(t_rr) = IB(k_index)+n;
      }else{
        gammax.col(t_rr) = rep_cpp(0,gammax.n_rows).t();
        gammax(k_index,t_rr) = -1;
        gammax(n,t_rr)  =  weights(IB[k_index]-(1+nvar+n));
        r1(t_rr) = IB(k_index)-n;
        r2(t_rr) = IB(k_index);
      }

      gammax = gammax-ee*gammax.row(k_index);
      b = b-ee*arma::as_scalar(b[k_index]);
      IB(k_index) = t;
      //cout << "t is " << t << endl;

      j++;
    }

    if(j==maxit){
      //cout << "WARNING:May not converge (tau = "<< tau <<")"<< endl;
    }

    tmp = find(IB<nvar+1);
    arma::vec estimate_temp = b(tmp);
    estimate_temp = estimate_temp(sort_index(IB(tmp)));

    if(estimate_temp.n_elem!=nvar+1){
      estimate.zeros();
      estimate(sort(IB(tmp))) = estimate_temp;
    }else{
      estimate = estimate_temp;
    }

    est_beta = join_rows(est_beta,estimate);
    tau_list.push_back(tau-tau_min);

    u = in_cpp(u_temp,IB);
    v = in_cpp(v_temp,IB);

    arma::mat xh = gammaxb.cols(find(u==0&&v==0));
    arma::mat xbarh = gammaxb.cols(find(u==1||v==1));
    dbarh = u.t()%rep_cpp(tau-tau_min,u.n_elem)+
      v.t()%rep_cpp(tau-tau_min-1,v.n_elem);
    dbarh = dbarh%weights;
    //cout<<"dbarh dimension is "<< dbarh.n_elem<<endl;

    arma::vec dh(sum(u==1||v==1),arma::fill::zeros);
    try{
      dh = solve(xh,-xbarh*dbarh.cols(find(u==1||v==1)).t());
    }
    catch(const runtime_error& error){
      dh = -arma::pinv(xh)*xbarh*dbarh.cols(find(u==1||v==1));
    }
    //cout << "dh dimension is "<<dh.n_rows<<endl;

    dh = dh/weights.elem(find(u==0&&v==0)).as_col()+1-tau;

    temp1 = u;
    if(use_residual==false){
      dh.elem(find(dh==tau-tau_min)).ones();
      dh.elem(find(dh==tau-tau_min-1)).zeros();
    }
    temp1(find(u==0&&v==0)) = dh;
    if(tau_t>0){
      dual_sol.col(tau_t-1) = temp1-temp;
    }

    temp = temp1;

    tau_t++;
  }

  // start test
  // test find function
  const int iters = 1000000;
  r1j_temp = c0.cols(IB)*gammax(arma::span(0,n-1),arma::span::all);
  r1j = join_rows(r1j_temp-1,1-r1j_temp);
  r0j = join_rows(gammax.row(n),weights.elem(r1-(1+nvar)).as_row()-gammax.row(n));
  theta = r0j/r1j;
  ProfilerStart("find_speed.prof");
  test_find(theta, r1j, iters);
  test_nofind(theta, r1j, nvar, iters);
  ProfilerStop();
}
#endif
