// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "math.h"
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#ifdef DEBUG
#include <RInside.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <gperftools/profiler.h>
#endif
typedef unsigned int uint;

using namespace Rcpp;
//using namespace arma;
using namespace std;

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(Matrix)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]



// repetition function which returns a row vector with times num;
// @param num a number;
// @param times an integer-valued number of times to repeat num;
arma::rowvec rep_cpp(double num,uint times){
  arma::rowvec result(times);
  for(uint i = 0;i<times;i++){
    result(i) = num;
  }
  return move(result);
}


// a function which returns a binary vector indicating if
//there is a match or not in vector a for vector b;
arma::vec in_cpp(const arma::vec &a, const arma::uvec &b){
  arma::vec result(a.n_elem);
  for(uint i = 0;i<a.n_elem;i++){
    if(sum(b.elem(find(b==a(i))))!=0){
      result(i) = 1;
    }else{
      result(i) = 0;
    }
  }
  return move(result);
}

void save_mat(arma::mat X, string filename) {
  ofstream f(filename);
  if (f.is_open()) {
    for (int i = 0; i < X.n_rows; ++i) {
      for (int j = 0;j < X.n_cols; ++j) {
        f << X(i,j) << " ";
      }
      f << endl;
    }
    f.close();
  }
}

void save_vector(vector<double> x, string filename) {
  ofstream f(filename);
  if (f.is_open()) {
    for (int i = 0;i < x.size(); ++i) {
      f << x[i] << endl;
    }
    f.close();
  }
}

void save_spmat(arma::sp_mat X, string filename) {
  ofstream f(filename);
  if (f.is_open()) {
    for (auto it = X.begin(); it != X.end(); ++it) {
      f << (*it) << " " << it.row() << " " << it.col() << endl;
    }
    f.close();
  }
}


//main function to implement quantile regression with simplex
//method;
//@param x design matrix;
//@param y a vector of response variable;
//@param weights a vector with the same length as y
//@param taurange the range of tau for quantile process;
//@param tau_min minimum tau value. The default value is 1e-10
//               to approach tau = 0;
//@param tol threshold to check convergence. The default value is
//           1e-14;
//@param maxit the maximum number of inerations at each tau.
//             Usually, larger sample size requires larger
//             numbers of iteration to converge. The default
//             value is 1e5;
//@param max_number_tau the number of tau to be tested in the
//                      quantile process. The default value is 1e5;
//@use_residual an logical value indicating using residual;

// This function will return: the primary solution as estimate,
// a list of quantile levels as tau and the difference of dual
//solution as diff_dual_sol;


//[[Rcpp::export]]
List qr_tau_para_diff_cpp(const arma::mat x, const arma::colvec y, const arma::rowvec weights,
                          const arma::colvec taurange, const double tau_min = 1e-10,
                          const double tol = 1e-14,
                          const uint maxit = 100000, const uint max_num_tau = 1000,
                          const bool use_residual = true){
#ifdef DEBUG
  ProfilerStart("tau.prof");
#endif
  //n: number of obs;
  uint n = x.n_rows;

  //if(n>max_num_tau) max_num_tau = 2*n;
  //nvar: number of covariates;
  uint nvar = x.n_cols;
  double tau = tau_min+taurange(0);
  //cout << tau << endl;
  //cc: last row in the linear programming tableau of QR;
  arma::rowvec cc(nvar+1+2*n,arma::fill::zeros);
  for(uint j = nvar+1;j<nvar+n+1;j++){
    cc[j] = tau*weights[j];
    cc[j+n] = (1-tau)*weights[j];
  }

  arma::colvec col_one(n);
  col_one.fill(1.0);
  arma::mat gammax_org = join_rows(col_one,x);
  arma::mat gammaxb = gammax_org.t();
  //b: last column in the linear programming tableau of QR;
  arma::colvec col_zero(1);
  col_zero.fill(0.0);
  arma::colvec b = join_cols(y,col_zero);
  //flip the sign if y<0;
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
  //cout << gammax(gammax.n_rows-1,0) << endl;

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
  //cout << last_flag <<endl;

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

  while(tau_t<max_num_tau){
     //cout << "tau_t is "<< tau_t<< endl;

    if(tau_t>0){
      r1j_temp = c0.cols(IB)*gammax(arma::span(0,n-1),arma::span::all);
      //cout << "r1j_temp dimension is " << r1j_temp.n_elem<<endl;
      r1j = join_rows(r1j_temp-1,1-r1j_temp);
      //cout << "r1j dimension is" << r1j.n_elem<<endl;
      r0j = join_rows(gammax.row(n),weights.elem(r1-(1+nvar)).as_row()-gammax.row(n));
      //cout << "r0j dimension is" << r0j.n_elem<<endl;
      theta = r0j/r1j;
      //cout<< "theta dimension is "<< theta.n_elem <<endl;
      bool choose = false;
      double theta_min = 0;
      for (int i = 0;i < (1+nvar)*2; i++)
        if (r1j[i] > 0 && (!choose || theta_min > theta[i]))
        {
          choose = true;
          theta_min = theta[i];
        }
      // double theta_min = arma::as_scalar(theta.cols(find(r1j>0)).min());
      //cout << "theta_min is "<<theta_min<< endl;

      tau = tau + theta_min + tau_min;
      //cout << "tau is "<< tau<< endl;

      if(tau>taurange[1]){
        if(last_flag)
          break;
        if(tau-theta_min<=taurange[1]){
          tau = taurange[1]-tau_min;
          last_flag = true;
        }else
          break;
      }

      cc.cols(1+nvar,nvar+n) = rep_cpp(tau,n)%weights;
      cc.cols(1+nvar+n,nvar+2*n) = rep_cpp(1-tau,n)%weights;
      gammax.row(n) = rep_cpp(tau,nvar+1)-
        cc.cols(IB)*gammax(arma::span(0,n-1),arma::span::all);
    }

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
  //cout << "est_beta dimension" << est_beta.n_cols << endl;
  //cout << "tau_t is "<< tau_t<< endl;
  if(last_flag==true&&j==0){
    est_beta.shed_col(tau_t);
    tau_list.erase(tau_list.begin()+tau_t-2);
    dual_sol.shed_cols(tau_t-1,max_num_tau-1);
    dual_sol.shed_col(tau_t-3);
  }else{
    est_beta.shed_col(tau_t);
    dual_sol.shed_cols(tau_t-1,max_num_tau-1);
  }


  //cout<<est_beta.n_cols<<endl;
  //cout<< "tau_t is "<<tau_t<<endl;

  //est_beta.shed_col(tau_t);
  //dual_sol.shed_cols(tau_t-1,max_num_tau-1);
  est_beta.shed_col(0);
#ifdef DEBUG
  ProfilerStop();
  save_mat(est_beta, "est_beta.txt");
  save_vector(tau_list, "tau_list.txt");
  save_spmat(dual_sol, "dual_sol.txt");
  cout << "finish" << endl;
#endif

  return List::create(Named("estimate") = est_beta,
                      Named("tau") = tau_list,
                      Named("diff_dual_sol") = dual_sol);
  // Named("gammax") = gammax,
  // Named("cc") = cc,
  // Named("IB") = IB);
}
