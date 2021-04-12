#include "quantreg.h"
#include "common.h"

double quantreg::G_func(uint& order, arma::vec& matY,arma::vec& delta) const{

  uint n = matY.n_elem;
  arma::uvec indice = arma::sort_index(matY);
  double prod = 1;
  if(order>1){

    order = order-1;

  }
  for(uint j = 0;j<order;j++){

    uint indj = indice(j);
    if(delta(indj)==0){

      prod = prod*((n-1-j)/(n-j));

    }

  }

  return prod;

}

double quantreg::KM_weight(uint& order, arma::vec& matY, arma::vec& delta) const{

  double G_c = 1-G_func(order,matY,delta);
  double weight = delta(order)/(matY.n_elem+1)*G_c;

  return weight;

}

double quantreg::KM_fun(const double& time, const arma::vec& matY, const arma::vec& delta)const{

  arma::uvec indice = arma::sort_index(matY);
  arma::vec Y1 = arma::sort(matY);
  arma::vec delta1 = delta;
  for(uint j = 0;j<matY.n_elem;j++){

    uint indj = indice(j);
    delta1(j) = delta(indj);

  }

  double sum = 0;

  for(uint i = 0;i<matY.n_elem;i++){

    if(Y1(i)<time){

      sum = sum+KM_weight(i,Y1,delta1);

    }

  }

  return sum;

}


// repetition function which returns a row vector with times num;
// @param num a number;
// @param times an integer-valued number of times to repeat num;
arma::rowvec quantreg::rep_cpp(double num,uint times) const{
  arma::rowvec result(times);
  for(uint i = 0;i<times;i++){
    result(i) = num;
  }
  return move(result);
}


// a function which returns a binary vector indicating if
//there is a match or not in vector a for vector b;
arma::vec quantreg::in_cpp(const arma::vec &a, const arma::uvec &b) const{
  arma::vec result = arma::zeros<arma::vec>(a.n_elem);
  for(uint i = 0;i<a.n_elem;i++){
    if(sum(b.elem(find(b==a(i))))!=0){
      result(i) = 1;
    }else{
      result(i) = 0;
    }
  }
  return move(result);
}

void quantreg::in(uint us,
                  uint ue,
                  uint vs,
                  uint ve,
                  const arma::uvec &IB,
                  arma::vec &u,
                  arma::vec &v) const {
  int n = v.n_elem;
  u.fill(arma::fill::zeros);
  v.fill(arma::fill::zeros);

  for (int i = 0;i < n; i++) {
    int x = IB(i);
    if (x >= us && x <= ue)
      u(x-us) = 1;
    else if (x >= vs && x <= ve)
      v(x-vs) = 1;
  }
}


//main function to implement quantile regression with simplex
//method;
//@param x design matrix;
//@param y a vector of response variable;
//@param weights a vector with the same length as y
//@param tau a fix quantile level;
//@param tol threshold to check convergence. The default value is
//           1e-14;
//@param maxit the maximum number of inerations at each tau.
//             Usually, larger sample size requires larger
//             numbers of iteration to converge. The default
//             value is 1e5;


// This function will return: the primary solution as estimate,
// a list of quantile levels as tau and the difference of dual
//solution as dual_sol;

void quantreg::qr_tau_para_diff_fix_cpp(const arma::mat& matZ,
                                        const arma::colvec& matY,
                                        const arma::rowvec& weights,
                                        const double& tau,
                                        arma::mat& est_beta,
                                        double tol,
                                        uint maxit) const{

  //n: number of obs;
  uint n = matZ.n_rows;

  //if(n>max_num_tau) max_num_tau = 2*n;
  //nvar: number of covariates;
  uint nvar = matZ.n_cols;
  //cout << tau << endl;
  //cc: last row in the linear programming tableau of QR;
  arma::rowvec cc = arma::zeros<arma::rowvec>(nvar+1+2*n);
  for(uint j = nvar+1;j<nvar+n+1;j++){

    cc[j] = tau*weights[j];cc[j+n] = (1-tau)*weights[j];

  }

  arma::colvec col_one= arma::ones<arma::colvec>(n);
  arma::mat gammax_org = matZ;
  arma::mat gammaxb = gammax_org.t();
  //b: last column in the linear programming tableau of QR;
  arma::colvec col_zero = arma::zeros<arma::colvec>(1);
  arma::colvec b = join_cols(matY,col_zero);
  //flip the sign if y<0;
  for(uint i = 0;i<n;i++){

    if(matY[i]<0){

      gammax_org.row(i) = -gammax_org.row(i);
      b.row(i) = -b.row(i);

    }

  }


  //IB: index of variables in the basic set;
  arma::uvec IB = arma::zeros<arma::uvec>(n);
  for(uint j = 0;j<n;j++){

    if(matY[j]>=0) IB[j] = nvar+j+1;
    else IB[j] = nvar+n+j+1;

  }

  //transformation of the LP tableau to initialize optimization;
  arma::rowvec cc_trans = -cc.cols(IB);
  arma::mat gammax = join_cols(gammax_org,cc_trans*gammax_org);
  //cout << gammax(gammax.n_rows-1,0) << endl;

  //once beta is pivoted to basic set, it cannot be pivoted out;
  arma::uvec free_var_basic = arma::zeros<arma::uvec>(n+1);
  free_var_basic[n] = 1;

  //r1,r2: index of positive or negative beta in the basic set;
  arma::uvec r1 = arma::regspace<arma::uvec>(0,nvar);
  arma::uvec r2 = arma::zeros<arma::uvec>(nvar+1);
  arma::mat rr = arma::zeros<arma::mat>(2, 1+nvar);

  //c0: a vector helps to generate the next tau level;
  arma::rowvec c0 = arma::zeros<arma::rowvec>(1+nvar+2*n);
  c0.subvec(nvar+1,nvar+n) = weights;
  c0.subvec(nvar+n+1,nvar+2*n) = -weights;

  // variables used in the while loop;
  uint j = 0;
  arma::vec yy = arma::zeros<arma::vec>(gammax.n_rows);
  arma::vec ee = arma::zeros<arma::vec>(gammax.n_rows);
  uint t_rr = 0;
  uint tsep = 0;
  arma::uvec nokeep = arma::zeros<arma::uvec>(gammax.n_rows);
  double t = 0;
  arma::vec k = arma::zeros<arma::vec>(gammax.n_rows);
  uint k_index = 0;
  arma::vec estimate = arma::zeros<arma::vec>(nvar+1);
  uint tau_t = 0;
  arma::rowvec r1j_temp = arma::zeros<arma::rowvec>(1+nvar);
  arma::rowvec r1j = arma::zeros<arma::rowvec>(2*(1+nvar));
  arma::rowvec r0j = arma::zeros<arma::rowvec>(2*(1+nvar));
  arma::vec u = arma::zeros<arma::vec>(n);
  arma::vec u_temp = arma::regspace<arma::vec>(1+nvar,nvar+n);
  arma::vec v = arma::zeros<arma::vec>(n);
  arma::vec v_temp = arma::regspace<arma::vec>(1+nvar+n,nvar+2*n);
  arma::vec temp1 = arma::zeros<arma::vec>(n);
  arma::vec temp = arma::zeros<arma::vec>(n);

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

      if(rr.min()>-tol) break;
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
      std::cout <<"min k is "<<k.min()<<std::endl;

      free_var_basic(k_index) = 1;

    }

    ee = yy/yy(k_index);
    ee(k_index) = 1-1/yy(k_index);

    std::cout <<"IB(k_index) is "<<IB(k_index)<<std::endl;
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

    std::cout << "WARNING:May not converge (tau = "<< tau <<")"<< std::endl;

  }

  arma::uvec tmp = find(IB<nvar+1);
  arma::vec estimate_temp = b(tmp);
  estimate_temp = estimate_temp(sort_index(IB(tmp)));

  if(estimate_temp.n_elem!=nvar+1){

    estimate.zeros();
    estimate(sort(IB(tmp))) = estimate_temp;

  }else{

    estimate = estimate_temp;
  }

  est_beta = estimate;


  // u = in_cpp(u_temp,IB);
  // v = in_cpp(v_temp,IB);
  //
  // arma::mat xh = gammaxb.cols(find(u==0&&v==0));
  // arma::mat xbarh = gammaxb.cols(find(u==1||v==1));
  // dbarh = u.t()%rep_cpp(tau,u.n_elem)+v.t()%rep_cpp(tau-1,v.n_elem);
  // dbarh = dbarh%weights;
  // //cout<<"dbarh dimension is "<< dbarh.n_elem<<endl;
  //
  // arma::vec dh = arma::zeros<arma::vec>(sum(u==1||v==1));
  // try{
  //
  //   dh = solve(xh,-xbarh*dbarh.cols(find(u==1||v==1)).t());
  //
  // }
  // catch(const std::runtime_error& error){
  //
  //   dh = -arma::pinv(xh)*xbarh*dbarh.cols(find(u==1||v==1));
  //
  // }
  // //cout << "dh dimension is "<<dh.n_rows<<endl;
  //
  // dh = dh/weights.elem(find(u==0&&v==0)).as_col()+1-tau;
  //
  // temp1 = u;
  // if(use_residual==false){
  //
  //   dh.elem(find(dh==tau)).ones();
  //   dh.elem(find(dh==tau-1)).zeros();
  //
  // }
  // temp1(find(u==0&&v==0)) = dh;
  // dual_sol = temp1;

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


void quantreg::qr_tau_para_diff_cpp(const arma::mat& x,
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
                                    const bool use_residual) const{
#ifdef DEBUG
  ProfilerStart("tau.prof");
#endif
  print_enter("simplex:");
  //n: number of obs;
  uint n = x.n_rows;

  //if(n>max_num_tau) max_num_tau = 2*n;
  //nvar: number of covariates;
  uint nvar = x.n_cols;
  double tau = tau_min+taurange(0);
  //cout << tau << endl;
  //cc: last row in the linear programming tableau of QR;
  arma::rowvec cc = arma::zeros<arma::rowvec>(nvar+1+2*n);
  for(uint j = nvar+1;j<nvar+n+1;j++){
    uint index_weight = j-nvar-1;
    cc[j] = tau*weights[index_weight];
    cc[j+n] = (1-tau)*weights[index_weight];
  }

  arma::colvec col_one = arma::ones<arma::colvec>(n);
  arma::mat gammax_org = join_rows(col_one,x);
  arma::mat gammaxb = gammax_org.t();
  //b: last column in the linear programming tableau of QR;
  arma::colvec col_zero = arma::zeros<arma::colvec>(1);
  arma::colvec b = join_cols(y,col_zero);
  //flip the sign if y<0;

  for (int i = 0;i < y.n_elem; i++)
    if (y(i) < 0)
    {
      gammax_org.row(i) = -gammax_org.row(i);
      b(i) = -b(i);
    }
    /*
     gammax_org.rows(find(y<0)) = -gammax_org.rows(find(y<0));
     b.rows(find(y<0)) = -b.rows(find(y<0));
     */


    //IB: index of variables in the basic set;
    arma::uvec IB = arma::zeros<arma::uvec>(n);
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
    arma::uvec free_var_basic = arma::zeros<arma::uvec>(n+1);
    free_var_basic[n] = 1;

    //r1,r2: index of positive or negative beta in the basic set;
    arma::uvec r1 = arma::regspace<arma::uvec>(0,nvar);
    arma::uvec r2 = arma::zeros<arma::uvec>(nvar+1);
    arma::mat rr = arma::zeros<arma::mat>(2, 1+nvar);

    //Initialize estimation output matrix;
    //est_beta: estimation matrix;
    // arma::mat est_beta(nvar+1,1,arma::fill::zeros);
    // est_beta = arma::regspace(1,nvar+1);

    //dual_sol: dual solution matrix a sparse matrix
    // arma::sp_mat dual_sol(n,max_num_tau);

    //tau_list:: a list of tau, automatically generated in alg;
    // vector<double> tau_list;

    //c0: a vector helps to generate the next tau level;
    arma::rowvec c0 = arma::zeros<arma::rowvec>(1+nvar+2*n);
    c0.subvec(nvar+1,nvar+n) = weights;
    c0.subvec(nvar+n+1,nvar+2*n) = -weights;


    //terminate loop indicator;
    bool last_flag = false;
    //cout << last_flag <<endl;

    // variables used in the while loop;
    uint j = 0;
    arma::vec yy = arma::zeros<arma::vec>(gammax.n_rows);
    arma::vec ee = arma::zeros<arma::vec>(gammax.n_rows);
    uint t_rr = 0;
    uint tsep = 0;
    arma::uvec nokeep = arma::zeros<arma::uvec>(gammax.n_rows);
    double t = 0;
    arma::vec k = arma::zeros<arma::vec>(gammax.n_rows);
    uint k_index = 0;

    arma::vec estimate = arma::zeros<arma::vec>(nvar+1);
    uint tau_t = 0;
    arma::rowvec r1j_temp = arma::zeros<arma::rowvec>(1+nvar);
    arma::rowvec r1j = arma::zeros<arma::rowvec>(2*(1+nvar));
    arma::rowvec r0j = arma::zeros<arma::rowvec>(2*(1+nvar));
    arma::rowvec theta = arma::zeros<arma::rowvec>(2*(1+nvar));
    arma::vec u = arma::zeros<arma::vec>(n);
    arma::vec u_temp = arma::regspace<arma::vec>(1+nvar,nvar+n);
    arma::vec v = arma::zeros<arma::vec>(n);
    arma::vec v_temp = arma::regspace<arma::vec>(1+nvar+n,nvar+2*n);
    arma::rowvec dbarh = arma::zeros<arma::rowvec>(n);
    arma::vec temp1 = arma::zeros<arma::vec>(n);
    arma::vec temp = arma::zeros<arma::vec>(n);
    print(0);
    while(tau_t<max_num_tau){
      // std::cout<<"tau_t is "<<tau_t<<std::endl;
      if(tau_t>0){
        r1j_temp = c0.cols(IB)*gammax(arma::span(0,n-1),arma::span::all);
        r1j = join_rows(r1j_temp-1,1-r1j_temp);
        r0j = join_rows(gammax.row(n),weights.elem(r1-(1+nvar)).as_row()-gammax.row(n));
        theta = r0j/r1j;
        bool choose = false;
        double theta_min = 0;
        for (int i = 0;i < (1+nvar)*2; i++)
          if (r1j[i] > 0 && (!choose || theta_min > theta[i]))
          {
            choose = true;
            theta_min = theta[i];
          }
          // double theta_min = arma::as_scalar(theta.cols(find(r1j>0)).min());

          tau = tau + theta_min + tau_min;

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
        // std::cout << "j is "<< j << std::endl;

        if(tau_t==0&&j<nvar+1){
          rr.row(0) = gammax.row(n);
          for(uint i = 0;i<r2.n_elem;i++){
            // std::cout<<"i is "<<i<<std::endl;
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
          print(1);
          //cout << rr.n_cols<< endl;
          //cout << gammax.n_cols <<endl;

          rr.row(0) = gammax.row(n);

          for(uint i = 0;i<r2.n_elem;i++){
            std::cout<<"r2i is "<<r2(i)<<std::endl;
            if(r2(i)==0){
              rr(1,i) = 0;
              rr(0,i) = -abs(rr(0,i));
            }else{
              rr(1,i) = weights(r2(i)-(1+nvar+n))-rr(0,i);
            }
          }
          print(2);
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

        // std::cout<<"DONE3"<<std::endl;
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

          std::cout<<"min k is "<<k.min()<<std::endl;

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
          std::cout<<"min k is "<<k.min()<<std::endl;
          free_var_basic(k_index) = 1;
        }
        print(3);
        ee = yy/yy(k_index);
        ee(k_index) = 1-1/yy(k_index);

        std::cout <<"IB(k_index) is "<<IB(k_index)<<std::endl;
        std::cout <<"free_var_basic(k_index) is "<<free_var_basic(k_index)<<std::endl;
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
        std::cout <<"r2(t_rr) is "<<r2(t_rr)<<std::endl;
        gammax = gammax-ee*gammax.row(k_index);
        b = b-ee*arma::as_scalar(b[k_index]);
        IB(k_index) = t;
        //cout << "t is " << t << endl;
        // std::cout<<"DONE2"<<std::endl;
        j++;
      }



      if(j==maxit){
        //cout << "WARNING:May not converge (tau = "<< tau <<")"<< endl;
      }

      arma::uvec tmp = find(IB<nvar+1);
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

      in(1+nvar, n+nvar, 1+nvar+n, nvar+2*n, IB, u, v);

      /*
       arma::vec u2(IB.n_elem);
       arma::vec v2(IB.n_elem);
       u2 = in_cpp(u_temp,IB);
       v2 = in_cpp(v_temp,IB);

       if (arma::any(u != u2))
       cout << "in cpp failed" << endl;
       if (arma::any(v != v2))
       cout << "in cpp failed" << endl;
       */

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
      catch(const std::runtime_error& error){
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
  if(last_flag==true&&j==0&&tau_t>=3){
    est_beta.shed_col(tau_t);
    tau_list.erase(tau_list.begin()+tau_t-2);
    dual_sol.shed_cols(tau_t-1,max_num_tau-1);
    dual_sol.shed_col(tau_t-3);
  }else if(last_flag==true&&j==0){
    est_beta.shed_col(tau_t);
    tau_list.erase(tau_list.begin()+tau_t-2);
    dual_sol.shed_cols(tau_t-1,max_num_tau-1);
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
    cout << "finish" << endl;
#endif

}

arma::vec quantreg::ranks_cpp(const arma::vec& matY,
                         const arma::mat& matZ,
                         const arma::rowvec& weights,
                         const arma::vec& taurange,
                         uint max_num_tau) const{

  // quantreg qr(taurange);
  print_enter("ranks:");
  uint n_Z = matZ.n_cols;
  uint n = matY.n_elem;
  std::cout<<"n is "<<n<<std::endl;
  arma::vec ranks = arma::zeros<arma::vec>(n);

  //Initialize estimation output matrix;
  //est_beta: estimation matrix;
  arma::mat est_beta(n_Z+1,1,arma::fill::zeros);
  est_beta = arma::regspace(1,n_Z+1);

  //dual_sol: dual solution matrix a sparse matrix
  arma::sp_mat dual_sol(n,max_num_tau);

  //tau_list:: a list of tau, automatically generated in alg;
  vector<double> tau_list;
  print(0);
  qr_tau_para_diff_cpp(matZ,matY,weights,taurange,est_beta,dual_sol,
                       tau_list,1e-10,1e-14,100000,max_num_tau);
  print(1);
  uint J = tau_list.size();


  // std::cout<<"A2 is "<<A2<<std::endl;
  arma::vec phi = arma::zeros<arma::vec>(J);
  arma::vec dt = arma::zeros<arma::vec>(J-1);
  dt(0) = tau_list.at(1)-tau_list.at(0);
  for(uint i = 1;i<J-1;i++){

    // std::cout<<"tau_i is "<<tau_list.at(i)<<std::endl;
    phi(i) = 0.5*tau_list.at(i)-0.5*tau_list.at(i)*tau_list.at(i);
    dt(i) = tau_list.at(i+1)-tau_list.at(i);

  }
  // std::cout<<"J is "<<J<<std::endl;
  arma::vec dphi = arma::zeros<arma::vec>(J-1);
  dphi = phi(arma::span(1,J-1))-phi(arma::span(0,J-2));

  ranks = dual_sol*(dphi/dt);

  return ranks;


}


double quantreg::rankscore_cpp(const arma::mat& matX,
                               const arma::mat& matZ,
                               const arma::rowvec& weights,
                               const arma::vec& taurange,
                               const arma::vec& ranks,
                               uint max_num_tau) const{

  quantreg qr(taurange);

  double A2 = 1.0/12.0;
  uint n = matX.n_elem;


  arma::colvec col_one = arma::ones<arma::colvec>(n);
  arma::mat design_Z = join_rows(col_one,matZ);
  arma::mat design_X = join_rows(col_one,matX);
  arma::mat matZ_new = matZ.t()*matZ;
  arma::mat proj_Z = arma::zeros<arma::mat>(n,n);
  try{
    proj_Z = matZ*arma::inv(matZ_new)*matZ.t();
  }
  catch(const std::runtime_error& error){
    proj_Z = matZ*arma::pinv(matZ_new)*matZ.t();
  }

  arma::mat sn = (design_X.t()-design_X.t()*proj_Z)*ranks;
  arma::mat Qn = (design_X.t()-design_X.t()*proj_Z)*design_X;
  // std::cout<<"DONE2"<<std::endl;
  double Tn = 0.0;
  arma::vec col_one_qn = arma::ones<arma::vec>(Qn.n_cols);
  arma::mat Qn_iv = arma::zeros<arma::mat>(Qn.n_cols,Qn.n_cols);

  try{
    Qn_iv = arma::solve(Qn,arma::diagmat(col_one_qn));
  }
  catch(const std::runtime_error& error){
    Qn_iv = arma::pinv(Qn);
  }

  Tn = arma::as_scalar(sn.t()*Qn_iv*sn)/A2;

  return Tn;

}
