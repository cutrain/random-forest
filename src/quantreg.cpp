#include "quantreg.h"
#include "common.h"

const double eps = 1e-14;

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
      // BUG?
      // prod = prod*((n-1-j)/(n-j));
      prod *= static_cast<double>(n-1-j) / (n-j);
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
extern arma::rowvec rep_cpp(double num,uint times);

// a function which returns a binary vector indicating if
//there is a match or not in vector a for vector b;
extern void in(uint us,
    uint ue,
    uint vs,
    uint ve,
    const arma::uvec &IB,
    arma::vec &u,
    arma::vec &v);

extern arma::vec get_dh(const arma::mat& gammaxb,
    const arma::vec& u,
    const arma::vec& v,
    const double tau,
    const double tau_min,
    const arma::rowvec& weights);

extern void update_dual_sol(int tau_t, double tau, double tau_min, bool use_residual,
    arma::vec& pre, arma::vec& now,
    arma::vec& u, arma::vec& v,
    arma::vec& dh, arma::sp_mat& dual_sol);

//main function to implement quantile regression with simplex
//method at a fixed quantile level;
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
// solution as dual_sol;
void quantreg::qr_tau_para_diff_fix_cpp(const arma::mat& x,
                                        const arma::colvec& y,
                                        const arma::rowvec& weights,
                                        const double& tau,
                                        arma::vec& est_beta,
                                        arma::vec& residual,
                                        double tol,
                                        uint maxit) const{
  //n: number of obs;
  uint n = x.n_rows;

  //nvar: number of covariates;
  uint nvar = x.n_cols;
  //cc: last row in the linear programming tableau of QR;
  arma::rowvec cc(nvar+1+2*n,arma::fill::zeros);
  for(uint j = nvar+1;j<nvar+n+1;j++){
    uint index_weight = j-nvar-1;
    cc[j] = tau*weights[index_weight];
    cc[j+n] = (1-tau)*weights[index_weight];
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
  for (int i = 0;i < y.n_elem; i++)
    if (y(i) < 0)
    {
      gammax_org.row(i) = -gammax_org.row(i);
      b(i) = -b(i);
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
    int tau_t = 0;
    arma::vec estimate(nvar+1,arma::fill::zeros);
    arma::vec u(n,arma::fill::zeros);
    arma::vec v(n,arma::fill::zeros);
    arma::vec pre(n,arma::fill::zeros);
    arma::vec now(n,arma::fill::zeros);
    j = 0;
    while (j < maxit) {
      int tsep;
      int t_rr;
      double t;
      if(j<nvar+1) {
        rr.row(0) = gammax.row(n);
        for(uint i = 0;i<r2.n_elem;i++) {
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
        rr.row(0) = gammax.row(n);
        for(uint i = 0;i<r2.n_elem;i++) {
          if(r2(i)==0){
            rr(1,i) = 0;
            rr(0,i) = -abs(rr(0,i));
          }else{
            rr(1,i) = weights(r2(i)-(1+nvar+n))-rr(0,i);
          }
        }

        if(rr.min()>-tol)
          break;
        int index = rr.index_min();
        t_rr = index / 2;
        tsep = index % 2;
        // t_rr = floor(tsep/2);
        // tsep = tsep-floor(tsep/2)*2;

        if(tsep==0) {
          t = r1(t_rr);
        }else{
          t = r2(t_rr);
        }
      }
      int k_index;
      arma::vec yy; // (gammax.n_rows, )
      arma::vec k;
      if (r2(t_rr)!=0) {
        if(tsep==0) {
          yy = gammax.col(t_rr);
        }else{
          yy = -gammax.col(t_rr);
        }

        k = b/yy; // gammax.n_rows
        arma::uvec nokeep = arma::find(yy<=eps || free_var_basic==1);
        k(nokeep) = arma::ones<arma::vec>(nokeep.n_elem)*DBL_MAX;
        k_index = k.index_min();

        if(tsep!=0) {
          yy(n) = yy(n) + weights(r2[t_rr] - (1+nvar+n));
        }
      } else {
        yy = gammax.col(t_rr);
        if (yy(n) < 0) {
          k = b/yy; // gammax.n_rows
          arma::uvec nokeep = arma::find(yy<=eps || free_var_basic==1);
          k(nokeep) = arma::ones<arma::vec>(nokeep.n_elem)*DBL_MAX;
          k_index = k.index_min();
        } else {
          k = -b/yy; // gammax.n_rows
          arma::uvec nokeep = arma::find(yy>=-eps || free_var_basic==1);
          k(nokeep) = arma::ones<arma::vec>(nokeep.n_elem)*DBL_MAX;
          k_index = k.index_min();
        }

        free_var_basic(k_index) = 1;
      }
      if(k.min()==DBL_MAX)
        break;

      arma::vec ee = yy/yy(k_index);
      ee(k_index) = 1-1/yy(k_index);

      // cout <<"IB(k_index) is "<<IB(k_index)<<endl;
      // cout <<"k_index is "<<k_index<<endl;
      if(IB(k_index)<=nvar+n){
        gammax.col(t_rr) = rep_cpp(0, gammax.n_rows).t();
        gammax(k_index,t_rr) = 1;
        r1(t_rr) = IB(k_index);
        r2(t_rr) = IB(k_index) + n;
      }else{
        gammax.col(t_rr) = rep_cpp(0, gammax.n_rows).t();
        gammax(k_index,t_rr) = -1;
        gammax(n,t_rr)  =  weights(IB[k_index] - (1+nvar+n));
        r1(t_rr) = IB(k_index) - n;
        r2(t_rr) = IB(k_index);
      }

      gammax = gammax - ee * gammax.row(k_index);
      b = b - ee * arma::as_scalar(b[k_index]);
      IB(k_index) = t;

      j++;
    }

  if(j==maxit){
    std::cout << "WARNING:May not converge (tau = "<< tau <<")"<< std::endl;
  }

  arma::uvec tmp = find(IB<nvar+1);
  arma::vec estimate_temp = b(tmp);
  estimate_temp = estimate_temp(sort_index(IB(tmp)));

  if(estimate_temp.n_elem!=nvar+1){
    est_beta.zeros();
    est_beta(sort(IB(tmp))) = estimate_temp;
  }else{
    est_beta = estimate_temp;
  }

  arma::uvec tmp_res = find(IB>=nvar+1);
  arma::vec res_temp = b(tmp_res);
  arma::uvec index_temp = IB(tmp_res);
  for(uint i = 0;i<index_temp.n_elem;i++){
    if(index_temp(i)>nvar+n){
      index_temp(i) = index_temp(i)-n;
      res_temp(i) = -res_temp(i);
    }
  }
  res_temp = res_temp(sort_index(index_temp));
  residual.zeros();
  residual(sort(index_temp-nvar-1)) = res_temp;
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
  // print_enter("simplex:");
  //n: number of obs;
  uint n = x.n_rows;
  // std::cout<<"n is "<<n<<std::endl;

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

    //c0: a vector helps to generate the next tau level;
    arma::rowvec c0 = arma::zeros<arma::rowvec>(1+nvar+2*n);
    c0.subvec(nvar+1,nvar+n) = weights;
    c0.subvec(nvar+n+1,nvar+2*n) = -weights;


    //terminate loop indicator;
    bool last_flag = false;
    //cout << last_flag <<endl;

    // variables used in the while loop;
    uint j = 0;

    arma::vec estimate = arma::zeros<arma::vec>(nvar+1);
    int tau_t = 0;
    arma::vec u = arma::zeros<arma::vec>(n);
    arma::vec v = arma::zeros<arma::vec>(n);
    arma::vec pre = arma::zeros<arma::vec>(n);
    arma::vec now = arma::zeros<arma::vec>(n);
    // print(-1);
    while(tau_t<max_num_tau){
      // std::cout<<"tau_t is "<<tau_t<<std::endl;
      // std::cout<<"r1 is "<<r1<<std::endl;
      // print(tau_t);
      if(tau_t>0){
        arma::rowvec r1j_temp = c0.cols(IB)*gammax(arma::span(0,n-1),arma::span::all);
        arma::rowvec r1j = join_rows(r1j_temp-1,1-r1j_temp);
        arma::rowvec r0j = join_rows(gammax.row(n),-gammax.row(n));
          for(uint g = 0;g<gammax.n_cols;g++){
            if(r1(g)>=nvar+1){
              r0j(g+gammax.n_cols) = weights(r1(g)-(1+nvar))+r0j(g+gammax.n_cols);
            } 
          }
        // print(-6);
        arma::rowvec theta = r0j/r1j;
        bool choose = false;
        double theta_min = 0;
        for (int i = 0;i < (1+nvar)*2; i++)
          if (r1j[i] > 0 && (!choose || theta_min > theta[i]))
          {
            choose = true;
            theta_min = theta[i];
          }
          // double theta_min = arma::as_scalar(theta.cols(find(r1j>0)).min());
          // print(-7);
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
      // print(-2);
      j = 0;
      while(j<maxit){
        // std::cout << "j is "<< j << std::endl;
        int tsep;
        int t_rr;
        double t;

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
          // print(-3);
          //cout << rr.n_cols<< endl;
          //cout << gammax.n_cols <<endl;

          rr.row(0) = gammax.row(n);

          for(uint i = 0;i<r2.n_elem;i++){
            // std::cout<<"r2i is "<<r2(i)<<std::endl;
            if(r2(i)==0){
              rr(1,i) = 0;
              rr(0,i) = -abs(rr(0,i));
            }else{
              rr(1,i) = weights(r2(i)-(1+nvar+n))-rr(0,i);
            }
          }
          // print(-4);
          if(rr.min()>-tol)
            break;
          int index = rr.index_min();
          //cout << "tsep is "<< tsep << endl;
          t_rr = index/2;
          //cout << "t_rr is "<< t_rr << endl;
          tsep = index % 2;
          // cout << "tsep is "<< tsep << endl;

          if(tsep==0){
            t = r1(t_rr);
          }else{
            t = r2(t_rr);
          }
        }

        // std::cout<<"DONE3"<<std::endl;
        int k_index;
        arma::vec yy;
        arma::vec k;
        if (r2(t_rr)!=0){
          if(tsep==0){
            yy = gammax.col(t_rr);
          }else{
            yy = -gammax.col(t_rr);
          }

          k = b/yy;
          arma::uvec nokeep = arma::find(yy<=eps || free_var_basic==1);
          k(nokeep) = arma::ones<arma::vec>(nokeep.n_elem)*DBL_MAX;
          k_index = k.index_min();

          if(tsep!=0){
            yy(n) = yy(n)+weights(r2[t_rr]-(1+nvar+n));
          }

          // std::cout<<"min k is "<<k.min()<<std::endl;

        }else{
          //cout << t_rr<< endl;
          yy = gammax.col(t_rr);
          // if (abs(yy(n)) < 1e-10) {
          //   cout << "eps found!!!" << endl;
          // }
          if(yy(n)<0){

            k = b/yy;
            arma::uvec nokeep = arma::find(yy<=eps || free_var_basic==1);
            k(nokeep) = arma::ones<arma::vec>(nokeep.n_elem)*DBL_MAX;
            k_index = k.index_min();

          }else{
            k = -b/yy;
            arma::uvec nokeep = arma::find(yy>=-eps || free_var_basic==1);
            k(nokeep) = arma::ones<arma::vec>(nokeep.n_elem)*DBL_MAX;
            k_index = k.index_min();
          }
          // std::cout<<"min k is "<<k.min()<<std::endl;
          free_var_basic(k_index) = 1;
        }
        // std::cout <<"k min is "<<k.min()<<std::endl;
        if(k.min()==DBL_MAX) break;
        // print(-5);
        arma::vec ee = yy/yy(k_index);
        ee(k_index) = 1-1/yy(k_index);

        // cout <<"k_index is "<<k_index<<endl;
        // std::cout <<"IB(k_index) is "<<IB(k_index)<<std::endl;
        // std::cout <<"free_var_basic(k_index) is "<<free_var_basic(k_index)<<std::endl;
        // cout <<"k min is "<<k.min()<<endl;
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
        // std::cout<<"r1 is "<<r1<<std::endl;
        // std::cout <<"r2(t_rr) is "<<r2(t_rr)<<std::endl;
        gammax = gammax-ee*gammax.row(k_index);
        b = b-ee*arma::as_scalar(b[k_index]);
        IB(k_index) = t;
        //cout << "t is " << t << endl;
        // std::cout<<"DONE2"<<std::endl;
        j++;
      }



      if(j==maxit){
        cout << "WARNING:May not converge (tau = "<< tau <<")"<< endl;
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

      arma::vec dh = get_dh(gammaxb,u,v,tau,tau_min,weights);

      update_dual_sol(tau_t, tau, tau_min, use_residual,
          pre, now,
          u, v,
          dh, dual_sol);

      tau_t++;
    }
    //cout << "est_beta dimension" << est_beta.n_cols << endl;
    //cout << "tau_t is "<< tau_t<< endl;
  if(last_flag==true&&j==0&&tau_t>=3){
    est_beta.shed_col(tau_t);
    tau_list.erase(tau_list.begin()+tau_t-2);
    dual_sol.shed_cols(tau_t-1,max_num_tau-1);
    dual_sol.shed_col(tau_t-3);
  }else if(last_flag==true&&j==0&&tau_t>=2){
    est_beta.shed_col(tau_t);
    tau_list.erase(tau_list.begin()+tau_t-2);
    dual_sol.shed_cols(tau_t-1,max_num_tau-1);
  }else{
    est_beta.shed_col(tau_t);
    dual_sol.shed_cols(tau_t-1,max_num_tau-1);
  }
  
  // print_leave();


    //cout<<est_beta.n_cols<<endl;
    //cout<< "tau_t is "<<tau_t<<endl;

    //est_beta.shed_col(tau_t);
    //dual_sol.shed_cols(tau_t-1,max_num_tau-1);
    est_beta.shed_col(0);
}

arma::vec quantreg::ranks_cpp(const arma::vec& matY,
                         const arma::mat& matZ,
                         const arma::rowvec& weights,
                         const arma::vec& taurange,
                         uint max_num_tau) const{
  uint n_Z = matZ.n_cols;
  uint n = matY.n_elem;
  arma::vec ranks = arma::zeros<arma::vec>(n);

  //Initialize estimation output matrix;
  //est_beta: estimation matrix;
  arma::mat est_beta(n_Z+1,1,arma::fill::zeros);
  est_beta = arma::regspace(1,n_Z+1);

  //dual_sol: dual solution matrix a sparse matrix
  arma::sp_mat dual_sol(n,max_num_tau);

  //tau_list:: a list of tau, automatically generated in alg;
  vector<double> tau_list;
  qr_tau_para_diff_cpp(matZ,matY,weights,taurange,est_beta,dual_sol,
                       tau_list,1e-10,1e-14,100000,max_num_tau);
  uint J = tau_list.size();


  arma::vec phi = arma::zeros<arma::vec>(J);
  arma::vec dt = arma::zeros<arma::vec>(J-1);
  dt(0) = tau_list.at(1)-tau_list.at(0);
  for(uint i = 1;i<J-1;i++){
    phi(i) = 0.5*tau_list.at(i)-0.5*tau_list.at(i)*tau_list.at(i);
    dt(i) = tau_list.at(i+1)-tau_list.at(i);
  }
  arma::vec dphi = arma::zeros<arma::vec>(J-1);
  dphi = phi(arma::span(1,J-1))-phi(arma::span(0,J-2));

  return dual_sol*(dphi/dt);
}

arma::vec quantreg::ranks_cpp_marginal(const arma::vec& matY) const
{
  uint n = matY.n_elem;
  arma::vec ranks = arma::zeros<arma::vec>(n);

  arma::vec phi = arma::zeros<arma::vec>(n+1);
  double dt = (double) 1/n;
  for(uint i = 1;i<n;i++){
    double tau_temp = (double) i/n;
    phi(i) = 0.5*tau_temp*(1-tau_temp);
  }
  arma::vec dphi = phi(arma::span(1,n))-phi(arma::span(0,n-1));
  arma::uvec ranks_Y = arma::sort_index(arma::sort_index(matY));

  ranks = dphi/dt;
  ranks = -ranks(ranks_Y);
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
  arma::mat matZ_new = design_X.t()*design_X;
  arma::mat proj_Z = arma::zeros<arma::mat>(n,n);
  try{
    proj_Z = design_X*arma::inv(matZ_new)*design_X.t();
  }
  catch(const std::runtime_error& error){
    proj_Z = design_X*arma::pinv(matZ_new)*design_X.t();
  }

  arma::mat sn = (design_X.t()-design_X.t()*proj_Z)*ranks;
  arma::mat Qn = (design_X.t()-design_X.t()*proj_Z)*design_X;
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

double quantreg::rankscore_cpp_marginal(const arma::mat& matX,
                                        const arma::vec& ranks) const
  {
  double A2 = 1.0/12.0;
  uint n = matX.n_elem;

  arma::colvec col_one = arma::ones<arma::colvec>(n);
  arma::mat design_X = join_rows(col_one,matX);
  arma::mat proj_Z = arma::ones<arma::mat>(n,n)/(double) n;

  arma::mat sn = (design_X.t()-design_X.t()*proj_Z)*ranks;
  arma::mat Qn = (design_X.t()-design_X.t()*proj_Z)*design_X;

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
