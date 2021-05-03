#include "quantreg.h"
#include "common.h"

const double eps = 1e-14;

/**********************/
/*Functional functions*/
/**********************/

double quantreg::G_func(uint& order, 
                        arma::vec& matY,
                        arma::vec& delta) const
  {

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

double quantreg::KM_weight(uint& order, 
                           arma::vec& matY, 
                           arma::vec& delta) const
  {

  double G_c = 1-G_func(order,matY,delta);
  double weight = delta(order)/(matY.n_elem+1)*G_c;

  return weight;

}

double quantreg::KM_fun(const double& time, 
                        const arma::vec& matY, 
                        const arma::vec& delta)const
  {

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
arma::rowvec quantreg::rep_cpp(double num,uint times) const
  {
  arma::rowvec result(times);
  for(uint i = 0;i<times;i++){
    result(i) = num;
  }
  return move(result);
}


// a function which returns a binary vector indicating if
//there is a match or not in vector a for vector b;
arma::vec quantreg::in_cpp(const arma::vec &a, const arma::uvec &b) const
  {
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

arma::vec quantreg::get_dh(const arma::mat& gammaxb,
                           const arma::vec& u,
                           const arma::vec& v,
                           const double tau,
                           const double tau_min,
                           const arma::rowvec& weights) const {
  auto u0 = u==0;
  auto u1 = u==1;
  auto v0 = v==0;
  auto v1 = v==1;
  arma::uvec f_uv0 = find(u0 && v0);
  arma::uvec f_uv1 = find(u1 || v1);
  
  arma::mat xh = gammaxb.cols(f_uv0);
  arma::mat xbarh = gammaxb.cols(f_uv1);
  arma::rowvec dbarh = u.t()%rep_cpp(tau-tau_min,u.n_elem)+
    v.t()%rep_cpp(tau-tau_min-1,v.n_elem);
  dbarh = dbarh%weights;

  arma::vec dh;
  try{
    dh = solve(xh,-xbarh*dbarh.cols(find(u==1||v==1)).t());
  }
  catch(const runtime_error& error){
    dh = -arma::pinv(xh)*xbarh*dbarh.cols(find(u==1||v==1));
  }

  dh = dh/weights.elem(find(u==0&&v==0)).as_col()+1-tau;
  return dh;
}

double quantreg::tau_inc(uint n, uint nvar, 
                         const arma::rowvec& c0, 
                         const arma::uvec& IB, 
                         const arma::mat& gammax, 
                         const arma::rowvec& weights, 
                         const arma::uvec& r1) const
{
  arma::rowvec r1j_temp = c0.cols(IB)*gammax(arma::span(0,n-1),arma::span::all); // 1+nvar
  arma::rowvec r1j = join_rows(r1j_temp-1,1-r1j_temp); // 2*(1+nvar)
  arma::rowvec r0j = join_rows(gammax.row(n),weights.elem(r1-(1+nvar)).as_row()-gammax.row(n)); // 2*(1+nvar)
  arma::rowvec theta = r0j/r1j; // 2*(1+nvar)
  
  bool choose = false;
  double theta_min = 0;
  for (int i = 0;i < (1+nvar)*2; i++)
    if (r1j[i] > eps && (!choose || theta_min > theta[i]))
    {
      choose = true;
      theta_min = theta[i];
    }
    if (!choose) {
      cout << "all r1j <= eps" << endl;
    }
    return theta_min;
}

void quantreg::inner_loop(int iter, uint n, uint nvar, int tsep, int t_rr, double t,
                          arma::mat& gammax,
                          arma::colvec& b,
                          arma::uvec& free_var_basic,
                          arma::uvec& r1,
                          arma::uvec& r2,
                          arma::uvec& IB,
                          const arma::rowvec& weights) const 
{
  int k_index;
  arma::vec yy; // gammax.n_rows
  
  if (r2(t_rr)!=0) {
    if(tsep==0) {
      yy = gammax.col(t_rr);
    }else{
      yy = -gammax.col(t_rr);
    }
    
    arma::vec k = b/yy; // gammax.n_rows
    arma::uvec nokeep = arma::find(yy<=eps || free_var_basic==1);
    k(nokeep) = rep_cpp(INFINITY, nokeep.n_elem).t();
    k_index = k.index_min();
    
    if(tsep!=0) {
      yy(n) = yy(n) + weights(r2[t_rr] - (1+nvar+n));
    }
  } else {
    yy = gammax.col(t_rr);
    if (yy(n) < -eps) {
      arma::vec k = b/yy; // gammax.n_rows
      arma::uvec nokeep = arma::find(yy<=eps || free_var_basic==1);
      k(nokeep) = rep_cpp(INFINITY, nokeep.n_elem).t();
      k_index = k.index_min();
    } else {
      arma::vec k = -b/yy; // gammax.n_rows
      arma::uvec nokeep = arma::find(yy>=-eps || free_var_basic==1);
      k(nokeep) = rep_cpp(INFINITY, nokeep.n_elem).t();
      k_index = k.index_min();
    }
    
    free_var_basic(k_index) = 1;
  }
  
  arma::vec ee = yy/yy(k_index);
  ee(k_index) = 1-1/yy(k_index);
  
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
}

void quantreg::update_dual_sol_(arma::sp_mat& dual_sol, 
                                arma::vec&& x, 
                                int p) const 
{
  // dual_sol.col(p) = x;
  for (int i = 0;i < x.n_elem; ++i) {
    if (fabs(x(i)) > eps)
      dual_sol(i,p) = x(i);
  }
}

void quantreg::update_dual_sol(int tau_t, double tau, 
                               double tau_min, bool use_residual,
                               arma::vec& pre, arma::vec& now,
                               arma::vec& u, arma::vec& v,
                               arma::vec& dh, arma::sp_mat& dual_sol) const 
{
  now = u;
  if(use_residual==false){
    dh.elem(find(dh==tau-tau_min)).ones();
    dh.elem(find(dh==tau-tau_min-1)).zeros();
  }
  now(find(u==0 && v==0)) = dh;
  if(tau_t>0){
    update_dual_sol_(dual_sol, now-pre, tau_t-1);
  }
  
  pre = now;
}


/***********************************************/
/*Rankscore at a fixed quantile level functions*/
/***********************************************/

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
//solution as dual_sol;

void quantreg::qr_tau_para_diff_fix_cpp(const arma::mat& x,
                                        const arma::colvec& y,
                                        const arma::rowvec& weights,
                                        const double& tau,
                                        arma::vec& est_beta,
                                        arma::mat& dual_sol,
                                        arma::vec& residual,
                                        double tol,
                                        uint maxit,
                                        const bool use_residual) const
  {
  
#ifdef DEBUG
  ProfilerStart("tau.prof");
#endif
  
  // print_enter("simplex:");
  //n: number of obs;
  uint n = x.n_rows;
  
  //if(n>max_num_tau) max_num_tau = 2*n;
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
    
    
    
    //c0: a vector helps to generate the next tau level;
    arma::rowvec c0(1+nvar+2*n,arma::fill::zeros);
    c0.subvec(nvar+1,nvar+n) = weights;
    c0.subvec(nvar+n+1,nvar+2*n) = -weights;
    
    
    // variables used in the while loop;
    uint j = 0;
    int tau_t = 0;
    arma::vec estimate(nvar+1,arma::fill::zeros);
    arma::vec u(n,arma::fill::zeros);
    arma::vec v(n,arma::fill::zeros);
    arma::vec pre(n,arma::fill::zeros);
    arma::vec now(n,arma::fill::zeros);
    // print(-1);
    bool notrun = true;
    for (int iter = 0;iter < maxit; ++iter) {
      rr.row(0) = gammax.row(n);
      for(uint i = 0;i<r2.n_elem;i++) {
        if(r2(i)==0){
          rr(1,i) = 0;
          rr(0,i) = -abs(rr(0,i));
        }else{
          rr(1,i) = weights(r2(i)-(1+nvar+n))-rr(0,i);
        }
      }
      
      int tsep;
      int t_rr;
      double t;
      if(tau_t==0 && iter<nvar+1) {
        t_rr = iter;
        tsep = 0;
        t = arma::as_scalar(r1(t_rr));
      } else {
        if(rr.min() > -tol)
          break;
        int index = rr.index_min();
        t_rr = index / 2;
        tsep = index % 2;
        
        if(tsep==0) {
          t = r1(t_rr);
        } else {
          t = r2(t_rr);
        }
      }
      inner_loop(iter, n, nvar, tsep, t_rr, t,
                 gammax, b, free_var_basic,
                 r1, r2, IB, weights);
      
      notrun = false;
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
    
    // std::cout<<"DONE2"<<std::endl;
    in(1+nvar, n+nvar, 1+nvar+n, nvar+2*n, IB, u, v);
    double tau_min = 0.0;
    arma::vec dh = get_dh(gammaxb,u,v,tau,tau_min,weights);
    
    now = u;
    if(use_residual==false){
      dh.elem(find(dh==tau)).ones();
      dh.elem(find(dh==tau-1)).zeros();
    }
    now(find(u==0&&v==0)) = dh;
    dual_sol = now;
    
    arma::uvec tmp_res = find(IB>=nvar+1);
    arma::vec res_temp = b(tmp_res);
    arma::uvec index_temp = IB(tmp_res);
    for(uint i = 0;i<index_temp.n_elem;i++){
      if(index_temp(i)>nvar+n){
        index_temp(i) = index_temp(i)-n;
        res_temp(i) = -res_temp(i);
      }
    }
    // std::cout<<"DONE3"<<std::endl;
    res_temp = res_temp(sort_index(index_temp));
    // std::cout<<"index_max"<<index_temp.max()<<std::endl;
    residual.zeros();
    residual(sort(index_temp-nvar-1)) = res_temp;
    // print_leave();
    
    
#ifdef DEBUG
    ProfilerStop();
    cout << "finish" << endl;
#endif
    
}

arma::vec quantreg::ranks_tau_cpp(const arma::vec& matY,
                                  const arma::mat& matZ,
                                  const arma::rowvec& weights,
                                  const double tau) const 
  {
  
  // quantreg qr(taurange);
  // print_enter("ranks:");
  uint n_Z = matZ.n_cols;
  uint n = matY.n_elem;
  // std::cout<<"n is "<<n<<std::endl;
  arma::vec ranks = arma::zeros<arma::vec>(n);
  
  //Initialize estimation output matrix;
  //est_beta: estimation matrix;
  arma::vec est_beta(n_Z+1,arma::fill::zeros);
  est_beta = arma::regspace(1,n_Z+1);
  
  //dual_sol: dual solution matrix a sparse matrix
  arma::mat dual_sol = arma::zeros<arma::mat>(n,1);
  arma::vec residual = arma::zeros<arma::vec>(n);
  
  // print(0);
  qr_tau_para_diff_fix_cpp(matZ,matY,weights,tau,est_beta,dual_sol,residual,
                           1e-14,1e5,true);
  
  // print(1);
  ranks = dual_sol-(1-tau);
  // print_leave();
  
  return ranks;
  
  
}

double quantreg::rankscore_tau_cpp(const arma::mat& matX,
                                   const arma::mat& matZ,
                                   const arma::rowvec& weights,
                                   const double tau,
                                   const arma::vec& ranks) const 
{
  
  
  double A2 = tau * (1 - tau);
  uint n = matX.n_elem;
  
  
  arma::colvec col_one = arma::ones<arma::colvec>(n);
  arma::mat design_Z = join_rows(col_one,matZ);
  for(uint j = 0;j<design_Z.n_cols;j++){
    design_Z.col(j) = design_Z.col(j)%weights.as_col();
  }
  arma::mat design_X = design_Z;
  for(uint j = 0;j<design_Z.n_cols;j++){
    design_X.col(j) = design_Z.col(j)%matX;
  }
  
  arma::mat matZ_new = design_Z.t()*design_Z;
  arma::mat proj_Z = arma::zeros<arma::mat>(n,n);
  try{
    proj_Z = design_Z*arma::inv(matZ_new)*design_Z.t();
  }
  catch(const std::runtime_error& error){
    proj_Z = design_Z*arma::pinv(matZ_new)*design_Z.t();
  }

  
  arma::mat proj_ZX = (design_X.t()-design_X.t()*proj_Z);
  arma::mat sn = proj_ZX*ranks;
  arma::mat Qn = proj_ZX*design_X;
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



/***********************************************/
/*Rankscore at whole quantile process functions*/
/***********************************************/

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
                                    const bool use_residual) const
{
#ifdef DEBUG
  ProfilerStart("tau.prof");
#endif
  // print_enter("simplex:");
  //n: number of obs;
  uint n = x.n_rows;
  
  //if(n>max_num_tau) max_num_tau = 2*n;
  //nvar: number of covariates;
  uint nvar = x.n_cols;
  double tau = tau_min+taurange(0);
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
    
    //c0: a vector helps to generate the next tau level;
    arma::rowvec c0(1+nvar+2*n,arma::fill::zeros);
    c0.subvec(nvar+1,nvar+n) = weights;
    c0.subvec(nvar+n+1,nvar+2*n) = -weights;
    
    
    //terminate loop indicator;
    bool last_flag = false;
    
    // variables used in the while loop;
    int tau_t = 0;
    bool notrun = true;
    arma::vec estimate(nvar+1,arma::fill::zeros);
    arma::vec u(n,arma::fill::zeros);
    arma::vec v(n,arma::fill::zeros);
    arma::vec pre(n,arma::fill::zeros);
    arma::vec now(n,arma::fill::zeros);
    
    for (tau_t = 0; tau_t < max_num_tau; ++tau_t){
      if(tau_t>0){
        double theta_min = tau_inc(n, nvar, c0, IB, gammax, weights, r1);
        tau = tau + theta_min + tau_min;
        
        if (tau > taurange[1]) {
          if (last_flag)
            break;
          if (tau-theta_min <= taurange[1]) {
            tau = taurange[1] - tau_min;
            last_flag = true;
          } else
            break;
        }
        
        cc.cols(1+nvar,nvar+n) = rep_cpp(tau,n)%weights;
        cc.cols(1+nvar+n,nvar+2*n) = rep_cpp(1-tau,n)%weights;
        gammax.row(n) = rep_cpp(tau,nvar+1)-
          cc.cols(IB)*gammax(arma::span(0,n-1),arma::span::all);
      }
      
      notrun = true;
      for (int iter = 0;iter < maxit; ++iter) {
        rr.row(0) = gammax.row(n);
        for(uint i = 0;i<r2.n_elem;i++) {
          if(r2(i)==0){
            rr(1,i) = 0;
            rr(0,i) = -abs(rr(0,i));
          }else{
            rr(1,i) = weights(r2(i)-(1+nvar+n))-rr(0,i);
          }
        }
        
        int tsep;
        int t_rr;
        double t;
        if(tau_t==0 && iter<nvar+1) {
          t_rr = iter;
          tsep = 0;
          t = arma::as_scalar(r1(t_rr));
        } else {
          if(rr.min() > -tol)
            break;
          int index = rr.index_min();
          t_rr = index / 2;
          tsep = index % 2;
          
          if(tsep==0) {
            t = r1(t_rr);
          } else {
            t = r2(t_rr);
          }
        }
        inner_loop(iter, n, nvar, tsep, t_rr, t,
                   gammax, b, free_var_basic,
                   r1, r2, IB, weights);
        
        notrun = false;
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
      
      arma::vec dh = get_dh(gammaxb, u, v, tau, tau_min, weights);
      
      update_dual_sol(tau_t, tau, tau_min, use_residual,
                      pre, now,
                      u, v,
                      dh, dual_sol);
    }
    
  if(last_flag && notrun&&tau_t>=3){
    est_beta.shed_col(tau_t);
    tau_list.erase(tau_list.begin()+tau_t-2);
    dual_sol.shed_cols(tau_t-1,max_num_tau-1);
    dual_sol.shed_col(tau_t-3);
  }else if(last_flag && notrun&&tau_t>=2){
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
#ifdef DEBUG
    ProfilerStop();
    cout << "finish" << endl;
#endif

}

// rankscore with nonempty Z;
arma::vec quantreg::ranks_cpp(const arma::vec& matY,
                              const arma::mat& matZ,
                              const arma::rowvec& weights,
                              const arma::vec& taurange,
                              uint max_num_tau) const
  {

  // quantreg qr(taurange);
  // print_enter("ranks:");
  uint n_Z = matZ.n_cols;
  uint n = matY.n_elem;
  // std::cout<<"n is "<<n<<std::endl;
  arma::vec ranks = arma::zeros<arma::vec>(n);

  //Initialize estimation output matrix;
  //est_beta: estimation matrix;
  arma::mat est_beta(n_Z+1,1,arma::fill::zeros);
  est_beta = arma::regspace(1,n_Z+1);

  //dual_sol: dual solution matrix a sparse matrix
  arma::sp_mat dual_sol(n,max_num_tau);

  //tau_list:: a list of tau, automatically generated in alg;
  vector<double> tau_list;
  // print(0);
  qr_tau_para_diff_cpp(matZ,matY,weights,taurange,est_beta,dual_sol,
                       tau_list,1e-10,1e-14,100000,max_num_tau);
  // print(1);
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
  // print_leave();

  return ranks;


}

double quantreg::rankscore_cpp(const arma::mat& matX,
                               const arma::mat& matZ,
                               const arma::rowvec& weights,
                               const arma::vec& taurange,
                               const arma::vec& ranks,
                               uint max_num_tau) const
  {
  
  
  double A2 = 1.0/12.0;
  uint n = matX.n_elem;
  
  
  arma::colvec col_one = arma::ones<arma::colvec>(n);
  arma::mat design_Z = join_rows(col_one,matZ);
  for(uint j = 0;j<design_Z.n_cols;j++){
    design_Z.col(j) = design_Z.col(j)%weights.as_col();
  }
  arma::mat design_X = design_Z;
  for(uint j = 0;j<design_X.n_cols;j++){
    design_X.col(j) = design_X.col(j)%matX;
  }
  
  arma::mat matZ_new = design_Z.t()*design_Z;
  arma::mat proj_Z = arma::zeros<arma::mat>(n,n);
  try{
    proj_Z = design_Z*arma::inv(matZ_new)*design_Z.t();
  }
  catch(const std::runtime_error& error){
    proj_Z = design_Z*arma::pinv(matZ_new)*design_Z.t();
  }
  
  
  arma::mat proj_ZX = (design_X.t()-design_X.t()*proj_Z);
  arma::mat sn = proj_ZX*ranks;
  arma::mat Qn = proj_ZX*design_X;
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

// rankscore with empty Z;
arma::vec quantreg::ranks_cpp_marginal(const arma::vec& matY) const
{
  
  // quantreg qr(taurange);
  // print_enter("ranks:");
  uint n = matY.n_elem;
  // std::cout<<"n is "<<n<<std::endl;
  arma::vec ranks = arma::zeros<arma::vec>(n);
  
  // std::cout<<"A2 is "<<A2<<std::endl;
  arma::vec phi = arma::zeros<arma::vec>(n+1);
  double dt = (double) 1/n;
  for(uint i = 1;i<n;i++){
    // std::cout<<"tau_i is "<<tau_list.at(i)<<std::endl;
    double tau_temp = (double) i/n;
    phi(i) = 0.5*tau_temp*(1-tau_temp);
  }
  // std::cout<<"J is "<<J<<std::endl;
  arma::vec dphi = phi(arma::span(1,n))-phi(arma::span(0,n-1));
  
  arma::uvec ranks_Y = arma::sort_index(arma::sort_index(matY));
  
  ranks = dphi/dt;
  ranks = -ranks(ranks_Y);
  // print_leave();
  
  return ranks;
  
  
}



double quantreg::rankscore_cpp_marginal(const arma::mat& matX,
                                        const arma::rowvec& weights,
                                        const arma::vec& ranks) const
  {
  
  double A2 = 1.0/12.0;
  uint n = matX.n_elem;
  
  arma::mat design_X = matX;
  for(uint j = 0;j<design_X.n_cols;j++){
    design_X.col(j) = design_X.col(j)%weights.as_col();
  }
  arma::mat proj_Z = arma::ones<arma::mat>(n,n)/(double) n;

  
  arma::mat sn = (design_X.t()-design_X.t()*proj_Z)*ranks;
  arma::mat Qn = (design_X.t()-design_X.t()*proj_Z)*design_X;
  
  double Tn = 0.0;
  arma::vec col_one_qn = arma::ones<arma::vec>(Qn.n_cols);
  arma::mat Qn_iv = arma::zeros<arma::mat>(Qn.n_cols,Qn.n_cols);
  // std::cout<<"DONE2"<<std::endl;
  try{
    Qn_iv = arma::solve(Qn,arma::diagmat(col_one_qn));
  }
  catch(const std::runtime_error& error){
    Qn_iv = arma::pinv(Qn);
  }
  
  Tn = arma::as_scalar(sn.t()*Qn_iv*sn)/A2;
  
  return Tn;
  
}
