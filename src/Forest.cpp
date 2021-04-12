#include "common.h"
#include "Forest.h"
#include <typeinfo>
#include <memory>
using std::make_shared;
using std::shared_ptr;

const double eps=1e-14;

int Forest::trainRF(std::vector<std::shared_ptr<Tree> >& trees,
                    const arma::mat& matZ,
                    const arma::mat& matX,
                    const arma::mat& matY,
                    const arma::vec& delta,
                    const double& tau,
                    const arma::vec& weight_rf,
                    const arma::mat& rfsrc_time_surv,
                    const arma::vec& time_interest,
                    const arma::vec& quantile_level,
                    const arma::umat& ids) {
  // int n = matZ.n_rows;
  print_enter("trainRF:");
  print(-1);
  for(size_t i = 0; i != NUM_TREE; i++) {
    // print(i);
    trees.push_back(train_tree(matZ.rows( ids.col(i) ),
                            matX.rows( ids.col(i) ),
                            matY( ids.col(i) ),
                            delta( ids.col(i) ),
                            tau,
                            weight_rf(ids.col(i)),
                            rfsrc_time_surv.rows( ids.col(i) ),
                            time_interest,
                            quantile_level));
  }
  print_leave();
  return 1;
}


std::shared_ptr<Tree> Forest::train_tree(const arma::mat& matZ,
                                         const arma::mat& matX,
                                         const arma::mat& matY,
                                         const arma::vec& delta,
                                         const double& tau,
                                         const arma::vec& weight_rf,
                                         const arma::mat& rfsrc_time_surv,
                                         const arma::vec& time_interest,
                                         const arma::vec& quantile_level) const{
  print_enter("train tree:");
  int n_obs = matX.n_rows;
  // int n_X = matX.n_cols;
  // int n_Z = matZ.n_cols;

  arma::uvec left_childs = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec right_childs = arma::zeros<arma::uvec>(MAX_NODE);
  arma::vec split_vars = arma::zeros<arma::vec>(MAX_NODE);
  arma::vec split_values = arma::zeros<arma::vec>(MAX_NODE);
  arma::uvec isLeaf = arma::zeros<arma::uvec>(MAX_NODE);

  arma::field<arma::uvec> nodeSample(MAX_NODE);
  nodeSample(0) = arma::regspace<arma::uvec>(0, n_obs-1);

  uint ndcount = 0;
  uint countsp = 0;
  uint end = 0;

  print(1);
  while(end==0&&countsp<=ndcount){
    end = split_generalized_MM(matZ,matX,matY,delta,tau,weight_rf,rfsrc_time_surv,
                               time_interest,nodeSample,isLeaf,split_vars,split_values,
                               left_childs,right_childs,countsp,ndcount,quantile_level);
    std::cout<<"end is "<<end<<std::endl;
    cout << "max_node="  <<  MAX_NODE << endl;
    if(ndcount + 2 >= MAX_NODE) {
      isLeaf.elem(arma::find(left_childs == 0)).ones();
      break;
    }
  }
  print(2);

  // std::cout<<"ndcount is "<<ndcount<<std::endl;

  arma::uvec nonEmpty = arma::regspace<arma::uvec>(0, ndcount);
  arma::vec split_values_temp(split_vars(nonEmpty));
  // std::cout<<"split_values number is "<<split_values_temp.n_elem<<std::endl;

  print(3);
  shared_ptr<Tree> tr = make_shared<Tree>(left_childs(nonEmpty),
                                    right_childs(nonEmpty),
                                    split_vars(nonEmpty),
                                    split_values(nonEmpty),
                                    isLeaf(nonEmpty));
  print(4);
  shared_ptr<arma::vec> sv = tr->split_vars;
  cout << "real split vars=" << sv->n_elem << endl;
  shared_ptr<arma::vec> sv2 = tr->get_split_vars();
  cout << "split vars=" << sv2->n_elem << endl;
  cout<<"split_values is "<<tr->get_split_values()->n_elem<<endl;

  // List tr = List::create(Named("split_vars") = split_vars.subvec(0,ndcount-1),
  //                        Named("split_values") = split_values.subvec(0,ndcount-1),
  //                        Named("nodesample") = nodeSample);
  print_leave();
  return tr;
}

uint Forest::split_generalized_MM(const arma::mat& matZ,
                          const arma::mat& matX,
                          const arma::mat& matY,
                          const arma::vec& delta,
                          const double& tau,
                          const arma::vec& weight_rf,
                          const arma::mat& rfsrc_time_surv,
                          const arma::vec& time_interest,
                          arma::field<arma::uvec>& nodeSample,
                          arma::uvec& isLeaf,
                          arma::vec& split_vars,
                          arma::vec& split_values,
                          arma::uvec& left_childs,
                          arma::uvec& right_childs,
                          uint& countsp,
                          uint& ndcount,
                          const arma::vec& quantile_level) const {
  uint end = 0;
  double varsp = -1;
  double cutsp = 0;
  uint nd = countsp;
  uint n_obs = matX.n_rows;
  uint ndc1 = 0;
  uint ndc2 = 0;

  while(fabs(varsp+1)<eps && countsp<=ndcount){
    nd = countsp;
    arma::vec best_split = find_split_generalized_MM(nd,matZ,matX,matY,delta,tau,weight_rf,
                                                     rfsrc_time_surv,time_interest,nodeSample,
                                                     quantile_level);
    varsp = best_split(0);
    cutsp = best_split(1);
    if (fabs(varsp+1) < eps) {
      isLeaf(nd) = 1;
      while (countsp <= ndcount) {
        countsp++;
        if (isLeaf(countsp) == 0)
          break;
      }
    }
  }
  if (fabs(varsp+1) > eps){
    split_vars(nd) = varsp;
    split_values(nd) = cutsp;
    ndc1 = ndcount + 1;
    ndc2 = ndcount + 2;
    left_childs(nd) = ndc1;
    right_childs(nd) = ndc2;

    arma::uvec nodeSamplend = std::move(nodeSample(nd));
    arma::vec xvarspsub = matX(varsp*n_obs + nodeSamplend);
    nodeSample(ndc1) = nodeSamplend(find(xvarspsub <=cutsp));
    nodeSample(ndc2) = nodeSamplend(find(xvarspsub >cutsp));

    if(nodeSample(ndc1).size() < MIN_SPLIT1) {
      isLeaf(ndc1) = 1;
    } else {
      isLeaf(ndc1) = 0;
    }
    if(nodeSample(ndc2).size() < MIN_SPLIT1) {
      isLeaf(ndc2) = 1;
    } else {
      isLeaf(ndc2) = 0;
    }
    ndcount += 2;
    while(countsp <= ndcount) {
      countsp++;
      if(isLeaf(countsp) == 0) break;
    }
  }else{
    end = 1;
  }

  std::cout<< "ndcount is "<<ndcount<<std::endl;
  std::cout<< "countsp is "<<countsp<<std::endl;
  return end;
}

arma::vec Forest::find_split_generalized_MM(uint nd,
                                            const arma::mat& matZ,
                                            const arma::mat& matX,
                                            const arma::mat& matY,
                                            const arma::vec& delta,
                                            const double& tau,
                                            const arma::vec& weight_rf,
                                            const arma::mat& rfsrc_time_surv,
                                            const arma::vec& time_interest,
                                            const arma::field<arma::uvec>& nodeSample,
                                            const arma::vec& quantile_level) const {
  arma::mat nodeSampleX = matX.rows(nodeSample(nd));
  arma::mat nodeSampleY = matY.rows(nodeSample(nd));
  arma::mat nodeSampleZ = matZ.rows(nodeSample(nd));
  arma::mat nodedelta = delta(nodeSample(nd));
  arma::mat node_rfsrc_time_surv = rfsrc_time_surv.rows(nodeSample(nd));
  arma::vec node_weight_rf = weight_rf(nodeSample(nd));
  uint n_obs = nodeSampleX.n_rows;
  uint n_Z = nodeSampleZ.n_cols;
  uint n_X = nodeSampleX.n_cols;
  arma::uvec spSet = arma::shuffle( arma::regspace<arma::uvec>(0,n_X-1) );

  arma::vec G_mat = arma::zeros<arma::vec>(n_obs);
  arma::vec beta = arma::zeros<arma::vec>(n_Z);
  arma::vec r_vec = arma::zeros<arma::vec>(n_obs);

  arma::vec beta_init = arma::ones<arma::vec>(n_Z);
  uint iteration = 0;
  arma::vec vecsp = arma::zeros<arma::vec>(2);
  if(arma::rank(nodeSampleZ)<n_Z){
    vecsp(0) = -1.0;
    vecsp(1) = 0.0;
  }else{
    MMLQR_cens_rfsrc_cpp(nodeSampleZ,nodeSampleY,nodedelta,tau,node_weight_rf,
                         node_rfsrc_time_surv,time_interest,beta_init,
                         G_mat,beta,r_vec,iteration,1e-10,5000);
    if(iteration>5000){
      vecsp(0) = -1.0;
      vecsp(1) = 0.0;
    }else{
      arma::mat tau_new = tau+(1-tau)*G_mat;
      arma::colvec indict = arma::zeros<arma::colvec>(n_obs);
      indict.elem(arma::find(r_vec<=0)).ones();

      arma::mat gradient = arma::zeros<arma::mat>(n_obs,n_Z);
      for(uint i = 0;i<n_Z;i++){
        gradient.col(i) = nodeSampleZ.col(i)%(tau_new-indict);
      }

      arma::mat general_cri(quantile_level.n_elem,n_X);
      general_cri.fill(0.0);
      arma::mat general_cri_cutsp(quantile_level.n_elem,n_X);
      general_cri_cutsp.fill(0.0);
      uint index_max = 0;
      double varsp = 0.0;
      double cutsp = 0.0;
      uint n_quantile = 0;

      for(auto i :spSet.head(MTRY)){
        n_quantile = quantile_level.n_elem;
        arma::vec quantile_x = rep_cpp(INFINITY,n_quantile);
        if(n_obs<quantile_level.n_elem)  {
          quantile_x.subvec(0,(n_obs-1)) = sort(nodeSampleX.col(i));
          n_quantile = n_obs-2;
        }
        else{
          quantile_x = quantile(nodeSampleX.col(i),quantile_level);
          n_quantile = quantile_x.n_elem-1;
        }

        for(uint j = 1;j < n_quantile;j++){
          arma::uvec left_node = find(nodeSampleX.col(i)<=quantile_x(j));
          arma::uvec right_node = find(nodeSampleX.col(i)>quantile_x(j));
          arma::mat mean_left = mean(gradient.rows(left_node),0);
          arma::mat mean_right = mean(gradient.rows(right_node),0);

          arma::mat diff_nodes = mean_left-mean_right;
          general_cri(j,i) = as_scalar(diff_nodes*diff_nodes.t())*left_node.n_elem*right_node.n_elem/(n_obs*n_obs);
          general_cri_cutsp(j,i) = quantile_x(j);
        }
      }

      double cri_first = general_cri(1);
      bool unique_cri = all(vectorise(general_cri)==cri_first);

      if(unique_cri==true){
        vecsp(0) = -1.0;
        vecsp(1) = 0.0;
      }else{
        index_max = general_cri.index_max();
        // cout<<"max cri is "<< general_cri.max()<<endl;
        // cout<<"index_max is"<< index_max<<endl;
        varsp = std::floor(index_max/(double) quantile_level.n_elem);
        cutsp = general_cri_cutsp(index_max);
        // std::cout<<"cutsp is"<<cutsp<<std::endl;
        vecsp(0) = varsp;
        vecsp(1) = cutsp;
      }
    }
  }
  return vecsp;
  // return List::create(Named("general_cri") = general_cri,
  //                     Named("index") = general_cri_index);
  // Named("gradient") = gradient,
  // Named("G_mat") = G_mat,
  // Named("tau_new") = tau_new,
  // Named("residual") = r_vec);
}

void Forest::MMLQR_cens_rfsrc_cpp(const arma::mat& nodematZ,
                                  const arma::mat& nodematY,
                                  const arma::vec& nodedelta,
                                  const double& tau,
                                  const arma::vec& nodeweight_rf,
                                  const arma::mat& noderfsrc_time_surv,
                                  const arma::vec& nodetime_interest,
                                  arma::vec beta_init,
                                  arma::vec& G_mat,
                                  arma::vec& beta,
                                  arma::vec& r_vec,
                                  uint& iteration,
                                  double toler,
                                  uint maxit) const{
  iteration = 0;
  uint n = nodematY.n_rows;

  arma::mat df_mat = nodematZ;
  arma::mat df_mat_trans = df_mat.t();
  arma::mat matY_trans = nodematY.t();

  // Calculation of epsilon
  double tn = toler/n;
  double e0 = -tn/log(tn);
  double eps = (e0-tn)/(1+log(e0));

  bool cond = true;

  beta = beta_init;
  arma::vec beta_prev = beta;
  arma::vec diff = abs(beta-beta_prev);

  arma::mat matY_fit = df_mat*beta_init;
  arma::vec A_entries = arma::zeros<arma::vec>(n);
  arma::vec B_mat = arma::zeros<arma::vec>(n);

  while(cond){
    beta_prev = beta;

    matY_fit = df_mat*beta;
    r_vec = nodematY-matY_fit;
    A_entries = 1/(eps+arma::abs(r_vec))%nodeweight_rf/2;
    B_mat = (tau-0.5)*nodeweight_rf;

    G_mat = arma::zeros<arma::vec>(n);
    for (arma::uword i = 0;i<n;i++){
      G_mat(i) = nodeweight_rf(i)*G_hat_rfsrc_cpp(arma::as_scalar(matY_fit(i)),
            noderfsrc_time_surv,nodetime_interest,i);
    }

    arma::mat matZ_proj = ADiag(df_mat_trans,A_entries)*df_mat;
    arma::mat matY_new = ADiag(matY_trans,A_entries).t()+B_mat+(1-tau)*G_mat;
    try{
      beta = solve(matZ_proj,df_mat.t()*matY_new);
      // inv(matZ_proj)*df_mat.t()*matY_new;
    }
    catch(const std::runtime_error& error){
      beta = pinv(matZ_proj)*df_mat.t()*matY_new;
    }

    diff = abs(beta-beta_prev);
    cond = diff.max() > toler;

    iteration++;

    if(iteration>maxit){
      std::cout << "WARNING: Algorithm did not converge"<<std::endl;
      break;
    }
  }

  // std::cout<<"iteration is "<<iteration<<std::endl;

  matY_fit = df_mat*beta;
  r_vec = nodematY-matY_fit;
  G_mat = arma::zeros<arma::vec>(n);
  for (arma::uword i = 0;i<n;i++){
    G_mat(i) = nodeweight_rf(i)*G_hat_rfsrc_cpp(arma::as_scalar(matY_fit(i)),
          noderfsrc_time_surv,nodetime_interest,i);
  }
  // return List::create(Named("beta") = beta,
  //                     Named("residual") = r_vec);
}


double Forest::G_hat_rfsrc_cpp(double time,
                               const arma::mat& rfsrc_time_surv,
                               const arma::vec& time_interest,
                               arma::uword i) const {
  arma::rowvec rowone = arma::ones<arma::rowvec>(1);
  arma::rowvec time_rfsrc = arma::join_rows(rowone,rfsrc_time_surv.row(i));
  arma::uword index = arma::sum(time_interest<=time);
  double G_hat = 1-arma::as_scalar(time_rfsrc(index));
  return G_hat;
}

arma::mat Forest::ADiag(arma::mat& A,
                        arma::vec& diag_entries) const{
  uint n = A.n_rows;
  uint p = A.n_cols;

  arma::mat ADiagmat = arma::zeros<arma::mat>(n,p);

  for(uint i = 0;i<n;i++){
    for(uint j = 0;j<p;j++){
      ADiagmat(i,j) = arma::as_scalar(A(i,j)*diag_entries(j));
    }
  }
  return ADiagmat;
}

arma::vec Forest::rep_cpp(double num,uint times) const {
  arma::vec result = arma::zeros<arma::vec>(times);
  for(uint i = 0;i<times;i++){
    result(i) = num;
  }
  return result;
}

split_info Forest::find_split_rankscore(arma::uword nd,
                                        const arma::mat& matZ,
                                        const arma::mat& matX,
                                        const arma::mat& matY,
                                        const arma::vec& taurange,
                                        const arma::field<arma::uvec>& nodeSample,
                                        const arma::vec& quantile_level,
                                        uint max_num_tau) const {

  // print_enter("find split:");
  arma::mat nodeSampleX = matX.rows(nodeSample(nd));
  arma::mat nodeSampleY = matY.rows(nodeSample(nd));
  arma::mat nodeSampleZ = matZ.rows(nodeSample(nd));
  // arma::vec node_weights = weights(nodeSample(nd));
  // print(0);
  uint n_obs = nodeSampleX.n_rows;
  uint n_Z = nodeSampleZ.n_cols;
  // std::cout<<"nvar is "<<n_Z<<std::endl;
  uint n_X = nodeSampleX.n_cols;
  arma::uvec spSet = arma::shuffle( arma::regspace<arma::uvec>(0,n_X-1) );
  arma::rowvec weights = arma::ones<arma::rowvec>(n_obs);

  quantreg qr(taurange);
  // print(1);

  double A2 = 1.0/12.0;
  uint n = nodeSampleY.n_elem;
  arma::vec ranks = arma::zeros<arma::vec>(n);


  uint iteration = 0;
  arma::vec vecsp = arma::zeros<arma::vec>(2);
  split_info sp_info;
  sp_info.status = -1;
  // print(2);
  if(arma::rank(nodeSampleZ)<n_Z){

    // vecsp(0) = -1;
    // vecsp(1) = 0;
    sp_info.status = -1;
    sp_info.varsp = 0;
    sp_info.cutsp = 0;


  }else{

    arma::vec ranks = qr.ranks_cpp(nodeSampleY,nodeSampleZ,weights,taurange,
                                   max_num_tau);

    arma::vec rankscore = arma::zeros<arma::vec>(n_X);
    // print(3);

    uint index_max = 0;
    double max_value = 0.0;
    for(auto i :spSet.head(MTRY)){

      rankscore(i) = qr.rankscore_cpp(nodeSampleX.col(i),nodeSampleZ,weights,
                taurange,ranks,max_num_tau);
      // std::cout<<"rankscore is "<<rankscore(i)<<std::endl;
      if(rankscore(i)>max_value){
        max_value = rankscore(i);
        index_max = i;
      }

    }
    // print(4);

    uint varsp = index_max;
    arma::vec nodeSplitX = nodeSampleX.col(index_max);
    double cutsp = 0.0;
    uint n_quantile = 0;
    double rankscore_split = 0.0;
    double rankscore_split_temp = 0.0;

    n_quantile = quantile_level.n_elem;
    arma::vec quantile_x = arma::zeros<arma::vec>(n_quantile);
    if(n_obs<quantile_level.n_elem)  {
      quantile_x.subvec(0,(n_obs-1)) = arma::sort( nodeSplitX);
      n_quantile = n_obs-2;

    }
    else{

      quantile_x = arma::linspace<arma::vec>(arma::max(nodeSplitX),
                                             arma::min(nodeSplitX),
                                             quantile_x.n_elem);
      n_quantile = quantile_x.n_elem-1;

    }
    // print(5);
    for(uint j = 1;j < n_quantile;j++){

      // std::cout<<"j is "<<j<<std::endl;
      arma::vec matXnode_ind = arma::zeros<arma::vec>(n);

      for(uint i = 0;i<n;i++){

        if(nodeSplitX(i)<=quantile_x(j)){
          matXnode_ind(i) = 1;
        }else{
          matXnode_ind(i) = 0;
        }

      }

      rankscore_split_temp = qr.rankscore_cpp(matXnode_ind,nodeSampleZ,weights,
                                              taurange,ranks,
                                              max_num_tau);
      if(rankscore_split_temp>rankscore_split){

        rankscore_split = rankscore_split_temp;
        cutsp = quantile_x(j);

      }
    }
    // print(6);

    // double cri_first = general_cri(1);
    // bool unique_cri = all(vectorise(general_cri)==cri_first);
    // cout<<unique_cri<<endl;

    // vecsp(0) = varsp;
    // vecsp(1) = cutsp;
    sp_info.status = 1;
    sp_info.varsp = varsp;
    sp_info.cutsp = cutsp;







  }

  return sp_info;

  // return List::create(Named("general_cri") = general_cri,
  //                     Named("index") = general_cri_index);
  // Named("gradient") = gradient,
  // Named("G_mat") = G_mat,
  // Named("tau_new") = tau_new,
  // Named("residual") = r_vec);

}

uint Forest::split_rankscore(const arma::mat& matZ,
                             const arma::mat& matX,
                             const arma::mat& matY,
                             const arma::vec& taurange,
                             arma::field<arma::uvec>& nodeSample,
                             arma::uvec& isLeaf,
                             arma::vec& split_vars,
                             arma::vec& split_values,
                             arma::uvec& left_childs,
                             arma::uvec& right_childs,
                             uint& countsp,
                             uint& ndcount,
                             const arma::vec& quantile_level,
                             uint max_num_tau) const {
  uint end = 0;
  int status = -1;
  uint varsp = 0;
  double cutsp = 0;
  uint nd = countsp;
  uint n_obs = matX.n_rows;
  uint ndc1 = 0;
  uint ndc2 = 0;

  while(status==-1 && countsp<=ndcount){
    nd = countsp;
    split_info best_split = find_split_rankscore(nd,matZ,matX,matY,
                                                 taurange,
                                                 nodeSample,
                                                 quantile_level,
                                                 max_num_tau);
    status = best_split.status;
    // std::cout<<"stauts is "<<status<<std::endl;
    varsp = best_split.varsp;
    cutsp = best_split.cutsp;
    if (status==-1) {
      isLeaf(nd) = 1;
      while (countsp <= ndcount) {
        countsp++;
        if (isLeaf(countsp) == 0)
          break;
      }
    }
  }
  if (status != -1){
    split_vars(nd) = varsp;
    split_values(nd) = cutsp;
    ndc1 = ndcount + 1;
    ndc2 = ndcount + 2;
    left_childs(nd) = ndc1;
    right_childs(nd) = ndc2;

    arma::uvec nodeSamplend = std::move(nodeSample(nd));
    arma::vec xvarspsub = matX(varsp*n_obs + nodeSamplend);
    nodeSample(ndc1) = nodeSamplend(find(xvarspsub <=cutsp));
    nodeSample(ndc2) = nodeSamplend(find(xvarspsub >cutsp));

    if(nodeSample(ndc1).size() < MIN_SPLIT1) {
      isLeaf(ndc1) = 1;
    } else {
      isLeaf(ndc1) = 0;
    }
    if(nodeSample(ndc2).size() < MIN_SPLIT1) {
      isLeaf(ndc2) = 1;
    } else {
      isLeaf(ndc2) = 0;
    }
    ndcount += 2;
    while(countsp <= ndcount) {
      countsp++;
      if(isLeaf(countsp) == 0) break;
    }
  }else{
    end = 1;
  }

  // std::cout<< "ndcount is "<<ndcount<<std::endl;
  // std::cout<< "countsp is "<<countsp<<std::endl;
  return end;
}

std::shared_ptr<Tree> Forest::train_tree(const arma::mat& matZ,
                                         const arma::mat& matX,
                                         const arma::mat& matY,
                                         const arma::vec& taurange,
                                         const arma::vec& quantile_level,
                                         uint max_num_tau) const{
  print_enter("train tree:");
  int n_obs = matX.n_rows;
  // int n_X = matX.n_cols;
  // int n_Z = matZ.n_cols;

  arma::uvec left_childs = arma::zeros<arma::uvec>(MAX_NODE);
  arma::uvec right_childs = arma::zeros<arma::uvec>(MAX_NODE);
  arma::vec split_vars = arma::zeros<arma::vec>(MAX_NODE);
  arma::vec split_values = arma::zeros<arma::vec>(MAX_NODE);
  arma::uvec isLeaf = arma::zeros<arma::uvec>(MAX_NODE);

  arma::field<arma::uvec> nodeSample(MAX_NODE);
  nodeSample(0) = arma::regspace<arma::uvec>(0, n_obs-1);

  uint ndcount = 0;
  uint countsp = 0;
  uint end = 0;

  print(1);
  while(end==0&&countsp<=ndcount){
    end = split_rankscore(matZ,matX,matY,taurange,nodeSample,
                          isLeaf,split_vars,split_values,left_childs,
                          right_childs,countsp,ndcount,quantile_level,
                          max_num_tau);
    // std::cout<<"end is "<<end<<std::endl;
    // cout << "max_node="  <<  MAX_NODE << endl;
    if(ndcount + 2 >= MAX_NODE) {
      isLeaf.elem(arma::find(left_childs == 0)).ones();
      break;
    }
  }
  print(2);

  std::cout<<"ndcount is "<<ndcount<<std::endl;

  arma::uvec nonEmpty = arma::regspace<arma::uvec>(0, ndcount);
  arma::vec split_values_temp(split_vars(nonEmpty));
  std::cout<<"split_values number is "<<split_values_temp.n_elem<<std::endl;

  print(3);
  shared_ptr<Tree> tr = make_shared<Tree>(left_childs(nonEmpty),
                                          right_childs(nonEmpty),
                                          split_vars(nonEmpty),
                                          split_values(nonEmpty),
                                          isLeaf(nonEmpty));
  print(4);
  shared_ptr<arma::vec> sv = tr->split_vars;
  // cout << "real split vars=" << sv->n_elem << endl;
  shared_ptr<arma::vec> sv2 = tr->get_split_vars();
  // cout << "split vars=" << sv2->n_elem << endl;
  // cout<<"split_values is "<<tr->get_split_values()->n_elem<<endl;

  // List tr = List::create(Named("split_vars") = split_vars.subvec(0,ndcount-1),
  //                        Named("split_values") = split_values.subvec(0,ndcount-1),
  //                        Named("nodesample") = nodeSample);
  print_leave();
  return tr;
}

int Forest::trainRF(std::vector<std::shared_ptr<Tree> >& trees,
                    const arma::mat& matZ,
                    const arma::mat& matX,
                    const arma::mat& matY,
                    const arma::vec& taurange,
                    const arma::vec& quantile_level,
                    uint max_num_tau,
                    const arma::umat& ids) {
  // int n = matZ.n_rows;
  print_enter("trainRF:");
  print(-1);
  arma::rowvec weights = arma::ones<arma::rowvec>(matY.n_rows);
  for(size_t i = 0; i != NUM_TREE; i++) {
    print(i);
    trees.push_back(train_tree(matZ.rows( ids.col(i) ),
                               matX.rows( ids.col(i) ),
                               matY( ids.col(i) ),
                               taurange,
                               quantile_level,
                               max_num_tau));
  }
  print_leave();
  return 1;
}
