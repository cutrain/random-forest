#include "Forest.h"
#include <typeinfo>


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
  for(size_t i = 0; i != NUM_TREE; i++) {

    std::cout<<"tree "<<i<<std::endl;
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

  while(end==0&&countsp<=ndcount){

    end = split_generalized_MM(matZ,matX,matY,delta,tau,weight_rf,rfsrc_time_surv,
                               time_interest,nodeSample,isLeaf,split_vars,split_values,
                               left_childs,right_childs,countsp,ndcount,quantile_level);

    std::cout<<"end is "<<end<<std::endl;
    if(ndcount >= MAX_NODE - 2) {

      isLeaf.elem(arma::find(left_childs == 0)).ones();
      break;
    }


  }

  std::cout<<"ndcount is "<<ndcount<<std::endl;
  arma::uvec nonEmpty = arma::regspace<arma::uvec>(0, ndcount);
  // std::count<<"nonEmpty is "<<nonEmpty(0)<<std::endl;
  // std::cout<<"split_var is "<<split_vars(nonEmpty).n_elem<<std::endl;
  arma::vec split_values_temp(split_vars(nonEmpty));
  // std::cout<<typeid(split_vars_temp).name()<<std::endl;
  std::cout<<"split_values number is "<<split_values_temp.n_elem<<std::endl;

  std::shared_ptr<Tree> tr(new Tree(left_childs(nonEmpty),
                                    right_childs(nonEmpty),
                                    split_vars(nonEmpty),
                                    split_values(nonEmpty),
                                    isLeaf(nonEmpty)));
  std::cout<<"split_values is "<<tr->get_split_values().n_elem<<std::endl;



  // List tr = List::create(Named("split_vars") = split_vars.subvec(0,ndcount-1),
  //                        Named("split_values") = split_values.subvec(0,ndcount-1),
  //                        Named("nodesample") = nodeSample);

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

  while(varsp==-1&&countsp<=ndcount){
    nd = countsp;
    // std::cout<<"nd is "<<nd<<std::endl;
    arma::vec best_split = find_split_generalized_MM(nd,matZ,matX,matY,delta,tau,weight_rf,
                                                     rfsrc_time_surv,time_interest,nodeSample,
                                                     quantile_level);
    varsp = best_split(0);
    // std::cout<<"varsp is "<<varsp<<std::endl;
    cutsp = best_split(1);
    // cout<<"cutsp is "<<cutsp<<endl;

    if(varsp==-1){

      isLeaf(nd)=1;
      while(countsp <= ndcount) {
        countsp++;
        if(isLeaf(countsp) == 0) break;
      }

    }

  }

  if(varsp!=-1){

    split_vars(nd) = varsp;
    split_values(nd) = cutsp;
    ndc1 = ndcount + 1;
    ndc2 = ndcount + 2;
    left_childs(nd) = ndc1;
    right_childs(nd) = ndc2;
    // std::cout<<"ndc1 is "<<ndc1<<std::endl;


    arma::uvec nodeSamplend = std::move(nodeSample(nd));
    arma::vec xvarspsub = matX(varsp*n_obs + nodeSamplend);
    nodeSample(ndc1) = nodeSamplend(find(xvarspsub <=cutsp));
    nodeSample(ndc2) = nodeSamplend(find(xvarspsub >cutsp));

    // std::cout<<"right node size is "<<nodeSample(ndc1).size()<<std::endl;
    // cout<<"left node size is "<<nodeSample(ndc2).size()<<endl;

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
  // std::cout<< "# in nodes "<<n_obs<<std::endl;
  uint n_Z = nodeSampleZ.n_cols;
  uint n_X = nodeSampleX.n_cols;
  arma::uvec spSet = arma::shuffle( arma::regspace<arma::uvec>(0,n_X-1) );
  // cout<<"var 1 is in splitting "<<find(spSet==0)<<endl;

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

    // std::cout<<"beta0 is "<<beta(0)<<std::endl;
    // cout<<"beta1 is "<<beta(1)<<endl;

    if(iteration>5000){

      vecsp(0) = -1.0;
      vecsp(1) = 0.0;

    }else{

      arma::mat tau_new = tau+(1-tau)*G_mat;
      // cout << "tau_new number of elem is "<< tau_new.n_rows<<endl;
      arma::colvec indict = arma::zeros<arma::colvec>(n_obs);
      indict.elem(arma::find(r_vec<=0)).ones();
      // cout<<"sum ind is "<<sum(indict)<<endl;

      arma::mat gradient = arma::zeros<arma::mat>(n_obs,n_Z);
      for(uint i = 0;i<n_Z;i++){

        gradient.col(i) = nodeSampleZ.col(i)%(tau_new-indict);

      }
      // std::cout<<"graident 0 is "<<mean(gradient.col(0))<<std::endl;

      arma::mat general_cri(quantile_level.n_elem,n_X);
      general_cri.fill(0.0);
      arma::mat general_cri_cutsp(quantile_level.n_elem,n_X);
      general_cri_cutsp.fill(0.0);
      uint index_max = 0;
      double varsp = 0.0;
      double cutsp = 0.0;
      uint n_quantile = 0;

      for(auto i :spSet.head(MTRY)){
        // std::cout<<"i is "<<i<<std::endl;
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

        // cout << "qunatile_elem " << quantile_x.n_elem << endl;

        for(uint j = 1;j < n_quantile;j++){

          // std::cout<<"j is "<<j<<std::endl;
          arma::uvec left_node = find(nodeSampleX.col(i)<=quantile_x(j));
          arma::uvec right_node = find(nodeSampleX.col(i)>quantile_x(j));
          arma::mat mean_left = mean(gradient.rows(left_node),0);
          arma::mat mean_right = mean(gradient.rows(right_node),0);

          arma::mat diff_nodes = mean_left-mean_right;
          // std::cout << diff_nodes.n_cols<<std::endl;
          general_cri(j,i) = as_scalar(diff_nodes*diff_nodes.t())*left_node.n_elem*right_node.n_elem/(n_obs*n_obs);
          // cout <<"left node number is "<< left_node.n_elem<<endl;
          general_cri_cutsp(j,i) = quantile_x(j);
          // cout << "cutsp is "<< quantile_x(j)<<endl;
        }

      }

      double cri_first = general_cri(1);
      bool unique_cri = all(vectorise(general_cri)==cri_first);
      // cout<<unique_cri<<endl;

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
  std::cout<<"n is "<<n<<std::endl;

  arma::mat df_mat = nodematZ;
  arma::mat df_mat_trans = df_mat.t();
  arma::mat matY_trans = nodematY.t();

  // Calculation of epsilon
  double tn = toler/n;
  double e0 = -tn/log(tn);
  double eps = (e0-tn)/(1+log(e0));
  // cout<<"eps is "<<eps<<endl;

  bool cond = true;

  beta = beta_init;
  arma::vec beta_prev = beta;
  arma::vec diff = abs(beta-beta_prev);

  arma::mat matY_fit = df_mat*beta_init;
  arma::vec A_entries = arma::zeros<arma::vec>(n);
  arma::vec B_mat = arma::zeros<arma::vec>(n);

  while(cond){
    // cout << "iteration is "<< iteration<<endl;
    beta_prev = beta;

    matY_fit = df_mat*beta;
    r_vec = nodematY-matY_fit;
    A_entries = 1/(eps+arma::abs(r_vec))%nodeweight_rf/2;
    // cout<<"A_entries is"<<A_entries.n_rows<<endl;

    B_mat = (tau-0.5)*nodeweight_rf;


    G_mat = arma::zeros<arma::vec>(n);
    for (arma::uword i = 0;i<n;i++){
      // std::cout<<"i is "<<i<<std::endl;

      G_mat(i) = nodeweight_rf(i)*G_hat_rfsrc_cpp(arma::as_scalar(matY_fit(i)),
            noderfsrc_time_surv,nodetime_interest,i);

    }
    // std::cout<<"G_mat is" <<G_mat.n_rows<<std::endl;


    arma::mat matZ_proj = ADiag(df_mat_trans,A_entries)*df_mat;
    // cout<<"x_proj is "<<x_proj.n_rows<<endl;
    arma::mat matY_new = ADiag(matY_trans,A_entries).t()+B_mat+(1-tau)*G_mat;
    try{

      beta = solve(matZ_proj,df_mat.t()*matY_new);
      // inv(matZ_proj)*df_mat.t()*matY_new;

    }
    catch(const std::runtime_error& error){

      beta = pinv(matZ_proj)*df_mat.t()*matY_new;

    }

    // std::cout<<"beta0 is "<<beta(0)<<std::endl;

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
    // cout<<"i is "<<i<<endl;

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
  //cout << i <<endl;
  double G_hat = 1-arma::as_scalar(time_rfsrc(index));
  // cout << G_hat<< endl;
  return G_hat;

}

arma::mat Forest::ADiag(arma::mat& A,
                        arma::vec& diag_entries) const{

  uint n = A.n_rows;
  // cout<<"n is "<<n<<endl;
  uint p = A.n_cols;
  // cout<<"p is "<<p<<endl;



  arma::mat ADiagmat = arma::zeros<arma::mat>(n,p);

  for(uint i = 0;i<n;i++){
    // cout<<"i is "<<i<<endl;
    for(uint j = 0;j<p;j++){
      // cout<<"j is "<<j<<endl;
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



