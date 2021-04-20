#ifndef ForestPrediction_h
#define ForestPrediction_h

#include "Tree.h"
#include "Forest.h"
#include "common.h"

class ForestPrediction {
public:

  ForestPrediction(const arma::mat& matXnew,
                   const arma::mat& matX0,
                   const arma::mat& matY0,
                   const arma::mat& matZ0,
                   const arma::umat& ids,
                   const double tau,
                   const std::vector<std::shared_ptr<Tree> >& trees) {
    // print_enter("prediction:");
    uint n_X = matXnew.n_rows;
    uint n_X0 = matX0.n_rows;
    uint n_Z = matZ0.n_cols;
    uint NUM_TREE = trees.size();
    arma::umat ndy = arma::zeros<arma::umat>(NUM_TREE, n_X0);
    arma::umat ndy_test = arma::zeros<arma::umat>(NUM_TREE, n_X);
    arma::field<arma::uvec> ndsz(NUM_TREE);
    arma::field<arma::uvec> tnd3B(NUM_TREE);
    std::vector<std::shared_ptr<Tree> >::const_iterator it;
    // print(-1);
    int i = 0;
    // std::cout<<"DONE1"<<std::endl;
    for(it = trees.begin(); it != trees.end(); it++, i++) {
      // take the i th tree
      // print(i);
      arma::uvec idi = ids.col(i);
      std::shared_ptr<Tree> tt = *it;
      arma::vec vars =  *tt->get_split_vars();
      arma::uvec lcs = *tt->get_left_childs();
      arma::uvec rcs = *tt->get_right_childs();
      arma::vec values = *tt->get_split_values();
      arma::uvec il = *tt->get_isLeaf();
      int nNd = arma::accu(il);
      arma::uvec ndszi = arma::zeros<arma::uvec>(nNd);

      // tnd: leaf number;
      arma::uvec tnd = arma::zeros<arma::uvec>(il.n_elem);
      arma::uvec tnd2 = (arma::find(il == 1));
      tnd.elem( tnd2 ) = arma::regspace<arma::uvec>(0, nNd-1);
      // std::cout<<"DONE2"<<std::endl;
      for(size_t j = 0; j != n_X0; j++) {
        arma::rowvec matX0j = matX0.row( j );
        int isl = 0;
        int varsp = 0;
        double cutsp = 0;
        size_t k = 0;
        while(isl == 0) {
          varsp = vars(k);
          // std::cout<<"varsp is "<<varsp<<std::endl;
          cutsp = values(k);
          // std::cout<<"cutsp is "<<cutsp<<std::endl;
          if(matX0j(varsp) > cutsp) {
            k = rcs(k);
          } else {
            k = lcs(k);
          }
          isl = il(k);
        }
        // std::cout<<"DONE3"<<std::endl;
        ndy(i,j) = k;
        ndszi(tnd(k))++;
      }

      ndsz(i) = ndszi;
      tnd3B(i) = tnd;

      for(size_t j = 0; j != n_X; j++) {
        arma::rowvec matXnewj = matXnew.row( j );
        int isl = 0;
        int varsp = 0;
        double cutsp = 0;
        size_t k = 0;
        while(isl == 0) {
          varsp = vars(k);
          // std::cout<<"varsp is "<<varsp<<std::endl;
          cutsp = values(k);
          // std::cout<<"cutsp is "<<cutsp<<std::endl;
          if(matXnewj(varsp) > cutsp) {
            k = rcs(k);
          } else {
            k = lcs(k);
          }
          isl = il(k);
        }
        // std::cout<<"DONE3"<<std::endl;
        ndy_test(i,j) = k;
      }

    }

    // print(-2);
    arma::mat rfweights = arma::zeros<arma::mat>(n_X,n_X0);
    for(uint it = 0;it<NUM_TREE;it++){
      // print(it);
      arma::uvec ndsz_temp = ndsz(it);
      // std::cout<<"DONE1"<<std::endl;
      arma::urowvec nodelabel_test = ndy_test.row(it);
      arma::urowvec nodelabel = ndy.row(it);
      // std::cout<<"nodelable is "<<nodelabel.n_elem<<std::endl;
      arma::uvec nodemap = tnd3B(it);
      // print(-3);
      // print_leave();
      for(uint i = 0;i<n_X;i++){
        // std::cout<<"i is "<<i<<std::endl;
        uint nl = nodelabel_test(i);
        uint nd = nodemap(nl);
        // std::cout<<"nd is "<<nd<<std::endl;
        arma::uvec id_temp = ids.col(it);
        // print(i);
        for(uint l:id_temp){
          // std::cout<<"l is "<<l<<std::endl;
          if(nodelabel(l)==nl){
            rfweights(l*n_X+i) = rfweights(l*n_X+i)+1/(double) ndsz_temp(nd);
          }

        }
      }
    }
    print(-3);
    quantreg qr(tau);
    arma::mat est_beta = arma::zeros<arma::mat>(n_X,n_Z+1);
    arma::vec residual = arma::zeros<arma::vec>(n_X0);
    for(uint i = 0; i < n_X;i++){
      arma::rowvec weight_i = rfweights.row((i));
      arma::vec est_beta_i = est_beta.row(i).as_col();
      qr.qr_tau_para_diff_fix_cpp(matZ0,matY0,rfweights.row(i),
                                  tau,est_beta_i,
                                  residual,1e-14,100000);
      est_beta.row(i) = est_beta_i.as_row();
    }


    this->nodeLabelB = ndy;
    this->nodeLabel_test = ndy_test;
    this->nodeSizeB = ndsz;
    this->tndB = tnd3B;
    this->rfweights = rfweights;
    this->estimate = est_beta;
  }
  
  ForestPrediction(const arma::mat& matXnew,
                   const arma::mat& matX0,
                   const arma::mat& matY0,
                   const arma::mat& matZ0,
                   const arma::umat& ids,
                   const arma::vec taurange,
                   uint max_num_tau,
                   const std::vector<std::shared_ptr<Tree> >& trees) {
    print_enter("prediction:");
    uint n_X = matXnew.n_rows;
    uint n_X0 = matX0.n_rows;
    uint n_Z = matZ0.n_cols;
    uint NUM_TREE = trees.size();
    arma::umat ndy = arma::zeros<arma::umat>(NUM_TREE, n_X0);
    arma::umat ndy_test = arma::zeros<arma::umat>(NUM_TREE, n_X);
    arma::field<arma::uvec> ndsz(NUM_TREE);
    arma::field<arma::uvec> tnd3B(NUM_TREE);
    std::vector<std::shared_ptr<Tree> >::const_iterator it;
    // print(-1);
    int i = 0;
    // std::cout<<"DONE1"<<std::endl;
    for(it = trees.begin(); it != trees.end(); it++, i++) {
      // take the i th tree
      // print(i);
      arma::uvec idi = ids.col(i);
      std::shared_ptr<Tree> tt = *it;
      arma::vec vars =  *tt->get_split_vars();
      arma::uvec lcs = *tt->get_left_childs();
      arma::uvec rcs = *tt->get_right_childs();
      arma::vec values = *tt->get_split_values();
      arma::uvec il = *tt->get_isLeaf();
      int nNd = arma::accu(il);
      arma::uvec ndszi = arma::zeros<arma::uvec>(nNd);
      
      // tnd: leaf number;
      arma::uvec tnd = arma::zeros<arma::uvec>(il.n_elem);
      arma::uvec tnd2 = (arma::find(il == 1));
      tnd.elem( tnd2 ) = arma::regspace<arma::uvec>(0, nNd-1);
      // std::cout<<"DONE2"<<std::endl;
      for(size_t j = 0; j != n_X0; j++) {
        arma::rowvec matX0j = matX0.row( j );
        int isl = 0;
        int varsp = 0;
        double cutsp = 0;
        size_t k = 0;
        while(isl == 0) {
          varsp = vars(k);
          // std::cout<<"varsp is "<<varsp<<std::endl;
          cutsp = values(k);
          // std::cout<<"cutsp is "<<cutsp<<std::endl;
          if(matX0j(varsp) > cutsp) {
            k = rcs(k);
          } else {
            k = lcs(k);
          }
          isl = il(k);
        }
        // std::cout<<"DONE3"<<std::endl;
        ndy(i,j) = k;
        ndszi(tnd(k))++;
      }
      
      ndsz(i) = ndszi;
      tnd3B(i) = tnd;
      
      for(size_t j = 0; j != n_X; j++) {
        arma::rowvec matXnewj = matXnew.row( j );
        int isl = 0;
        int varsp = 0;
        double cutsp = 0;
        size_t k = 0;
        while(isl == 0) {
          varsp = vars(k);
          // std::cout<<"varsp is "<<varsp<<std::endl;
          cutsp = values(k);
          // std::cout<<"cutsp is "<<cutsp<<std::endl;
          if(matXnewj(varsp) > cutsp) {
            k = rcs(k);
          } else {
            k = lcs(k);
          }
          isl = il(k);
        }
        // std::cout<<"DONE3"<<std::endl;
        ndy_test(i,j) = k;
      }
      
    }
    
    print(-2);
    arma::mat rfweights = arma::zeros<arma::mat>(n_X,n_X0);
    for(uint it = 0;it<NUM_TREE;it++){
      // print(it);
      arma::uvec ndsz_temp = ndsz(it);
      // std::cout<<"DONE1"<<std::endl;
      arma::urowvec nodelabel_test = ndy_test.row(it);
      arma::urowvec nodelabel = ndy.row(it);
      // std::cout<<"nodelable is "<<nodelabel.n_elem<<std::endl;
      arma::uvec nodemap = tnd3B(it);
      // print(-3);
      for(uint i = 0;i<n_X;i++){
        // std::cout<<"i is "<<i<<std::endl;
        uint nl = nodelabel_test(i);
        uint nd = nodemap(nl);
        // std::cout<<"nd is "<<nd<<std::endl;
        arma::uvec id_temp = ids.col(it);
        // print(i);
        for(uint l:id_temp){
          // std::cout<<"l is "<<l<<std::endl;
          if(nodelabel(l)==nl){
            rfweights(l*n_X+i) = rfweights(l*n_X+i)+1/(double) ndsz_temp(nd);
          }
          
        }
      }
    }
    print(-3);
    quantreg qr(taurange);
    arma::field<arma::mat> est_beta(n_X);
    arma::vec residual = arma::zeros<arma::vec>(n_X0);
    arma::vec tau_pred = arma::regspace<arma::vec>(1,99)/100;
    for(uint i = 0; i < n_X;i++){
      print(i);
      vector<double> tau_list;
      arma::mat est_beta_i  = arma::zeros<arma::mat>(tau_pred.n_elem,
                                                     n_Z+1);
      for(uint t = 0;t<tau_pred.n_elem;t++){
        arma::vec est_beta_i_temp = arma::regspace<arma::vec>(1,n_Z+1);
        qr.qr_tau_para_diff_fix_cpp(matZ0,matY0,rfweights.row(i),
                                    tau_pred(t),est_beta_i_temp,
                                    residual,1e-14,100000);
        est_beta_i.row(t) = est_beta_i_temp.as_row();
      }
      est_beta(i) = est_beta_i;
    }
    print_leave();
    
    this->nodeLabelB = ndy;
    this->nodeLabel_test = ndy_test;
    this->nodeSizeB = ndsz;
    this->tndB = tnd3B;
    this->rfweights = rfweights;
    this->estimate_process = est_beta;
  }
  
  ForestPrediction(const arma::mat& matXnew,
                   const arma::mat& matX0,
                   const arma::mat& matY0,
                   const arma::umat& ids,
                   const arma::vec taurange,
                   uint max_num_tau,
                   const std::vector<std::shared_ptr<Tree> >& trees) {
    print_enter("prediction:");
    uint n_X = matXnew.n_rows;
    uint n_X0 = matX0.n_rows;
    uint NUM_TREE = trees.size();
    arma::umat ndy = arma::zeros<arma::umat>(NUM_TREE, n_X0);
    arma::umat ndy_test = arma::zeros<arma::umat>(NUM_TREE, n_X);
    arma::field<arma::uvec> ndsz(NUM_TREE);
    arma::field<arma::uvec> tnd3B(NUM_TREE);
    std::vector<std::shared_ptr<Tree> >::const_iterator it;
    // print(-1);
    int i = 0;
    // std::cout<<"DONE1"<<std::endl;
    for(it = trees.begin(); it != trees.end(); it++, i++) {
      // take the i th tree
      // print(i);
      arma::uvec idi = ids.col(i);
      std::shared_ptr<Tree> tt = *it;
      arma::vec vars =  *tt->get_split_vars();
      arma::uvec lcs = *tt->get_left_childs();
      arma::uvec rcs = *tt->get_right_childs();
      arma::vec values = *tt->get_split_values();
      arma::uvec il = *tt->get_isLeaf();
      int nNd = arma::accu(il);
      arma::uvec ndszi = arma::zeros<arma::uvec>(nNd);
      
      // tnd: leaf number;
      arma::uvec tnd = arma::zeros<arma::uvec>(il.n_elem);
      arma::uvec tnd2 = (arma::find(il == 1));
      tnd.elem( tnd2 ) = arma::regspace<arma::uvec>(0, nNd-1);
      // std::cout<<"DONE2"<<std::endl;
      for(size_t j = 0; j != n_X0; j++) {
        arma::rowvec matX0j = matX0.row( j );
        int isl = 0;
        int varsp = 0;
        double cutsp = 0;
        size_t k = 0;
        while(isl == 0) {
          varsp = vars(k);
          // std::cout<<"varsp is "<<varsp<<std::endl;
          cutsp = values(k);
          // std::cout<<"cutsp is "<<cutsp<<std::endl;
          if(matX0j(varsp) > cutsp) {
            k = rcs(k);
          } else {
            k = lcs(k);
          }
          isl = il(k);
        }
        // std::cout<<"DONE3"<<std::endl;
        ndy(i,j) = k;
        ndszi(tnd(k))++;
      }
      
      ndsz(i) = ndszi;
      tnd3B(i) = tnd;
      
      for(size_t j = 0; j != n_X; j++) {
        arma::rowvec matXnewj = matXnew.row( j );
        int isl = 0;
        int varsp = 0;
        double cutsp = 0;
        size_t k = 0;
        while(isl == 0) {
          varsp = vars(k);
          // std::cout<<"varsp is "<<varsp<<std::endl;
          cutsp = values(k);
          // std::cout<<"cutsp is "<<cutsp<<std::endl;
          if(matXnewj(varsp) > cutsp) {
            k = rcs(k);
          } else {
            k = lcs(k);
          }
          isl = il(k);
        }
        // std::cout<<"DONE3"<<std::endl;
        ndy_test(i,j) = k;
      }
      
    }
    
    print(-2);
    arma::mat rfweights = arma::zeros<arma::mat>(n_X,n_X0);
    for(uint it = 0;it<NUM_TREE;it++){
      // print(it);
      arma::uvec ndsz_temp = ndsz(it);
      // std::cout<<"DONE1"<<std::endl;
      arma::urowvec nodelabel_test = ndy_test.row(it);
      arma::urowvec nodelabel = ndy.row(it);
      // std::cout<<"nodelable is "<<nodelabel.n_elem<<std::endl;
      arma::uvec nodemap = tnd3B(it);
      // print(-3);
      for(uint i = 0;i<n_X;i++){
        // std::cout<<"i is "<<i<<std::endl;
        uint nl = nodelabel_test(i);
        uint nd = nodemap(nl);
        // std::cout<<"nd is "<<nd<<std::endl;
        arma::uvec id_temp = ids.col(it);
        // print(i);
        for(uint l:id_temp){
          // std::cout<<"l is "<<l<<std::endl;
          if(nodelabel(l)==nl){
            rfweights(l*n_X+i) = rfweights(l*n_X+i)+1/(double) ndsz_temp(nd);
          }
          
        }
      }
    }
    print(-3);
    print_leave();
    
    this->nodeLabelB = ndy;
    this->nodeLabel_test = ndy_test;
    this->nodeSizeB = ndsz;
    this->tndB = tnd3B;
    this->rfweights = rfweights;
  }


  const arma::field<arma::uvec>& get_nodeSize() const
  {
    return nodeSizeB;
  };
  const arma::umat& get_nodeLabel() const
  {
    return nodeLabelB;
  };
  const arma::umat& get_nodeLabel_test() const
  {
    return nodeLabel_test;
  };
  const arma::field<arma::uvec>& get_nodeMap() const
  {
    return tndB;
  };
  const arma::mat& get_weights() const
  {
    return rfweights;
  };
  const arma::mat& get_estimate() const
  {
    return estimate;
  };
  const arma::field<arma::mat>& get_estimate_process() const
  {
    return estimate_process;
  };
private:
  arma::field<arma::uvec> nodeSizeB;

  arma::umat nodeLabelB; // Each column is for one tree
  
  arma::umat nodeLabel_test; 

  // vector that defines a map between node number and its location in matrix nodeSize
  arma::field<arma::uvec> tndB;

  //Random Forest Weights;
  arma::mat rfweights;

  //best estimation for Xnew;
  arma::mat estimate;
  
  arma::field<arma::mat> estimate_process;

};

#endif /* ForestPrediction_h */
