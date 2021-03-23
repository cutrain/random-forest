#' This is the function to calculate quantile progess using parametric linear programming
#' It will return the primary solution as beta, and the difference of dual solution as diff_dsol
#' It won't return the dual solution directly, since the matrix is too large (n by nlog(n))


#More specific illustration for each parameter you can adjust:
#@param Y A vector of response, size \eqn{n}.
#@param C A vector or matrix of covariates.
#@param taurange The range of tau for quantile process. The default value is (0,1).
#@param tau_min Minimum tau value. The default value is 1e-10 to approach tau = 0.
#@param tol Threshold to check convergence. The default value is 1e-14.
#@param maxit The maximum number of inerations at each tau. Usually, larger sample size requires larger numbers of iteration to converge. The default value is 1e5.
#@param max_number_tau The number of tau to be tested in the quantile process.
#Based on our experience, it is safe to set it as twice as the sample size.
#The default value is 1e5. With the increase of sample size \eqn{n}, this number
#is theoretically \eqn{nlogn}.

#This function will return: the primary solution as beta, and the difference of dual solution as diff_dsol
#It won't return the dual solution directly, since the matrix is too large (n by nlog(n))

# x is an n\times p design matrix (does not include the intercept)
# y is n\times p response vector
# taurange, define an interval, (a, b), default is (0,1)
# tau_min = 1e-10, the smallest fitting tau level if (0, 1)
# max_number_tau The number of tau to be tested in the quantile process.
#return(list(beta = t(estbeta_all),diff_dsol = temp4))
# output: beta, (p+1+1) \times ntau,  first row is the fitted quantile levels,
#            starting from the second row, each row is the covaraite coefficents across quantile levels

# diff_dsol    n

#Y,

#X, Z


#testing Z coefficient =0

#regress Y~ X using this function

#give you Z, intergrated rank test statsitics






#' @import Matrix MASS
#' @export

qr_tau_para_diff <- function(x, y, taurange, tau_min = 1e-10, tol = 1e-14,
                             maxit = 100000, max_num_tau = 1000, bland = F,
                             use_residual = T){
  require('Matrix')
  require('MASS')
  # if (0 %in% taurange){
  #     library(quantreg)
  #     res0 <- rq(y~x,tau=0)
  # }
  # if (1 %in% taurange){
  #     library(quantreg)
  #     res1 <- rq(y~x,tau=1)
  # }
  x = as.matrix(x)
  m <- nrow(x) # subject number
  nvar <- ncol(x) # number of var
  tau <- taurange[1]+tau_min
  # rounds <- length(taulist)
  cc <- c(rep(0, 1+nvar), tau*rep(1, m), (1-tau)*rep(1, m))

  # A matrix, b matrix
  gammax <- cbind(matrix(1, nrow = m, ncol = 1), x)
  gammaxb <- t(gammax)
  b <- c(y, 0)
  gammax[y<0,] <- -gammax[y<0,]
  b[b<0] <- -b[b<0]

  # Ib
  IB <- (y>=0)*((1:m)+1+nvar)+(y<0)*((1:m)+1+nvar+m)

  gammax <- rbind(gammax, -cc[IB] %*% gammax)
  freevarrow <- c(vector(mode = 'logical', m), T) # these variables cannot be non-basic variables
  r1 <- 1:(nvar+1)
  r2 <- vector('numeric', nvar+1)
  rr <- matrix(0, nrow = 2, ncol = 1+nvar) # last row in the table

  estbeta_all <- matrix(0, nrow = nvar+1, ncol = max_num_tau)
  # d_all <- matrix(0, nrow = m, ncol = max_num_tau)
  temp4 <- Matrix(0, nrow = m, ncol = max_num_tau)
  taulist <- vector('numeric', max_num_tau)

  c0 <- c(rep(0, 1+nvar), rep(1, m), rep(-1, m))
  last_flag <- FALSE
  for (rd in 1:max_num_tau){
    # print(rd)
    if (rd>=2){
      # print(tau)
      # g <- gammax[-(m+1),]
      r1j <- 1 - crossprod(c(c0[IB],0), gammax)
      r1j <- c(r1j, 1-r1j)
      r0j <- c(gammax[(1+m),], 1-gammax[(1+m),])
      theta <- -r0j/r1j
      theta[r1j>=0] <- Inf
      # ib <- IB[IB>(1+nvar)]
      # ib <- ib - 1 - nvar

      # theta <- min(theta[-ib])
      theta <- min(theta)
      tau <- tau + theta + tau_min
      if (tau > taurange[2]){
        if (last_flag) {break}
        if ((tau-theta) <= taurange[2]){
          tau <- taurange[2]-tau_min
          last_flag <- TRUE
        }else{
          break
        }
      }

      # tau <- taulist[rd]
      cc[(2+nvar):(1+nvar+m)] <- tau
      cc[(2+nvar+m):(1+nvar+2*m)] <- 1-tau
      gammax[(m+1),] <- tau - crossprod(c(cc[IB],0), gammax)
    }

    j <- 0
    while (j<maxit) {
      # start iteration
      # step 2
      if (rd==1 && j<nvar+1){
        rr[1,] <- gammax[(1+m),]
        rr[2,] <- (1 - rr[1,]) * (r2!=0)
        rr[1, r2==0] <- -abs(rr[1, r2==0])

        # step 3
        # choose t
        t_rr <- j+1
        tsep <- 1
        t <- r1[t_rr]
      }else{
        rr[1,] <- gammax[(1+m),]
        rr[2,] <- (1 - rr[1,]) * (r2!=0)
        rr[1, r2==0] <- -abs(rr[1, r2==0])

        # terminate
        rrl <- min(rr)
        if (rrl>=-tol){
          break
        }

        # step 3
        # choose t
        if (bland){
          if (any(rr[1,]< -tol)){
            tsep <- which(rr[1,]< -tol)
            tmp <- r1[tsep]
            t <- min(tmp)
            t_rr <- tsep[which(tmp==t)]
            tsep <- 1
          }else{
            tsep <- which(rr[2,]< -tol)
            tmp <- r2[tsep]
            t <- min(tmp)
            t_rr <- tsep[which(tmp==t)]
            tsep <- 2
          }
        }else{
          tsep <- which(rr==rrl, T)[1,]
          t_rr <- tsep[2]
          tsep <- tsep[1]
          if (tsep==1){
            t <- r1[t_rr]
          }else{
            t <- r2[t_rr]
          }
        }
      }

      if (r2[t_rr]!=0){
        # choose k
        # step 4
        if (tsep==1){
          yy <- gammax[, t_rr]
        }else{
          yy <- -gammax[, t_rr]
        }

        # step 5
        k <- b/yy
        if (bland){
          keep <- yy>0 & !freevarrow
          k <- which((min(k[keep])==k)&keep)
          if (length(k)!=1){
            tmp <- IB[k]
            k <- k[which(tmp==min(tmp))]
          }
        }else{
          keep <- yy>0 & !freevarrow
          k <- which((min(k[keep])==k)&keep)[1]
        }

        # pivoting step 6'
        if (tsep!=1){
          yy[m+1] <- yy[m+1]+1
        }
      }else{
        yy <- gammax[, t_rr]
        if (yy[m+1]<0){
          # step 5
          k <- b/yy
          if (bland){
            keep <- yy>0 & !freevarrow
            k <- which((min(k[keep])==k)&keep)
            if (length(k)!=1){
              tmp <- IB[k]
              k <- k[which(tmp==min(tmp))]
            }
          }else{
            keep <- yy>0 & !freevarrow
            k <- which((min(k[keep])==k)&keep)[1]
          }
        }else{
          k <- -b/yy
          if (bland){
            keep <- yy<0 & !freevarrow
            k <- which((min(k[keep])==k)&keep)
            if (length(k)!=1){
              tmp <- IB[k]
              k <- k[which(tmp==min(tmp))]
            }
          }else{
            keep <- yy<0 & !freevarrow
            k <- which((min(k[keep])==k)&keep)[1]
          }
        }
        # pivoting step 6'
        freevarrow[k] <- T
      }

      ee <- yy/yy[k]
      ee[k] <- 1 - 1/yy[k]

      if (IB[k]<=(nvar+m+1)){
        gammax[, t_rr] <- 0
        gammax[k,t_rr] <- 1
        r1[t_rr] <- IB[k]
        r2[t_rr] <- IB[k]+m
      }else{
        gammax[, t_rr] <- 0
        gammax[k,t_rr] <- -1
        gammax[(m+1),t_rr] <- 1
        r1[t_rr] <- IB[k]-m
        r2[t_rr] <- IB[k]
      }

      # pivoting step 6
      gammax <- gammax - tcrossprod(ee, gammax[k,])
      b <- b - ee * b[k]
      IB[k] <- t

      j <- j+1
    }

    if (j==maxit){
      warning('May not converge')
    }


    tmp <- IB %in% 1:(nvar+1)
    estimate <- b[tmp][order(IB[tmp])]

    if (length(estimate)!=(1+nvar)){
      estimate <- vector('numeric', nvar+1)
      for (i in 1:(nvar+1)){
        try(estimate[i] <- b[IB==i], silent = T) # some b[IB==i] may not exist, which means this beta is 0.
      }
    }

    #
    u <- (nvar+1+(1:m)) %in% IB
    v <- (nvar+1+m+(1:m)) %in% IB
    xh <- gammaxb[,!(u|v)]
    xbarh <- gammaxb[,(u|v)]
    dbarh <- ((tau-tau_min)*u + (tau-tau_min-1)*v)
    tryCatch(
      dh <- solve(xh,-xbarh %*% dbarh[u|v]), error = function(c){
        dh <<- ginv(xh) %*% -xbarh %*% dbarh[u|v]
      }
    )
    ### skip this part
    # hatd <- dbarh
    # hatd[!(u|v)] <- dh
    # d_all[,rd] <- hatd
    # # gammaxb %*% hatd = 0  x^t %*% d is 0 !
    # #
    ###
    temp1 <- u
    if (!use_residual){
      dh[dh==(tau-tau_min)] <- 1
      dh[dh==(tau-tau_min-1)] <- 0
    }
    temp1[!(u|v)] <- dh
    if (rd!=1){
      temp4[, rd-1] <- temp1 - temp
    }
    temp <- temp1

    estbeta_all[, rd] <- estimate
    taulist[rd] <- tau
  }
  if (!last_flag){
    estbeta_all <- t(rbind(taulist-tau_min,estbeta_all))[1:(rd-1),]
    # d_all <- d_all[,1:(rd-1)]
    temp4 <- temp4[,1:(rd-2)]
  }else if (last_flag && j==0){
    estbeta_all <- t(rbind(taulist-tau_min,estbeta_all))[c(1:(rd-3),(rd-1)),]
    # d_all <- d_all[,c(1:(rd-3),(rd-1))]
    temp4 <- temp4[,c(1:(rd-4),(rd-2))]
  }else {
    estbeta_all <- t(rbind(taulist-tau_min,estbeta_all))[1:(rd-1),]
    # d_all <- d_all[,1:(rd-1)]
    temp4 <- temp4[,1:(rd-2)]
  }


  # if (0 %in% taurange){
  #     if (max(abs(res0$coefficients-estbeta_all[1,-1]))>1e-8){
  #         estbeta_all <- rbind(c(0,res0$coefficients), estbeta_all)
  #         estbeta_all[2,1] <- tau_min
  #         d_all <- cbind(d_all[,1],d_all)
  #         d_all[d_all[,2]==0,2] <- tau_min
  #         d_all[d_all[,2]==-1,2] <- tau_min - 1
  #     }
  # }
  # if (1 %in% taurange){
  #     if (max(abs(res1$coefficients-estbeta_all[nrow(estbeta_all),-1]))>1e-8){
  #         estbeta_all <- rbind(estbeta_all, c(1,res1$coefficients))
  #         d_all <- cbind(d_all,d_all[,ncol(d_all)])
  #         d_all[d_all[,ncol(d_all)]==estbeta_all[nrow(estbeta_all)-1,1],ncol(d_all)] <- 1
  #         d_all[d_all[,ncol(d_all)-1]==estbeta_all[nrow(estbeta_all)-1,1]-1,ncol(d_all)] <- 0
  #     }
  # }


  colnames(estbeta_all) <- c('tau_left',paste('beta',1:(nvar+1), sep = '_'))
  return(list(beta = t(estbeta_all),diff_dsol = temp4))
}

qr_tau_para_diff_weight <- function(x, y, taurange,weights, tau_min = 1e-10, tol = 1e-14,
                             maxit = 100000, max_num_tau = 1000, bland = F,
                             use_residual = T){
  require('Matrix')
  require('MASS')
  # if (0 %in% taurange){
  #     library(quantreg)
  #     res0 <- rq(y~x,tau=0)
  # }
  # if (1 %in% taurange){
  #     library(quantreg)
  #     res1 <- rq(y~x,tau=1)
  # }
  x = as.matrix(x)
  m <- nrow(x) # subject number
  if(max_num_tau<2*m) max_num_tau = 2*m
  nvar <- ncol(x) # number of var
  tau <- taurange[1]+tau_min
  # rounds <- length(taulist)
  # cc <- c(rep(0, 1+nvar), tau*rep(1, m), (1-tau)*rep(1, m))
  weights = weights/mean(weights)
  cc <- c(rep(0, 1+nvar), tau*weights, (1-tau)*weights)

  # A matrix, b matrix
  gammax <- cbind(matrix(1, nrow = m, ncol = 1), x)
  gammaxb <- t(gammax)
  b <- c(y, 0)
  gammax[y<0,] <- -gammax[y<0,]
  b[b<0] <- -b[b<0]

  # Ib
  IB <- (y>=0)*((1:m)+1+nvar)+(y<0)*((1:m)+1+nvar+m)

  gammax <- rbind(gammax, -cc[IB] %*% gammax)
  freevarrow <- c(vector(mode = 'logical', m), T) # these variables cannot be non-basic variables
  r1 <- 1:(nvar+1)
  r2 <- vector('numeric', nvar+1)
  rr <- matrix(0, nrow = 2, ncol = 1+nvar) # last row in the table

  estbeta_all <- matrix(0, nrow = nvar+1, ncol = max_num_tau)
  # d_all <- matrix(0, nrow = m, ncol = max_num_tau)
  temp4 <- Matrix(0, nrow = m, ncol = max_num_tau)
  taulist <- vector('numeric', max_num_tau)

  # c0 <- c(rep(0, 1+nvar), rep(1, m), rep(-1, m))
  c0 = c(rep(0,1+nvar),weights,-weights)
  last_flag <- FALSE
  for (rd in 1:max_num_tau){
    # print(rd)
    if (rd>=2){
      # print(tau)
      # g <- gammax[-(m+1),]
      r1j <- crossprod(c0[IB], gammax[-(m+1),])
      # r1j <- c(1-r1j, r1j)
      # r0j <- c(gammax[(1+m),], weights[r1-(1+nvar)]-gammax[(1+m),])
      # theta <- -r0j/r1j
      # theta[r1j>=0] <- Inf
      r1j = c(r1j-1,-r1j+1)
      r0j = c(gammax[(1+m),], weights[r1-(1+nvar)]-gammax[(1+m),])
      theta = (r0j/r1j)[r1j>=0]

      # ib <- IB[IB>(1+nvar)]
      # ib <- ib - 1 - nvar

      # theta <- min(theta[-ib])
      theta <- min(theta)
      tau <- tau + theta + tau_min
      if (tau > taurange[2]){
        if (last_flag) {break}
        if ((tau-theta) <= taurange[2]){
          tau <- taurange[2]-tau_min
          last_flag <- TRUE
        }else{
          break
        }
      }

      # tau <- taulist[rd]
      cc[(2+nvar):(1+nvar+m)] <- tau*weights
      cc[(2+nvar+m):(1+nvar+2*m)] <- (1-tau)*weights
      gammax[(m+1),] <- tau - crossprod(c(cc[IB],0), gammax)
    }

    j <- 0
    while (j<maxit) {
      # start iteration
      # step 2
      if (rd==1 && j<nvar+1){
        rr[1,] <- gammax[(1+m),]
        rr[2,r2!=0] <- (weights[r2[r2!=0]-(1+nvar+m)] - rr[1,r2!=0])
        rr[1, r2==0] <- -abs(rr[1, r2==0])

        # step 3
        # choose t
        t_rr <- j+1
        tsep <- 1
        t <- r1[t_rr]
      }else{
        rr[1,] <- gammax[(1+m),]
        rr[2,r2!=0] <- (weights[r2[r2!=0]-(1+nvar+m)] - rr[1,r2!=0])
        rr[1, r2==0] <- -abs(rr[1, r2==0])

        # terminate
        rrl <- min(rr)
        if (rrl>=-tol){
          break
        }

        # step 3
        # choose t
        if (bland){
          if (any(rr[1,]< -tol)){
            tsep <- which(rr[1,]< -tol)
            tmp <- r1[tsep]
            t <- min(tmp)
            t_rr <- tsep[which(tmp==t)]
            tsep <- 1
          }else{
            tsep <- which(rr[2,]< -tol)
            tmp <- r2[tsep]
            t <- min(tmp)
            t_rr <- tsep[which(tmp==t)]
            tsep <- 2
          }
        }else{
          tsep <- which(rr==rrl, T)[1,]
          t_rr <- tsep[2]
          tsep <- tsep[1]
          if (tsep==1){
            t <- r1[t_rr]
          }else{
            t <- r2[t_rr]
          }
        }
      }

      if (r2[t_rr]!=0){
        # choose k
        # step 4
        if (tsep==1){
          yy <- gammax[, t_rr]
        }else{
          yy <- -gammax[, t_rr]
        }

        # step 5
        k <- b/yy
        if (bland){
          keep <- yy>0 & !freevarrow
          k <- which((min(k[keep])==k)&keep)
          if (length(k)!=1){
            tmp <- IB[k]
            k <- k[which(tmp==min(tmp))]
          }
        }else{
          keep <- yy>0 & !freevarrow
          if(sum(keep)==0){

            tau = taurange[2]
            last_flag = T
            break

          }
          k <- which((min(k[keep])==k)&keep)[1]
        }

        # pivoting step 6'
        if (tsep!=1){
          yy[m+1] <- yy[m+1]+weights[r2[t_rr]-(1+nvar+m)]
        }
      }else{
        yy <- gammax[, t_rr]
        if (yy[m+1]<0){
          # step 5
          k <- b/yy
          if (bland){
            keep <- yy>0 & !freevarrow
            k <- which((min(k[keep])==k)&keep)
            if (length(k)!=1){
              tmp <- IB[k]
              k <- k[which(tmp==min(tmp))]
            }
          }else{
            keep <- yy>0 & !freevarrow
            k <- which((min(k[keep])==k)&keep)[1]
          }
        }else{
          k <- -b/yy
          if (bland){
            keep <- yy<0 & !freevarrow
            k <- which((min(k[keep])==k)&keep)
            if (length(k)!=1){
              tmp <- IB[k]
              k <- k[which(tmp==min(tmp))]
            }
          }else{
            keep <- yy<0 & !freevarrow
            k <- which((min(k[keep])==k)&keep)[1]
          }
        }
        # pivoting step 6'
        freevarrow[k] <- T
      }

      ee <- yy/yy[k]
      ee[k] <- 1 - 1/yy[k]

      if (IB[k]<=(nvar+m+1)){
        gammax[, t_rr] <- 0
        gammax[k,t_rr] <- 1
        r1[t_rr] <- IB[k]
        r2[t_rr] <- IB[k]+m
      }else{
        gammax[, t_rr] <- 0
        gammax[k,t_rr] <- -1
        gammax[(m+1),t_rr] <- weights[IB[k]-(1+nvar+m)]
        r1[t_rr] <- IB[k]-m
        r2[t_rr] <- IB[k]
      }

      # pivoting step 6
      gammax <- gammax - tcrossprod(ee, gammax[k,])
      b <- b - ee * b[k]
      IB[k] <- t

      j <- j+1
    }

    if (j==maxit){
      warning('May not converge')
    }


    tmp <- IB %in% 1:(nvar+1)
    estimate <- b[tmp][order(IB[tmp])]

    if (length(estimate)!=(1+nvar)){
      estimate <- vector('numeric', nvar+1)
      for (i in 1:(nvar+1)){
        try(estimate[i] <- b[IB==i], silent = T) # some b[IB==i] may not exist, which means this beta is 0.
      }
    }

    #
    u <- (nvar+1+(1:m)) %in% IB
    v <- (nvar+1+m+(1:m)) %in% IB
    xh <- gammaxb[,!(u|v)]
    xbarh <- gammaxb[,(u|v)]
    dbarh <- ((tau-tau_min)*u + (tau-tau_min-1)*v)*weights
    tryCatch(
      dh <- solve(xh,-xbarh %*% dbarh[u|v]), error = function(c){
        dh <<- ginv(xh) %*% -xbarh %*% dbarh[u|v]
      }
    )

    # dh = cc[IB]%*%gammax[-(m+1),]
    dh = dh/weights[!(u|v)]+1-tau

    ### skip this part
    # hatd <- dbarh
    # hatd[!(u|v)] <- dh
    # d_all[,rd] <- hatd
    # # gammaxb %*% hatd = 0  x^t %*% d is 0 !
    # #
    ###
    temp1 <- u
    if (!use_residual){
      dh[dh==(tau-tau_min)*weights[r1-1-nvar]] <- 1
      dh[dh==(tau-tau_min-1)*weights[r1-1-nvar]] <- 0
    }
    temp1[!(u|v)] <- dh
    if (rd!=1){
      temp4[, rd-1] <- temp1 - temp
    }
    temp <- temp1

    estbeta_all[, rd] <- estimate
    taulist[rd] <- tau
  }
  if (!last_flag){
    estbeta_all <- t(rbind(taulist-tau_min,estbeta_all))[1:(rd-1),]
    # d_all <- d_all[,1:(rd-1)]
    temp4 <- temp4[,1:(rd-2)]
  }else if (last_flag && j==0){
    estbeta_all <- t(rbind(taulist-tau_min,estbeta_all))[c(1:(rd-3),(rd-1)),]
    # d_all <- d_all[,c(1:(rd-3),(rd-1))]
    temp4 <- temp4[,c(1:(rd-4),(rd-2))]
  }else {
    estbeta_all <- t(rbind(taulist-tau_min,estbeta_all))[1:(rd-1),]
    # d_all <- d_all[,1:(rd-1)]
    temp4 <- temp4[,1:(rd-2)]
  }


  # if (0 %in% taurange){
  #     if (max(abs(res0$coefficients-estbeta_all[1,-1]))>1e-8){
  #         estbeta_all <- rbind(c(0,res0$coefficients), estbeta_all)
  #         estbeta_all[2,1] <- tau_min
  #         d_all <- cbind(d_all[,1],d_all)
  #         d_all[d_all[,2]==0,2] <- tau_min
  #         d_all[d_all[,2]==-1,2] <- tau_min - 1
  #     }
  # }
  # if (1 %in% taurange){
  #     if (max(abs(res1$coefficients-estbeta_all[nrow(estbeta_all),-1]))>1e-8){
  #         estbeta_all <- rbind(estbeta_all, c(1,res1$coefficients))
  #         d_all <- cbind(d_all,d_all[,ncol(d_all)])
  #         d_all[d_all[,ncol(d_all)]==estbeta_all[nrow(estbeta_all)-1,1],ncol(d_all)] <- 1
  #         d_all[d_all[,ncol(d_all)-1]==estbeta_all[nrow(estbeta_all)-1,1]-1,ncol(d_all)] <- 0
  #     }
  # }


  colnames(estbeta_all) <- c('tau_left',paste('beta',1:(nvar+1), sep = '_'))
  return(list(beta = t(estbeta_all),diff_dsol = temp4))
}

