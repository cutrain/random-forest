#' Conditional Quantile Regression Random Forest
#'
#' Fits a "\code{CenQRF}" model.
#'
#' The argument "control" defaults to a list with the following values:
#' \describe{
#'   \item{\code{tau}}{is the quantile level of interest; default value is 0.5.}
#'   \item{\code{maxTree}}{is the number of trees to be used in the forest.}
#'   \item{\code{maxNode}}{is the maximum node number allowed to be in the tree; the default value is 500.}
#'   \item{\code{minSplitNode}}{is the minimum number of baseline observations in each splitable node; the default value is 5.}
#'   \item{\codeP{mytry}{is the number of variables randomly selected as candidates for splitting a node; the default value is sqrt(p).}}
#' }
#'
#' @param matY is the matrix of response variable.
#' @param delta is a binary variable indicating the censoring information of the response variable.
#' @param matZ is the matrix of predictive variables.
#' @param matX is the matrix of smoothing variables.
#' @param control a list of control parameters. See 'details' for important special
#' features of control parameters.
#'
#' @export
#'
#' @return An object of S4 class "\code{CenQRF}" representig the fit, with the following components:
#'
#' @importFrom randomForestSRC rfsrc
#'
#'
CQRforest <- function(matY,delta,matZ = NULL,matX, control = list()) {

  control <- CQRforest.control(control)
  Call <- match.call()
  if (missing(matY)) stop("Argument 'matY' is required.")
  if (missing(delta)) stop("Argument 'delta' is required.")
  if (missing(matX)) stop("Argument 'matX' is required.")


  ## extract information
  Call <- match.call(expand.dots = FALSE)
  # mcall <- Call[c(1L, callName)]
  # mcall[[1L]] <- quote(stats::model.frame)
  # mf <- eval(mcall, parent.frame())
  # mt <- attr(mf, "terms")
  .Y <- matY
  .D <- delta
  .X <- matX
  if (length(grep("`|\\(", colnames(.X))) > 0)
    .X <- .X[,-grep("`|\\(", colnames(.X)), drop = FALSE]

  .p <- ncol(.X)
  .n <- length(.Y)
  data_temp_censor = as.data.frame(cbind(matY,delta,matX,matZ))
  names(data_temp_censor)[1:2] = c("time","status")
  if(is.null(matZ))  matZ = matrix(1,nrow = 1,ncol = .n)
  .Z0 <- matZ
  .X0 <- apply(.X,2,order)
  .Y0 <- .Y
  .D0 <- .D
  if (is.function(control$mtry)) control$mtry <- control$mtry(.p)
  tau = control$tau

  data_temp_censor$status = abs(data_temp$status-1)
  rfsrc_sim_censor_time = randomForestSRC::rfsrc(Surv(time,status)~.,data = data_temp_censor,ntree = 500,
                                nodesize = 20,do.trace = T,forest.wt = T,
                                splitrule = "random")
  surv_time_mat = rfsrc_sim_censor_time$survival
  time_interest = as.matrix(rfsrc_sim_censor_time$time.interest)

  print("DONE")

  out <- CenQRForest_C(.Z0,.X0,.Y0,.D0,tau,as.matrix(rep(1,.n)),surv_time_mat,
                                                    time_interest,
                                                    control$quantile_level,
                                                    control$numTree,
                                                    control$minSplitTerm,
                                                    control$maxNode,control$mtry)
  out$Frame <- lapply(out$trees, cleanTreeMat, cutoff = cutoff, .X0 = .X0, disc = disc)


  out$call <- Call
  out$data <- list(.X = .X0, .Z = .Z0, .Y = .Y0, .D = .D0)
  # out$rName <- all.vars(formula)[1]
  # out$vNames <- attr(mt, "term.labels")
  out$control <- control
  class(out) <- "CensorQRForest"
  return(out)
}


#' Censored Quantile Regression Random Forest
#'
#' Fits a "\code{CenQRF}" model.
#'
#' The argument "control" defaults to a list with the following values:
#' \describe{
#'   \item{\code{tau}}{is the quantile level of interest; default value is 0.5.}
#'   \item{\code{maxTree}}{is the number of trees to be used in the forest.}
#'   \item{\code{maxNode}}{is the maximum node number allowed to be in the tree; the default value is 500.}
#'   \item{\code{minSplitNode}}{is the minimum number of baseline observations in each splitable node; the default value is 5.}
#'   \item{\codeP{mytry}{is the number of variables randomly selected as candidates for splitting a node; the default value is sqrt(p).}}
#' }
#'
#' @param matY is the matrix of response variable.
#' @param delta is a binary variable indicating the censoring information of the response variable.
#' @param matZ is the matrix of predictive variables.
#' @param matX is the matrix of smoothing variables.
#' @param control a list of control parameters. See 'details' for important special
#' features of control parameters.
#'
#' @export
#'
#' @return An object of S4 class "\code{CenQRF}" representig the fit, with the following components:
#'
#' @importFrom randomForestSRC rfsrc
#'
#'
CensQRforest <- function(matY,delta,matZ = NULL,matX,newdata,
                         control = list()) {

  control <- CQRforest.control(control)
  Call <- match.call()
  if (missing(matY)) stop("Argument 'matY' is required.")
  if (missing(delta)) stop("Argument 'delta' is required.")
  if (missing(matX)) stop("Argument 'matX' is required.")


  ## extract information
  Call <- match.call(expand.dots = FALSE)
  # mcall <- Call[c(1L, callName)]
  # mcall[[1L]] <- quote(stats::model.frame)
  # mf <- eval(mcall, parent.frame())
  # mt <- attr(mf, "terms")
  namey = names(matY)
  namez = names(matZ)
  naemx = names(matX)

  .X = matX
  if (length(grep("`|\\(", namex)) > 0)
    .X <- .X[,-grep("`|\\(", namex), drop = FALSE]

  .p <- ncol(.X)
  .n <- length(matY)
  .data_temp_censor = as.data.frame(cbind(matY,delta,matX,matZ))
  names(.data_temp_censor) = c("time","status",namex,namez)
  if(is.null(matZ))  matZ = matrix(1,nrow = 1,ncol = .n)
  .Z0 <- matZ
  .X0 <- apply(.X,2,order)
  .Y0 <- matY
  .D0 <- delta
  if (is.function(control$mtry)) control$mtry <- control$mtry(.p)
  tau = control$tau


  rfsrc_censor_time = randomForestSRC::rfsrc(Surv(time,status)~.,data = .data_temp_censor,ntree = 500,
                                                 nodesize = 20,do.trace = T,forest.wt = T,
                                                 splitrule = "random")
  predict_rfsrc = predict(rfsrc_censor_time,newdata = .data_temp_censor)
  rfsrc_sim_surv = cbind(1,predict_rfsrc$survival)

  time_interest = predict_rfsrc$time.interest
  time_censor = .data_temp_censor$time[.data_temp_censor$status==0]
  F_est = sapply(1:length(time_censor),function(i)
    1-rfsrc_sim_surv[.data_temp_censor$status==0,][i,
                                       sum(time_interest<=time_censor[i])+1],
    simplify = T)

  weight_censor = ifelse(F_est>=tau,1,(tau-F_est)/(1-F_est))

  .data_surv = data.frame(rbind(.data_temp_censor[.data_temp_censor$status==0,],
                               .data_temp_censor[.data_temp_censor$status==1,],
                               data.frame(time = 100*max(.data_temp_censor$time),
                                          .data_temp_censor[.data_temp_censor$status==0,-1])),
                         id = c(1:nrow(.data_temp_censor),1:sum(.data_temp_censor$status==0)))
  .data_surv = data.frame(.data_surv,
                         weight = c(weight_censor,rep(1,sum(.data_temp_censor$status==1)),1-weight_censor))

  .Z0 = as.matrix(.data_surv[,namez,drop = FALSE])
  .X0 = as.matrix(.data_surv[,namex,drop = FALSE])
  print(ncol(.X0))
  .Y0 = as.matrix(.data_surv$time)
  .Xnew = as.matrix(newdata)
  delta = as.matrix(.data_surv$status)[1:.n,]
  weight_censor = matrix(.data_surv$weight,nrow = 1)
  weight_rf = matrix(rep(1,nrow(.data_surv)),ncol = 1)

  print("DONE")

  out <- CenQRForest_WW_C(.Z0,.X0,.Y0,.Xnew,delta = delta,
                          tau = tau,
                          weight_rf = weight_rf,
                          weight_censor = weight_censor,
                          quantile_level = control$quantile_level,
                          numTree = control$numTree,
                          minSplit1 = control$minSplitTerm,
                          maxNode = control$maxNode,
                          mtry = control$mtry)
  # out$Frame <- lapply(out$trees, .X0 = .X0, disc = disc)


  out$call <- Call
  out$data <- list(.X = .X0[1:.n,], .Z = .Z0[1:.n,],
                   .Y = .Y0[1:.n,], .D = .D0[1:.n,])
  # out$rName <- all.vars(formula)[1]
  # out$vNames <- attr(mt, "term.labels")
  out$control <- control
  class(out) <- "CensorQRForest"
  return(out)
}

#' rocTree controls
#'
#' @keywords internal
#' @noRd
CQRforest.control <- function(l) {
  ## default values
  dl <- list(numTree = 500,minSplitTerm = 15,maxNode = 500,tau = 0.5,
             mtry = function(x) ceiling(sqrt(x)),
             quantile_level = as.matrix(seq(0,1,length.out = 50)[2:49]))
  l.name <- names(l)
  if (!all(l.name %in% names(dl)))
    warning("unknown names in control are ignored: ", l.name[!(l.name %in% names(dl))])
  dl[match(l.name, names(dl))] <- l
  ## if (is.null(dl$hN)) dl$hN <- dl$tau / 20
  return(dl)
}

#' Clean the `treeMat` from tree and forests; make it easier to read and compatible with print function
#' @keywords internal
#' @noRd
cleanTreeMat <- function(treeMat, cutoff, .X0, disc) {
  ## prepraing treeMat
  ## Remove 0 rows and redefine child nodes
  ## 0 rows were produced from prunning
  treeMat <- data.frame(treeMat)
  names(treeMat) <- c("p", "cutOrd", "left", "right", "is.terminal")
  treeMat$p <- ifelse(treeMat$is.terminal == 1, NA, treeMat$p + 1)
  ## treeMat$p + (1 - treeMat$is.terminal)
  treeMat$left <- ifelse(treeMat$left == 0, NA, treeMat$left + 1)
  treeMat$right <- ifelse(treeMat$right == 0, NA, treeMat$right + 1)
  mv <- rowSums(treeMat[,2:5], na.rm = TRUE) == 0
  if (sum(mv) > 0) {
    treeMat <- treeMat[-which(mv),]
    treeMat$left <- match(treeMat$left, rownames(treeMat))
    treeMat$right <- match(treeMat$right, rownames(treeMat))
    rownames(treeMat) <- NULL
  }
  if (nrow(treeMat) > 1) {
    treeMat$cutVal <- cutoff[ifelse(treeMat$cutOrd > 0, treeMat$cutOrd, NA)]
    if (sum(treeMat$p %in% which(disc))) {
      for (i in which(disc)) {
        ind <- which(treeMat$p == i)
        treeMat$cutVal[ind] <- as.numeric(levels(as.factor(.X0[,i]))[treeMat$cutOrd[ind]])
      }
    }
    if (nrow(treeMat) <= 3) {
      treeMat$nd <- 1:3
    } else {
      nd <- 1:3
      for (i in 2:(nrow(treeMat) - 2)) {
        if (treeMat$is.terminal[i] == 0) nd <- c(nd, 2 * nd[i], 2 * nd[i] + 1)
      }
      treeMat$nd <- nd
    }
  } else {
    treeMat$cutVal <- NA
    treeMat$nd <- 1
  }
  treeMat$cutOrd <- ifelse(treeMat$is.terminal == 1, NA, treeMat$cutOrd)
  return(treeMat)
}
