library(caret)
library(data.table)
library(doParallel)
library(rattle)
library(plyr)
library(testthat)
library(checkmate)
library(ALEPlot)
library(segmented)
library(xtable)
library(quantmod)
library(lubridate)
library(GGally)
library(plm)
library(parallel)
library(plotly)

# library(iml)
# test_check("iml")

rm(list = ls())

par(mar=rep(3,4))

output.dir <- "/home/simaan/Dropbox/Publications/Working Research Paper/Book Chapter Akhtar/TeX/BC_ML_Int_2021_01_28"
fig.dir <- paste(output.dir,"/Figures/",sep = "")


# function to compute number of features
feature_used <- function(pred, feature, sample_size){
  dat <- pred$trainingData
  fvalues = dat[,feature] 
  # permute feature
  dat2 = dat[sample(1:nrow(dat), size = sample_size, replace = TRUE),]
  prediction1 = predict(pred,dat2)
  
  sampled_fvalues = sapply(dat2[,feature], function(x){
    sample(setdiff(fvalues, x), size = 1)
  })
  
  dat2 <- data.table(dat2)
  dat2 = dat2[, (feature) := sampled_fvalues]
  dat2$.outcome <- NULL
  prediction2 = predict(pred,dat2)
  
  plot(prediction1~prediction2)
  
  if (any(( prediction1 - prediction2) != 0)) 
    return(TRUE)
  FALSE
}


skp_bike <- TRUE 

if(!skp_bike) {
  
  # get bike data
  {
    file.i <- "https://raw.githubusercontent.com/compstat-lmu/paper_2019_iml_measures/master/data/bike-sharing-daily.csv"
    ds <- read.csv(file.i)
    # plot(cnt~casual + hum + temp,data = ds)
    # adjust the data in the same way as the in the paper/github
    bike <- ds
    bike$weekday = factor(bike$weekday, levels=0:6, labels = c('SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'))
    bike$holiday = factor(bike$holiday, levels = c(0,1), labels = c('NO HOLIDAY', 'HOLIDAY'))
    bike$workingday = factor(bike$workingday, levels = c(0,1), labels = c('NO WORKING DAY', 'WORKING DAY'))
    # otherwise its rank defiicient
    bike$workingday = NULL
    bike$season = factor(bike$season, levels = 1:4, labels = c('SPRING', 'SUMMER', 'FALL', 'WINTER'))
    bike$weathersit = factor(bike$weathersit, levels = 1:3, labels = c('GOOD', 'MISTY', 'RAIN/SNOW/STORM'))
    bike$mnth = factor(bike$mnth, levels = 1:12, labels = c('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OKT', 'NOV', 'DEZ'))
    bike$yr[bike$yr == 0] = 2011
    bike$yr[bike$yr == 1] = 2012
    bike$yr = factor(bike$yr)
    bike = dplyr::select(bike, -instant, -dteday, -registered, -casual, -atemp)
    ds <- bike
    rm(bike)
  }
  
  # fit model
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3,allowParallel = T)
  model_list <- c("lasso","glmnet","svmLinear","svmRadial","rf","rpart","rpart2")
  y_var <- "cnt"
  
  ds <- ds[,!names(ds) %in% names(Filter(is.factor,ds))]
  ds_train <- ds
  ds_test <- ds_train
  x_var <- names(ds)[!names(ds) %in% y_var]
  model_formula <- formula(paste(y_var, " ~ " ,paste(x_var,collapse = " + ")))
  
  
  
  
  
  cl <- makePSOCKcluster(detectCores())
  registerDoParallel(cl)
  
  NF_seq <- IAS_seq <- MEC_model_seq <- train_model_list <- c()
  
  for (model_i in model_list) {
    cat("This is model ",model_i,"\n")
    set.seed(13)
    train_model <- train(model_formula, data = ds_train, method = model_i,
                         trControl=trctrl,
                         # preProcess = c("center", "scale"),
                         tuneLength = 10)
    
    train_model_list <- c(train_model_list,list(train_model))
    
    y_hat_i <- predict(train_model,ds_test[,x_var])
    y_true <- ds_test[,y_var]
    plot(y_true~y_hat_i)
    summary(lm(y_true~y_hat_i))$adj
    
    # B <- predict(train_model$finalModel, type = 'coefficients')
    # B2 <- coef(train_model$finalModel)
    # B <- coef(train_model$finalModel, train_model$finalModel$lambdaOpt)
    # 
    # X <- cbind(1,train_model$trainingData )
    # y_hat2 <- as.matrix(X)%*%B[,1]
    # plot(y_hat_i,y_hat2)
    
    # }
    
    NF <- sapply(x_var, function(x) feature_used(train_model,x,500) )
    NF_seq <- c(NF_seq,sum(NF))
    
    # compute ALE
    yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata,
                                                          type = "raw"))
    
    {
      f_0 <- mean(y_hat_i)
      f_ale_1st  <- f_0
      
      MEC <- V <- c()
      
      for (v in x_var) {
        j <- which(v == x_var)
        ale_j <- ALEPlot(ds_train, train_model, pred.fun = yhat, J = j,
                         K = 10^2, NA.plot = TRUE)
        
        
        loess_j <- loess(ale_j$f.values ~ ale_j$x.values,span = 0.1)
        
        {
          file.i <- paste0(fig.dir,"ALE_",model_i,".pdf")
          pdf(file.i)
          plot(ale_j$f.values~ale_j$x.values,
               ylab = "ALE", xlab = expression(x[j]),
               pch = 20, cex = 0.5)
          lines(predict(loess_j,ale_j$x.values)~ale_j$x.values,col = 2)
          grid(10)
          dev.off()
          
        }
        x_j <- ds_train[,v]
        f_j_ale <- predict(loess_j,x_j) # based on the ale, we compute the approximation
        # f_j_ale <- f_j_ale - mean(f_j_ale)
        f_ale_1st <- f_ale_1st + f_j_ale
        
        # MEC using segmented regression
        epsilon <- 0.05 # set tolerance
        
        MEC_try_function <- function(ale_j) {
          x <- ale_j$x.values
          y <- ale_j$f.values
          K <- 1
          lm_j <- lm(y ~ x)
          Rsq_j <- summary(lm_j)$adj
          seg_fit_j <- fitted(lm_j)
          
          ds_plot <- data.frame(seg_fit_j,y,x)
          ds_plot <- ds_plot[order(ds_plot$x),]
          
          file.i <- paste0(fig.dir,"MEC_",model_i,"_",j,".pdf")
          pdf(file.i)
          plot(seg_fit_j~x,pch = 20,type = "l", data = ds_plot,
               ylab = "ALE", xlab = expression(x[j]), lwd = 2)
          lines(y ~ x, data = ds_plot,col = 2)
          grid(10)
          dev.off()
          
          while(Rsq_j < 1 - epsilon) {
            cat("This is ",K,"\n")
            seg_j <- segmented(lm_j,seg.Z = ~ x, npsi = K)
            seg_fit_j <- fitted(seg_j)
            Rsq_j <- summary(lm(y~seg_fit_j))$adj
            K <- K + 1
            
            ds_plot <- data.frame(seg_fit_j,y,x)
            ds_plot <- ds_plot[order(ds_plot$x),]
            
            pdf(file.i)
            plot(seg_fit_j~x,pch = 20,type = "l", data = ds_plot,
                 ylab = "ALE", xlab = expression(x[j]), lwd = 2)
            lines(y ~ x, data = ds_plot,col = 2)
            grid(10)
            dev.off()
          }
          
          return(K)
          
        }
        
        
        catch_error <- try(MEC_try_function(ale_j),silent  = T)
        try_i <- 1
        
        while(inherits(catch_error,"try-error")) {
          try_i <- try_i + 1
          cat("This is error trial ",try_i,"\n" )
          catch_error <- try(MEC_try_function(ale_j),silent  = T)
        }
        
        
        MEC_j <- catch_error
        V_j <- var(f_j_ale)
        
        MEC <- c(MEC,MEC_j)
        V <- c(V,V_j)
      }
      
      MEC_model <- sum(MEC*V/sum(V))
      
    }
    
    # compute IAS
    IAS1 <- sum((y_hat_i - f_ale_1st)^2)
    IAS2 <- sum((y_hat_i - f_0)^2)
    IAS <- IAS1/IAS2
    
    # alternatively we can do so using regressing
    1-summary(lm(y_hat_i~f_ale_1st))$adj
    IAS_seq <- c(IAS_seq,IAS)
    
    MEC_model_seq <- c(MEC_model_seq,MEC_model)
    
  }
  
  
  stopCluster(cl)
  registerDoSEQ()
  
  sum_df <- data.frame(model = model_list,  MEC = MEC_model_seq, IAS = round(IAS_seq,2),NF = NF_seq)
  xtable(sum_df)
  
  
}


#######################################################################################
#######################################################################################



############################################
######## FINANCIAL DATA ####################

# get the main data
{
  t1 <- "1990-01-01"
  v <- c("SPY","GLD","IEF","XLF")
  P.list <- lapply(v, function(sym) get(getSymbols(sym,from = t1)) )
  getSymbols("^VIX",from = t1)
  P.list <- c(P.list,list(VIX))
  
  # volume
  P.list5 <- lapply(P.list, function(x) x[,5])
  P5 <- na.omit(Reduce(function(...) merge(...),P.list5 ))
  # prices
  P.list6 <- lapply(P.list, function(x) x[,6])
  P6 <- na.omit(Reduce(function(...) merge(...),P.list6 ))
  
  names(P5) <- names(P6) <- c("SPY","GLD","IEF","XLF","VIX")
  names(P5) <- paste(names(P5),"vol",sep = "_")
  P5$VIX_vol <- NULL
  
  # adjust the time series
  P5 <- P5["2005-01-01/2020-12-31",]
  P6 <- P6["2005-01-01/2020-12-31",]
  
  # compute returns, realized volatility, and MAs
  R6 <- na.omit(P6/lag(P6)-1)
  R_m <- apply.monthly(R6,apply,2,function(x) prod(1+x) - 1 )
  V_m <- apply.monthly(R6,apply,2,function(x) sd(x)*sqrt(25) )
  names(V_m) <- paste(names(V_m),"V",sep = "_")
  Vol_m <- apply.monthly(P5,apply,2,last)
  Vol_m <- Vol_m/lag(Vol_m) - 1
  
  # add rolling difference 
  R_m_ma <- R_m - rollapply(R_m,12,mean)
  V_m_ma <- V_m/rollapply(V_m,12,mean)
  Vol_m_ma <- Vol_m - rollapply(Vol_m,12,mean)
  
  names(R_m_ma) <- paste(names(R_m_ma),"_roll",sep="")
  names(V_m_ma) <- paste(names(V_m_ma),"_roll",sep="")
  names(Vol_m_ma) <- paste(names(Vol_m_ma),"_roll",sep="")
  
  ds <-  na.omit(merge(R_m,V_m,Vol_m,R_m_ma,V_m_ma,Vol_m_ma))
  SPY_V_next  <- ds$SPY_V
  names(SPY_V_next) <- paste(names(SPY_V_next),"next",sep = "_")
  ds <- lag(ds,1)
  ds <- na.omit(merge(SPY_V_next,ds))
  
  
  # create some correlation lots
  cor_plot1 <- ggcorr(ds[,c("SPY_V_next",names(R_m))])
  cor_plot2 <- ggcorr(ds[,c("SPY_V_next",names(V_m))])
  cor_plot3 <- ggcorr(ds[,c("SPY_V_next",names(Vol_m))])
  cor_plot4 <- ggcorr(ds[,c("SPY_V_next",names(R_m_ma))])
  cor_plot5 <- ggcorr(ds[,c("SPY_V_next",names(V_m_ma))])
  cor_plot6 <- ggcorr(ds[,c("SPY_V_next",names(Vol_m_ma))])
  
  cor_plot_list <- list(cor_plot1,cor_plot2,cor_plot3,cor_plot4,cor_plot5,cor_plot6)
  for (i in 1:length(cor_plot_list)) {
    file.i <- paste(output.dir,"/Figures/corr_plot_",i,".pdf",sep = "")
    pdf(file.i)
    print(cor_plot_list[[i]])
    dev.off()
  }
  
}

####################################################
############ ML APPLICATION for Vol ################

# model_list <- c("lasso","glmnet","svmLinear","svmRadial","rf","rpart","rpart2","mlp","knn","BstLm","glmboost","xgboost")
model_list <- c("lasso","glmnet","svmLinear","svmRadial","rf","rpart","rpart2")


y_var <- "SPY_V_next"

# split data into training and testing
ds2 <- ds
ds2$SPY_V_next <- log(ds2$SPY_V_next)
ds_test <- last(ds2,60)
ds_train <- ds2[!date(ds2) %in% date(ds_test),]

x_var <- names(ds2)[!names(ds2) %in% y_var]
model_formula <- formula(paste(y_var, " ~ " ,paste(x_var,collapse = " + ")))

# consider tuning using time slice. In this case, we start with 60 and then use the next 12 months
# for tuning
# this for timeslice

T_win_seq <- (2:5)*12
h_win_seq <- c(3,6,9,12)
metric_seq <-  c("RMSE","Rsquared","MAE")

inputs_ds <- expand.grid(model_list,metric_seq,T_win_seq,h_win_seq)
names(inputs_ds) <- c("Model","Metric","Window","Horizon")
inputs_ds <- inputs_ds[order(inputs_ds$Model,inputs_ds$Metric,inputs_ds$Window,inputs_ds$Horizon),]
rownames(inputs_ds) <- NULL

# run one example
# inputs_ds <- inputs_ds[inputs_ds$Model == "rpart2",]
# MAIN_ML_FUNCTION(1)


# choose a sub-sample for illustration
# inputs_ds <- inputs_ds[inputs_ds$Metric == "RMSE" & inputs_ds$Window == 60 & inputs_ds$Horizon == 12,]

## ML running starts here <----------------


MAIN_ML_FUNCTION <- function(iter_n) {
  
  NF_seq <- IAS_seq <- MEC_model_seq <- train_model_list <- c()
  
  model_i <- as.character(inputs_ds$Model[iter_n])
  metric_i <- as.character(inputs_ds$Metric[iter_n])
  
  T_win <- inputs_ds$Window[iter_n]
  h_win <- inputs_ds$Horizon[iter_n]
  
  trctrl <- trainControl(method = "timeslice",
                         initialWindow = T_win,
                         horizon = h_win,
                         fixedWindow = TRUE,
                         allowParallel = TRUE)
  
  
  
  cat(" #################### This is model ",model_i," #####################","\n")
  train_model <- train(model_formula, data = data.frame(ds_train), method = model_i,
                       trControl=trctrl, metric = metric_i,
                       # preProcess = c("center", "scale"),
                       tuneLength = 10)
  
  train_model_list <- c(train_model_list,list(train_model))
  
  y_hat_i <- predict(train_model,ds_train[,x_var])
  y_true <- as.numeric(ds_train[,y_var])
  R_in <- summary(lm(y_true~y_hat_i))$adj
  
  y_hat_out_i <- predict(train_model,ds_test[,x_var])
  y_true_out <- as.numeric(ds_test[,y_var])
  plot(y_true_out~y_hat_out_i)
  R_out <- summary(lm(y_true_out~y_hat_out_i))$adj
  
  NF <- sapply(x_var, function(x) feature_used(train_model,x,500) )
  NF_seq <- c(NF_seq,sum(NF))
  
  # compute ALE
  yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata,
                                                        type = "raw"))
  
  f_0 <- mean(y_hat_i)
  f_ale_1st  <- f_0
  
  
  # run the following only for the features that come into the model
  
  if(sum(NF) > 0) {
    MEC <- V <- c()
    
    for (v in x_var[NF]) {
      j <- which(names(ds_train) == v)
      ale_j <- ALEPlot(data.frame(ds_train), train_model, pred.fun = yhat, J = j,
                       K = 10^2, NA.plot = TRUE)
      
      
      loess_j <- loess(ale_j$f.values ~ ale_j$x.values,span = 0.1)
      
      file.i <- paste0(fig.dir,model_i,"/ALE_",v,".pdf")
      pdf(file.i)
      plot(ale_j$f.values~ale_j$x.values,
           ylab = "ALE", xlab = expression(x[j]),
           pch = 20, cex = 0.5)
      lines(predict(loess_j,ale_j$x.values)~ale_j$x.values,col = 2)
      grid(10)
      dev.off()
      
      
      
      x_j <- as.numeric(ds_train[,v])
      f_j_ale <- predict(loess_j,x_j) # based on the ale, we compute the approximation
      # f_j_ale <- f_j_ale - mean(f_j_ale)
      f_ale_1st <- f_ale_1st + f_j_ale
      
      # MEC using segmented regression
      epsilon <- 0.05 # set tolerance
      
      MEC_try_function <- function(ale_j) {
        x <- ale_j$x.values
        y <- ale_j$f.values
        K <- 1
        lm_j <- lm(y ~ x)
        Rsq_j <- summary(lm_j)$adj
        seg_fit_j <- fitted(lm_j)
        
        ds_plot <- data.frame(seg_fit_j,y,x)
        ds_plot <- ds_plot[order(ds_plot$x),]
        
        file.i <- paste0(fig.dir,model_i,"/MEC_",v,".pdf")
        
        pdf(file.i)
        plot(seg_fit_j~x,pch = 20,type = "l", data = ds_plot,
             ylab = "ALE", xlab = expression(x[j]), lwd = 2,ylim = range(y))
        lines(y ~ x, data = ds_plot,col = 2)
        grid(10)
        dev.off()
        
        while(Rsq_j < 1 - epsilon) {
          cat("This is ",K,"\n")
          seg_j <- segmented(lm_j,seg.Z = ~ x, npsi = K)
          seg_fit_j <- fitted(seg_j)
          Rsq_j <- summary(lm(y~seg_fit_j))$adj
          K <- K + 1
          Rsq_j
          
          if(K > 30)
            return(K)
          
          ds_plot <- data.frame(seg_fit_j,y,x)
          ds_plot <- ds_plot[order(ds_plot$x),]
          
          pdf(file.i)
          plot(seg_fit_j~x,pch = 20,type = "l", data = ds_plot,
               ylab = "ALE", xlab = expression(x[j]), lwd = 2,ylim = range(y))
          lines(y ~ x, data = ds_plot,col = 2)
          grid(10)
          dev.off()
        }
        
        return(K)
        
      }
      
      
      catch_error <- try(MEC_try_function(ale_j),silent  = T)
      try_i <- 1
      
      while(inherits(catch_error,"try-error")) {
        try_i <- try_i + 1
        cat("This is error trial ",try_i,"\n" )
        catch_error <- try(MEC_try_function(ale_j),silent  = T)
        if(try_i > 10) 
          break
      }
      
      
      MEC_j <- catch_error
      V_j <- var(f_j_ale)
      
      MEC <- c(MEC,MEC_j)
      V <- c(V,V_j)
    }
    
    
    MEC_copy <- as.numeric(MEC)
    V_copy <- V
    
    keep_val1 <- which(is.na(MEC_copy))
    keep_val2 <- which(is.na(V_copy))
    keep_val <- 1:length(V)
    keep_val <- keep_val[!keep_val %in% union(keep_val1,keep_val2)]
    
    
    MEC_copy <- MEC_copy[keep_val]
    V_copy <- V_copy[keep_val]
    MEC_model <- sum(MEC_copy*V_copy/sum(V_copy))
    
    # compute IAS
    IAS1 <- sum((y_hat_i - f_ale_1st)^2,na.rm = T)
    IAS2 <- sum((y_hat_i - f_0)^2)
    IAS <- IAS1/IAS2
  }
  
  
  if(sum(NF) == 0) {
    MEC_model <- 0
    IAS <- 0
  }
  
  main_result <- list(MEC = MEC_model,IAS = IAS,NF = sum(NF),train_model = train_model)
  
  return(main_result)
}




# cl <- makePSOCKcluster(detectCores())
# registerDoParallel(cl)
MAIN_ML_mclapply <- mclapply(1:nrow(inputs_ds),MAIN_ML_FUNCTION, mc.cores = detectCores()-2)
gc()

# stopCluster(cl)
# registerDoSEQ()

MEC_model_seq <- sapply(MAIN_ML_mclapply, function(x) x$MEC)
IAS_seq <- sapply(MAIN_ML_mclapply, function(x) x$IAS)
NF_seq <- sapply(MAIN_ML_mclapply, function(x) x$NF)
train_model_list <- lapply(MAIN_ML_mclapply, function(x) x$train_model)

sum_df <- data.frame(model = inputs_ds,  MEC = MEC_model_seq, IAS = round(IAS_seq,2),NF = NF_seq)

# measure complexity using standadrized values
stand_score <- function(x) (x-mean(x,na.rm = T))/sd(x,na.rm = T)
sum_df$complexity <- stand_score(sum_df$MEC) + stand_score(sum_df$IAS) + stand_score(sum_df$NF)
# xtable(sum_df)


## let's add statistical performance
true_ys_in <- as.numeric(exp((ds_train[,y_var])))
true_ys_out <- as.numeric(exp((ds_test[,y_var])))
fitted_ys_in <- sapply(1:length(train_model_list), function(m) (exp(predict(train_model_list[[m]],ds_train[,x_var]))))
fitted_ys_out <- sapply(1:length(train_model_list), function(m) (exp(predict(train_model_list[[m]],ds_test[,x_var]))))

R_in_seq <- sapply(1:ncol(fitted_ys_in), function(i) summary( lm(true_ys_in ~ fitted_ys_in[,i]) )$adj  )
R_out_seq <- sapply(1:ncol(fitted_ys_out), function(i) summary( lm(true_ys_out ~ fitted_ys_out[,i]) )$adj  )

sum_df2 <- data.frame(sum_df,R_in_seq,R_out_seq)

# add the backtest results
W_mat <- fitted_ys_out
W_mat <- as.xts(W_mat)
# names(W_mat) <- model_list
W_mat <- W_mat*sqrt(12)
# let's add the back-testing results

port_sum_f <- function(x) {
  m <- mean(x)*12
  s <- sd(x)*sqrt(12)
  sr <- m/s
  VaR <- mean(x*12) - quantile(x*12,0.05)
  ES <- -mean(x[x < quantile(x,0.05)])*12
  return(c(m,s,sr,VaR,ES))
}

c_cons <- 0.01
TC <- 10/(100^2)

port_results_01 <- c()
PORT_ret_mat <- PORT_W_mat <- c()
for (i in 1:ncol(W_mat)) {
  W_i <- W_mat[,i]
  port_w1 <- c_cons*1/(W_i^2)
  port_w2 <- 1-port_w1
  port_W <- as.matrix(cbind(port_w1,port_w2))
  TO <- c(0,apply(port_W[-1,]-port_W[-nrow(port_W),],1,function(x) sum(abs(x))))
  
  ETF_ret <- R_m[,c("SPY","IEF")]
  ETF_ret <- ETF_ret[rownames(port_W),]
  RET_port <- apply(ETF_ret*port_W,1,sum)
  RET_port <- RET_port - TC*TO
  PORT_ret_mat <- cbind(PORT_ret_mat,RET_port)
  PORT_W_mat <- cbind(PORT_W_mat,port_w1)
  port_sum_i <- port_sum_f(RET_port)
  port_sum_i <- c(port_sum_i,mean(TO))
  port_results_01 <- rbind(port_results_01,port_sum_i)
}

colnames(port_results_01) <- c("Mean","Std","SR","VaR","ES","TO")
port_results_01 <- data.frame(port_results_01)
rownames(port_results_01) <- NULL

sum_df3 <- data.frame(sum_df2,port_results_01)
sum_df3

######################################################################################################################################

#########################################
########### SUMMARY OF MAIN RESULTS #####
#########################################

# summarize based on algorithm
DT <- data.table(sum_df3)
DT_sum_mean <- DT[,lapply(.SD,function(x) mean(x,na.rm = T) ), by  = list(model.Model),
                  .SDcol = c("MEC","IAS","NF","complexity","R_in_seq", "R_out_seq", "Mean","Std", "SR","VaR","ES","TO")  ]
DT_sum_sd <- DT[,lapply(.SD,function(x) sd(x,na.rm = T) ), by  = list(model.Model),
                .SDcol = c("MEC","IAS","NF","complexity","R_in_seq", "R_out_seq", "Mean","Std", "SR","VaR","ES","TO")  ]

xtable(DT_sum_mean)
xtable(DT_sum_sd)



# CREATE PLOTS

ggplot_function <- function(v) {
  ds_plot <- ddply(sum_df3,"model.Model", function(x) data.frame(Performance = x[,v], Complexity = x$complexity, Type = x$model.Model)   )
  ds_plot$Type <- as.factor(ds_plot$Type)
  # p <- ggplot(data = ds_plot)
  # p <- p + geom_smooth(data = ds_plot, aes(Complexity,Performance), method = "loess",se = TRUE)
  # p <- p + geom_point(aes(Complexity,Performance, colour = Type,shape = Type))
  # 
  p <- ggplot(data = ds_plot,aes(Complexity,Performance, colour = Type,shape = Type))
  p <- p + geom_point()
  p <- p + geom_smooth(method = "lm",se = TRUE)
  
  return(p)
}


v_seq <- c("R_in_seq","R_out_seq",
           "Std","SR","ES","TO")

ggplot_list <- list()
for (v in v_seq) {
  p_v <- ggplot_function(v)
  ggplot_list <- c(ggplot_list,list(p_v))
  file.i <- paste(fig.dir,"/perf_",v,".pdf",sep = "")
  pdf(file.i)
  print(p_v)
  dev.off()
}


## RUN REGRESSIONS WITH FIXED EFFECTS

# let's create a panel regression
lm_1 <- lm(R_in_seq ~complexity + as.factor(model.Model), data = sum_df3)
lm_2 <- lm(R_out_seq ~complexity + as.factor(model.Model), data = sum_df3)
lm_3 <- lm(Mean ~complexity + as.factor(model.Model), data = sum_df3)
lm_4 <- lm(Std ~complexity + as.factor(model.Model), data = sum_df3)
lm_5 <- lm(ES ~complexity + as.factor(model.Model), data = sum_df3)
lm_6 <- lm(TO ~complexity + as.factor(model.Model), data = sum_df3)
# 
# lm_1 <- lm(R_in_seq ~complexity , data = sum_df3)
# lm_2 <- lm(R_out_seq ~complexity, data = sum_df3)
# lm_3 <- lm(Mean ~complexity , data = sum_df3)
# lm_4 <- lm(Std ~complexity, data = sum_df3)
# lm_5 <- lm(ES ~complexity, data = sum_df3)
# lm_6 <- lm(TO ~complexity, data = sum_df3)


lm_list <- list(lm_1,lm_2,lm_3,lm_4,lm_5,lm_6)
stargazer::stargazer(lm_list)


lm_1 <- lm(R_in_seq ~complexity , data = sum_df3)
lm_2 <- lm(R_out_seq ~complexity, data = sum_df3)
lm_3 <- lm(Mean ~complexity , data = sum_df3)
lm_4 <- lm(Std ~complexity, data = sum_df3)
lm_5 <- lm(ES ~complexity, data = sum_df3)
lm_6 <- lm(TO ~complexity, data = sum_df3)


lm_list <- list(lm_1,lm_2,lm_3,lm_4,lm_5,lm_6)
stargazer::stargazer(lm_list)



lm_1 <- lm(R_in_seq ~ complexity + I(complexity^2), data = sum_df3)
lm_2 <- lm(R_out_seq ~complexity + I(complexity^2), data = sum_df3)
lm_3 <- lm(Mean ~complexity  + I(complexity^2), data = sum_df3)
lm_4 <- lm(Std ~complexity  + I(complexity^2), data = sum_df3)
lm_5 <- lm(ES ~complexity + I(complexity^2), data = sum_df3)
lm_6 <- lm(TO ~complexity + I(complexity^2), data = sum_df3)


lm_list <- list(lm_1,lm_2,lm_3,lm_4,lm_5,lm_6)
stargazer::stargazer(lm_list)

# let's take a look at cumulative returns for top performance
# colnames(PORT_ret_mat) <- model_list
PORT_ret_mat <- data.frame(PORT_ret_mat)
bench_ret <- function(w) w* ETF_ret$SPY + (1-w)*ETF_ret$IEF
plot(cumsum(PORT_ret_mat[,2]) ~ date(rownames(PORT_ret_mat)),type = "l", ylim = c(0,0.8))
lines(cumsum(bench_ret(1)) ~ date(rownames(PORT_ret_mat)),col = 2)
lines(cumsum(bench_ret(0.5)) ~ date(rownames(PORT_ret_mat)),col = 3)
lines(cumsum(bench_ret(0)) ~ date(rownames(PORT_ret_mat)),col = 4)



########################################################
############### MVC Efficient Frontier #################
########################################################


# add constraints
eps <- 0.001
BC_f <- function(mean_target,com_target) {
  # sum to one constraint
  A <- matrix(1,1,d)
  A <- rbind(A,-A)
  B <- c(1 -eps,-(1 + eps))
  
  # sum to mu_target
  A2 <- t(Mu)
  B2 <- mean_target
  
  # sum to gamma_target
  A3 <- t(Gamma)
  B3 <- -com_target
  
  
  # stack altogether in a list
  A_final <- rbind(A,A2,A3)
  B_final <- c(B,B2,B3)
  
  return(list(A_final,B_final))
}

ncol(PORT_ret_mat)
Mu <- apply(PORT_ret_mat,2,mean)
Sigma <- var(PORT_ret_mat)
Gamma <-  sum_df3$complexity
d <- nrow(Sigma)


### add MV as benchmark portfolio
OPT_PORT_MV_function <- function(mean_target,com_target) { 
  
  U <- function(X) {
    u2 <- t(X)%*%Sigma%*%X
    total <- u2
    return(c(total))
  }
  
  
  G <- function(X) {
    g2 <- 2*Sigma%*%X
    total <- g2 
    return(total)
  }
  
  BC <- BC_f(mean_target,com_target)
  A <- BC[[1]]
  B <- BC[[2]]
  
  # solving this should give the minimum variance portfolio (GMV)
  X0 <- rep(1/d,d)
  X_opt <- constrOptim(X0,U,grad = G ,ui = A,ci = B)
  X1 <- X_opt$par
  X1 <- X1/sum(X1)
  
  return(X1)
  
}


mean_seq <- seq(min(Mu),max(Mu), length = 10^2)
com_seq <- seq(min(Gamma),max(Gamma), length = 10^2)

mc_grid <- expand.grid(mean_seq,com_seq)

opt_port_grid <- function(i) {
  mean_target_i <- mc_grid[i,1]
  com_target_i <- mc_grid[i,2]
  X_i <- try(OPT_PORT_MV_function(mean_target_i,com_target_i),silent = T)
  return(X_i)  
}

port_list <- mclapply(1:nrow(mc_grid), opt_port_grid,mc.cores =  detectCores())
port_list <- port_list[!sapply(port_list,function(x) inherits(x,"try-error")  )]
port_mat <- Reduce(cbind,port_list)
range(port_mat)

mvc_f <- function(x) {
  mu_p <- t(x)%*%Mu
  sig_p <- t(x)%*%Sigma%*%x
  com_p <- t(x)%*%Gamma
  return(c(mu_p,sig_p,com_p))
}

mvc_ds <- data.frame(t(sapply(port_list,mvc_f)))
names(mvc_ds) <- c("mu_p","sig_p","gamma_p")

plot( I(sqrt(12)*mvc_ds$mu_p/sqrt(mvc_ds$sig_p)) ~ mvc_ds$gamma_p )


model <- loess(mu_p~sig_p+gamma_p,data = mvc_ds, control = loess.control(surface = "interpolate"))

sig_seq <- seq(range(mvc_ds$sig_p)[1],range(mvc_ds$sig_p)[2],length = 100)
gamma_seq <- seq(range(mvc_ds$gamma_p)[1],range(mvc_ds$gamma_p)[2],length = 100)

X <- expand.grid(sig_seq,gamma_seq)
names(X) <- c("sig_p","gamma_p")
Z <- predict(model,X)

{
  f <- list(
    family = "Courier New, monospace",
    size = 11,
    color = "#7f7f7f"
  )
  
  axx <- list(
    title = "Risk",
    titlefont = f
  )
  
  axy <- list(
    title = "Complexity",
    titlefont = f
    
  )
  
  axz <- list(
    title = "Mean",
    titlefont = f
    
  )
  
  
  p <- plot_ly(x = sqrt(sig_seq*(12)*100), y = gamma_seq, z = Z*12*100) %>% add_surface() 
  p <- layout(p,title = paste("Mean-Variance-Complexity Efficient Frontier"),scene = list(xaxis=axx,yaxis=axy,zaxis=axz))
  p
}



port_mean <- apply(port_mat,1,median)

sum_df4 <- data.frame(sum_df3,port_mean)

barplot_f <- function(v) {
  W_algo <- ddply(sum_df4,v,function(x) mean(x$port_mean) )
  W_algo <- W_algo[order(W_algo$V1),]
  W <- W_algo$V1
  names(W) <- W_algo[,v]
  
  x <-  barplot(W,ylim = c(-0.03,0.03),xaxt="n")
  labs <- names(W)
  text(cex=1, x=x, y=-0.025, labs, xpd=TRUE, srt=45)
}

file.i <- paste(fig.dir,"surface/port_model.pdf",sep = "")
pdf(file.i)
barplot_f("model.Model")
dev.off()


file.i <- paste(fig.dir,"surface/port_metric.pdf",sep = "")
pdf(file.i)
barplot_f("model.Metric")
dev.off()


















