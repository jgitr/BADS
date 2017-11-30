# Email: winjules2@gmail.com


wd = paste0(Sys.getenv("USERPROFILE"), "/BADS")
setwd(wd)

dat.input = read.csv("BADS_WS1718_known.csv")
testset = read.csv("BADS_WS1718_class.csv")

library("lubridate")

# Date Variables

process.input = function(dat, train = TRUE){
  
  # Extract years
  date.df = data.frame(dat$order_date, dat$delivery_date, dat$user_dob)
  
  date.imputed = lapply(date.df, function(z){
    
    date.z = as.Date(z, format = "%Y-%m-%d")
    mean.z = mean(date.z, na.rm = TRUE)
    date.z[!is.finite(date.z)] = mean.z
    
    return(date.z)
    
  })
  
  # Bind and delivery time
  clean.date.df = do.call(cbind.data.frame, date.imputed)
  names(clean.date.df) = c("order_date", "delivery_date", "user_dob")
  
  clean.date.df$today = as.Date("2017-11-14", format = "%Y-%m-%d")
  clean.date.df$user_age = clean.date.df$today - clean.date.df$user_dob
  
  # Use delivery time groups
  clean.date.df$delivery_time = clean.date.df$delivery_date - clean.date.df$order_date
  clean.date.df$delivery_time = round(as.numeric(clean.date.df$delivery_time),0)
  omit.delivery.idx = which(clean.date.df$delivery_time <= 0) # save
  
  clean.date.df$delivery_time[clean.date.df$delivery_time <= 7] = 7
  clean.date.df$delivery_time[clean.date.df$delivery_time > 7 & clean.date.df$delivery_time <= 14] = 14
  clean.date.df$delivery_time[clean.date.df$delivery_time > 14] = 20
  
  clean.date.df$delivery_time[omit.delivery.idx] = 20 # NA, 20 is assumed/imputed
  
  # Use age groups - inisignificant, omit
  clean.date.df$user_age = round(as.numeric(clean.date.df$user_age/365),0)
  under.18 = which(clean.date.df$user_age <= 18)
  under.30 = which(clean.date.df$user_age > 18 & clean.date.df$user_age <= 30)
  under.40 = which(clean.date.df$user_age > 30 & clean.date.df$user_age <= 40)
  under.50 = which(clean.date.df$user_age > 40 & clean.date.df$user_age <= 50)
  under.60 = which(clean.date.df$user_age > 50 & clean.date.df$user_age <= 60)
  under.70 = which(clean.date.df$user_age > 60 & clean.date.df$user_age <= 70)
  over.70 = which(clean.date.df$user_age > 70)
  
  clean.date.df$age_lev = 0
  clean.date.df$age_lev[under.18] = 18
  clean.date.df$age_lev[under.30] = 30
  clean.date.df$age_lev[under.40] = 40
  clean.date.df$age_lev[under.50] = 50
  clean.date.df$age_lev[under.60] = 60
  clean.date.df$age_lev[under.70] = 70
  clean.date.df$age_lev[over.70 ] = 100
  
  extr.dates = data.frame(as.factor(clean.date.df$age_lev), as.factor(clean.date.df$delivery_time))
  names(extr.dates) = c("age_lev", "delivery_time")
  
  # Categorical Variables
  # omit item_color and brand_id
  
  categ.df = data.frame(dat$item_size, dat$user_title, dat$user_state, factor(dat$item_id))
  names(categ.df) = c("item_size", "user_title", "user_state", "item_id")
  
  # User title is imputed as female where not reported
  categ.df$user_title[categ.df$user_title == "not reported"] = "Mrs"
  
  # Without item_size and item_color
  numerical.df = data.frame(dat$item_price)
  names(numerical.df) = c("item_price")
  
  if(train == TRUE)
  {
    training.df = data.frame("return" = dat$return, extr.dates, categ.df, numerical.df)
    
    return(training.df)
  } else{
    test.df = data.frame(extr.dates, categ.df, numerical.df)
    return(test.df)
  }
  
  
}

### Estimate conservatively! 
# Omit Age: Inisgnificant groups and 1900 dob is unrealistic
# Omit brand_id - different brands between test and training sample
# Omit user id - senseless 
# ...

# Use training and test 
training = process.input(dat = dat.input, train = TRUE)
test = process.input(dat = testset, train = FALSE)
test.it = subset(test, select = intersect(names(test), names(training)))

# Model - 4 variables are sufficient
g = glm(return ~ delivery_time  + user_title + user_state + item_price,data = training, family = "binomial")
model = g # copy 

# Predict testset
fitted.results = predict(model,newdata=test.it,type='response')
fitted.results = ifelse(fitted.results > 0.5,1,0)


results = data.frame("order_item_id" = as.integer(names(fitted.results)), "return" = fitted.results)
results$order_item_id= results$order_item_id + 100000

# write.csv(results, "insertid.csv")



################################## Tests #######################################################


# Checking correlation
DF = lapply(training,as.integer)
cordf = do.call("cbind.data.frame", DF)
cor(cordf)

# Pseudo R2
library(pscl)
pR2(g)

# Simple cross validation using fixed vector, dynamic version further down
library("ROCR")
choose = seq(100, 100000, by = 100) # generate sample

training.short = training[-choose,]
training.verify = training[choose,]
short.model = glm(return ~ delivery_time  +
                    user_title + user_state + item_price,
                  data = training.short, family = "binomial")

# Compute AUC for predicting Class with the model
prob = predict(short.model, newdata=training.verify, type="response")
pred = prediction(prob, training.verify$return)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
auc = performance(pred, measure = "auc")
auc = auc@y.values[[1]]
auc


########################################################################################################

# K-fold cross validation - dynamic version


# Select data
dat = training

library(plyr)   # progress bar
library(caret)  # confusion matrix

# False positive rate
fpr = NULL

# False negative rate
fnr = NULL

# Number of iterations
k = 50

# Initialize progress bar
pbar = create_progress_bar('text')
pbar$init(k)

# Accuracy
acc = NULL

set.seed(12345)

for(i in 1:k)
{
  # Split dataset in training and test
  n.sample = floor(0.5* nrow(dat))
  index = sample(seq_len(nrow(dat)),size=n.sample)
  train = dat[index, ]
  test = dat[-index, ]
  
  # Define model
  model = glm(return ~ delivery_time  + user_title + user_state + item_price, 
              data = train, family = "binomial")
  
  # Predict results
  raw.results = predict(model, test, type='response')
  
  # Binary predictions
  results = ifelse(raw.results > 0.5,1,0)
  
  # True return realizations
  true.returns = test$return
  
  # Accuracy calculation
  misClasificError = mean(true.returns != results)
  
  # Collecting results
  acc[i] = 1-misClasificError
  
  # Confusion matrix
  cm = caret::confusionMatrix(data=results, reference=true.returns)
  fpr[i] = cm$table[2]/(nrow(dat)-n.sample)
  fnr[i] = cm$table[3]/(nrow(dat)-n.sample)
  
  pbar$step()
}


# Average accuracy of the model
mean(acc)

par(mfcol=c(1,2))

# Histogram of accuracy
hist(acc,xlab='Accuracy',ylab='Freq',density=30)

# Boxplot of accuracy
boxplot(acc,col='red',horizontal=TRUE,xlab='Accuracy')

# Confusion matrix and plots of fpr and fnr
mean(fpr)
mean(fnr)
hist(fpr,xlab='fnr',ylab='Freq',main='FPR',density=30)

hist(fnr,xlab='fnr',ylab='Freq',main='FNR',density=30)


