# This was a predictive modeling exercise. The high level goal was to predict
# the price of a bottle of wine based on attributes like where it is from, what
# type of wine it is, and the written review of the wine (the "description"
# column). The assigned questions are given in the form "Q1.", "Q2.", etc.

library(data.table)

wine = fread("winemag-data-130k-v2.csv")

# Q1. The year that the wine was made is part of the "title" column. Write
# a function to extract it and add a "year" column to the dataset.

getYear = function(string) {
  newstring = sub("^.*([0-9]{4}).*", "\\1", string)
  if (nchar(newstring) != 4) return(0)
  return(newstring)
}

wine$year = lapply(wine$title, getYear)

# I used regex to extract 4 digit numbers that appeared anywhere in the string
# if there was not a year in the string, then I returned 0. Some numbers were
# two digit such as 14, but it was not possible to infer if those referred to
# 2014 or not.

# Q2. Build a model to predict the price of a bottle of wine only using the text
# in the "description" column. For each of the tasks below, the choice of what
# model to use and how to evaluate the model is up to you.
#
# A)	Regression: Predict a wine's price only using the text in the "description"
# column.
#
# B)	Classification: Predict whether a wine costs more than 42 dollars only
# using the "description column".

set.seed(123)
data = fread("data.csv")

# remove underscore and just combine words, in case cleaning later removes
# underscores and I want the combined words to stay as a single distinct
# entity

data$wiki = sub("_", "", data$wiki)

# Convert price by doing a log transform. Take ln of the numbers because price
# is exponential

data$logprice = log(data$price)

library(FeatureHashing)
library(Matrix)
library(xgboost)

# Use a hashed model for bag of words classification
d1 <- hashed.model.matrix(~ split(wiki, delim = " ", type = "tf-idf"),
                          data = data, hash.size = 2^16, signed.hash = FALSE)


smp_size <- floor(0.7 * nrow(data))

train <- sample(seq_len(nrow(data)), size = smp_size)
test <- c(1:nrow(data))[-train]
dtrain <- xgb.DMatrix(d1[train,], label = data$logprice[train])
dvalid <- xgb.DMatrix(d1[test,], label = data$logprice[test])
watch <- list(train = dtrain, test = dvalid)

m2 <- xgb.train(data = dtrain, booster = "gbtree", nrounds = 3000, eta = 0.02, 
                max.depth = 5, colsample_bytree = 0.9, min_child_weight = 1.5,
                subsample = 0.95, objective = "reg:linear",
                watchlist = watch, early_stopping_rounds=50,print_every_n=10)


xpred <- predict(m2, dvalid)
xpreds = cbind(data[test, ], xpred)
xpreds$predprice = exp(xpreds$xpred)
xpreds$fortytwo = "no"
xpreds$fortytwo[which(xpreds$price >  42)] = "yes"

# I thought that $42 was an arbitrary price to do the classification, and might
# result in information loss because (for all intents and purposes) there
# probably isn't words in the description that distinguish between a wine that
# costs more or less than $42 In addition to doing the classification model, I
# classified the predicted values using the linear predictions I did for part a
# of question 2

# I also wasn't sure if 42 as a cutoff point would be ideal, so I looked at
# prices between 30 and 70 as cutoffs and found that a different cutoff price
# yielded more accuracy for predicting whether a wine costs more or less than
# $42

xpreds2 = xpreds

for (i in 30:70) {
  xpreds2$predicted42 = "no"
  xpreds2$predicted42[which(xpreds$predprice >  i)] = "yes"
  confus = table(xpreds2$predicted42, xpreds2$fortytwo)
  print(cbind(sum(diag(confus))/sum(confus), i))
}

# 1 for under 42 (no) and 0 for over 42 (yes)

data$fortytwo = "no" 
data$fortytwo[which(data$price >  42)] = "yes"

data$fortytwonum = 1
data$fortytwonum[which(data$price >  42)] = 0

d1 <- hashed.model.matrix(~ split(wiki, delim = " ", type = "tf-idf"),
                          data = data, hash.size = 2^16, signed.hash = FALSE)

smp_size <- floor(0.7 * nrow(data))

train <- sample(seq_len(nrow(data)), size = smp_size)
test <- c(1:nrow(data))[-train]
dtrain <- xgb.DMatrix(d1[train,], label = data$fortytwonum[train])
dvalid <- xgb.DMatrix(d1[test,], label = data$fortytwonum[test])
watch <- list(train = dtrain, test = dvalid)

m2 <- xgb.train(data = dtrain, booster = "gbtree", nrounds = 3000, eta = 0.02, 
                max.depth = 5, colsample_bytree = 0.9, min_child_weight = 1.5,
                subsample = 0.95, objective = "binary:logistic",
                watchlist = watch, early_stopping_rounds=50,print_every_n=10)

xpred <- predict(m2, dvalid)
xpreds = cbind(data[test, ], xpred)

for (i in seq(.4, .6, .01)) {
  xpreds2$predicted42 = "no"
  xpreds2$predicted42[which(xpreds2$xpred <  i)] = "yes"
  confus = table(xpreds2$fortytwo, xpreds2$predicted42)
  print(cbind(sum(diag(confus))/sum(confus), i))
}

# The classification with wine prices had a similar accuracy to then linear
# regression predictions accuracy. It was about 80%

wine = fread("data.csv")
wine = wine[ , -c("description")]
wine = as.data.frame(wine)
wine = wine[!is.na(wine$price), ]

getYear = function(string) {
  newstring = sub("^.*([0-9]{4}).*", "\\1", string)
  if (nchar(newstring) != 4) return(0)
  return(newstring)
}

wine$year = lapply(wine$title, getYear)
wine$year = as.numeric(wine$year)

wine$province  = as.factor(wine$province)
levels(wine$province)[table(wine$province) < 300] <- "other"

# Question 3: Model Building

# Build a model to predict the price of a bottle of wine using columns except the "description" column. For each of the tasks below, the choice of what model to use and how to evaluate the model is up to you. 

# A)	Regression: Predict a wine's price using any column except for the "description" column.
# B)	Classification: Predict whether a wine costs more than 42 dollars any column except for the "description" column.

# There were too many different wineries listed so I dropped this column. There
# were also too many different provinces so I ended up dropping any value that
# didn't show up at least 300 times

# 'Taster twitter handle overlapped taster name'. 'Title' was dropped - it
# seems like only year would be important. 'Province' was informative enough to
# drop country, designation, region_1 and region_2.

drops = c("country", "designation", "region_1", "region_2", "taster_twitter_handle", "title",  "V1", "winery") 
wine = wine[ , !(colnames(wine) %in% drops)]

wine$taster_name[which(wine$taster_name == "")] = "unknown"
wine$points = as.numeric(wine$points)
wine$taster_name = as.factor(wine$taster_name)
wine$variety = as.factor(wine$variety)
wine$year = as.factor(wine$year)

wine$logprice = log(wine$price)
wine = wine[,  c(2, 7, 1, 3, 4, 5, 6)]

dummies = dummyVars(~ taster_name + variety + year + province , data = wine) 
wine2 = cbind(wine, predict(dummies, newdata = wine))

drops = c("taster_name", "variety", "year", "province")
wine2 = wine2[ , !(colnames(wine2) %in% drops)]

params = list()
params$objective = "reg:linear"
params$gamma = .02
params$booster = "gbtree"
params$eta = .05
params$colsample_bytree = .9
params$max_depth = 5
params$eval_metric = "rmse"
params$min_child_weight = 1.5
params$colsample_bytree = 0.9

train_ind <- sample(seq_len(nrow(wine2)), size = floor(0.7 * nrow(wine2)))

wine_train = wine2[train_ind, ]
wine_test = wine2[-train_ind, ]

dtrain <- xgb.DMatrix(as.matrix(wine_train[ , 3:940]),label=wine_train$logprice,missing=NA)
dtest <- xgb.DMatrix(as.matrix(wine_test[ ,3:940]),label=wine_test$logprice,missing=NA)
watchlist <- list(train = dtrain, eval = dtest)


XGB<-xgb.train( params=params,nrounds=3500,missing=NA,data=dtrain,
                watchlist,
                early_stopping_rounds=50,print_every_n=5, nthreads = -1)

xpred <- predict(XGB, dtest)
xpreds = cbind(wine_test[, 1:3], xpred)
xpreds$predprice = exp(xpreds$xpred)

xpreds$over42 = "yes"
xpreds$over42[ which(xpreds$price < 43)] = "no"

for (i in 30:70) {
  xpreds2$predicted42 = "no"
  xpreds2$predicted42[which(xpreds$predprice >  i)] = "yes"
  confus = table(xpreds2$predicted42, xpreds2$over42)
  print(cbind(sum(diag(confus))/sum(confus), i))
}

# Using a cutoff of 43 ended up being the optimal cutoff to find whether a wine
# is cheaper or equal to $42 It had about 85% accuracy

# Question 4: Discussion -	My colleague asserts that the writers of these
# reviews are biased against wines from Spain, in the sense their written
# reviews make the wines sound cheaper than they really are. Do you agree or
# disagree with this conclusion and why?

check = wine$V1[which(wine$country == "Spain")]

spainwines = xpreds[ xpreds$V1 %in% check, ]
spainwines$overprice = spainwines$price - spainwines$predprice

mean(spainwines$overprice) #4.743671
mean(spainwines$overprice[which(spainwines$price < 66)]) #-1.047933

# For all Spanish wines, prices predicted from descriptions are on average $4.70
# lower than the actual price. As my model has difficulty predicting the prices
# of very expensive wines; over $100, the prices are very different. Since the
# price values in the dataset are very skewed with most wines being in the $20
# range and small number of wines being extremely expensive in the over $2000
# range, the model was much better at predicting prices between $4 and about
# $66. For these lower-priced Spanish wines, actual prices of the wines are on
# average one dollar LOWER than the predicted price.

notspainwines = xpreds[ !xpreds$V1 %in% check, ]
notspainwines$diffprice = notspainwines$price - notspainwines$predprice
mean(notspainwines$diffprice) #5.354673
mean(notspainwines$diffprice[which(notspainwines$price < 66)]) #-0.2852838

# Looking at the rest of the test set, predicted prices are on average $5.30
# lower than actual prices, but when i look only at wines priced under $66
# dollars, predictions are on average slightly more than the actual price.


# We can go futher by using Spain as the holdout and make a model with the rest
# of the data and then see if the predictions are higher or lower on average

data = cbind(data, wine$country)

train <- which(data$V2 != "Spain")
test <- c(1:nrow(data))[-train]
dtrain <- xgb.DMatrix(d1[train,], label = data$logprice[train])
dvalid <- xgb.DMatrix(d1[test,], label = data$logprice[test])
watch <- list(train = dtrain, test = dvalid)

m2 <- xgb.train(data = dtrain, booster = "gbtree", nrounds = 1500, eta = 0.02, 
                max.depth = 5, colsample_bytree = 0.9, min_child_weight = 1.5,
                subsample = 0.95, objective = "reg:linear",
                watchlist = watch, early_stopping_rounds=50,print_every_n=10)

xpred <- predict(m2, dvalid)
xpreds = cbind(data[test, ], xpred)
xpreds$predprice = exp(xpreds$xpred)
xpreds$pricediff = xpreds$price - xpreds$predprice

mean(xpreds$pricediff) #5.835722
mean(xpreds$pricediff[which(xpreds$price < 66)]) #-0.838568

# When using description as a predictor, Spanish wines have a higher predicted
# price for medium to cheap wines and a lower predicted price for expensive
# wines. However, almost all high priced wines will tend to have a significantly
# lower predicted price than their actual price.
