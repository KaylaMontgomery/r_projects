# This project was for a skill assessment performed for a company that
# originates consumer unsecured loans. It involved analyzing a dataset
# consisting of two months of position records for loans used to provide retail
# purchase financing and answering questions (which are included inline as
# comments in the form "Q1." "Q2.", etc.) The dataset included loan origination
# characteristics (like state, loan purpose, etc) and payment/balance
# information.

library(dplyr)
library(data.table)
library(ggplot2)
library(tidyr)
library(lubridate)
library(devtools)
library(xgboost)
library(caret)

# note that one of the columns is misspelled  
data <- fread("client_data.csv") %>%
  rename(initial_loan_amount = intial_loan_amount)

# Q1a. How many distinct loans are there in the dataset?
data %>% distinct(loan_account_id) %>% summarise("distinct_loans" = n())

# Q1b. What is the total initial loan amount vs total outstanding principal at
# the end of May vs June?

data %>%
  group_by("month end" = as_of_date) %>%
  summarise(
    "total_initial_loan_amount" = sum(initial_loan_amount),
    "total_principal_outstanding" = sum(principal_outstanding)
  )

# Q2. What is the distribution of payment standing at month end and do you
# observe any notable changes from May to June?

payment_levels <-  c("PAID", "CURRENT", "30", "60", "90", "120", "CHARGED_OFF")

data$payment_standing <- factor(data$payment_standing,
                                levels = payment_levels)

data %>%
  ggplot +
  aes(payment_standing) +
  geom_bar() +
  facet_grid( ~ as_of_date)

# The most notable change is the increase in Current loans, but there are a lot
# of new originations too (see Q4).

# Q3. Please produce a transition matrix from May to June by payment standing by
# 1) number of loans and 2) outstanding principal dollars.

transition <- data %>%
  filter(as_of_date == "5/31/18") %>%
  merge((data %>% filter(as_of_date == "6/30/18")), by = "loan_account_id") %>%
  select(
    payment_standing.x,
    payment_standing.y,
    principal_outstanding.x,
    principal_outstanding.y
  ) %>%
  group_by("may_pmnt_standing" = payment_standing.x,
           "jun_pmnt_standing" = payment_standing.y) %>%
  summarise(
    "count" = n(),
    "may_outst_principal" = sum(principal_outstanding.x),
    "jun_outst_principal" = sum(principal_outstanding.y)
  )

View(transition)

# Q4. Let's say our CEO just came over to your desk and asked you how the
# profile and performance of our loan portfolio changed from May to June. Please
# show and explain the data that you would present to him.

# My response:

# I'm sure the CEO is interested in new originations - these are loans that
# started during June.

data %>%
  filter(as_of_date == "6/30/18") %>%
  mutate(loan_start_date = mdy(loan_start_date)) %>%
  filter(loan_start_date > "2018-05-31") %>%
  summarize("new_loans" = n(),
            "total_originated" = sum(initial_loan_amount))

# As a financing deal originator, the company needs access to capital markets to
# extend new loans. To do so it borrows against its portfolio of financing deals
# (which are essentially consumer unsecured loans). Institutional lenders
# impose eligibility criteria on these loans when extending credit and may not
# permit the company to borrow against loans that are too new, have borrowers that
# are too far behind on payments, etc.

# Institutional lenders may be leery of using new consumer unsecured loans as
# collateral. Thes loans in this dataset are especially hazardous - consumers
# who require financing typically don't have access to credit. It's considered
# an especially ominous sign to miss the FIRST payment - this is called a "first
# payment default".

data %>%
  filter(as_of_date == "6/30/18") %>%
  mutate(loan_start_date = mdy(loan_start_date)) %>%
  filter(loan_start_date > "2018-05-01", payment_standing == "30") %>%
  summarize("first_payment_default_count" = n(),
            "first_payment_default_amount" = sum(initial_loan_amount))

# Q5. (bonus question) What if anything are you able to find as explanatory
# factors to worsening payment standing?

# My process:

# Make a ranked lookup table for payment standing. If rank increases, that
# indicates worsening standing.

pLut <- c(
  "PAID" = 1,
  "CURRENT" = 2,
  "30" = 3,
  "60" = 4,
  "90" = 5,
  "120" = 6,
  "CHARGED_OFF" = 7
)

# State may be a predictor. Heal holes in 'shipping state' data. While I
# did most of the exercise with dplyr syntax, this particular operation is
# easier for me to implement with my native data.table syntax.

predict_table <- data

predict_table$shipping_state[which(predict_table$shipping_state.x == "")] <-
  predict_table$billing_state[which(predict_table$shipping_state == "")]

# Create a table for predictors and implement some predictors based on changes
# between May and June.

predict_table <- predict_table %>%
  filter(as_of_date == "5/31/18") %>%
  inner_join((data %>% filter(as_of_date == "6/30/18")), by = "loan_account_id") %>%
  mutate(
    ps.x = pLut[payment_standing.x],
    ps.y = pLut[payment_standing.y],
    worsened = ps.y > ps.x,
    b_state_same_as_s_state = billing_state.x == shipping_state.x,
    b_state_changed = billing_state.x == billing_state.y,
    s_state_changed = shipping_state.x == shipping_state.y,
    utilization_changed = utilization.x - utilization.y
  ) %>%
  select(
    loan_account_id,
    worsened,
    billing_state.x,
    shipping_state.x,
    b_state_same_as_s_state,
    b_state_changed,
    s_state_changed,
    utilization_changed,
    utilization.x,
    principal_outstanding.x,
    days_past_due.x,
    term.x,
    time_on_bureau.x,
    check_status.x,
    loan_reason = V14.x
  )

data = predict_table

predict_table$worse = predict_table$worsened*1

mylogit <- glm(worse ~ b_state_same_as_s_state + utilization_changed  + utilization.x + 
                 principal_outstanding.x + days_past_due.x + term.x + time_on_bureau.x + check_status.x + 
                 loan_reason,
               data = predict_table, family = "binomial")

# 0 means not worsened, 1 means worsened 

summary(mylogit)
data = predict_table

data$principal_outstanding.x = as.numeric(data$principal_outstanding.x)
data$days_past_due.x = as.numeric(data$days_past_due.x)
data$term.x = as.numeric(data$term.x)
data$time_on_bureau.x = as.numeric(data$time_on_bureau.x)

drops = c("loan_account_id", "worsened", "b_state_changed", "s_state_changed", "billing_state", "shipping_state", "billing_state.x", "shipping_state.x")
data = data[ , !(colnames(data) %in% drops)]
data$b_state_same_as_s_state = as.character(data$b_state_same_as_s_state)
data$check_status.x = as.character(data$check_status.x)


dummies = dummyVars(~ b_state_same_as_s_state + check_status.x + loan_reason, data = data) 
data2 = cbind(data, predict(dummies, newdata = data))

drops = c("b_state_same_as_s_state", "check_status.x" ,"loan_reason")
data2 = data2[ , !(colnames(data2) %in% drops)]

params = list()
params$objective = "binary:logistic"
params$gamma = .02
params$booster = "gbtree"
params$eta = .05
params$colsample_bytree = .9
params$max_depth = 5
params$eval_metric = "rmse"
params$min_child_weight = 1.5
params$colsample_bytree = 0.9

train_ind <- sample(seq_len(nrow(data2)), size = floor(0.7 * nrow(data2)))

data2 = data2%>%select(-worse,worse)

loan_train = data2[train_ind, ]
loan_test = data2[-train_ind, ]

dtrain <- xgb.DMatrix(as.matrix(loan_train[ , 1:22]),label=loan_train$worse,missing=NA)
dtest <- xgb.DMatrix(as.matrix(loan_test[ ,1:22]),label=loan_test$worse,missing=NA)
watchlist <- list(train = dtrain, eval = dtest)

XGB<-xgb.train( params=params,nrounds=3500,missing=NA,data=dtrain,
                watchlist,
                early_stopping_rounds=50,print_every_n=5, nthreads = -1)

importance_matrix <- xgb.importance(colnames(loan_train), model = XGB)

xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")

# I implemented a logistic regression model to determine predictors of worsening
# status during a month. I then implemented an xgboost model to find out the relative  
# importance of the predictors. 
#
# By far the most important predictor of worsening status is the duration that a
# delinquent loan has already been outstanding at the start of the month.
#
# Higher outstanding principals are also correlated with worsening status.
#
# Credit is important. High credit utilization and changes in credit utilization
# during the month are associated with worsening loan standing. A predictor of
# standing NOT worsening is the borrower's length of time on the bureau (credit age).
# 
# Loans with longer terms are more likely to worsen.
#
# I picked "accessories" as a baseline to compare other loan reasons to.
# Compared to these loans, loans taken out for hobbies, home or other reasons
# are more likely to worsen. Loans taken out for sports, kitchen, mattress,
# DIY or Gadgets are less likely to worsen
