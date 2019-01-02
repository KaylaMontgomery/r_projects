# This is  a two-part project using the National Consumer Complaint database.
# (https://data.consumerfinance.gov/api/views/s6ew-h6mp/rows.csv?accessType=DOWNLOAD)

# In the first part, I create a visualization of consumer complaint resolution
# pathways for the most common issues.

# In the second part of the project, I implement an XGBoost model to predict
# company response to consumer complain narratives.

## PART 1: VISUALIZATION
# This creates a littoral diagram. A littoral diagral is a Sankey diagram where
# flows can rejoin. networkD3 offers a subset of D3's network graph
# functionality within R for quick implementation.

library(data.table)
library(dplyr)
library(purrr)
library(igraph)
library(networkD3)
library(FeatureHashing)
library(Matrix)
library(xgboost)

complaints <- fread("data.csv")

# there are >100 issues. For simplicity, only show the top 10
top_issues <- complaints %>%
  group_by(`Issue`) %>%
  summarize(count = n_distinct(`Complaint ID`)) %>%
  arrange(desc(count)) %>%
  top_n(10)

# preprocess the data
complaints_f <- complaints %>%
  filter(`Issue` %in% top_issues$Issue) %>%
  mutate(`Company public response` = ifelse(`Company public response` == '',
                                             'No public response reported',
                                             `Company public response`),
         `Company response to consumer` = ifelse(`Company response to consumer` == '',
                                                  'No response to consumer reported',
                                                  `Company response to consumer`),
         `Consumer disputed?` = recode(`Consumer disputed?`, No = "No consumer dispute", Yes = "Consumer dispute", .default = "Dispute status unknown")
         )
  
# custom function to generate links
generate_links <- function(link1, link2, data){
  data %>%
    group_by(!!as.name(link1), !!as.name(link2)) %>%
    summarize(value = n_distinct(`Complaint ID`)) %>%
    rename(source_name = !!as.name(link1), target_name = !!as.name(link2))
}

nodes1 <- c("Issue", "Company public response", "Company response to consumer")
nodes2 <- c("Company public response", "Company response to consumer", "Consumer disputed?")

links_data <- map2(nodes1, nodes2, generate_links, data = complaints_f) %>% bind_rows %>% ungroup

# networkD3 requires us to pass in nodes as numbers, so we must recode

nodes_table <- data.table(name = unique(c(links_data$source_name, links_data$target_name)), number = 0:32)

links_data <- links_data %>%
  left_join(nodes_table, by = c("source_name" = "name")) %>%
  rename(source = number) %>%
  left_join(nodes_table, by = c("target_name" = "name")) %>%
  rename(target = number) %>%
  select(source, target, value)
  

# plot using the sankeyNetwork function from networkD3

sankeyNetwork(Links = links_data, Nodes = nodes_table, Source = 'source',
              Target = 'target', Value = 'value', NodeID = 'name', NodeGroup = NULL, fontSize = 10) 

# PART 2: PREDICTIVE MODEL
# In the second part of the project, I implement an XGBoost model to predict
# company response to consumer complain narratives based on the text of the
# complaint. I use a bag of words model for the text.

set.seed(123)
data <- fread("data.csv")
data2 <- data[which(data$`Consumer complaint narrative` != ""),  ]
data2 <- data2[which(data2$`Company response to consumer` != ""), ]

data <- data2[ , c(6, 15)]

names(data) = c("complaint", "response")

data$response2 = as.factor(data$response)
data$response3 = as.numeric(data$response2)

max <- max(data$response3)
data$response3[which(data$response3 == max)] = 0

# use a hashed model for bag of words because it's computationally more efficient 
d1 <- hashed.model.matrix(~ split(complaint, delim = " ", type = "tf-idf"),
                          data = data, hash.size = 2^16, signed.hash = FALSE)


smp_size <- floor(0.7 * nrow(data))

train <- sample(seq_len(nrow(data)), size = smp_size)
test <- c(1:nrow(data))[-train]
dtrain <- xgb.DMatrix(d1[train,], label = data$response3[train])
dvalid <- xgb.DMatrix(d1[test,], label = data$response3[test])
watch <- list(train = dtrain, test = dvalid)

param <- list(  objective           = "multi:softmax",
                gamma               = 0.02,
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.05,
                max_depth           = 3,
                subsample           = 0.9,
                colsample_bytree    = 0.5, 
                num_class           = max
                )

watchlist <- list(train = dtrain, eval = dvalid)

m2 <- xgb.train(
        params = param,
        nrounds = 100,
        missing = NA,
        data = dtrain,
        watchlist,
        early_stopping_rounds = 50,
        print_every_n = 5,
        nthreads = -1
      )

xpred <- predict(m2, dvalid)

xpreds <- matrix(xpred, nrow = (length(xpred)/5) ,ncol = 5, byrow = T)
xpreds <- cbind(data[test, ], xpred)

table(xpreds$response3, xpreds$xpred)