# This was a data analysis task. The analysis questions are given in the form "Q1", "Q2", etc.

library(data.table)
health = read.csv("data.csv")
nls = read.csv("data2.csv")

# Q1a. Check if there are any missing values in this data set. If there is/are,
# which variable(s) and what percentage of missing values are there?

View(colMeans(is.na(nls)))

# Q1b. This is a panel data set, and you want to use it as a cross-sectional
# data set. Drop any duplicated, old records by person. What is the sample size
# of this cross-sectional data?

attach(nls)
nls =  nls[order(idcode, year),]
detach(nls)
cross = nls[!rev(duplicated(rev(nls$idcode))),]
dim(cross)

# Sample size is 4711

# Q1c. What is the average ln(wage) of those aged between 20 and 40?

mean(cross$ln_wage[which(cross$age > 20 & cross$age < 40)])

# 1.745295

# Q1d. What is the variable type of race? Recode it properly.

class(cross$race)

# Categorical/factor

cross$race = as.factor(as.numeric(cross$race))

# Q1e. What is the relationship between total work experience and ln(wage)?
# Visualize the relationship.

library(ggplot2)

ggplot(cross, aes(x=ttl_exp, y=ln_wage)) +
  geom_point()

# It seems to be a positive correlation. As total work experience increase, wage
# increases.

# Q1f. How much does ln(wage) increase (or decrease) as a year of total work
# experience increases, when the respondent's race is controlled?

model <- lm(ln_wage ~ ttl_exp + race, data=cross)
summary(model)

# For each year increase in total work experience, ln(wage) is increased by
# 0.043095


# Q1g. Re-open nls_women_from14to26_1968.csv. Drop all of the variables other
# than idcode, year and ln_wage.  Reshape the data format from long to wide.
# Report the average ln(wage) every year between 1968 and 1988. Visualize the
# trend.

drops <- c("idcode", "year",  "ln_wage")
nls = nls[ , (names(nls) %in% drops)]
nls = reshape(nls, idvar = "idcode", timevar = "year", direction = "wide")
nls = nls[ , order(names(nls))]
means = colMeans(is.na(nls[, 2:16]))
View(means)

means = as.data.frame(means)
means$year = names(nls[2:16])
means$year = substr(means$year, 9, 10)
means$year = as.numeric(means$year)

ggplot(data = means, aes(x = year, y = means)) + geom_line(color = "blue") + geom_point()

# Q2. "Health_providers.csv" is the results of a health provider survey. In that
# survey, each patient was asked to report up to three health providers. You can
# see that the responses from the patients are pretty diverse. A single provider
# could be reported as multiple different names. For example, Gotham OBGYN could
# be reported as Gatham/gathom/G0tham/Getham/Gutham/Agotham OBGYN. Well, our
# patients were very cra.. creative. Please think of a way to help us get a list
# of refined final health providers, in which multiple similar raw provider
# names can be identified as a single final provider. You are free to use any
# approach you prefer, including manual editing, but some unsupervised learning
# approach might help. Turn in your program and two tables. The first table
# should contain final provider names (you decide the names) and the raw names
# under them. The second table (something like pic 3.2) should show which
# patient reported which provider(s). You are not expected to achieve a perfect
# classification result, and have the flexibility to decide whether two
# different names belong to a single provider.

# After performing some preliminary cleaning, I used unsupervised learning to
# classify providers using the string distance method.

health = read.csv("./Resume-cover letter sop/Interview Projects/Health_providers.csv")
library(dplyr)
health <- mutate_all(health, .funs=tolower)

health$Reported_provider2[which(health$Reported_provider2 == "same")] = health$Reported_provider1[which(health$Reported_provider2 == "same") ]
health$Reported_provider3[which(health$Reported_provider3 == "same")] = health$Reported_provider3[which(health$Reported_provider3 == "same") ]

navalues= c("0", "na", "n/a", "none", "dont know", "", "n", "dont know", "no", ".dont knew", "don't know", "don't knew", "dont known")
health <-  health %>%
  mutate(Reported_provider1 = replace(Reported_provider1, Reported_provider1 %in% navalues, NA),
         Reported_provider2 = replace(Reported_provider2, Reported_provider2 %in% navalues, NA),
         Reported_provider3 = replace(Reported_provider3, Reported_provider3 %in% navalues, NA)
         )

health = health[, 2:4]
health2 = data.frame(newcol = c(t(health)), stringsAsFactors=FALSE)

health2 = na.omit(health2)

library(stringdist)

uniquemodels <- unique(as.character(health2$newcol))

distancemodels <- stringdistmatrix(uniquemodels, uniquemodels, method = "jw")
rownames(distancemodels) <- uniquemodels

hc <- hclust(as.dist(distancemodels))
dfClust <- data.frame(uniquemodels, cutree(hc, k=40))
names(dfClust) <- c('modelname','cluster')

t <- table(dfClust$cluster)
t <- cbind(t,t/length(dfClust$cluster))
t <- t[order(t[,2], decreasing=TRUE),]
p <- data.frame(factorName=rownames(t), binCount=t[,1], percentFound=t[,2])

dfClust <- merge(x=dfClust, y=p, by.x = 'cluster', by.y='factorName', all.x=T)
dfClust <- dfClust[rev(order(dfClust$binCount)),]

names(dfClust) <-  c('cluster','modelname')
clustered = dfClust[c('cluster','modelname')]

View(clustered)

attach(clustered)
clustered =  clustered[order(cluster),]
detach(clustered)

first <- clustered[match(unique(clustered$cluster), clustered$cluster),]
first = first$modelname

clustered$final_name = first[clustered$cluster]

library(plyr)
data = ddply(clustered, .(cluster, final_name), summarize, modelname = toString(modelname))
data$modelname = sapply(data$modelname, gsub, pattern=",", replacement=" |") 
names(data) = c("Group_id", "Final_provider_name", "All_raw_names")

write.csv(data, "health_text.csv")
