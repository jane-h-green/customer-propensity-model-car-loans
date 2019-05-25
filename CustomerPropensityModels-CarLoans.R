####################################################################
#    Customer Propensity Models - Line for Credit for Car Purchase #
####################################################################

# Author: Jane Nikolova
# Occupation: Senior Consultant
# All Rights Reserved.
# Date: December, 2018

# Models & Data Mining Work completed as part of a Case Study in a Data Science Course (part of Harvard DS graduate program);
# Course: Data Science for Business - Harvard Extension School - Harvard University.

# Summary of methods & content:

# @ Basic EDA - Basic data exploration.
# @ Decision Trees - Using decision trees to evaluate the factors affecting propensity of a consumer to buy a car.
# @ Regression - Using a regression to evaluate the factors affecting propensity of a consumer to buy a car.
# @ Random Forest - Using random forest to evaluate the factors affecting propensity of a consumer to buy a car.
# @ Neural Networks - Using NNs to evaluate the factors affecting propensity of a consumer to buy a car.
# @ Evaluating the performance of all methods.
# @ K-Means - Cluster Analysis - Defining clustering of consumers.

####################################
# Importing all relevant libraries :
####################################

library(caret)
library(corrplot)
library(cowplot)
library(DataExplorer)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(ggthemes) 
library(knitr)
library(lubridate)
library(maps)
library(MLmetrics)
library(neuralnet, nnet)
library(parallel)
library(party)
library(partykit)
library(purrr)
library(plotly)
library(reshape)
library(radiant.data)
library(RColorBrewer)
library(pROC)
library(readr) 
library(rpart.plot) 
library(randomForest)
library(scales)
library(stringr) 
library(tidyr)
library(tidyselect)
library(viridis)
library(vtreat)

########################################################################
# Load all historical and supplemental data and examine column names : #
########################################################################

setwd("")# Set to the correct local working directory -
mktg <- read.csv('CurrentCustomerMktgResults.csv')
axiom <- read.csv('householdAxiomData.csv')
credit <- read.csv('householdCreditData.csv')
car <- read.csv('householdVehicleData.csv')
#
colnames(mktg)
colnames(axiom)
colnames(credit)
colnames(car)

################################################################
# Merge all datasets by ID, which corresponds to "HHuniqueID": #
################################################################

dataset <- merge(mktg, axiom, by = "HHuniqueID", match = "left")
dataset <- merge(dataset, credit, by = "HHuniqueID", match = "left")
dataset <- merge(dataset, car, by = "HHuniqueID", match = "left")
dataset <- subset(dataset, select=-c(HHuniqueID))
dataset <- subset(dataset, select=-c(dataID))
res <- hms(dataset$CallStart) # format to 'hours:minutes:seconds'
dataset$CallStart <- hour(res)*60 + minute(res) 
res <- hms(dataset$CallEnd) # format to 'hours:minutes:seconds'
dataset$CallEnd <- hour(res)*60 + minute(res) 
y <- as.factor(dataset$Y_AccetpedOffer)
dataset$Y_AccetpedOffer <- y
class(dataset$Y_AccetpedOffer)
colnames(dataset)


################################################################
#                       - Basic EDA -                          #
################################################################

# overall structure  & dimensions of the data -
str(dataset)
dim(dataset)
# data set class - 
class(dataset)
# classes for each column - 
sapply(dataset, class)
# look at the top 6 rows - 
head(dataset)
# levels -  
nlevels(dataset)
# column names - 
names(dataset)
# summary stats for each vector - 
summary(dataset)

plot_str(dataset)
plot_missing(dataset)
plot_histogram(dataset) 
plot_density(dataset) 
plot_bar(dataset)

####################################################################################
# Create and group dummy variables; examine outliers and highly correlated values. # 
####################################################################################

# Check for outliers (numerical predictors) - removing values that may be skewing the data trends - 
boxplot(dataset$NoOfContacts)#maybe we can cu the values above 40.
boxplot(as.numeric(dataset$annualDonations))#values above 30 are too far, may be userful to cut out.
boxplot(dataset$DaysPassed)#days above 600 seem like outliers
boxplot(dataset$Age)#age above 80 can be dropped
boxplot(dataset$PrevAttempts)#can drop above 20
boxplot(dataset$RecentBalance)#can drop above 60000

# Using more general approach to pre-process these columns - Capping the outliers to the lower and upper bound of
# the interval 0.025 and 0.975
dataset$NoOfContacts <- squish(dataset$NoOfContacts, quantile(dataset$NoOfContacts, c(0.025, 0.975)))
dataset$annualDonations <- squish(as.numeric(dataset$annualDonations), quantile(as.numeric(dataset$annualDonations), c(0.025, 0.975)))
dataset$DaysPassed <- squish(dataset$DaysPassed, quantile(dataset$DaysPassed, c(0.025, 0.975)))
dataset$Age <- squish(dataset$Age, quantile(dataset$Age, c(0.025, 0.975)))
dataset$PrevAttempts <- squish(dataset$PrevAttempts, quantile(dataset$PrevAttempts, c(0.025, 0.975)))
dataset$RecentBalance <- squish(dataset$RecentBalance, quantile(dataset$RecentBalance, c(0.025, 0.975)))


# Up to here we can create a function to load and pre-process the original training data set -
# Can be used to quickly reload the dataset and apply first steps of preprocessing. 
standardizedTrainigDataSet <- function(){
  setwd("")//# Set to the correct local working directory -
  mktg <- read.csv('CurrentCustomerMktgResults.csv')
  axiom <- read.csv('householdAxiomData.csv')
  credit <- read.csv('householdCreditData.csv')
  car <- read.csv('householdVehicleData.csv')
  # Joining the data - 
  data <- merge(mktg, axiom, by = "HHuniqueID", match = "left")
  data <- merge(data, credit, by = "HHuniqueID", match = "left")
  data <- merge(data, car, by = "HHuniqueID", match = "left")
  # Removing columns with no relevant info - 
  data <- subset(data, select=-c(HHuniqueID))
  data <- subset(data, select=-c(dataID))
  # Joining the data - 
  y <- as.factor(data$Y_AccetpedOffer)
  data$Y_AccetpedOffer <- y
  #
  res <- hms(data$CallStart) # format to 'hours:minutes:seconds'
  data$CallStart <- hour(res)*60 + minute(res) 
  res <- hms(data$CallEnd) # format to 'hours:minutes:seconds'
  data$CallEnd <- hour(res)*60 + minute(res) 
  #
  #Add column for call duration
  data$CallDuration <- (data$CallEnd - data$CallStart)
  #
  data$NoOfContacts <- squish(data$NoOfContacts, quantile(data$NoOfContacts, c(0.025, 0.975)))
  data$annualDonations <- squish(as.numeric(data$annualDonations), quantile(as.numeric(data$annualDonations), c(0.025, 0.975)))
  data$DaysPassed <- squish(data$DaysPassed, quantile(data$DaysPassed, c(0.025, 0.975)))
  data$Age <- squish(data$Age, quantile(data$Age, c(0.025, 0.975)))
  data$PrevAttempts <- squish(data$PrevAttempts, quantile(data$PrevAttempts, c(0.025, 0.975)))
  data$RecentBalance <- squish(data$RecentBalance, quantile(data$RecentBalance, c(0.025, 0.975)))
  
  return(data)
}

# We would like to remove call start and call end perdictors - call start does not vary for rejected and accepted offers;
# Naturally long calls signal customer interest and chances the offer will be accpeted so this information is of no
# value and can be removed; If we were about to notice a difference in the call-start for accepted and rejected
# offers we could have included a recommended start time for a call for the future prospects:

dataset <- standardizedTrainigDataSet()
only.accpeted.offers <- subset(dataset, dataset$Y_AccetpedOffer == 1)
only.rejected.offers <- subset(dataset, dataset$Y_AccetpedOffer == 0)

median(only.accpeted.offers$CallStart)#812
mean(only.accpeted.offers$CallStart)#812.0368
mean(only.accpeted.offers$CallEnd)#821.2438
mean(only.accpeted.offers$CallDuration)#9.206983
range(only.accpeted.offers$CallDuration)#0 to 54
range(only.accpeted.offers$CallStart)#540 1079

#Calls start at approximately the same time. 
median(only.rejected.offers$CallStart)#817
mean(only.rejected.offers$CallStart)#811.2796
mean(only.rejected.offers$CallEnd)#814.8944
mean(only.rejected.offers$CallDuration)#3.614775 
range(only.rejected.offers$CallDuration)#0 to 51
range(only.rejected.offers$CallStart)#540 1079

hist(only.rejected.offers$CallStart)
hist(only.accpeted.offers$CallStart)

hist(only.rejected.offers$CallDuration)
hist(only.accpeted.offers$CallDuration)# Accecpted offers are naturally associated with long calls. 


standardizedTrainigDataSetNoTime <- function(){
  setwd("")
  mktg <- read.csv('CurrentCustomerMktgResults.csv')
  axiom <- read.csv('householdAxiomData.csv')
  credit <- read.csv('householdCreditData.csv')
  car <- read.csv('householdVehicleData.csv')
  # Joining the data - 
  data <- merge(mktg, axiom, by = "HHuniqueID", match = "left")
  data <- merge(data, credit, by = "HHuniqueID", match = "left")
  data <- merge(data, car, by = "HHuniqueID", match = "left")
  # Removing columns with no relevant info - 
  data <- subset(data, select=-c(HHuniqueID))
  data <- subset(data, select=-c(dataID))
  # Joining the data - 
  y <- as.factor(data$Y_AccetpedOffer)
  data$Y_AccetpedOffer <- y
  #
  data <- subset(data, select=-c(CallStart))
  data <- subset(data, select=-c(CallEnd))
  #
  data$NoOfContacts <- squish(data$NoOfContacts, quantile(data$NoOfContacts, c(0.025, 0.975)))
  data$annualDonations <- squish(as.numeric(data$annualDonations), quantile(as.numeric(data$annualDonations), c(0.025, 0.975)))
  data$DaysPassed <- squish(data$DaysPassed, quantile(data$DaysPassed, c(0.025, 0.975)))
  data$Age <- squish(data$Age, quantile(data$Age, c(0.025, 0.975)))
  data$PrevAttempts <- squish(data$PrevAttempts, quantile(data$PrevAttempts, c(0.025, 0.975)))
  data$RecentBalance <- squish(data$RecentBalance, quantile(data$RecentBalance, c(0.025, 0.975)))
  
  #interactive vars:
  data$BalanceAge <- (data$RecentBalance * data$Age)
  data$ContactsAge <- (data$NoOfContacts * data$Age)
  data$ContactsLastDay <- (data$NoOfContacts * data$LastContactDay)
  
  return(data)
}

# Creating dummies, i.e. treating the data with automatic features' engineering - 

# Automated variable processing
# for **categorical** outcomes 

# We are trying to predict a binary outcome 'offer accepted / rejectected' - 
# or column "Y_AccetpedOffer". 
plan <- designTreatmentsC(dataset, 
                          names(dataset),
                          'Y_AccetpedOffer', 
                          1)

# Apply the plan
treated.dataset <- prepare(plan, dataset)
#summary(treated.dataset)
#colnames(treated.dataset)

# All missing values are treated & all dummy varibales are included - 
plot_missing(treated.dataset)
plot_histogram(treated.dataset) 


# A function to easily split and treat the data - 
treatedSets <- function(split=0.70, data){
  splitPercent <- round(nrow(data) %*% split)
  totalRecords <- 1:nrow(data)
  idx <- sample(totalRecords, splitPercent)
  trainDat <- data[idx,]
  testDat  <- data[-idx,]
  options(scipen=999)
  set.seed(1234)
  plan         <- designTreatmentsC(trainDat, names(trainDat), 'Y_AccetpedOffer',1)
  treatedTrain <- prepare(plan, trainDat)
  treatedTest  <- prepare(plan, testDat)
  return(list(treatedTrain, treatedTest))
}

########################################################################
# Prediction models for binary outcomes - 
#
# 1. Random Forest - most powerful method (code in full detail)
#
# 2. Decision Tree - most competitive method (code in full detail)
#
# 3. Log regression - examinign for comparative purposes
#
# 4. Neural Networks - examining predictions using the insights derived in
#    the previous steps. 
#
#########################################################################

# Splitting the data - 70% training & 30% not validation sets - 
dataset <- standardizedTrainigDataSetNoTime()
splitPercent <- round(nrow(dataset) %*% 0.7)
totalRecords <- 1:nrow(dataset)
idx <- sample(totalRecords, splitPercent)
trainDat <- dataset[idx,]
testDat  <- dataset[-idx,]
options(scipen=999)
set.seed(1234)
plan         <- designTreatmentsC(trainDat, names(trainDat), 'Y_AccetpedOffer',1)
treatedTrain <- prepare(plan, trainDat)
treatedTest  <- prepare(plan, testDat)

#head(treatedTrain)
### 1. RANDOM FOREST ###

# Fit a random forest model with Caret
downSampleFit <- train(Y_AccetpedOffer ~ .,
                       data = treatedTrain,
                       method = "rf",
                       verbose = FALSE,
                       ntree = 5,tuneGrid = data.frame(mtry = 1))
downSampleFit
# Accuracy  Kappa    
# 0.66  0.20

predProbs   <- predict(downSampleFit,  treatedTrain, type = c("prob"))
predClasses <- predict(downSampleFit,  treatedTrain)

# Confusion Matrix; MLmetrics has the same function but use CARET!!
caret::confusionMatrix(predClasses, treatedTrain$Y_AccetpedOffer)

# Other interesting model artifacts
varImp(downSampleFit)
plot(varImp(downSampleFit), top = 20)

# Adding more trees to the forest with the randomForest package - 
moreTrees <-randomForest(Y_AccetpedOffer ~ .,
                         data = treatedTrain, 
                         ntree=1000)


# Confusion Matrix.
trainClass<-predict(moreTrees, treatedTrain)
confusionMatrix(trainClass, treatedTrain$Y_AccetpedOffer)

# Look at improved var importance
varImpPlot(moreTrees)
# plot the RF with a legend - 
layout(matrix(c(1,2),nrow=1),
       width=c(4,1)) 
par(mar=c(5,4,4,0)) #No margin on the right side
plot(moreTrees, log="y")
par(mar=c(5,0,4,2)) #No margin on the left side
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(moreTrees$err.rate),col=1:4,cex=0.8,fill=1:4)

# Checking if some optimization will help - 
optimizedTrees <-randomForest(Y_AccetpedOffer ~ .,data = treatedTrain, ntree=300)

# Confusion Matrix - 
trainClass<-predict(optimizedTrees, treatedTrain)
confusionMatrix(trainClass, treatedTrain$Y_AccetpedOffer)

### Now let's apply to the validation test set
threesPreds        <- predict(downSampleFit, treatedTest)
moreTreesPreds <- predict(moreTrees,    treatedTest)
optimizedTreesPreds  <- predict(optimizedTrees,    treatedTest)

# Accuracy Comparison from MLmetrics
Accuracy(treatedTest$Y_AccetpedOffer, threesPreds)
Accuracy(treatedTest$Y_AccetpedOffer, moreTreesPreds)
Accuracy(treatedTest$Y_AccetpedOffer, optimizedTreesPreds)#Highest accuracy - 

varImp(optimizedTrees)
varImpPlot(optimizedTrees)

# Some of the features are not rated high - we can try to prune them - 
dataset <- standardizedTrainigDataSetNoTime()
dataset <- subset(dataset, select=-c(headOfhouseholdGender))
dataset <- subset(dataset, select=-c(Marital))
dataset <- subset(dataset, select=-c(DaysPassed))
dataset <- subset(dataset, select=-c(PrevAttempts))
dataset <- subset(dataset, select=-c(annualDonations))
dataset <- subset(dataset, select=-c(PetsPurchases))
dataset <- subset(dataset, select=-c(DigitalHabits_5_AlwaysOn))
dataset <- subset(dataset, select=-c(AffluencePurchases))
dataset <- subset(dataset, select=-c(DefaultOnRecord))
dataset <- subset(dataset, select=-c(RecentBalance))
dataset <- subset(dataset, select=-c(CarLoan))

#head(dataset)

# Splitting the data - 80% training & 20% not validation sets - 
set.seed(12345)
splitPercent <- round(nrow(dataset) %*% .70)
totalRecords <- 1:nrow(dataset)
idx <- sample(totalRecords, splitPercent)

trainDat <- dataset[idx,]
testDat  <- dataset[-idx,]

plan <- designTreatmentsC(dataset, 
                          names(dataset),
                          'Y_AccetpedOffer', 
                          1)

# Apply the plan
treatedTrain <- prepare(plan, trainDat)
treatedTest <- prepare(plan, testDat)

# Fit a random forest model with Caret
downSampleFit <- train(Y_AccetpedOffer ~ .,
                       data = treatedTrain,
                       method = "rf",
                       verbose = FALSE,
                       ntree = 5,tuneGrid = data.frame(mtry = 1))
downSampleFit
#
predProbs   <- predict(downSampleFit,  treatedTrain, type = c("prob"))
predClasses <- predict(downSampleFit,  treatedTrain)

# Confusion Matrix; MLmetrics has the same function but use CARET!!
caret::confusionMatrix(predClasses, treatedTrain$Y_AccetpedOffer)

# Other interesting model artifacts
varImp(downSampleFit)
plot(varImp(downSampleFit), top = 20)

# Adding more trees to the forest with the randomForest package - 
moreTrees <-randomForest(Y_AccetpedOffer ~ .,
                         data = treatedTrain, 
                         ntree=1000)


# Confusion Matrix.
trainClass<-predict(moreTrees, treatedTrain)
confusionMatrix(trainClass, treatedTrain$Y_AccetpedOffer)

# Look at improved var importance
varImpPlot(moreTrees)
# plot the RF with a legend - 
layout(matrix(c(1,2),nrow=1),
       width=c(4,1)) 
par(mar=c(5,4,4,0)) #No margin on the right side
plot(moreTrees, log="y")
par(mar=c(5,0,4,2)) #No margin on the left side
plot(c(0,1),type="n", axes=F, xlab="", ylab="")
legend("top", colnames(moreTrees$err.rate),col=1:4,cex=0.8,fill=1:4)

# Checking if some optimization will help - 
optimizedTrees <-randomForest(Y_AccetpedOffer ~ .,data = treatedTrain, ntree=250)

# Confusion Matrix - 
trainClass<-predict(optimizedTrees, treatedTrain)
confusionMatrix(trainClass, treatedTrain$Y_AccetpedOffer)

### Now let's apply to the validation test set
threesPreds        <- predict(downSampleFit, treatedTest)
moreTreesPreds <- predict(moreTrees,    treatedTest)
optimizedTreesPreds  <- predict(optimizedTrees,    treatedTest)

# Accuracy Comparison from MLmetrics
Accuracy(treatedTest$Y_AccetpedOffer, threesPreds) # ~ 0.63
Accuracy(treatedTest$Y_AccetpedOffer, moreTreesPreds)# ~ 0.75
Accuracy(treatedTest$Y_AccetpedOffer, optimizedTreesPreds) # ~ 0.76.

varImp(optimizedTrees)
varImpPlot(optimizedTrees)

# Accuracy level is higher with the best performer in the RF model group scoring at .76.

### 2. DECISION TREES ###

#Reloading the data since it was pruned in the RF section - 
dataset <- standardizedTrainigDataSetNoTime()

#
set.seed(1234)

# Training with 70% of the data - 
splitPercent <- round(nrow(dataset) %*% .7)
totalRecords <- 1:nrow(dataset)
idx <- sample(totalRecords, splitPercent)

trainDat <- dataset[idx,]
testDat  <- dataset[-idx,]

plan <- designTreatmentsC(dataset, 
                          names(dataset),
                          'Y_AccetpedOffer', 
                          1)

# Apply the plan
treatedTrain <- prepare(plan, trainDat)
treatedTest <- prepare(plan, testDat)

# Force a full tree (override default parameters)
overFit <- rpart(Y_AccetpedOffer ~ ., 
                 data = treatedTrain, 
                 method = "class", 
                 minsplit = 1, 
                 minbucket = 1, 
                 cp=-1)
prp(overFit, extra = 1)
#overFit$splits

# Fit a decision tree with caret
trctrl <- trainControl(method = "cv", number = 10)
set.seed(3333)
dtree_fit <- train(as.factor(Y_AccetpedOffer) ~., 
                   data = treatedTrain, 
                   method = "rpart",
                   #parms = list(split = "information"),
                   trControl=trctrl,
                   tuneLength = 10)

dtree_fit
dtree_fit$bestTune

# Plot the CP Accuracy Relationship to adust the tuneGrid inputs
plot(dtree_fit)

# Plot a pruned tree
prp(dtree_fit$finalModel, extra = 1)

# Make some predictions on the training set
trainCaret<-predict(dtree_fit, treatedTrain)
head(trainCaret)

# Get the conf Matrix
confusionMatrix(trainCaret, as.factor(treatedTrain$Y_AccetpedOffer))
#Accuracy : 0.799

testCaret<-predict(dtree_fit,treatedTrain)
confusionMatrix(testCaret,as.factor(treatedTrain$Y_AccetpedOffer))
#Accuracy : 0.799 

varImp(dtree_fit)
plot(varImp(dtree_fit), top = 8)

# So far the DT model is the best candidate with accuracy of about .80. 

### 3. LOGISTIC REGRESSION ###

#We can use parallel to speed up computations - 
set.seed(1234)
fit <- glm(Y_AccetpedOffer ~ ., treatedTrain, family='binomial')
length(coefficients(fit))

#bestFit <- step(fit, direction='both')
#saveRDS(bestFit, 'bestFitBankOffers.rds')
bestFit <-readRDS('bestFitBankOffers.rds')
summary(bestFit)


# Compare model size
length(coefficients(fit))
length(coefficients(bestFit))

# Predictions
preds <- predict(bestFit, type='response')
# Classify 
cutoff <- 0.5
classes <- ifelse(preds >= cutoff, 1,0)
# Organize w/Actual
results <- data.frame(actual = treatedTrain$Y_AccetpedOffer, classes = classes)
head(results)
# Get a confusion matrix
(confMat <- ConfusionMatrix(results$classes, results$actual))
sum(diag(confMat)) / sum(confMat)
# Accuracy - 
Accuracy(results$classes, results$actual)#0.78
# Visually -
ggplot(results, aes(x=preds, color=as.factor(actual))) +
geom_density() + 
geom_vline(aes(xintercept = cutoff), color = 'green')


### 5. NEURAL NETS - EXAMINING PREDICTIONS ###

# Using the results from the RF, DTs and LogReg we can examine NNs training -
# more precisely we can try to specify few input nodes based on predictors importance derived above:

dataset <- standardizedTrainigDataSetNoTime()
set.seed(1234)
# Training with 70% of the data - 
splitPercent <- round(nrow(dataset) %*% .7)
totalRecords <- 1:nrow(dataset)
idx <- sample(totalRecords, splitPercent)

trainDat <- dataset[idx,]
testDat  <- dataset[-idx,]

plan <- designTreatmentsC(dataset, 
                          names(dataset),
                          'Y_AccetpedOffer', 
                          1)

# Apply the plan
treatedTrain <- prepare(plan, trainDat)
treatedTest <- prepare(plan, testDat)
treatedTrain$Y_AccetpedOffer <- as.integer(treatedTrain$Y_AccetpedOffer)

# However, we can use the most important predictors according to the RF optimized tree results above - 
# we can pick all with level of importance above 10:
#Communication_catP             12.537427939
#Communication_catB             17.826531649
#LastContactDay_clean           67.664004359
#LastContactMonth_catP          48.982007827
#LastContactMonth_catB          55.848571703
#NoOfContacts_clean             39.844323337
#Outcome_catP                   26.394598676
#Outcome_catB                   37.861509927
#EstRace_catP                   19.064823817
#EstRace_catB                   35.778600383
#Age_clean                      78.213271392
#Job_catP                       29.218621649
#Job_catB                       36.305034151
#Education_catP                 13.962872524
#Education_catB                 14.169942604
#HHInsurance_clean              24.154886820
#carMake_catP                   50.387603957
#carMake_catB                   58.452223175
#carModel_catP                  56.509466293
#carModel_catB                 205.569219307
#carYr_clean                    59.428637018
#Communication_lev_NA           14.904969417
#Communication_lev_x_cellular   10.805736207
#Outcome_lev_NA                 11.832250502
#Outcome_lev_x_success          33.613044356


nn.rf.layers4 <- neuralnet(treatedTrain$Y_AccetpedOffer ~ 
                              treatedTrain$Communication_catP +
                             treatedTrain$Communication_catB +
                             treatedTrain$LastContactDay_clean +
                             treatedTrain$LastContactMonth_catP + 
                             treatedTrain$LastContactMonth_catB +
                             treatedTrain$NoOfContacts_clean +
                             treatedTrain$Outcome_catP +
                             treatedTrain$Outcome_catB +
                             treatedTrain$EstRace_catP +
                             treatedTrain$EstRace_catB  +
                             treatedTrain$Age_clean +
                             treatedTrain$Job_catP  +
                             treatedTrain$Job_catB  +
                             treatedTrain$Education_catP +
                             treatedTrain$Education_catB +
                             treatedTrain$HHInsurance_clean +
                             treatedTrain$carMake_catP  +
                             treatedTrain$carMake_catB +
                             treatedTrain$carModel_catP  +
                             treatedTrain$carModel_catB  +
                             treatedTrain$carYr_clean +
                             treatedTrain$Communication_lev_NA +
                             treatedTrain$Communication_lev_x_cellular +
                             treatedTrain$Outcome_lev_NA +
                             treatedTrain$Outcome_lev_x_success,
                              data = treatedTrain, 
                              hidden = 4)#  

nn.rf.layers4$weights
nn.rf.layers4$model.list
plot(nn.rf.layers4)

test <- subset(treatedTrain, select = c("Communication_catP",            
                                        "Communication_catB",             
                                        "LastContactDay_clean",          
                                        "LastContactMonth_catP",         
                                        "LastContactMonth_catB",          
                                        "NoOfContacts_clean",             
                                        "Outcome_catP",                   
                                        "Outcome_catB",                   
                                        "EstRace_catP",                   
                                        "EstRace_catB",                   
                                        "Age_clean",                      
                                        "Job_catP",                      
                                        "Job_catB",                       
                                        "Education_catP",                 
                                        "Education_catB",                
                                        "HHInsurance_clean",              
                                        "carMake_catP",                   
                                        "carMake_catB",                   
                                        "carModel_catP",                  
                                        "carModel_catB",               
                                        "carYr_clean",                    
                                        "Communication_lev_NA",           
                                        "Communication_lev_x_cellular",   
                                        "Outcome_lev_NA",               
                                        "Outcome_lev_x_success"))

pr.nn <- compute(nn.rf.layers4, test)
pr.nn_ <- pr.nn$net.result
# Accuracy (training set)
pr.nn_2 <- max.col(pr.nn_)
Accuracy(treatedTrain$Y_AccetpedOffer, pr.nn_2) # ~ 0.59.

#Removing communication and education - 
nn.rf.layers2 <- neuralnet(treatedTrain$Y_AccetpedOffer ~ 
                             treatedTrain$LastContactDay_clean +
                             treatedTrain$LastContactMonth_catP + 
                             treatedTrain$LastContactMonth_catB +
                             treatedTrain$NoOfContacts_clean +
                             treatedTrain$Outcome_catP +
                             treatedTrain$Outcome_catB +
                             treatedTrain$EstRace_catP +
                             treatedTrain$EstRace_catB  +
                             treatedTrain$Age_clean +
                             treatedTrain$Job_catP  +
                             treatedTrain$Job_catB  +
                             treatedTrain$HHInsurance_clean +
                             treatedTrain$carMake_catP  +
                             treatedTrain$carMake_catB +
                             treatedTrain$carModel_catP  +
                             treatedTrain$carModel_catB  +
                             treatedTrain$carYr_clean +
                             treatedTrain$Outcome_lev_NA +
                             treatedTrain$Outcome_lev_x_success,
                           data = treatedTrain, 
                           hidden = 2) 

nn.rf.layers2$weights
nn.rf.layers2$model.list
plot(nn.rf.layers2)

test <- subset(treatedTrain, select = c("LastContactDay_clean",          
                                        "LastContactMonth_catP",         
                                        "LastContactMonth_catB",          
                                        "NoOfContacts_clean",             
                                        "Outcome_catP",                   
                                        "Outcome_catB",                   
                                        "EstRace_catP",                   
                                        "EstRace_catB",                   
                                        "Age_clean",                      
                                        "Job_catP",                      
                                        "Job_catB",                      
                                        "HHInsurance_clean",              
                                        "carMake_catP",                   
                                        "carMake_catB",                   
                                        "carModel_catP",                  
                                        "carModel_catB",               
                                        "carYr_clean",                  
                                        "Outcome_lev_NA",               
                                        "Outcome_lev_x_success"))

pr.nn <- compute(nn.rf.layers2, test)
pr.nn_ <- pr.nn$net.result
# Accuracy (training set)
pr.nn_2 <- max.col(pr.nn_)

Accuracy(treatedTrain$Y_AccetpedOffer, pr.nn_2) # ~ 0.60.

# Nerual nets perform really bad for this classification problem.

###############################################################
# Summary of results, ordered best to worst performing model: #
###############################################################

# 1) Decision trees seems to be the top option in terms of accuracy - ~80%.

# 2) Logistic regression comes second with almost equal level of accuracy as DTs - ~76%.

# 3) Random forest is 3d. It is worthwhile noting that if we include call start-time(s) and call duration, the forest returns 86% of accuracy, which is very high prediction score. However, as shown in the EDA part, for this data set the start time of successful calls does not differ on average from the start time for unsuccessful calls. 
#    So there is no way to detect a pattern of 'better time to call during the day' and include as a recommendation. 
#    The only notable difference is that call length is on average much longer for accepted offers. Therefore, it may make sense to include a study where the agents making the calls try to engage the users for longer periods. If this change does indeed increase the chances of accepting an offer, then the analysis may be redone call-duration and recommended call start time factored in as recommendations in the future prospects' dataset. 
#    However, based on the current data set there is no sufficient evidence to confirm that call length is not just the result of discussion happening after the offer had already been accepted. More evidence, such as agent approach and time of acceptance of the offer during the call is needed to make sense out of the call-length which seems to be a very strong factor in the random forest model accuracy outcome.

# 4) Neural networks – the current setup performs poorly compared to the models 1) to 3), with an accuracy level barely reaching to the 60%. 

############################################################
# Estimating top 100 prospects                             #
############################################################

# Load future prospects' set:
loadProspects <- function(){
  setwd("")
  prospects <- read.csv('ProspectiveCustomers.csv')
  setwd("")
  mktg <- read.csv('CurrentCustomerMktgResults.csv')
  axiom <- read.csv('householdAxiomData.csv')
  credit <- read.csv('householdCreditData.csv')
  car <- read.csv('householdVehicleData.csv')
  # Repeat the same pre-processing steps - 
  prospects <- merge(mktg, axiom, by = "HHuniqueID", match = "left")
  prospects <- merge(prospects, credit, by = "HHuniqueID", match = "left")
  prospects <- merge(prospects, car, by = "HHuniqueID", match = "left")
  prospects <- subset(prospects, select=-c(HHuniqueID))
  prospects <- subset(prospects, select=-c(dataID))
  #y <- as.factor(prospects$Y_AccetpedOffer)
  #prospects$Y_AccetpedOffer <- y
  prospects <- subset(prospects, select=-c(CallStart))
  prospects <- subset(prospects, select=-c(CallEnd))
  #
  prospects$NoOfContacts <- squish(prospects$NoOfContacts, quantile(prospects$NoOfContacts, c(0.025, 0.975)))
  prospects$annualDonations <- squish(as.numeric(prospects$annualDonations), quantile(as.numeric(prospects$annualDonations), c(0.025, 0.975)))
  prospects$DaysPassed <- squish(prospects$DaysPassed, quantile(prospects$DaysPassed, c(0.025, 0.975)))
  prospects$Age <- squish(prospects$Age, quantile(prospects$Age, c(0.025, 0.975)))
  prospects$PrevAttempts <- squish(prospects$PrevAttempts, quantile(prospects$PrevAttempts, c(0.025, 0.975)))
  prospects$RecentBalance <- squish(prospects$RecentBalance, quantile(prospects$RecentBalance, c(0.025, 0.975)))
  
  #interactive vars:
  prospects$BalanceAge <- (prospects$RecentBalance * prospects$Age)
  prospects$ContactsAge <- (prospects$NoOfContacts * prospects$Age)
  prospects$ContactsLastDay <- (prospects$NoOfContacts * prospects$LastContactDay)
  
  return(prospects)
  
}

prospects <- loadProspects()

# Predict:
options(scipen=999)
set.seed(1234)
validationTrain <- prepare(plan, prospects)
testCaret<-predict(dtree_fit,validationTrain, type="prob")
testCaret
head(validationTrain)
# Top 100 prospects:
validationTrain$Y_AccetpedOffer.0 <- testCaret$`0`
validationTrain$Y_AccetpedOffer.1 <- testCaret$`1`
predicted.accpeted.offers <- subset(validationTrain, validationTrain$Y_AccetpedOffer.1 >= 0.9)
predicted.rejected.offers <- subset(validationTrain, validationTrain$Y_AccetpedOffer.0 >= 0.9)  

head(predicted.accpeted.offers)
head(predicted.rejected.offers)
count(predicted.accpeted.offers)#287
count(predicted.rejected.offers)#419

# Top 100 prospects:
sorted.accepted.offers <- predicted.accpeted.offers[order(predicted.accpeted.offers$Y_AccetpedOffer.1, decreasing=TRUE),]
top100 <- (sorted.accepted.offers[1:100,])

############################################################
# Estimating top 100 prospects from best group model       #
############################################################

setwd("")
prospects.probs <- read.csv('top100.csv')
setwd("")
prospects.full <- read.csv('ProspectiveCustomers.csv')
head(prospects.full)
colnames(prospects.probs)
# Repeat the same pre-processing steps - 
prospects <- merge(prospects.probs, prospects.full, by = "HHuniqueID", match = "left")

setwd("")
mktg <- read.csv('CurrentCustomerMktgResults.csv')
axiom <- read.csv('householdAxiomData.csv')
credit <- read.csv('householdCreditData.csv')
car <- read.csv('householdVehicleData.csv')
# Repeat the same pre-processing steps - 
prospects <- merge(prospects, axiom, by = "HHuniqueID", match = "left")
prospects <- merge(prospects, axiom, by = "HHuniqueID", match = "left")
prospects <- merge(prospects, credit, by = "HHuniqueID", match = "left")
prospects <- merge(prospects, car, by = "HHuniqueID", match = "left")
prospects <- subset(prospects, select=-c(HHuniqueID))
prospects <- subset(prospects, select=-c(dataID))

prospects <- subset(prospects, select=-c(X))
prospects <- subset(prospects, select=-c(dataID))
prospects <- subset(prospects, select=-c(XGBClass))
prospects <- subset(prospects, select=-c(SGBClass))
prospects <- subset(prospects, select=-c(RFClass))
prospects <- subset(prospects, select=-c(XGBScore))
prospects <- subset(prospects, select=-c(SGBScore))
prospects <- subset(prospects, select=-c(RFScore))

# overall structure  & dimensions of the data -
str(prospects)
dim(prospects)
# data set class - 
class(prospects)
# classes for each column - 
sapply(prospects, class)
# look at the top 6 rows - 
head(prospects)
# levels -  
nlevels(prospects)
# column names - 
names(prospects)
# summary stats for each vector - 
summary(prospects)

plot_density(prospects) 

historical <- standardizedTrainigDataSetNoTime()

# Call center / Agent behavior - 

mean(prospects$LastContactDay)#14.53
mean(historical$LastContactDay)#15.72125

mean(prospects$NoOfContacts)#1.9
mean(historical$NoOfContacts)#2.4

mean(prospects$DaysPassed)#153.01
mean(historical$DaysPassed)#46.7

mean(prospects$PrevAttempts)#2.27
mean(historical$PrevAttempts)#0.61

# User group related - 

mean(prospects$Age)#NA
mean(historical$Age)#41

mean(prospects$DefaultOnRecord)#0.01
mean(historical$DefaultOnRecord)#0.01

mean(prospects$AffluencePurchases)#NA
mean(historical$AffluencePurchases)#0.49

mean(prospects$RecentBalance)#2053.937
mean(historical$RecentBalance)#1341.451

mean(prospects$PetsPurchases)#NA
mean(historical$PetsPurchases)#0.48

mean(prospects$HHInsurance)#0.2
mean(historical$HHInsurance)#0.49

mean(prospects$annualDonations)#NA
mean(historical$annualDonations)#28.87638 - almost none.

mean(prospects$CarLoan)#0.03
mean(historical$CarLoan)#0.13

#colnames(prospects)


#############################################################
# Clustering analysis - 100 prospects from best group model #
#############################################################

library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

options(scipen=999)
set.seed(1234)

prospects$Y_AccetpedOffer <- prospects.probs$XGBScore
plan         <- designTreatmentsN(prospects, names(prospects), 'Y_AccetpedOffer')
treatedTrain <- prepare(plan, prospects)

# Optimal number of clusters is 2:
fviz_nbclust(treatedTrain, kmeans, method = "silhouette")

# Plotting the clusters:
k2 <- kmeans(treatedTrain, centers = 2, nstart = 5)#Optimal clustering.
k3 <- kmeans(treatedTrain, centers = 3, nstart = 5)
k4 <- kmeans(treatedTrain, centers = 4, nstart = 5)
k5 <- kmeans(treatedTrain, centers = 5, nstart = 5)
#
p2 <- fviz_cluster(k2, geom = "point", data = treatedTrain) + ggtitle("k = 2")
p3 <- fviz_cluster(k3, geom = "point", data = treatedTrain) + ggtitle("k = 3")
p4 <- fviz_cluster(k4, geom = "point", data = treatedTrain) + ggtitle("k = 4")
p5 <- fviz_cluster(k5, geom = "point", data = treatedTrain) + ggtitle("k = 5")
library(gridExtra)
grid.arrange(p2, p3, p4, p5, nrow = 2)


# We can see that the customer base is concentric in terms of clusters. With a wider cluster containing
# a smaller one. This is indicative to the fact that all users accepting the offer will share completely
# certain set of characteristics with small portion of the users having additional characteristics. 

treatedTrain$cluster <- k2$cluster 
data_clus_1 <- treatedTrain[treatedTrain$cluster == 1,]
data_clus_2 <- treatedTrain[treatedTrain$cluster == 2,]

#plot_density(data_clus_1)
#plot_density(data_clus_2)

#plot_density(data_clus_1$Age_clean)
#plot_density(data_clus_2$Age_clean)

#mean(data_clus_1$Age_clean)
#mean(data_clus_2$Age_clean)
