---
title: "Practical-machine-learning"
author: "areyou76"
date: "December 27, 2015"
output: html_document
---



###Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

###Data

The training data for this project are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv]

The test data are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv]

The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har]. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.

###Preliminary Work

####Choosing the prediction algorithm

Steps Taken

1.Tidy data. Remove columns with little/no data.

2.Create Training and test data from traing data for cross validation checking

3.Trial 3 methods Random Forrest, Gradient boosted model and Linear discriminant analysis

Fine tune model through combinations of above methods, reduction of input variables or similar. The fine tuning will take into account accuracy first and speed of analysis second.


```{r, message=FALSE, warning=FALSE }
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)
library(gbm)
```

```{r, message=FALSE, warning=FALSE}
library(doParallel)
library(survival)
library(splines)
library(plyr)
```


###Load data

* Load data.

* Remove "#DIV/0!", replace with an NA value.


```{r}
# load data
training <- read.csv("pml-training.csv", na.strings=c("#DIV/0!"), row.names = 1)
testing <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!"), row.names = 1)
```


###Predictions


```{r}
training <- training[, 6:dim(training)[2]]

treshold <- dim(training)[1] * 0.95
#Remove columns with more than 95% of NA or "" values
goodColumns <- !apply(training, 2, function(x) sum(is.na(x)) > treshold  || sum(x=="") > treshold)

training <- training[, goodColumns]

badColumns <- nearZeroVar(training, saveMetrics = TRUE)

training <- training[, badColumns$nzv==FALSE]

training$classe = factor(training$classe)

#Partition rows into training and crossvalidation
inTrain <- createDataPartition(training$classe, p = 0.6)[[1]]
crossv <- training[-inTrain,]
training <- training[ inTrain,]
inTrain <- createDataPartition(crossv$classe, p = 0.75)[[1]]
crossv_test <- crossv[ -inTrain,]
crossv <- crossv[inTrain,]


testing <- testing[, 6:dim(testing)[2]]
testing <- testing[, goodColumns]
testing$classe <- NA
testing <- testing[, badColumns$nzv==FALSE]
```


```{r}
#Train 3 different models
mod1 <- train(classe ~ ., data=training, method="rf")
#mod2 <- train(classe ~ ., data=training, method="gbm")
#mod3 <- train(classe ~ ., data=training, method="lda")

pred1 <- predict(mod1, crossv)
#pred2 <- predict(mod2, crossv)
#pred3 <- predict(mod3, crossv)
```


```{r}
#show confusion matrices
confusionMatrix(pred1, crossv$classe)
```


```{r}
#confusionMatrix(pred2, crossv$classe)
#confusionMatrix(pred3, crossv$classe)

#Create Combination Model

#predDF <- data.frame(pred1, pred2, pred3, classe=crossv$classe)
#predDF <- data.frame(pred1, pred2, classe=crossv$classe)

#combModFit <- train(classe ~ ., method="rf", data=predDF)
#in-sample error
#combPredIn <- predict(combModFit, predDF)
#confusionMatrix(combPredIn, predDF$classe)



#out-of-sample error
pred1 <- predict(mod1, crossv_test)
#pred3 <- predict(mod3, crossv_test)
accuracy <- sum(pred1 == crossv_test$classe) / length(pred1)
```

Based on results, the Random Forest prediction was far better than either the GBM or lsa models. The RF model will be used as the sole prediction model. The confusion matrix created gives an accuracy of 99.6%. This is excellent.

As a double check the out of sample error was calculated. This model achieved 99.7449 % accuracy on the validation set.

###Fine Tuning

Assess Number of relevant variables


```{r}
varImpRF <- train(classe ~ ., data = training, method = "rf")
varImpObj <- varImp(varImpRF)
# Top 40 plot
plot(varImpObj, main = "Importance of Top 40 Variables", top = 40)
```


```{r}
# Top 25 plot
plot(varImpObj, main = "Importance of Top 25 Variables", top = 25)
```


###Conclusions

It can be concluded that, The Random Forest method worked very well. The Confusion Matrix achieved 99.6% accuracy. The Out of Sample Error achieved 99.7449 %. This model will be used for the final calculations.

The logic behind using the random forest method as the predictor rather than other methods or a combination of various methods is:

1. Random forests are suitable when to handling a large number of inputs, especially when the interactions between variables are unknown.
2. Random forest's built in cross-validation component that gives an unbiased estimate of the forest's out-of-sample (or bag) (OOB) error rate.
3. A Random forest can handle unscaled variables and categorical variables. This is more forgiving with the cleaning of the data.
4. It worked


####*******************************************************************


####Prepare the submission. (using COURSERA provided code)

```{r}
pml_write_files = function(x){
n = length(x)
for(i in 1:n){
filename = paste0("problem_id_",i,".txt")
write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}
}
x <- testing

answers <- predict(mod1, newdata=x)
answers
```


```{r}
pml_write_files(answers)
```





