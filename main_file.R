
disease_data <- read.csv("data.csv")

View(disease_data)

summary(disease_data)

str(disease_data)

# Creating a subset of the predictor variables 

data_predictors <- disease_data[, -c(1, 2,33)]
response_variable <- as.factor(disease_data$diagnosis)
(table(response_variable))

head(data_predictors)

# Plot histograms for each predictor

par(mfrow = c(2, 3)) 
for (i in 1:ncol(data_predictors)) {
  hist(data_predictors[[i]], main = colnames(data_predictors)[i], 
       xlab = colnames(data_predictors)[i], col = 'orange', border = 'black')
}

# Plot boxplots for each predictor
par(mfrow = c(2, 3)) # Create a 3x3 grid of plots
par(mai=c(.25,.25,.25,.25))
for (i in 1:ncol(data_predictors)) {
  boxplot(data_predictors[[i]], main = colnames(data_predictors)[i], 
          col = 'lightblue', border = 'black')
  
}
# Calculate the skewness of each predictor
library(e1071)
skewness_values <- apply(data_predictors, 2, skewness)
skewness_table <- data.frame(
  Predictor = colnames(data_predictors),
  Skewness = skewness_values
)
skewness_table

#plot frequency distributions for each individual variables

# Identify categorical columns with fewer than 10 unique values
categorical_cols <- names(disease_data)[sapply(disease_data, function(col) length(unique(col)) < 10)]

# Adjust the plotting area and bar width
par(mar = c(5, 4, 4, 2) + 0.1)  # Adjusting margins: bottom, left, top, right

# Create the frequency table for diagnosis
diagnosis_table <- table(disease_data$diagnosis)

# Create the bar plot with customized dimensions
barplot(diagnosis_table, 
        main = "Diagnosis Barplot", 
        xlab = "Diagnosis", 
        ylab = "Frequency", 
        cex.names = 0.8,      # Adjust size of category labels
        cex.axis = 0.8,       # Adjust size of axis labels
        col = "lightblue",    # Custom bar color
        width = 1.5)          # Adjust bar width




# Install and load the caret package if not already installed
library(caret)

# Find near-zero variance features
nzv <- nearZeroVar(disease_data, saveMetrics = TRUE)

print(nzv)

nzv_variables <- nzv[nzv$nzv == TRUE, ]
print(nzv_variables)


#create a corelation plot

library(corrplot)
correlations <- cor(data_predictors)
dim(correlations)
correlations[1:4, 1:4]

## To visually examine the correlation structure of the data, the corrplot package
## contains an excellent function of the same name. 

corrplot(correlations, order = "hclust")

## To filter based on correlations, the findCorrelation function will apply the
## algorithm in Sect. 3.5. For a given threshold of pairwise correlations, the function
## returns column numbers denoting the predictors that are recommended
## for deletion:
highCorr <- findCorrelation(correlations, cutoff = .85)
length(highCorr)
highCorr

#predictors_filtered <- data_predictors[,-highCorr]

#######################Pre-processing ########
predictors_preprocessed <- preProcess(data_predictors, method = c("center", "scale", "BoxCox", "spatialSign"))
predictors <- predict(predictors_preprocessed, data_predictors)

###############Train and Test sets ###########
set.seed(123)
train_index <- createDataPartition(response_variable, p =0.8, list = FALSE)
train_data <- predictors[train_index, ]
test_data <- predictors[-train_index, ]
train_y <- response_variable[train_index]
test_y <- response_variable[-train_index]

ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = TRUE)

###########Linear Classification Models ##################

################# LOGISTIC REGRESSION ####################
set.seed(143)
logistic_model <- train(x = train_data, y = train_y, method = "glm", trControl = ctrl, preProcess = c("pca"), metric="ROC")
print(logistic_model)
summary(logistic_model)
logistic_predictions <- predict(logistic_model, train_data)
logistic_confusion <- confusionMatrix(logistic_predictions, train_y)
print(logistic_confusion)

logistic_test_predictions <- predict(logistic_model, test_data)
logistic_test_confusion <- confusionMatrix(logistic_test_predictions, test_y)
print(logistic_test_confusion)
logistic_kappa_testing <- logistic_test_confusion$overall["Kappa"]
print(logistic_kappa_testing)

###################### LDA ######################

lda_model <- train(x = train_data, y = train_y , method ="lda", metric = "ROC", trControl = ctrl, preProcess = c("pca"))
print(lda_model)
lda_training_predictions <- predict(lda_model, train_data)
lda_test_predictions <- predict(lda_model, test_data)

lda_training_confusion <- confusionMatrix(lda_training_predictions, train_y)
lda_testing_confusion <- confusionMatrix(lda_test_predictions, test_y)

print(lda_testing_confusion)

###################### PLS Discriminant Analysis #################
library(klaR)
pls_model <- train(
  x = train_data, 
  y = train_y, 
  method = "pls",  
  trControl = ctrl,
  tuneGrid = expand.grid(.ncomp = 1:20)  # Tune for 1 to 5 components
)
print(pls_model)

pls_training_predictions <- predict(pls_model, train_data)
pls_testing_predictions <- predict(pls_model, test_data)

pls_testing_confusion <- confusionMatrix(pls_testing_predictions, test_y)

print(pls_testing_confusion)


######################### Penalized models ##########################

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
set.seed(476)
penalized_enet_model <- train(x= train_data,
                   y = train_y,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   metric = "ROC",
                   trControl = ctrl)

penalized_enet_training_preditions <- predict(penalized_enet_model, train_data)
penalized_enet_testing_predictions <- predict(penalized_enet_model, test_data)

penalized_enet_testing_confusion <- confusionMatrix(penalized_enet_testing_predictions, test_y)
print(penalized_enet_testing_confusion)

#Lasso model
glmnGrid <- expand.grid(.alpha = 1,
                        .lambda = seq(.01, .2, length = 10))
set.seed(476)
penalized_lasso_model <- train(x= train_data,
                         y = train_y,
                         method = "glmnet",
                         tuneGrid = glmnGrid,
                         metric = "ROC",
                         trControl = ctrl)

penalized_lasso_testing_predictions <- predict(penalized_lasso_model, test_data)

penalized_lasso_confusion <- confusionMatrix(penalized_lasso_testing_predictions, test_y)

print(penalized_lasso_confusion)

###Ridge Model

glmGrid <- expand.grid(.alpha=0, .lambda = seq(.01, .2, length= 10))
set.seed(476)
penalized_ridge_model <- train(x = train_data, 
                               y = train_y, method="glmnet",
                               tuneGrid = glmGrid, metric = "ROC", trControl = ctrl)

penalized_ridge_test_predictions <- predict(penalized_ridge_model, test_data)

penalized_ridge_confusion <- confusionMatrix(penalized_ridge_test_predictions, test_y)

print(penalized_ridge_confusion)


###########################Nearest Shrunken Centroids ###############################

library(pamr)
nscGrid <- data.frame(.threshold = 0:10)
set.seed(476)
nsc_model <- train(x = train_data,
                  y = train_y,
                  method = "pam",
                  tuneGrid = nscGrid,
                  metric = "ROC",
                  trControl = ctrl)

nsc_test_predictions <- predict(nsc_model, test_data)

nsc_confusion_matrix <- confusionMatrix(nsc_test_predictions, test_y)

print(nsc_confusion_matrix)

############################### NON LINEAR MODELS #######################################3

########################### NEURAL NETWORKS #################################

neural_grid <- expand.grid(.size = 1 : 20, .decay = c(0, .1, 1, 2))

max_size <- max(neural_grid$.size)

max_no_of_weights <- 1*(max_size * (30 + 1) + max_size + 1 )

set.seed(123)

neural_model <- train(x = train_data, y = train_y, 
                      method="nnet",
                      metric="ROC",
                      tuneGrid = neural_grid,
                      trace = FALSE,
                      maxi = 2000,
                      MaxNWts = max_no_of_weights,
                      trControl = ctrl)

neural_test_predictions <- predict(neural_model, test_data)

neural_confusion_matrix <- confusionMatrix(neural_test_predictions, test_y)
print(neural_confusion_matrix)


################# Support vector machines ##############################

set.seed(165)
library(kernlab)

sigma_range <- sigest(as.matrix(train_data))

svm_grid <- expand.grid(.sigma = sigma_range[1], .C= 2^(seq(-4, 8)))

svm_model <- train(x = train_data, y = train_y,
                   method="svmRadial",
                   metric="ROC",
                   tuneGrid = svm_grid,
                   fit = FALSE,
                   trControl = ctrl)

svm_test_predictions <- predict(svm_model, test_data)

svm_confusion_matrix <- confusionMatrix(svm_test_predictions, test_y)

print(svm_confusion_matrix)

############### Flexible discriminant analysis ################

library(earth)
library(mda)

fda_grid <- expand.grid(.degree=1:2, .nprune=2:30)
set.seed(250)
fda_model <- train(x = train_data, y = train_y, method="fda",
                   metric="ROC",
                   trControl = ctrl,
                   preProcess = c("pca"), tuneGrid = fda_grid)


fda_test_predictions <- predict(fda_model, test_data)

fda_confusion_matrix <- confusionMatrix(fda_test_predictions, test_y)

print(fda_confusion_matrix)


########################## Naive Bayes #######################
library(klaR)
naive_bayes_model <- train(x = train_data, y = train_y, 
                           trControl = ctrl,
                           metric = "ROC",
                           method = "nb",
                           preProcess = c("pca"), useKernel = TRUE, fl = 2)
naive_bayes_predictions <- predict(naive_bayes_model, test_data)
naive_bayes_confusion <- confusionMatrix(naive_bayes_predictions, test_y)

print(naive_bayes_model)
print(naive_bayes_confusion)
plot(naive_bayes_model)

###################### K- nearest neighbours ########################
set.seed(455)
knn_model <- train(x = train_data, y = train_y,
                   method="knn",
                   metric="ROC",
                   trControl= ctrl,
                   tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:7)+1)),
                   preProcess = c("pca"))

                   

knn_predictions <- predict(knn_model, test_data)

knn_confusion_matrix <- confusionMatrix(knn_predictions, test_y)
print(knn_confusion_matrix)
