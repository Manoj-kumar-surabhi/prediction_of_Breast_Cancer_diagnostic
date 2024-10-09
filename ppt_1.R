
disease_data <- read.csv("data.csv")

View(disease_data)

summary(disease_data)

str(disease_data)

# Creating a subset of the predictor variables 



data_predictors <- disease_data[, -c(1, 2,33)]
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
library(corrplot)
 
corrplot(correlations, order = "hclust")

## To filter based on correlations, the findCorrelation function will apply the
## algorithm in Sect. 3.5. For a given threshold of pairwise correlations, the function
## returns column numbers denoting the predictors that are recommended
## for deletion:
highCorr <- findCorrelation(correlations, cutoff = .85)
length(highCorr)
highCorr
filteredSegData <- data_predictors[, -highCorr]
length(filteredSegData)



