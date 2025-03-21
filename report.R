set.seed(123)


install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")
install.packages("xgboost")
install.packages("pROC")
install.packages("corrplot")
install.packages("ROCR")




library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)
library(corrplot)
library(ROCR)

telecom <- read.csv("https://www.louisaslett.com/Courses/MISCADA/telecom.csv")

dim(telecom)
str(telecom)
summary(telecom)

colSums(is.na(telecom))

telecom$Churn <- as.factor(telecom$Churn)

ggplot(telecom, aes(x = Churn)) +
  geom_bar(fill = "steelblue") +
  ggtitle("Churn Distribution") +
  theme_minimal()

ggplot(telecom, aes(x = tenure, fill = Churn)) +
  geom_histogram(binwidth = 1, position = "fill") +
  ggtitle("Churn by Tenure") +
  theme_minimal()

numericVars <- telecom %>% select_if(is.numeric)
corr_matrix <- cor(numericVars, use = "complete.obs")
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.7)

trainIndex <- createDataPartition(telecom$Churn, p = 0.7, list = FALSE)
trainData <- telecom[trainIndex, ]
testData  <- telecom[-trainIndex, ]

trainData_clean <- na.omit(trainData)
testData_clean  <- na.omit(testData)

logitModel <- glm(Churn ~ ., data = trainData_clean, family = "binomial")

logitPredProb <- predict(logitModel, testData_clean, type = "response")
logitPred <- ifelse(logitPredProb > 0.5, "Yes", "No") %>% as.factor()

confusionMatrix(logitPred, testData_clean$Churn)

roc_logit <- roc(testData_clean$Churn, logitPredProb)
auc_logit <- auc(roc_logit)
plot(roc_logit, main = "Logistic Regression ROC Curve")

rfModel <- randomForest(Churn ~ ., data = trainData_clean, ntree = 500, importance = TRUE)

varImpPlot(rfModel, main = "Feature Importance - Random Forest")

train_matrix <- model.matrix(Churn ~ . -1, data = trainData_clean)
test_matrix  <- model.matrix(Churn ~ . -1, data = testData_clean)
train_label  <- ifelse(trainData_clean$Churn == "Yes", 1, 0)
test_label   <- ifelse(testData_clean$Churn == "Yes", 1, 0)

xgbTrain <- xgb.DMatrix(data = train_matrix, label = train_label)
xgbTest  <- xgb.DMatrix(data = test_matrix, label = test_label)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc"
)

xgbModel <- xgb.train(params = params,
                      data = xgbTrain,
                      nrounds = 100,
                      watchlist = list(eval = xgbTest),
                      early_stopping_rounds = 10,
                      verbose = 1)

xgbPredProb <- predict(xgbModel, xgbTest)
xgbPred <- ifelse(xgbPredProb > 0.5, "Yes", "No") %>% as.factor()

confusionMatrix(xgbPred, testData_clean$Churn)

roc_xgb <- roc(testData_clean$Churn, xgbPredProb)
auc_xgb <- auc(roc_xgb)
plot(roc_xgb, main = "XGBoost ROC Curve")

importance_matrix <- xgb.importance(model = xgbModel)
xgb.plot.importance(importance_matrix, top_n = 10, main = "XGBoost Feature Importance")

print(paste("Logistic Regression Model AUC:", round(auc_logit, 3)))
print(paste("XGBoost Model AUC:", round(auc_xgb, 3)))

