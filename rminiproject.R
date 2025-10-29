library(readr)
library(caret)
library(randomForest)
library(pROC)
library(dplyr)
library(class)
library(e1071)

setwd("C:/Users/Muhil/Downloads")

email_phishing_data <- read_csv("email_phishing_data.csv")
email_phishing_data$label <- as.factor(email_phishing_data$label)

set.seed(123)
train_index <- createDataPartition(email_phishing_data$label, p = 0.8, list = FALSE)
train_data <- email_phishing_data[train_index, ]
test_data  <- email_phishing_data[-train_index, ]

#############################
# Model 1: Logistic Regression
#############################

log_model <- glm(label ~ ., data = train_data, family = binomial)
log_prob <- predict(log_model, test_data, type = "response")
log_pred <- ifelse(log_prob > 0.5, 1, 0)
log_pred <- factor(log_pred, levels = c(0,1))
cat("\n Logistic Regression Results \n")
confusionMatrix(log_pred, test_data$label)
log_roc <- roc(test_data$label, log_prob)
cat("AUC:", auc(log_roc), "\n")
plot(log_roc, main = "ROC - Logistic Regression")


#############################
# Model 2: KNN
#############################


library(FNN)  # Faster + more stable KNN

train_knn <- train_data[, -9]
test_knn <- test_data[, -9]

preProcValues <- preProcess(train_knn, method = c("center", "scale"))
train_knn <- predict(preProcValues, train_knn)
test_knn <- predict(preProcValues, test_knn)

train_label <- as.numeric(as.character(train_data$label))
test_label <- as.numeric(as.character(test_data$label))

set.seed(123)
knn_pred <- knn(train_knn, test_knn, train_label, k = 11, prob = TRUE)

knn_pred <- factor(knn_pred, levels = c(0,1))

cat("\n KNN Results \n")
confusionMatrix(knn_pred, test_data$label)


#############################
# Downsampling for Random Forest
#############################
minority_size <- sum(train_data$label == 1)
down_train <- 
  train_data %>%
  group_by(label) %>%
  sample_n(minority_size) %>%
  ungroup()

cat("\nBalanced Samples:\n")
table(down_train$label)

#############################
# Model 3: Random Forest (Balanced)
#############################


set.seed(123)
rf_model <- randomForest(label ~ ., data = down_train, ntree = 200, mtry = 2)

cat("\n Random Forest Results  \n")
rf_pred <- predict(rf_model, test_data)
confusionMatrix(rf_pred, test_data$label)

rf_prob <- predict(rf_model, test_data, type = "prob")[,2]
rf_roc <- roc(test_data$label, rf_prob)
cat("AUC:", auc(rf_roc), "\n")
plot(rf_roc, main = "ROC - Random Forest Downsampled")


#############################
# MODEL 4: Support Vector Machine (SVM)
#############################
svm_model <- svm(label ~ ., data=train_data, kernel="radial", probability=TRUE)

svm_pred <- predict(svm_model, test_data)
cat("\n SVM Results \n")
confusionMatrix(svm_pred, test_data$label)

svm_prob <- attr(predict(svm_model, test_data, probability=TRUE), "probabilities")[,2]
svm_roc <- roc(test_data$label, svm_prob)
cat("AUC:", auc(svm_roc), "\n")
plot(svm_roc, main="ROC - SVM")


#############################
# MODEL 5: XGBoost (Best Model)
#############################
library(xgboost)
library(Matrix)

train_matrix <- xgb.DMatrix(data = as.matrix(train_knn), label = train_label)
test_matrix  <- xgb.DMatrix(data = as.matrix(test_knn), label = test_label)

params <- list(objective="binary:logistic",
               eval_metric="auc",
               scale_pos_weight = (sum(train_label==0)/sum(train_label==1)))

xgb_model <- xgb.train(params=params, 
                       data=train_matrix, 
                       nrounds=100)

xgb_prob <- predict(xgb_model, test_matrix)
xgb_pred <- as.factor(ifelse(xgb_prob >= 0.5, 1, 0))

cat("\n XGBoost Results \n")
confusionMatrix(xgb_pred, test_data$label)

xgb_roc <- roc(test_data$label, xgb_prob)
cat("AUC:", auc(xgb_roc), "\n")
plot(xgb_roc, main="ROC - XGBoost")


#############################
# final Summary
#############################


cat("\n FINAL MODEL PERFORMANCE SUMMARY\n")
cat("-----------------------------------------\n")
cat("Logistic Regression AUC:", auc(log_roc), "\n")
cat("KNN Balanced Accuracy:", 0.54331, "\n") # From your result
cat("Random Forest AUC:", auc(rf_roc), "\n")
cat("SVM AUC:", auc(svm_roc), "\n")
cat("XGBoost AUC:", auc(xgb_roc), "<< BEST MODEL\n")
cat("-----------------------------------------\n")




