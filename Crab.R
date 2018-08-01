df <- read.csv("C:/Users/OmarVr/Downloads/small_train.data", sep = '\t', 
               header=T,encoding = 'UTF-8')
label <- read.csv("C:/Users/OmarVr/Downloads/small_train_upselling.labels", sep = '\t',
                  header = F, encoding = 'UTF-8')
new_data <- read.csv("C:/Users/OmarVr/Downloads/small_test.data", sep = '\t',
                  header = T, encoding = 'UTF-8')

count_na <- sapply(df, function(x) sum(length(which(is.na(x)))))
names(count_na) <- names(df)
count_na <- data.frame(count_na)
summary(count_na)
hist(as.numeric(count_na$count_na),breaks = 1000)
a <- ggplot(count_na, aes(as.numeric(count_na)) ) +
  geom_histogram(binwidth = 500)

percent_na_accepted = 0.80
df_na_reduced <- sapply(df, function(x) sum(length(which(is.na(x)))) < 50000*(1-percent_na_accepted))
df_na_reduced <- df[,df_na_reduced]

count_na <- sapply(df_na_reduced, function(x) sum(length(which(is.na(x)))))
names(count_na) <- names(df_na_reduced)
count_na <- data.frame(count_na)
summary(count_na)
hist(as.numeric(count_na$count_na),breaks = 100)
a <- ggplot(count_na, aes(as.numeric(count_na)) ) +
  geom_histogram(binwidth = 500)

summary(df_na_reduced)
str(df_na_reduced[complete.cases(df_na_reduced),])


library(mlbench)
library(caret)

correlation_matrix <- cor(df_na_reduced[complete.cases(df_na_reduced),1:38])
print(correlation_matrix)
high_corr <- findCorrelation(correlation_matrix, cutoff=0.5)
library(corrplot)
corrplot(correlation_matrix, type = "upper", order = "original", 
         tl.col = "black", tl.srt = 45)

str(df_na_reduced[complete.cases(df_na_reduced),-high_corr])

df_label <- cbind(df_na_reduced,label)
df_label <- df_label[complete.cases(df_label),-high_corr]
Large_factor_level_columns <- names(which(sapply(df_label, 
                                                 function(x) if (is.factor(x) & length(levels(x)) > 53 ) 
                                                   TRUE  
                                                 else 
                                                   FALSE)))

df_label <- df_label[,!(names(df_label) %in% Large_factor_level_columns)]
df_label$V1 <- as.factor(df_label$V1)


summary(as.factor(df_label$V1))


train <- sample(1:nrow(df_label), nrow(df_label)*0.7)
training <- df_label[train,]
testing <- df_label[-train,]

x_train <- training[,-47]
y_train <- training[,47]
y_train <- as.data.frame(y_train)
y_train$y_train <- as.factor(y_train$y_train)

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1 )
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE)

fit <- train(x_train,y_train$y_train, method="rf",trControl = fitControl)

stopCluster(cluster)
registerDoSEQ()

library(plyr)
var_importance <- (varImp(fit,scale = FALSE))
var_importance_sorted <- var_importance$importance
var_importance_sorted$var <- row.names.data.frame(var_importance_sorted)
var_importance_sorted <- arrange(var_importance_sorted, desc(Overall))
var_importance_sorted[1:18,]$var

df_label <- df_label[,(names(df_label) %in% c(var_importance_sorted[1:5,]$var,'V1'))]
df_label <- df_label[,!(names(df_label) %in% c('Var193'))]
df_label <- df_label[,!(names(df_label) %in% c('Var226'))]
df_label$Var219 <- gsub("[[:blank:]]", "", df_label$Var219)
df_label$Var219 <- sub("^$", "N", df_label$Var219)
df_label$Var219 <- as.factor(df_label$Var219)

scenarios_under <- df_label %>%
  group_by(Var226)%>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  filter(n > 4)
df_label <- semi_join(df_label,scenarios_under)
df_label <- droplevels(df_label)

set.seed(681)
train <- sample(1:nrow(df_label), nrow(df_label)*0.7)
#label <- t(df_label[-train,]$V1)
 logistic <- glm(V1~., data=df_label[train,], family = binomial)
# mats <- Map(function(x, y)
#   if (is.factor(x) & any(is.na(match(unique(x),unique(y)))))
#     x <- replace(x, which(x == dplyr::setdiff(levels(x), levels(y))[1]), NA)
#   else
#     x,
#   df_label[-train,],
#   df_label[train,])
# new_data <- do.call(cbind, lapply(mats, data.frame, stringsAsFactors=FALSE))
# names(new_data) <- names(df_label)
# new_data <- droplevels(new_data)


prob_logistic <- (predict.glm(logistic, df_label[-train,], type = 'response'))
predict_logistic <- rep('-1',nrow(df_label[-train,]))
predict_logistic[prob_logistic > 0.5] <- '1'
table(predict_logistic, df_label[-train,]$V1)
summary(predict_logistic)
error <- sum(label!=predict_logistic)/nrow(df_label[-train,])

library(pROC)
g <- roc(V1 ~ predict_logistic, data = df_label[-train,])
plot(g)

train <- sample(1:nrow(df_label), nrow(df_label)*0.7)

cluster <- makeCluster(detectCores() - 1 )
registerDoParallel(cluster)

fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 10, 
                           verboseIter = FALSE,
                           sampling = "up", 
                           allowParallel = TRUE)

mod_fit <- train(V1~.,  data=df_label[train,], method="glm", family="binomial",
                 trControl = fitControl, tuneLength = 5)

pred = predict(mod_fit, newdata=df_label[-train,])
confusionMatrix(data=pred, df_label[-train,]$V1)

stopCluster(cluster)
registerDoSEQ()


cluster <- makeCluster(detectCores() - 1 )
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", 
                           number = 10, 
                           verboseIter = FALSE,
                           sampling = "down", 
                           allowParallel = TRUE)

mod_fit_rf <- train(V1~.,  data=df_label[train,], method="rf",
                 trControl = fitControl, prox = TRUE)

pred_rf = predict(mod_fit_rf, newdata=df_label[-train,],type = 'raw')
confusionMatrix(data=pred_rf, df_label[-train,]$V1)

stopCluster(cluster)
registerDoSEQ()


cluster <- makeCluster(detectCores() - 1 )
registerDoParallel(cluster)

fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 10, 
                           verboseIter = FALSE,
                           sampling = "up", 
                           allowParallel = TRUE)

mod_fit_svm <- train(V1 ~., data=df_label[train,], method = "svmLinear",
                    trControl=fitControl,
                    tuneLength = 5)
pred_svm = predict(mod_fit_svm, newdata=df_label[-train,],type = 'raw')
confusionMatrix(data=pred_svm, df_label[-train,]$V1)




#Best performer

best_pred <- predict(mod_fit, newdata=df_label[-train,], type = 'prob')
best_pred <- as.data.frame(best_pred[,2])
best_pred <- round((best_pred - min(best_pred)) / (max(best_pred) - min(best_pred)) * 10,digits = 0)
names(best_pred) <- 'Calificacion'
best_pred$Predictions <- pred


# New data

new_data <- (new_data[,(names(new_data) %in% c(var_importance_sorted[1:5,]$var))])
new_data <- (new_data[complete.cases(new_data),])
scenarios_under <- new_data %>%
  group_by(Var226)%>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  filter(n > 1)
new_data <- semi_join(new_data,scenarios_under)
new_data <- droplevels(new_data)

new_pred <- predict(mod_fit, newdata=new_data, type = 'prob')
new_pred_logist <- predict(mod_fit, newdata=new_data, type = 'raw')
new_pred <- as.data.frame(new_pred[,2])
new_pred <- round((new_pred - min(new_pred)) / (max(new_pred) - min(new_pred)) * 10,digits = 0)
names(new_pred) <- 'Score'
new_pred$Predictions <- new_pred_logist

write.table(new_pred,"C:/Users/OmarVr/Downloads/small_test_pred__and_score.data", sep = '\t', row.names = FALSE)



