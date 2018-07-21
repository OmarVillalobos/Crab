

df <- read.csv("/Users/omar/Downloads/small_train.data", sep = '\t', 
                 header=T,encoding = 'UTF-8')
label <- read.csv("/Users/omar/Downloads/small_train_upselling.labels", sep = '\t',
                  header = F, encoding = 'UTF-8')

count_na <- sapply(df, function(x) sum(length(which(is.na(x)))))
names(count_na) <- names(df)
count_na <- data.frame(count_na)
summary(count_na)
hist(as.numeric(count_na$count_na),breaks = 1000)

percent_na_accepted = 0.80
df_na_reduced <- sapply(df, function(x) sum(length(which(is.na(x)))) < 50000*(1-percent_na_accepted))
df_na_reduced <- df[,df_na_reduced]

count_na <- sapply(df_na_reduced, function(x) sum(length(which(is.na(x)))))
names(count_na) <- names(df_na_reduced)
count_na <- data.frame(count_na)
summary(count_na)
hist(as.numeric(count_na$count_na),breaks = 100)

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


control <- trainControl(method="repeatedcv", number=3, repeats=1)

model <- train(V1~., data=df_label, method="logreg", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)




