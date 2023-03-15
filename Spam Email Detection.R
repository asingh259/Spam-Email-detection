library(caret)
library(MASS)
library(gains)

#Reading the data from URL
spammail.df <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"), header = TRUE)

colname.df <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"), header = TRUE, row.names = NULL)

#Reading the values as Columns for Spammail.df columns
column.n <- colname.df[31:87,1]
column.n <- append(column.n,"Spam")

#Assigning column values to spammail.df
colnames(spammail.df) <- column.n

#Separating Spams and non-spams into two different dataframes
spammail.yes <- spammail.df[which(spammail.df$Spam==1),]
spammail.no <- spammail.df[which(spammail.df$Spam==0),]

#Average of Spams and non-spams
avg.spammail.yes <- colMeans(spammail.yes[1:57])
avg.spammail.no <- colMeans(spammail.no[1:57])

#Difference between Averages of Spams and non Spams
diff_average <- abs(avg.spammail.yes - avg.spammail.no)
diff_average

#10 Predictors with highest difference between Spam class and non spam class average
max_diff <- sort.list(diff_average, decreasing = TRUE)
head(max_diff,10)

# Predictors at columns 57,56,55,27,19,21,25,16,26,52 are the highest in terms of difference between
#their spams and respective non spam columns

highest_10 <- spammail.df[,c(57,56,55,27,19,21,25,16,26,52,58)]

set.seed(1)

#Partitioning data into training (80%) and validation(20%)
train.index <- createDataPartition(highest_10$Spam, p= 0.8, list = FALSE)
train.df <- highest_10[train.index,]
valid.df <- highest_10[-train.index,]

#Linear discriminant analysis on training data
spam.lda <- lda(Spam~., data = train.df, scale=T)

#Linear discriminant analysis on validation data
pred <- predict(spam.lda, valid.df, type="response")
summary(pred)

#Accuracy - confusion matrix

table(Predicted = pred$class, Actual = valid.df$Spam)
mean(pred$class==valid.df$Spam)

#gain
gain <- gains(valid.df$Spam, pred$posterior[,2], groups = 10)
gain
names(gain)

#lift chart
plot(c(0,gain$cume.pct.of.total*sum(valid.df$Spam))~c(0,gain$cume.obs), 
     xlab = "# cases", ylab = "Cumulative", main = "", type = "l")
lines(c(0,sum(valid.df$Spam))~c(0, dim(valid.df)[1]), lty = 5)


#decile chart and values
heights <- gain$mean.resp/mean(valid.df$Spam)
midpoints <- barplot(heights, names.arg = gain$depth,  ylim = c(0,9), col = "blue",  
                     xlab = "Percentile", ylab = "Decile lift", 
                     main = "Decile-chart")
text(midpoints, heights+0.5, labels=round(heights, 1), cex = 0.8)


# As per the confusion matrix, lift chart and Decile chart, we can conclude that the model is effective 
#in identifying spams and non spams 83% of the times. 
#Also, Lift has sufficiently good area under its curve compared to the naive rule line. 
#Even though the first decile is lower than the second decile, the chart seems to follow an ideal trend from second 
#decile onwards. First 5 deciles of the chart covers 90% of the variation. 
#Hence, we can conclude from the above parameters that the model is good in identifying spams. 
