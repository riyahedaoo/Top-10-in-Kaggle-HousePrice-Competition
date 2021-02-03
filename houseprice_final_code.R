#libraries
library(DMwR)
library(readr)
library(dplyr)
library(caret)
library(caretEnsemble)
library(doParallel)
library(xgboost)
library(Matrix)
library(randomForest)


#import data:
train <- read_csv("C:/Users/Saniya and Family/Downloads/train.csv")
test <- read_csv("C:/Users/Saniya and Family/Downloads/test.csv")
result <- read_csv("C:/Users/Saniya and Family/Downloads/sample_submission.csv")

#outliers in train:
train$LotArea[train$LotArea > quantile(train$LotArea,0.95)] <- quantile(train$LotArea,0.95)
train$BsmtFinSF1[train$BsmtFinSF1 > quantile(train$BsmtFinSF1,0.95)] <- quantile(train$BsmtFinSF1,0.95)
train$BsmtFinSF2[train$BsmtFinSF2 > quantile(train$BsmtFinSF2,0.95)] <- quantile(train$BsmtFinSF2,0.95)
train$BsmtUnfSF[train$BsmtUnfSF > quantile(train$BsmtUnfSF,0.95)] <- quantile(train$BsmtUnfSF,0.95)
train$GrLivArea[train$GrLivArea > quantile(train$GrLivArea,0.95)] <- quantile(train$GrLivArea,0.95)
train$GarageArea[train$GarageArea > quantile(train$GarageArea,0.95)] <- quantile(train$GarageArea,0.95)

train$LotArea[train$LotArea < quantile(train$LotArea,0.05)] <- quantile(train$LotArea,0.05)
train$BsmtFinSF1[train$BsmtFinSF1 < quantile(train$BsmtFinSF1,0.05)] <- quantile(train$BsmtFinSF1,0.05)
train$BsmtFinSF2[train$BsmtFinSF2 < quantile(train$BsmtFinSF2,0.05)] <- quantile(train$BsmtFinSF2,0.05)
train$BsmtUnfSF[train$BsmtUnfSF < quantile(train$BsmtUnfSF,0.05)] <- quantile(train$BsmtUnfSF,0.05)
train$GrLivArea[train$GrLivArea < quantile(train$GrLivArea,0.05)] <- quantile(train$GrLivArea,0.05)
train$GarageArea[train$GarageArea < quantile(train$GarageArea,0.05)] <- quantile(train$GarageArea,0.05)


#combine data
test$SalePrice <- result$SalePrice
data <- rbind(train,test)
data <- data[,-1]

names(data)[names(data) == "2ndFlrSF"]  <- "SndFlrSF"
names(data)[names(data) == "1stFlrSF"]  <- "FrstFlrSF"
names(data)[names(data) == "3SsnPorch"] <- "TrdSsnPorch"


#Visualise y
hist(train$SalePrice)
hist(log(train$SalePrice))

data$SalePrice <- log(data$SalePrice)

#Separate num and chr
datanum <- select_if(data,is.numeric)
datachr <- select_if(data,is.character)

#Fill NA for numeric
names(which(sapply(datanum, anyNA)))
datanum[is.na(datanum)] <- 0

#Fill NA for chr
names(which(sapply(datachr, anyNA)))

datachr$MSZoning[is.na(datachr$MSZoning)] <- "None"
datachr$Exterior1st[is.na(datachr$Exterior1st)] <- "None"
datachr$Utilities[is.na(datachr$Utilities)] <- "None"
datachr$Exterior2nd[is.na(datachr$Exterior2nd)] <- "None"
datachr$MasVnrType[is.na(datachr$MasVnrType)] <- "None"
datachr$Electrical[is.na(datachr$Electrical)] <- "None"
datachr$KitchenQual[is.na(datachr$KitchenQual)] <- "None"
datachr$Functional[is.na(datachr$Functional)] <- "None"
datachr$SaleType[is.na(datachr$SaleType)] <- "None"
datachr$ExterCond[is.na(datachr$ExterCond)] <- "None"
datachr$BsmtQual[is.na(datachr$BsmtQual)] <- "None"
datachr$BsmtCond[is.na(datachr$BsmtCond)] <- "None"
datachr$GarageQual[is.na(datachr$GarageQual)] <- "None"
datachr$GarageCond[is.na(datachr$GarageCond)] <- "None"
datachr$PoolQC[is.na(datachr$PoolQC)] <- "None"
datachr$HeatingQC[is.na(datachr$HeatingQC)] <- "None"
datachr$FireplaceQu[is.na(datachr$FireplaceQu)] <- "None"
datachr$BsmtFinType1[is.na(datachr$BsmtFinType1)] <- "None"
datachr$BsmtFinType2[is.na(datachr$BsmtFinType2)] <- "None"
datachr$BsmtExposure[is.na(datachr$BsmtExposure)] <- "None"



#Combining variables:
#pool
datachr$Pool <- ifelse(is.na(datachr$PoolQC),0,1)
#fireplace
datachr$Fire <- ifelse(is.na(datachr$FireplaceQu),0,1)
#Garage
datachr$Garage <- ifelse(is.na(datachr$GarageType),0,1)
#Basement
datachr$Basement1 <- ifelse(is.na(datachr$BsmtFinType1),0,1)
#Basement
datachr$Basement2 <- ifelse(is.na(datachr$BsmtFinType2),0,1)
#Total Area
datanum$TotalArea <- datanum$FrstFlrSF+datanum$SndFlrSF+datanum$TotalBsmtSF

#Total Bath
datanum$TBaths <- datanum$FullBath + datanum$BsmtFullBath + 0.5*(datanum$HalfBath + datanum$BsmtHalfBath)
#Total Porch Area
datanum$TPorch <- datanum$OpenPorchSF + datanum$ScreenPorch + datanum$TrdSsnPorch + datanum$EnclosedPorch + datanum$WoodDeckSF

#removing not required
colnames(datachr)
datanum <- datanum[,-c(12,13,14,17,18,19,20,28:32)]
datachr <- datachr[,-c(34:37,39,33)]
#too much NA - Alley, MiscFeatures, all Bsmt, 
datachr <- datachr[,-c(3,34,35,41)]


#converting to num:
table(datachr$Fence)
datachr$Fence <- data$Fence
datachr$Fence <- ifelse(is.na(datachr$Fence),0,1)

vars2recode <- c("ExterCond","BsmtQual","BsmtCond","GarageQual",
                 "GarageCond","FireplaceQu","KitchenQual","HeatingQC","PoolQC")

for(var2recode in vars2recode) {
  datachr[unlist(datachr[,var2recode] == 'Ex'),var2recode]    <- "5"
  datachr[datachr[,var2recode] == 'Gd',var2recode]            <- "4"
  datachr[datachr[,var2recode] == 'TA',var2recode]            <- "3"
  datachr[datachr[,var2recode] == 'Fa',var2recode]            <- "2"
  datachr[datachr[,var2recode] == 'Po',var2recode]            <- "1"
  datachr[datachr[,var2recode] == 'None',var2recode]          <- "0"
}

datachr$ExterCond <- as.numeric(datachr$ExterCond)
datachr$BsmtQual <- as.numeric(datachr$BsmtQual)
datachr$BsmtCond <- as.numeric(datachr$BsmtCond)
datachr$GarageQual <- as.numeric(datachr$GarageQual)
datachr$GarageCond <- as.numeric(datachr$GarageCond)
datachr$FireplaceQu <- as.numeric(datachr$FireplaceQu)
datachr$KitchenQual <- as.numeric(datachr$KitchenQual)
datachr$HeatingQC <- as.numeric(datachr$HeatingQC)
datachr$PoolQC <- as.numeric(datachr$PoolQC)

datachr[datachr[,'BsmtExposure'] == 'Gd','BsmtExposure']    <-"4"
datachr[datachr[,'BsmtExposure'] == 'Av','BsmtExposure']    <-"3"
datachr[datachr[,'BsmtExposure'] == 'Mn','BsmtExposure']    <-"2"
datachr[datachr[,'BsmtExposure'] == 'No','BsmtExposure']    <-"1"
datachr[datachr[,'BsmtExposure'] == 'None','BsmtExposure']  <-"0"
datachr$BsmtExposure <- as.numeric(datachr$BsmtExposure)

datachr[datachr[,'BsmtFinType1'] == 'GLQ','BsmtFinType1']   <-"6"
datachr[datachr[,'BsmtFinType1'] == 'ALQ','BsmtFinType1']   <-"5"
datachr[datachr[,'BsmtFinType1'] == 'BLQ','BsmtFinType1']   <-"4"
datachr[datachr[,'BsmtFinType1'] == 'Rec','BsmtFinType1']   <-"3"
datachr[datachr[,'BsmtFinType1'] == 'LwQ','BsmtFinType1']   <-"2"
datachr[datachr[,'BsmtFinType1'] == 'Unf','BsmtFinType1']   <-"1"
datachr[datachr[,'BsmtFinType1'] == 'None','BsmtFinType1']  <-"0"
datachr$BsmtFinType1 <- as.numeric(datachr$BsmtFinType1)

datachr[datachr[,'BsmtFinType2'] == 'GLQ','BsmtFinType2']   <-"6"
datachr[datachr[,'BsmtFinType2'] == 'ALQ','BsmtFinType2']   <-"5"
datachr[datachr[,'BsmtFinType2'] == 'BLQ','BsmtFinType2']   <-"4"
datachr[datachr[,'BsmtFinType2'] == 'Rec','BsmtFinType2']   <-"3"
datachr[datachr[,'BsmtFinType2'] == 'LwQ','BsmtFinType2']   <-"2"
datachr[datachr[,'BsmtFinType2'] == 'Unf','BsmtFinType2']   <-"1"
datachr[datachr[,'BsmtFinType2'] == 'None','BsmtFinType2']  <-"0"
datachr$BsmtFinType2 <- as.numeric(datachr$BsmtFinType2)


datachr[datachr[,'Functional'] == 'Typ','Functional']   <-"8"
datachr[datachr[,'Functional'] == 'Min1','Functional']   <-"7"
datachr[datachr[,'Functional'] == 'Min2','Functional']   <-"6"
datachr[datachr[,'Functional'] == 'Mod','Functional']   <-"5"
datachr[datachr[,'Functional'] == 'Maj1','Functional']   <-"4"
datachr[datachr[,'Functional'] == 'Maj2','Functional']   <-"3"
datachr[datachr[,'Functional'] == 'Sev','Functional']  <-"2"
datachr[datachr[,'Functional'] == 'None','Functional']  <-"1"
datachr$Functional <- as.numeric(datachr$Functional)




dmy <- dummyVars(" ~ .", data = datachr, fullRank = T)
dataf <- data.frame(predict(dmy, newdata = datachr))

str(dataf)

#Train & Test combine:
data <- cbind(dataf,datanum)
trainD <- data[c(1:1460),]
testD <- data[c(1461:2919),]


preProcValues    <- preProcess(as.data.frame(trainD[,names(trainD) %in% names(datanum)]), method = c("nzv","BoxCox","center", "scale"))
trainTransformed <- predict(preProcValues, as.data.frame(trainD))
testTransformed  <- predict(preProcValues, as.data.frame(testD))

#attach saleprice and removing old 
testTransformed <- testTransformed[,-141]
trainTransformed <- trainTransformed[,-141]
trainTransformed$SalePrice <- train$SalePrice
trainTransformed$SalePrice <- log(trainTransformed$SalePrice)

#variable importance:
rf       <- randomForest(modformula, data=trainTransformed, method="rf", ntrees=1000)

var <- varImp(rf)

df            <- varImp(rf)
indices_imp   <- order(df[,1], decreasing=T)
importance_df <- data.frame(var=rownames(df)[indices_imp],imp=df[,1][indices_imp])

nrow(importance_df)
vars <- c("Id",rownames(df)[indices_imp][1:150],"SalePrice")

trainTransformed <- trainTransformed[,names(trainTransformed) %in% vars]
testTransformed  <- testTransformed[,names(testTransformed) %in% vars]

# Feature plot
featurePlot(x=trainTransformed[,-82], y=trainTransformed$SalePrice, between= list(x=1, y=1),type=c("g","p","smooth"))

#formula
modformula <- as.formula(paste("SalePrice ~ .+ I(GrLivArea^2) + I(TBaths^2) + I(YearBuilt^2) +
                                I(YearRemodAdd^2) + I(TotalArea^2) + I(LotArea^2) + I(OverallQual^2)"))



#model list

registerDoParallel(4)
getDoParWorkers()

ctrl <- trainControl(method="cv",number=3, savePredictions="final", allowParallel=T, index=createFolds(trainTransformed$SalePrice,5))

model_list <- caretList( modformula, data=trainTransformed, trControl=ctrl,
                         tuneList = list(
                           modxgb=caretModelSpec(method="xgbTree", 
                                                 tuneGrid=expand.grid(eta=c(0.05), 
                                                                      gamma=0,
                                                                      max_depth=c(100),
                                                                      nrounds=c(1000),
                                                                      colsample_bytree=c(0.5),
                                                                      min_child_weight=c(4),
                                                                      subsample=c(0.5))),
                           modsvm =caretModelSpec(method="svmRadial",
                                                  tuneGrid= expand.grid(C=c(0.25,0.5,0.75,1,1.25,1.5),
                                                                        sigma=c(0.01,0.15,0.2))),
                           modgbm =caretModelSpec(method="gbm",
                                                  tuneGrid=expand.grid(n.trees = seq(100,1000,by=400),
                                                                       interaction.depth = c(1,2),
                                                                       shrinkage = c(0.01,0.1),
                                                                       n.minobsinnode = c(10,30,50))),
                           modrf =caretModelSpec(method="parRF",
                                                 tuneGrid=expand.grid(mtry=c(2,8,12,16,20,30,90))),
                           modrdg=caretModelSpec(method="glmnet",
                                                 tuneGrid=expand.grid(alpha = 0, 
                                                                      lambda = seq(0.01, 0.9, by=0.01))),
                           modlas =caretModelSpec(method="glmnet",
                                                  tuneGrid=expand.grid(alpha = 1, 
                                                                       lambda = c(0.0045)))))



#predictions
ensemble <- caretEnsemble(model_list, trControl=ctrl, metric="RMSE")


pe <- predict(ensemble, trainTransformed)
pxg <- predict(model_list$modxgb, trainTransformed)
psvm<- predict(model_list$modsvm, trainTransformed)
prf<- predict(model_list$modrf, trainTransformed)
prdg <- predict(model_list$modrdg, trainTransformed) 
plas <- predict(model_list$modlas, trainTransformed)

regr.eval(trainTransformed$SalePrice, pxg)



pe <- predict(ensemble, testTransformed)
pxg <- predict(model_list$modxgb, testTransformed)
psvm<- predict(model_list$modsvm, testTransformed)
prf<- predict(model_list$modrf, testTransformed)
prdg <- predict(model_list$modrdg, testTransformed) 
plas <- predict(model_list$modlas, testTransformed)

p <-  0.5*prf + 0.5*pxg
range(p)
ptest <- exp(p)
result$SalePrice <- ptest

result[result$SalePrice > exp(12.5) ,"SalePrice"]   <- result$SalePrice[result$SalePrice > exp(12.5)] * 1.01
result[result$SalePrice < exp(11.2) ,"SalePrice"]   <- result$SalePrice[result$SalePrice < exp(11.2)] * 0.89

write.csv(result,"house.csv", row.names = F)















