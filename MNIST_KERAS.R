library(keras)
library(ggplot2)
library(tidyr)
library(ramify)

mnist <- dataset_mnist()

X_train <- mnist$train$x
X_test <- mnist$test$x

y_train <- mnist$train$y
y_test <- mnist$test$y

dim(X_train)
nrow(X_train)
ncol(X_train)


image_1 <- as.data.frame(X_train[1, , ])
image_1
colnames(image_1) <- seq_len(ncol(image_1))
image_1
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x","value",-y)
image_1$x <- as.integer(image_1$x)
image_1

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 2) +
  xlab("") +
  ylab("")

######## Flattern array 2 dimantion to 1 dimention

X_train <- array_reshape(X_train,c(nrow(X_train),28*28))
X_train <- X_train/255 #scale value
dim(X_train)

X_test <- array_reshape(X_test,c(nrow(X_test),28*28))
X_test <- X_test/255 #scale value
dim(X_test)

## Basic Neural Network

#training dataset
model <- keras_model_sequential() %>%
         layer_dense(200,activation = 'relu') %>%
         layer_dense(10,activation='sigmoid')

model %>% compile( optimizer = 'adam',
                   loss = 'sparse_categorical_crossentropy',
                   metrics = c('accuracy'))

model %>% fit(X_train,y_train,epochs=5)

#test dataset
model %>% evaluate(X_test,y_test)

#predict

y_predicted <- model %>% predict(X_test)

y_predicted[1,]

#look at particular predicted value
which.max(y_predicted[2,])

#look at all predicted values
argmax(y_predicted)

#
y_test
