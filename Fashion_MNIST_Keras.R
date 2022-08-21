library (tensorflow)
library(keras)
library(ggplot2)
library(tidyr)

#fashion MNIST dataset Ketas

fashion <- dataset_fashion_mnist()

#define X_train, y_train, X_test, y_test

X_train <- fashion$train$x
X_test <- fashion$test$x

y_train <- fashion$train$y
y_test <- fashion$test$y

dim(X_train)

image_1 <- as.data.frame(X_train[1,, ])
colnames(image_1) <- seq_len(ncol(image_1)) #column names to dataset (1:28)
image_1$y <- seq_len(nrow(image_1)) #y value rows
image_1 <- gather(image_1,"x","values",-y) #wider to longer format
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x=x,y=y,fill=values)) +
       geom_tile()+
       scale_y_reverse() +
       theme_minimal()
 
####### Modelling Data

#1. Lets reshape(2*2 to 1 array) and scale dataset
X_train <- array_reshape(X_train,c(nrow(X_train),28*28))
X_train <- X_train/255

X_test <- array_reshape(X_test,c(nrow(X_test),28*28))
X_test <- X_test/255

model <- keras_model_sequential() %>%
         layer_dense(400,activation='relu',input_shape = c(784)) %>%
         layer_dense(300,activation='relu') %>%
         layer_dense(64, activation = "relu") %>%
         layer_dense(10,activation='sigmoid')

model %>% compile(optimizer = 'adam',
                 loss = 'sparse_categorical_crossentropy',
                 metrics = c('accuracy'))
                 
model %>% fit(X_train,y_train,epochs=25)

#run model on test
model %>% evaluate(X_test,y_test)

y_predicted <- model %>%
               predict(X_test)

#predict class for first object
which.max(y_predicted[1,])

#cross check with test value.
y_test[1]


