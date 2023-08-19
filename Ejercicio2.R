#Librerias
library(keras)
library(tensorflow)
#install_tensorflow()
library("readxl")
library(dplyr)
library(ggplot2)
library(recipes)
library(forecast)
library(lubridate)

#Leer xls files
consumo<-read_excel("Consumo.xlsx")

consumo$Fecha<-as.Date(consumo$Fecha, "%Y/%m/%d")
str(consumo)

#Serie de tiempo con LSTM
#Consumo de Diesel
fecha<-consumo[,'Fecha']
diesel<-consumo[,'Diesel alto azufre']
dieselc<-consumo[c('Fecha','Diesel alto azufre')]

diesel_ts<-ts(dieselc$`Diesel alto azufre`, start = c(2001,1),frequency = 12)

#Gráfico de la serie a utilizar
serie<-diff(diesel_ts) #Quitar esto para ver la predicción sin estacionarizar la serie en media
plot(serie)

#Normalizar la serie
serie_norm<-scale(serie)

#Transformar a una serie supervisada
lagged<-c(rep(NA,1),serie_norm[1:(length(serie_norm)-1)])
supervisada<-as.data.frame(cbind(lagged,serie_norm))
colnames(supervisada)<-c("x-1","x")
supervisada[is.na(supervisada)]<-0

#Cantidad de elementos de cada conjunto
entrenamiento<-round(0.6*length(serie))
val_prueba<-round(0.2*length(serie))

#El test son los últimos
test<-tail(supervisada,val_prueba)
#Se corta la matriz
supervisada<-supervisada %>% head(nrow(.)-val_prueba)
#Se saca el conjunto de validación y se corta nuevamente
validation<-supervisada %>% tail(val_prueba)
supervisada<-head(supervisada,nrow(supervisada)-val_prueba)
#El train son los que quedan
train<-supervisada
rm(supervisada)

#Division en entrenamiento, prueba y validación
y_train<-train[,2]
x_train<-train[,1]
y_val<-validation[,2]
x_val<-validation[,1]
y_test<-test[,2]
x_test<-test[,1]

#Convertir a matrices
paso <- 1
caracteristicas<-1 #es univariada
dim(x_train) <- c(length(x_train),paso,caracteristicas)
dim(y_train) <- c(length(y_train),caracteristicas)
dim(x_test) <- c(length(x_test),paso,caracteristicas)
dim(y_test) <- c(length(y_test),caracteristicas)
dim(x_val) <- c(length(x_val),paso,caracteristicas)
dim(y_val) <- c(length(y_val),caracteristicas)

#Creando el modelo
lote = 1
unidades<-1
modelo1<-keras_model_sequential()
modelo1 %>% 
  layer_lstm(unidades, batch_input_shape=c(lote,paso,caracteristicas),
             stateful = T) %>%
  layer_dense(units = 1)

summary(modelo1)

#Compilar el modelo
modelo1 %>%
  compile(
    optimizer = "rmsprop",
    loss = "mse"
  )

#Entrenar el modelo
epocas <- 50
history <- modelo1 %>% fit(
  x = x_train,
  y = y_train,
  validation_data = list(x_val, y_val),
  batch_size = lote,
  epochs = epocas,
  shuffle = FALSE,
  verbose = 0
)

#Graficar el modelo
plot(history)

#Evaluar el modelo
print("Entrenamiento")
modelo1 %>% evaluate(
  x = x_train,
  y = y_train
)
print("Validación")
modelo1 %>% evaluate(
  x = x_val,
  y = y_val
)
print("Prueba")
modelo1 %>% evaluate(
  x = x_test,
  y = y_test
)

#Predicción modelo 1
prediccion_fun <- function(data,modelo, batch_size,scale,center,dif=F, Series=NULL,n=1){
  prediccion <- numeric(length(data))
  if (dif==F){
    for(i in 1:length(data)){
      X = data[i]
      dim(X) = c(1,1,1)
      yhat = modelo %>% predict(X, batch_size=batch_size)
      # invert scaling
      yhat = yhat*scale+center
      # store
      prediccion[i] <- yhat
    }
  }else{
    for(i in 1:length(data)){
      X = data[i]
      dim(X) = c(1,1,1)
      yhat = modelo1 %>% predict(X, batch_size=batch_size)
      # invert scaling
      yhat = yhat*scale+center
      # invert differencing
      yhat  = yhat + Series[(n+i)]
      # store
      prediccion[i] <- yhat
    }
  }
  
  return(prediccion)
}

prediccion_val <- prediccion_fun(x_val,modelo1,1,attr(serie_norm,"scaled:scale"),
                                 attr(serie_norm,"scaled:center"),dif=T,diesel_ts,entrenamiento              
)
prediccion_test <- prediccion_fun(x_test,modelo1,1,attr(serie_norm,"scaled:scale"),
                                  attr(serie_norm,"scaled:center"),dif=T,diesel_ts,entrenamiento+val_prueba               
)

#Gráfica de la predicción
serie<-diesel_ts
serie_test<- tail(serie,val_prueba)
serie<-head(serie,length(serie)-val_prueba)
serie_val<-tail(serie,val_prueba)
serie<-head(serie,length(serie)-val_prueba)
serie_train <- serie


df_serie_total<-data.frame(pass=as.matrix(diesel_ts), date=zoo::as.Date(time(diesel_ts)))
df_serie_val<-data.frame(pass=prediccion_val, date=zoo::as.Date(time(serie_val)))
df_serie_test<-data.frame(pass=prediccion_test, date=zoo::as.Date(time(serie_test)))


df_serie_total$class <- 'real'
df_serie_val$class <- 'validacion'
df_serie_test$class <- 'prueba'

df_serie<-rbind(df_serie_total, df_serie_val,df_serie_test)
df_serie$class<-factor(df_serie$class,levels = c('real','validacion','prueba'))
ggplot(df_serie,aes(x = date, y = pass, colour = class)) +
  geom_line()

#Modelo 2 (más complejo)
# Capa lstm 1 
unit_lstm1 <- 64
dropout_lstm1 <- 0.01
recurrent_dropout_lstm1 <- 0.01

# capa lstm 2 settings
unit_lstm2 <- 32
dropout_lstm2 <- 0.01
recurrent_dropout_lstm2 <- 0.01

timesteps=1


# initiate model sequence
modelo2 <- keras_model_sequential()

modelo2 %>%
  
  # lstm1
  layer_lstm(
    name = "lstm1",
    units = unit_lstm1,
    input_shape = c(timesteps, 1),
    dropout = dropout_lstm1,
    recurrent_dropout = recurrent_dropout_lstm1,
    return_sequences = TRUE
  ) %>%
  
  # lstm2
  layer_lstm(
    name = "lstm2",
    units = unit_lstm2,
    dropout = dropout_lstm2,
    recurrent_dropout = recurrent_dropout_lstm2,
    return_sequences = FALSE
  ) %>%
  
  
  # output layer
  layer_dense(
    name = "output",
    units = 1
  )


# compile the model
modelo2 %>%
  compile(
    optimizer = "rmsprop",
    loss = "mse"
  )

# model summary
summary(modelo2)

#Entrenamos el modelo 2
epocas <- 50
history <- modelo2 %>% fit(
  x = x_train,
  y = y_train,
  validation_data = list(x_val, y_val),
  batch_size = lote,
  epochs = epocas,
  shuffle = FALSE,
  verbose = 0
)

#Evaluamos el modelo 2
#Entrenamiento
modelo2 %>% evaluate(
  x = x_train,
  y = y_train
)

#Validacion 
modelo2 %>% evaluate(
  x = x_val,
  y = y_val
)

#Prueba
modelo2 %>% evaluate(
  x = x_test,
  y = y_test
)

#Predicción modelo 2
prediccion_val_2 <- prediccion_fun(x_val,modelo2,1,attr(serie_norm,"scaled:scale"),
                                   attr(serie_norm,"scaled:center"),dif=T,diesel_ts,entrenamiento              
)
prediccion_test_2 <- prediccion_fun(x_test,modelo2,1,attr(serie_norm,"scaled:scale"),
                                    attr(serie_norm,"scaled:center"),dif=T,diesel_ts,entrenamiento+val_prueba               
)

#Gráfica de la predicción
df_serie_val_2<-data.frame(pass=prediccion_val_2, date=zoo::as.Date(time(serie_val)))
df_serie_test_2<-data.frame(pass=prediccion_test_2, date=zoo::as.Date(time(serie_test)))


df_serie_total$class <- 'real'
df_serie_val_2$class <- 'validacion'
df_serie_test_2$class <- 'prueba'

df_serie_2<-rbind(df_serie_total, df_serie_val_2,df_serie_test_2)
df_serie$class<-factor(df_serie_2$class,levels = c('real','validacion','prueba'))
ggplot(df_serie,aes(x = date, y = pass, colour = class)) +
  geom_line()
