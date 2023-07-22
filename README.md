# Cancer-Prediction-System
import pandas as pd
dataset=pd.read_csv("Cancer_Data.csv")
y=dataset['diagnosis']
import seaborn as sns
y_cat=pd.get_dummies(y)
x=dataset.drop("diagnosis",axis=1)
from keras.models import Sequential
model=Sequential()
from keras.layers import Dense
model.get_config()
model.add(  Dense(
                kernel_initializer="zero",
                units=4,
                input_shape=(32,),
                activation="relu",
                bias_initializer="zero",
       )
         )
model.add(  Dense(
                kernel_initializer="zero",
                units=3,
                input_shape=(32,),
                activation="relu",
                bias_initializer="zero",
       )
         )
model.add(  Dense(
                kernel_initializer="zero",
                units=2,
                input_shape=(32,),
                activation="relu",
                bias_initializer="zero",
       )
         )
model.compile(loss="categorical_crossentropy")
model.fit(x,y_cat)
dataset=pd.read_csv("Cancer_Data.csv")
y=dataset['diagnosis']
import seaborn as sns
y_cat=pd.get_dummies(y)
x=dataset.drop("diagnosis",axis=1)
from keras.models import Sequential
model=Sequential()
from keras.layers import Dense
model.get_config()
model.add(  Dense(
                kernel_initializer="zero",
                units=4,
                input_shape=(32,),
                activation="relu",
                bias_initializer="zero",
       )
         )
         model.add(Dense(
                kernel_initializer="zero",
                units=3,
                input_shape=(32,),
                activation="relu",
                bias_initializer="zero",))
                model.add(Dense(
                kernel_initializer="zero",
                units=2,
                input_shape=(32,),
                activation="relu",
                bias_initializer="zero",))
                model.compile(loss="categorical_crossentropy")
                model.fit(x,y_cat)
