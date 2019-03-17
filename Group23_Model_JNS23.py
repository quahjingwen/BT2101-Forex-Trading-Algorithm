import numpy as np
import decimal
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers
from keras import metrics

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

import pandas as pd


class Crypto_Trade(QCAlgorithm):
    
    def Initialize(self):
  
        #self.Debug("START: Initialize")
        self.SetStartDate(2018,10,30)    #Set Start Date
        self.SetEndDate(2018,11,3)     #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Cash)
        self.currency = "GBPJPY 8G" #base currency
        self.addcurrency1 = "CHFJPY 8G"
        self.addcurrency2 = "GBPCHF 8G"
        self.AddForex(self.currency,Resolution.Hour)
        self.AddForex(self.addcurrency1,Resolution.Hour)
        self.AddForex(self.addcurrency2,Resolution.Hour)
        self.long_list =[]
        self.short_list =[]
        self.model =Sequential()
        self.datalist = np.array([])
        self.x=0
        
        #retraining variables
        self.count = 0
        self.currPeak = 100000 #curr portfolio value
        self.prevPV = 100000 #previous portfolio value
        self.currLoss = 0 #accumulated loss
        self.once = 0
        
        #self.Debug("End: Initialize")

    def OnData(self, data): #This function runs on every resolution of data mentioned. 
                            #(eg if resolution = daily, it will run daily, if resolution = hourly, it will run hourly.)
        
        #self.Debug("START: Ondata")
        currency_data = self.History([self.currency], 24, Resolution.Hour) # Asking for last 10 days of data
        currency_data['mid'] = pd.Series((currency_data['high']+currency_data['low'])/2)
        
        self.Debug("History is : " + str(currency_data))
        #L= len(currency_data)
        #self.Debug("The length is " + str (L))
        
        ## to add the initial datalist
        if self.once == 0:
            self.datalist = np.array([currency_data.mid]) #Get the inital close prices and make an array
        L = len(currency_data)
        self.Debug("The length is " + str (L))
        
        #Start of algo
        if not currency_data.empty: # Making sure the data is not empty and then only proceed with the algo
            
            data = np.array([currency_data.mid])  #Get the close prices and make an array
            # self.Debug("Close prices after making an array" + str(data))
            
            add_data = currency_data.tail(1)
            self.Debug("Checking data is : " + str(add_data))
            add_data = add_data.mid
            add_data = np.array([add_data])
            
            ##### THE RETRAINING CODE HERE#####
            
            # retraining of first dataset (first 24 hours)
            if self.once == 0:
                self.x =0
                self.once = 1
            
            ## if portfolio value > 100k don’t retrain (assume that we start with 100k)
            elif self.Portfolio.TotalPortfolioValue > 100000:
                # self.count += 1
                #currency_data.df['GBPJPY'].append(self.History(['GBPJPY'], 1, Resolution.Hour))
                ## append one more period
                self.datalist = np.concatenate([self.datalist, add_data], axis=1)
                self.x= 1
            
            # if portfolio value greater than currPeak (best performance of current model), reset currPeak
            elif self.Portfolio.TotalPortfolioValue > self.currPeak:
                # set the currPeak & prevPV value
                self.currPeak = self.Portfolio.TotalPortfolioValue
                self.prevPV = self.Portfolio.TotalPortfolioValue
                # self.count += 1
                for i in range(L//10):
                    ## positive reinforcement
                    self.datalist = np.concatenate([self.datalist, add_data], axis=1)
                    self.x= 1
            
            elif self.Portfolio.TotalPortfolioValue <= self.currPeak:
                loss = self.Portfolio.TotalPortfolioValue - self.prevPV
                # update currLoss
                self.currLoss += loss #accumulated loss
                self.prevPeak = self.Portfolio.TotalPortfolioValue
                ## If portfolio value goes down by x%, we retrain (x will be determined by backtesting)
                # if portfolio value decreases by 0.3%, retrain 
                if (self.currLoss/ self.currPeak) > 0.003:
                    self.x= 0
                    # reset currLoss
                    self.currPeak = 0
                    self.count = 0
                    self.currLoss = 0
                else:
                    # self.count += 1
                    self.x= 1
                    #currency_data.df['GBPJPY'].append(self.History(['GBPJPY'], 1, Resolution.Hour))
                    ## append one more period
                    self.datalist = np.concatenate([self.datalist, add_data], axis=1)
            if self.count == 10:
                self.x= 0
                self.count = 0
                
            elif self.x == 1:
                self.x= 1
                self.count += 1
                #currency_data.df['GBPJPY'].append(self.History(['GBPJPY'], 1, Resolution.Minute))
                ## append one more period
                self.datalist = np.concatenate([self.datalist, add_data], axis=1)
            self.Debug("Count:" + str(self.count))
            # data = self.datalist
            
            
            #Data preprocessing
            def moving_average(a, n=3) :
                ret = np.cumsum(a, dtype=float)
                ret[n:] = ret[n:] - ret[:-n]
                return ret[n - 1:] / n
            
            data_size = self.datalist.shape[1]
            self.Debug("Data size " + str(data_size))
            data_train = self.datalist[0:int(data_size*0.7)]
            data_test = self.datalist[int(data_size*0.7):]
            # Rolling time window, and calculate mean and standard error
            rolmean = np.array(moving_average(self.datalist, n=3))
            data_log_moving_avg_diff = data_train[:,2:] - [rolmean]
            data = data_log_moving_avg_diff
            ###################################
            
            
            #Data Preparation for input to LSTM
            X1 = data[:,0:L-5] #(0 to 5 data)
            self.Debug("X1 is " + str(X1))
            X2 = data[:,1:L-4] #(1 to 6 data)
            self.Debug("X2 is " + str(X2))
            X3 = data[:,2:L-3] #(#2 to 7 data) 
            self.Debug("X3 is " + str(X3))
        
            X= np.concatenate([X1,X2,X3],axis=0) # concatenate to join X1 X2 X3
            self.Debug("X after concatenate:  " + str(X))
            X_data= np.transpose(X) # # transpose to get in the form [0,1,2],[1,2,3],[2,3,4],[3,4,5]...
            self.Debug("X after transpose:  " + str(X_data))
        
            Y_data = np.transpose(data[:,3:L-2]) # to grt in form [ [3],[4],[5]....
            self.Debug("Y :  " + str(Y_data))
            
            #Normalize the data 
            scaler = MinMaxScaler() 
            scaler.fit(X_data)
            X_data = scaler.transform(X_data)
            self.Debug("X after transformation is " + str(X_data))
         
            scaler1 = MinMaxScaler()
            scaler1.fit(Y_data)
            Y_data = scaler1.transform(Y_data)
            self.Debug("Y data shpae " + str(Y_data.shape))
            self.Debug("Y after transformation is " + str(Y_data))
            
            if self.x==0:  #To make sure the model is build only once and avoid computation at every new data
                
                # self.Debug("RETRAINING!!!!")
                
                #USE TimeSeriesSplit to split data into n sequential splits
                tscv = TimeSeriesSplit(n_splits=2)
                
                # Make cells and epochs to be used in grid search.
                cells = [200,400]
                epochs  = [200,400]
                
                # creating a datframe to store final results of cross validation for different combination of cells and epochs
                df = pd.DataFrame(columns= ['cells','epoch','mse'])
                
                #Loop for every combination of cells and epochs. In this setup, 4 combinations of cells and epochs [100, 100] [ 100,200] [200,100] [200,200]
                for i in cells:
                    for j in epochs:
                        
                        cvscores = []
                        # to store CV results
                        #Run the LSTM in loop for every combination of cells an epochs and every train/test split in order to get average mse for each combination.
                        for train_index, test_index in tscv.split(X_data):
                            #self.Debug("TRAIN:", train_index, "TEST:", test_index)
                            X_train, X_test = X_data[train_index], X_data[test_index]
                            Y_train, Y_test = Y_data[train_index], Y_data[test_index]
                            
                            self.Debug("X_train input before reshaping :  " + str(X_train))
                            #self.Debug("X_test is" + str(X_test))
                            self.Debug("Y input before reshaping:  "+ str(Y_train))
                            #self.Debug("Y_test is" + str(Y_test))
                            
                            #self.Debug ( " X train [0] is " + str (X_train[0]))
                            #self.Debug ( " X train [1] is " + str (X_train[1]))
                            
                            
                            X_train= np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
                            self.Debug("X input to LSTM :  " + str(X_train))
                            X_test= np.reshape(X_test, (X_test.shape[0],1,X_test.shape[1]))
                            self.Debug("Y input to LSTM :  "+ str(Y_train))
                 
                            #self.Debug("START: LSTM Model")
                            #self.Debug(i)
                            #self.Debug(j)
                            model = Sequential()
                            model.add(LSTM(i, input_shape = (1,3), return_sequences = True))
                            model.add(Dropout(0.10))
                            model.add(LSTM(i,return_sequences = True))
                            model.add(LSTM(i))
                            model.add(Dropout(0.10))
                            model.add(Dense(1))
                            model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
                            model.fit(X_train,Y_train,epochs=j,verbose=0)
                            #self.Debug("END: LSTM Model")
                            
                            scores = model.evaluate(X_test, Y_test, verbose=0)
                            #self.Debug("%s: %f " % (model.metrics_names[1], scores[1]))
                            cvscores.append(scores[1])
                                
                        MSE= np.mean(cvscores)
                        #self.Debug("MSE" + str(MSE))
                        
                        #Create a dataframe to store output from each combination and append to final results dataframe df.
                        df1 = pd.DataFrame({ 'cells': [i], 'epoch': [j], 'mse': [MSE]})
                        self.Debug("Individual run ouput DF1" + str(df1))
                        #Appending individual ouputs to final dataframe for comparison
                        df = df.append(df1) 
                        
                
                self.Debug("Final table of DF"+ str(df))
                
                #Check the optimised values obtained from cross validation
                #This code gives the row which has minimum mse and store the values to O_values
                O_values = df[df['mse']==df['mse'].min()]
                
                # Extract the optimised  values of cells and epochs from above row (having min mse )
                O_cells = O_values.iloc[0][0]
                O_epochs = O_values.iloc[0][1]

                #Build model for whole data:
                # Repeating the model but for optimised cells and epochs
                
                X_data1= np.reshape(X_data, (X_data.shape[0],1,X_data.shape[1]))
                
                self.Debug("shape" + str(X_data1.shape))
                self.Debug("shape" + str(Y_data.shape))
                self.Debug("START: Final_LSTM Model")
                self.model = Sequential()
                self.model.add(LSTM(O_cells, input_shape = (1,3), return_sequences = True))
                self.model.add(Dropout(0.10))
                self.model.add(LSTM(O_cells,return_sequences = True))
                self.model.add(LSTM(O_cells))
                self.model.add(Dropout(0.10))
                self.model.add(Dense(1))
                self.model.compile(loss= 'mean_squared_error',optimizer = 'rmsprop', metrics = ['mean_squared_error'])
                self.model.fit(X_data1,Y_data,epochs=O_epochs,verbose=0)
                self.Debug("END: Final_LSTM Model")
                
            self.x=1    
                
            #Prepare new data for prediction based above model
            # Similar to as we did initially ( data prep for input to LSTM)
            
            X1_new = data[:,-3]
            #self.Debug(X1_new)
            X2_new = data[:,-2]
            #self.Debug(X2_new)
            X3_new = data[:,-1]
            #self.Debug(X3_new)
            
            X_new= np.concatenate([X1_new,X2_new,X3_new],axis=0)
            X_new= np.transpose(X_new)
            #self.Debug(X_new)
            
            scaler = MinMaxScaler() 
            scaler.fit(X_data)
            X_new = scaler.transform([X_new])
            #self.Debug(X_new)
            
            X_new= np.reshape(X_new,(X_new.shape[0],1,X_new.shape[1]))
            #self.Debug(X_new)

            self.Debug("Xnew" + str(X_new))
            self.Debug("Xnew shape" + str(X_new.shape))
            
            # Predicting with the LSTM model
            Predict = self.model.predict(X_new)
            
            #Needs to inverse transform as we transformed the data for LSTM input
            output = scaler1.inverse_transform(Predict)

            self.Debug("Output from LSTM model is" + str(output))
            
            
            #Checking the current price 
            price = currency_data.mid[-1]
            # self.Debug("Current price is" + str(price))
            
            #Make decision for trading based on the output from LSTM and the current price.
            #If output ( forecast) is greater than current price , we will buy the currency; else, do nothing.
            # Only one trade at a time and therefore made a list " self.long_list". 
            #As long as the currency is in that list, no further buying can be done.
            # Risk and Reward are defined: Ext the trade at 1% loss or 1 % profit.
            # Generally the LSTM model can predict above/below the current price and hence a random value is used
            #to scale it down/up. Here the number is 1.1 but can be backtested and optimised.
            
            # if output == "UP"
            # if output X% higher than price, in this case 5%
            if 1.05*output > price  and self.currency not in self.long_list and self.currency not in self.short_list :
                self.Debug("output is greater")
                # Buy the currency with X% of holdings, in this case 90%
                self.SetHoldings(self.currency, 0.9*0.5) 
                self.SetHoldings(self.addcurrency1, 0.9*0.3) 
                self.SetHoldings(self.addcurrency2, 0.9*0.2)
                self.long_list.append(self.currency)
                self.Debug("long")
                
            for currency in self.long_list:
                cost_basis = self.Portfolio[currency].AveragePrice
                #self.Debug("cost basis is " +str(cost_basis))
                # if 0.5% loss -> stop loss, if 1% profit, we will take it
                if  ((price <= float(0.995) * float(cost_basis)) or (price >= float(1.01) * float(cost_basis))):
                    #self.Debug("SL-TP reached")
                    #self.Debug("price is" + str(price))
                    #If true then sell the currencies in long positive
                    self.SetHoldings(currency, 0)
                    self.long_list.remove(currency)
                    # self.Debug("squared long")
                    
            ## if output == "DOWN"
            if 0.9*output < price  and self.currency not in self.long_list and self.currency not in self.long_list:
            #self.Debug("output is lesser")
            #Buy the currency with X% of holding in this case 90%x
                self.SetHoldings(self.currency, -0.9*0.5)
                self.SetHoldings(self.addcurrency1, -0.9*0.3)
                self.SetHoldings(self.addcurrency2, -0.9*0.2)
                self.short_list.append(self.currency)
                self.Debug("short")
                
            # if self.currency in self.short_list:
            #     cost_basis = self.Portfolio[self.currency].AveragePrice
            for currency in self.short_list:
                cost_basis = self.Portfolio[currency].AveragePrice
                
            #self.Debug("cost basis is " +str(cost_basis))
                #if curr price < average price by 1% or if curr price > 0.5% of average pric
                if  ((price <= float(0.99) * float(cost_basis)) or (price >= float(1.005) * float(cost_basis))):
                #self.Debug("SL-TP reached")
                #self.Debug("price is" + str(price))
                #If true then buy back
                    self.SetHoldings(currency, 0)
                    self.short_list.remove(currency)
                    self.Debug("squared short")
                #self.Debug("END: Ondata")
