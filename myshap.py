import pickle
import pandas as pd
import numpy as np
import shap
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

from data_preprocess import Data_preprocessing


class Shap_Interpretability:
    '''
    Class have AIX Classification functionality and other helper functions
    '''
    def __init__(self):
        pass
    
        
    def get_shap_tree(self):
        
        # TREE
        
        try :
            
            explainer = shap.TreeExplainer(Data_preprocessing.model)
        
        except Exception :
            model = Data_preprocessing.model.best_estimator_
            explainer = shap.TreeExplainer(model)

        
        return explainer, 'Tree'
    
    
    def get_shap_linear(self):
        
        # LINEAR
        
        X_train_bd = Data_preprocessing.X_train.copy()
        X_train_bd_process, all_features = Data_preprocessing.pre_process(X_train_bd)
        
        try:
            explainer = shap.LinearExplainer(Data_preprocessing.model,X_train_bd_process )
            
        except Exception:
            model = Data_preprocessing.model.best_estimator_
            explainer = shap.LinearExplainer(model,X_train_bd_process )
        
        return explainer, 'Linear'
    
    
    def get_shap_kernel(self):
        
        """
        This method is used to automate Kernel Shap
        """
        
        X_train_bd = Data_preprocessing.X_train.copy()
        X_train_bd_process, all_features = Data_preprocessing.pre_process(X_train_bd)
        
        datax = shap.kmeans(X_train_bd_process,10)
        
        k_explainer = shap.KernelExplainer(Data_preprocessing.model.predict_proba, datax)
        
        return k_explainer, 'Kernel'
    
        
        
    
    def model_type(self):
        '''
        Handling model type and Hyperparameter tuning issues
        '''
        try:
            # Tree
            explainer = self.get_shap_tree()
            
        except (Exception, AttributeError):
            try:
                # Linear
                explainer = self.get_shap_linear()
            
            except(Exception, AttributeError):
                # Kernel
                explainer = self.get_shap_kernel()
            
        finally:
            return explainer
        
        
    @staticmethod
    def shap_helper(limit, explainer):
        
        '''Helper function'''
        
        X_global = Data_preprocessing.X_test.copy()
        shap_preprocessed, all_features =  Data_preprocessing.pre_process(X_global)
        
        if limit == None:
            start = 0
            end =   X_global.shape[0]
        else:
        
            start = limit[0]
            end   = limit[1]

        feature_display = pd.DataFrame(shap_preprocessed.iloc[start:end,:], columns=all_features)
        shap_values = explainer.shap_values(shap_preprocessed.iloc[start:end,:])
        
        return shap_values, feature_display
    
    
    def shap_global(self, limit = None,which_class=None):

        
        """
        This method is used to visualize summary plot using Shap.
        
        Parameters
        -----------------------------------------------------------------------------------
        
        limit     : Default is to use whole dataset, limit can be provided to use the subset of a dataset
                    eg. limit = [0,12]
                    
        which_class : (Default is None) To see the interpretations of class wise one can utilize which_class argument
                       Default for binary class: which class will be 1
                        & for Multi-class it will be the most frequent class in Y_train
                        
                        Y_train also require in dataframe format.
        
        
        Returns
        -----------------------------------------------------------------------------------
        Summary plot
        
        """
        
        print("/SHAP GLOBAL")
        explainer = self.model_type()
        
        shap_values, feature_display = self.shap_helper(limit,explainer[0])
        
        
        print(shap.summary_plot(shap_values, feature_display,plot_type = 'bar'))
        
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        print(shap_values.shape)


        if shap_values.ndim > 2 and Data_preprocessing.y_test.nunique()[0] == 2:
            print("ndim > 2")

            if which_class == None:
                print("Default : Global data for class 1")
                which_class = 1

            
            return shap.summary_plot(shap_values[which_class], feature_display,  plot_type = 'dot')
        
        elif shap_values.ndim > 2 and Data_preprocessing.y_test.nunique()[0] > 2:
            
            if which_class == None:
                print("----x------x---x-Multiclass-x------x------x")
                print("Default : Global data for most freq class ")
                label_column = Data_preprocessing.y_test.columns[0]
                most_freq_label = Data_preprocessing.y_test[label_column].value_counts().index[0]  
                which_class = most_freq_label
                
            return shap.summary_plot(shap_values[which_class], feature_display,  plot_type = 'dot')
                
        else: 
            print("ndim 1")

            return shap.summary_plot(shap_values, feature_display, plot_type = 'dot')
        
        
        
    def shap_pdp(self, name_list, limit=None,  which_class = None ):
        
        """
        This method is used to visualize partial dependence plot using Shap.
        
        Parameters
        -----------------------------------------------------------------------------------
        name_list : Can provide list having two feature names
                    eg. name_list = ['Age','Salary']
        
        limit     : Default is to use whole dataset, limit can be provided to use the subset of a dataset
                    eg. limit = [0,12]
                    
        which_class : (Default is None) To see the interpretations of class wise one can utilize which_class argument
                       Default for binary class: which class will be 1
                        & for Multi-class it will be the most frequent class in Y_train
                        
                        Y_train also require in dataframe format.
        
        
        Returns
        -----------------------------------------------------------------------------------
        Partial dependence plot
        
        """
        
        print("\tSHAP PDP")
        
        explainer = self.model_type()
        
        shap_values, feature_display = self.shap_helper(limit,explainer[0])
        
        
        x = name_list[0]
        y = name_list[1]    
        
                
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        print(shap_values.shape)
        
        
        if shap_values.ndim > 2 and Data_preprocessing.y_test.nunique()[0] == 2:
            print("ndim > 2")
            if which_class == None:
                print("Default : Global data for class 1")
                which_class = 1

            return shap.dependence_plot(x, shap_values[which_class], feature_display,interaction_index=y)
        
        elif shap_values.ndim > 2 and Data_preprocessing.y_test.nunique()[0] > 2:
            if which_class == None:
                print("----x------x---x-Multiclass-x------x------x")
                print("Default : Global data for most freq class ")
                label_column = Data_preprocessing.y_test.columns[0]
                most_freq_label = Data_preprocessing.y_test[label_column].value_counts().index[0]  
                which_class = most_freq_label
            return shap.dependence_plot(x, shap_values[which_class], feature_display,interaction_index=y)
            

        else: 
            print("ndim 1")

            return shap.dependence_plot(x, shap_values, feature_display,interaction_index=y)
          
        
    def get_shap_instance(self, index,which_class = None):
        
        """
        This method is used to interpret a single data point's predictions
        
        Parameters:
        -----------------------------------------------------------------------------------
        
        index :  Index of that record (index of X_test which is assumed to be pandas dataframe )
        
        which_class : (Default is None) To see the interpretations of class wise one can utilize which_class argument
                       Default for binary class: which class will be 1
                        & for Multi-class it will be the most frequent class in Y_train
                        
                        Y_train also require in dataframe format.
        Returns:
        A force plot and decision plot
        -----------------------------------------------------------------------------------
        
        """
        
        explainer, model_type = self.model_type()
        
        
        shap_preprocessed, all_features =  Data_preprocessing.pre_process(Data_preprocessing.X_test.iloc[[index],:])
       
        feature_display = pd.DataFrame(shap_preprocessed, columns=all_features)
        
        try:
            instance_prediction = (Data_preprocessing.model.predict(shap_preprocessed))[0]
            iprediction = (Data_preprocessing.model.predict_proba(shap_preprocessed))[0]
            print("Prediction ",instance_prediction)
            print("\tPredit_proba",iprediction)
        except ValueError:
            instance_prediction = (Data_preprocessing.model.predict(shap_preprocessed.values))[0]
            iprediction = (Data_preprocessing.model.predict_proba(shap_preprocessed.values))[0]
            print("Prediction ",instance_prediction)
            print("\tPredit_proba",iprediction)
            
        
        shap_values = explainer.shap_values(shap_preprocessed)
                    
        print("Base Value",explainer.expected_value)
        print("\tModel_Type ", model_type)
        print("----xx----"*10)
        
        if isinstance(explainer.expected_value, list):
                expected_value = np.array(explainer.expected_value)
        else:
            expected_value = np.array(explainer.expected_value)
        
                
        if isinstance(shap_values, list):
                 shap_values = np.array(shap_values)

        if shap_values.ndim > 2 and Data_preprocessing.y_test.nunique()[0] == 2:
            
            if which_class == None:

                which_class = 1
            
        
            if model_type == 'Kernel':
                print(shap.decision_plot(expected_value[which_class], shap_values[which_class], feature_display))

                shap.initjs()
                return shap.force_plot(expected_value[which_class],shap_values[which_class], feature_display)
                
            else:
                print(shap.decision_plot(expected_value[which_class], shap_values[which_class], feature_display,link='logit'))

                shap.initjs()
                return shap.force_plot(expected_value[which_class],shap_values[which_class], feature_display, link='logit')
            
            
        elif shap_values.ndim > 2 and Data_preprocessing.y_test.nunique()[0] > 2:
            print("----x-----Multi-classification----x-----")
            
            
            if which_class == None:

                which_class = instance_prediction
                
            if isinstance(which_class, list):
                which_class = (which_class[0])
                
            if isinstance(which_class, (np.ndarray,np.generic)):
                which_class = int(which_class.tolist()[0])
                
            print(which_class)
                
        
            print(shap.decision_plot(expected_value[which_class], shap_values[which_class], feature_display))

            return shap.force_plot(expected_value[which_class],shap_values[which_class], feature_display, matplotlib=True )  
        
        else:
            
            print("Single Shap value")
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            if model_type == 'Kernel':
                print(shap.decision_plot(explainer.expected_value, shap_values, feature_display))
                return shap.force_plot(explainer.expected_value, shap_values, feature_display, matplotlib=True)
        
            else:

                print(shap.decision_plot(explainer.expected_value, shap_values, feature_display,link='logit'))

                return shap.force_plot(explainer.expected_value, shap_values, feature_display, link='logit', matplotlib=True)
        
###------------------------------------x--------------x-Multi-classification-x----------x-----------------------------------------------x----------##

class Shap_multiclass(Shap_Interpretability):
    '''
    SHAP muticlass that inherits some funtionality form Parent class
    Changes passing every model in kernel explainer
    '''
    
    def __init__(self):
        pass
    
    
    def model_type(self):
        '''
        JUST USING Kernel Explainer 
        '''

        explainer = self.get_shap_kernel()
        
        return explainer
            
      
    
    
           
###------------------------------------x--------------x-Regression-x----------x-----------------------------------------------x----------##       
            
class Shap_regression(Shap_Interpretability):
    '''
    SHAP Regression that inherits some funtionality form Parent class
    '''
    
    def __init__(self):
        pass
        
        
    def get_shap_kernel(self):
        
        """
        This method is used to automate Kernel Shap
        """
        
        X_train_bd = Data_preprocessing.X_train.copy()
        X_train_bd_process, all_features = Data_preprocessing.pre_process(X_train_bd)
        
        datax = shap.kmeans(X_train_bd_process,10)
        
        k_explainer = shap.KernelExplainer(Data_preprocessing.model.predict, datax)
        
        return k_explainer, 'Kernel'
    
    
    
    def get_shap_instance(self, index):
        
        explainer, model_type = self.model_type()
        
        
        shap_preprocessed, all_features =  Data_preprocessing.pre_process(Data_preprocessing.X_test.iloc[[index],:])
       
        feature_display = pd.DataFrame(shap_preprocessed, columns=all_features)
        
        try:
            instance_prediction = (Data_preprocessing.model.predict(shap_preprocessed))[0]
            print("Prediction ",instance_prediction)
        
        except ValueError:
            instance_prediction = (Data_preprocessing.model.predict(shap_preprocessed.values))[0]
            print("Prediction ",instance_prediction)
        
        
        shap_values = explainer.shap_values(shap_preprocessed)
                    
        print("Base Value",explainer.expected_value)
        print("\tModel_Type ", model_type)
        print("----xx----"*10)
        
        if isinstance(explainer.expected_value, list):
                expected_value = np.array(explainer.expected_value)
        else:
            expected_value = np.array(explainer.expected_value)
        
                
        if isinstance(shap_values, list):
                 shap_values = np.array(shap_values)

        try:
            return shap.force_plot(explainer.expected_value, shap_values, feature_display)
        except AssertionError:
            return shap.force_plot(explainer.expected_value, shap_values[0], feature_display,matplotlib=True)
    
    
        
                
                
            
    