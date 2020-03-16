from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

class Preprocessing():
    '''
    This class is used to preprocess features
    Method 1: preprocess data in group level, one feature by one feature.
    Method 2: preprocess data in subject level.
    Parameters:
    ----------
    TODO

    '''

    def __init__(self, data_preprocess_method='StandardScaler', data_preprocess_level='subject'):
        self.data_preprocess_method = data_preprocess_method
        self.data_preprocess_level = data_preprocess_level

    def data_preprocess(self, feature_train, feature_test):
        '''
        This function is used to preprocess features
        Method 1: preprocess data in group level, one feature by one feature.
        Method 2: preprocess data in subject level.
        '''
        # Method 1: Group level preprocessing.
        if self.data_preprocess_level == 'group':
            feature_train, model = self.scaler(feature_train, self.data_preprocess_method)
            feature_test = model.transform(feature_test)
        elif self.data_preprocess_level == 'subject':
            # Method 2: Subject level preprocessing.
            scaler = preprocessing.StandardScaler().fit(feature_train.T)
            feature_train = scaler.transform(feature_train.T) .T
            scaler = preprocessing.StandardScaler().fit(feature_test.T)
            feature_test = scaler.transform(feature_test.T) .T
        else:
            print('Please provide which level to preprocess features\n')
            return

        return feature_train, feature_test

    def scaler(self, X, method):
        if method == 'StandardScaler':
            model = StandardScaler()
            stdsc_x = model.fit_transform(X)
            return stdsc_x, model
        
        elif method == 'MinMaxScaler':
            model = MinMaxScaler()
            mima_x = model.fit_transform(X)
            return mima_x, model
        else:
            print(f'Please specify the standardization method!')
            return
        
    def scaler_apply(self, train_x, test_x, scale_method):
        """
        Apply model to test data
        """
        train_x, model = self.scaler(train_x, scale_method)
        test_x = model.transform(test_x)
        return train_x, test_x
