def data_preprocess(sel, feature_train, feature_test, data_preprocess_method, data_preprocess_level):
        '''
        This function is used to preprocess features
        Method 1: preprocess data in group level, one feature by one feature.
        Method 2: preprocess data in subject level.
        Method 5:
        '''
        # Method 1: Group level preprocessing.
        if data_preprocess_level == 'group':
            feature_train, model = elscaler.scaler(feature_train, data_preprocess_method)
            feature_test = model.transform(feature_test)
        elif data_preprocess_level == 'subject':
            # Method 2: Subject level preprocessing.
            scaler = preprocessing.StandardScaler().fit(feature_train.T)
            feature_train = scaler.transform(feature_train.T) .T
            scaler = preprocessing.StandardScaler().fit(feature_test.T)
            feature_test = scaler.transform(feature_test.T) .T
        else:
            print('Please provide which level to preprocess features\n')
            return

        return feature_train, feature_test