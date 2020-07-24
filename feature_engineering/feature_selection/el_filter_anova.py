    def dimReduction_filter(feature_train, label_train, feature_test, p_thrd = 0.05):
        """
        This function is used to Univariate Feature Selection: ANOVA
        """
        from sklearn.feature_selection import f_classif
        f, p = f_classif(feature_train, label_train)
        mask_selected = p < p_thrd
        feature_train = feature_train[:,mask_selected]
        feature_test = feature_test[:, mask_selected]
        return feature_train, feature_test, mask_selected