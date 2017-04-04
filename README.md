# Cost-sensitive-Boosting-Tutorial
The tutorial introduces the concepts of asymmetric (cost-sensitive and/or imbalanced class) learning, decision theory and boosting. It briefly describes the results of the paper:

Nikolaou, N., Edakunni, N. U., Kull, M., Flach, P. A., and Brown, G., 'Cost-sensitive boosting algorithms: Do we really need them?', Machine Learning, 104(2), pages 359-384, 2016.

It presents the Calibrated AdaMEC method (AdaBoost with calibrated probability estimates and a shifted decision threshold) found to be the most flexible, empirically successful and theoretically valid way to handle asymmetric classification with adaboost ensembles.

The code provided allows the user to reproduce the papers experiments, but also to extend them by choosing different calibration techniques, weak learners, ensemble sizes, AdaBoost variants, train\calibration splits, etc.
