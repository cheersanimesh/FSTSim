# FSTSim
FSTSim: A Federated semi-supervised classification model using Teacher-Student learning framework.

### Goal:

 Since manual supervision and supervised training do not scale very well, we intend to propose a semi-supervised learning method, for classification of data points. 
In recent years, data privacy is of utmost importance, especially when it comes to privacy in the matters of legal advice, healthcare and national security. 
Most users are not comfortable with sharing personal data with software applications such as this. Keeping user data in a central database could be insecure and at risk of being used for malicious intent.
Hence, sharing user data for training models could lead to a serious issue. Thus we also propose a federated learning setup to perform the classification task.

## Proposed Solution:

Our approach proceeds towards implementing a teacher student model for semi supervised classification and extend this technique in a federated setup.

**Local Training Model**

[!Alt text](./images/local_training_model.png)

**Global training models**

[!Alt text](./images/overall_model.png)

## Overall Results

| Workflow | Accuracy Student | Accuracy Teacher |
| :---: | :---: | :---: |
| Federated | 81.53 | 96.81 |
| Non Federated | 91.21 | 97.32 |

[!Alt text](https://github.com/cheersanimesh/FSTSim/images/fed_non_fed.png)

[!Alt text](https://github.com/cheersanimesh/FSTSim/images/fed_non_fed_2.png)

## Conclusion

The performance drop is not significant as the number of unlabelled data increases, suggesting that the workflow is robust towards the amount of unlabelled data points.

There is the still significant	difference in performance between non federated setup and federated setup requiring the need of better aggregation method for global models.

As the number of clients increases, there is a significant change in performance suggesting that aggregation needs to be more robust.







