
# Classification using Deep Neural Network on Red Wine Quality
## **Introduction**

This project is used as an evaluation for Certified Deep Learning Engineer course. Our team was tasked on using Deep Neural Network on structured data to solved it as a regression or classification problem.

## **Problem Statement**
The intricacies of deciding a fine wine when you are selecting a dish to pair it with or whether to judge the quality of a certain wine for the massess could be easier for the sommelier. 

## **The Idea**
It is better if we can in some way, use machine learning to help predict the quality of wine by the thousands of barrels by just sampling its chemical properties, quatifying it and run it through the deep neural network.

## **The Journey**

- **Data Collection**
    - The datasets are retrieved from Kaggle (https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009).
    - It is related to red and white variants of the Portuguese "Vinho Verde" wine that is produced in Northern Portugal. Due to privacy issues, only       physicochemical that is important in characterising a wine and sensory output classes are available.
    - It have 11 unique labels and can be categorised into 11 classes. However, the dataset only contain 6 quality classes which is rating of 3, 4, 5, 6, 7, 8.
    - The data is imbalance because there is no representation of lower and higher quality wine.
    
    
    
- **Data Processing**
   - Using Knime to apply SMOTE onto the dataset to try make it equal representation of all the classes by having it filled up the imbalance data synthetically.
   
   ![SMOTE](https://user-images.githubusercontent.com/76154147/106477420-1bb2f980-64e3-11eb-8d03-8a8205aef607.jpg)
   
     - **Before SMOTE**
      ![Before Smote](https://user-images.githubusercontent.com/76154147/106477493-2e2d3300-64e3-11eb-8332-f769f17475a0.jpg)

     - **After SMOTE**
     ![After SMOTE](https://user-images.githubusercontent.com/76154147/106477682-5e74d180-64e3-11eb-8a9b-f586d6d9ecaf.jpg)

  
- **Architecture**
  - We improved over **Normal Deep Neural Network** by applying **K-Fold**.  



- **Evaluation**

    - Train Normal

    ![Train Normal](https://user-images.githubusercontent.com/63250608/164382689-5b847d93-586f-4ab0-9a1d-e97316847027.png)
    
    - Test Normal
    
    ![Test Normal](https://user-images.githubusercontent.com/63250608/164382754-881c1f42-1d45-42ab-a657-d33ba200db4e.png)


    - Train K-Fold
    
    ![Train K5](https://user-images.githubusercontent.com/63250608/164382844-171bf913-476b-444e-b4d2-c4403f17ea00.png)

    
    - Test K-Fold
    
    ![Test K5](https://user-images.githubusercontent.com/63250608/164382919-1d60cca8-eaec-4b78-b0e5-78d9ac21180e.png)

    
    - Average F1 Score 
    
    ![Average F1 all fold](https://user-images.githubusercontent.com/63250608/164382977-a3b62b84-d086-490a-a47e-c5259480df28.png)
    
    
## **Conclusion**
 
  - K-Folding improve the overall score but longer processing time.
  - The effect of lack of equal representation for all the classes cannot be improve greatly using synthetic data. 

    
## **Future Improvement**
  - Pick a more robust and comprehensive data as the model cannot be improved using such an imbalance data.
  - Needs more representation of other classes.
  - Using arbiter to test more variables and hyperparameter.



## Getting Started 

- Clone a copy of the repository. 

```

git clone https://github.com/farisk263/Red-Wine-Quality-Check.git

```
- Using an IDE and open the RedWineQualityCheck.java




## Built on

* [Maven](https://maven.apache.org/) - Dependency Management
* [DL4J](https://deeplearning4j.org/) - Deep Learning Library


## FAQ 

If you are having problems on installing Maven dependencies. Try to reload it by right-clicking the POM.xml


