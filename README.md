# Adaptive-systems-double-Q-learning

## Installation
To install the required packages, run the following command:
```bash
pip install .
```

## Results:

Based on `models/deep_q_model_92.pth`
x-axis: amount of episodes,   
y-axis: Mean Squared Error  

![Graph of the results](https://github.com/Max2411/Adaptive-systems-double-Q-learning/blob/main/graph_of_results.PNG)

#### Decaying epsilon
The current code for decaying the epsilon does not work properly. Fixing this issue would probably help train better models.

#### Hyperparameters
We can still mess a lot with hyperparameters, there is likely to be a better combination of hyperparameters for training our model.

#### Better save
The current code saves the last known model. In the graph above you see that the last model had reward of 92
which is not the highest scoring model of what we have trained. That's why a good change would be to save the model with
the highest reward.

#### Longer training
This is a result based on `models/deep_q_model_101.pth` where the moon lander
lands somewhere on the right most of the time this might be because the model has first learned
how to safely land and would start learning where after this point. This could also be a result of saving a model that is not the best out
of its batch.

![wrong landing](https://github.com/Max2411/Adaptive-systems-double-Q-learning/blob/main/wrong_landing_space.PNG)


### Helpful sources:
..* [Copy network](https://gist.github.com/abhinavsagar/a7cd1d67ad5a71589d576cd84f15b648 "Github link")  
..* [General knowledge](https://www.youtube.com/watch?v=wc-FxNENg9U "Youtube link")     
..* [General knowledge](https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html "Web page")     
..* [General knowledge](https://jfking50.github.io/lunar/ "Web page")       
..* Pytorch Documentation       
..* StackOverflow for troubleshooting       
