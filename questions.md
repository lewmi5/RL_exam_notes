# Questions
## Reinforcement-learning basics:
1. What is RL formalism (MDP), policy, value function, rewards? Give examples.
2. What are the basic components of a reinforcement learning algorithm?

## Value-based methods
1. What is the relation of value function and a policy?
2. What is the difference between value function and q-function?
3. Formulate Bellman equation. How many are there?
4. What is a policy iteration algorithm and what are its stages?
5. What are the differences between model-free and model-based methods?
6. What is the difference between prediction and control?
7. Describe the differences between Monte-Carlo and Temporal-Difference? 
8. What is TD(k) and TD(λ)?
9. Describe the SARSA algorithm.
10. Describe the Q-learning algorithm.
11. What is the difference between on-policy and off-policy algorithms?

## Function approximation 
1. Describe model-free prediction using function approximators and Monte-Carlo and Temporal Difference.
2. Describe potential problems when using function approximators in RL. What is the deadly triad?
3. Describe the LSMC and LSTD algorithms.
4. Describe the DQN algorithm. What problems of using function approximators in RL DQN aims to solve?

## Policy gradient-methods
1. Why/when use policy methods and value methods?
2. List two gradient-free policy algorithms?
3. Evolutionary Strategies, what are their strengths and weaknesses? 
4. Compare on-policy methods and off-policy methods. List a few algorithms in each class.
5. Describe policy gradient theorem (show the proof - for ambitious students).
6. What is the Reinforce algorithm?
7. Describe the A3C algorithm. What are its strengths and weaknesses? (we may ask about the pseudo-code).
8. What is IMPALA?
9. Derive the TRPO algorithm. What are its strengths and weaknesses? (we may ask about the pseudo-code).
10. Derive the PPO algorithm. What are its strengths and weaknesses? (we may ask about the pseudo-code).
11. Derive the DDPG/TD3 algorithms. What are their strengths and weaknesses? (we may ask about the pseudo-code).
12. Describe the replay buffer technique? Where is it used?
13. What is double Q-learning? Why and when is it used?
14. What are methods of ensuring exploration in policy algorithms?
15. Derive the SAC algorithms. What are their strengths and weaknesses? (we may ask about the pseudo-code).
16. Describe briefly the maximum entropy RL formalism (compare with the “standard” approach).

## Exploration-exploitation:
1. Multi-armed bandits model.
2. Regret and regret decomposition.
3. Formulate asymptotic and worst case lower bounds on regret.
4. Discuss optimality properties of the bandit algorithms.
5. Define UCB and discuss its properties.
6. Define Bayesian multi-armed bandits. Characterize Bernoulli with Beta prior bandits.
7. Discuss Thompson sampling.

## Model-based methods
1. What is model-based RL, compare with model-free. What are its strengths and weaknesses?
2. Describe the Dyna-algorithm (DynaQ).
3. What is the back-propagation-through time algorithm? Why is it not used?
4. Derive the LQR and iLQR algorithms. What are their strengths and weaknesses?
5. Compare open-loop vs closed-loop algorithms.
6. Describe the MCTS algorithms. What are their strengths and weaknesses? (we may ask about the pseudo-code).
7. Describe the World models/Simple algorithms. What are their strengths and weaknesses? 
8. Describe the I2A algorithm.
9. Describe the MBMF algorithm.
10. Describe the MBVE algorithm.
11. Describe the TreeQN algorithm/architecture.
12. Describe the UPN algorithm/architecture.
13. Describe selected three RL benchmarks.

## Advanced topics in RL
1. Describe Reinforcement Learning for Improving Agent Design.
2. Discuss open-endedness on the example of POET.
3. Discuss morphological approach of Learning to Control Self-Assembling Morphologies.
4. What are inductive biases, give definition and examples.
5. Relational inductive bias on the example of Relational Deep Reinforcement Learning.
6. Discuss the topic of causality and adversarial examples (generally and in RL).
7. Describe briefly the causal calculus.
8. Casual confusion on the example of Causal Confusion in Imitation Learning
9. Discuss End to End Learning for Self-Driving Cars approach (note the problems with imitation learning).
10. Discuss random exploration on the example of Learning to Fly by Crashing.
11. Discuss the domain randomisation technique on the example of OpenAi Rubik’s cube project.
12. Discuss general/universal random functions and the hindsight replay buffer technique.

## Distributional RL
1. Formulate the distributional RL operators.
2. Describe C51.
3. Describe Quantile DRL.
4. Describe IQN.
5. What are the components of Rainbow. Describe them.
