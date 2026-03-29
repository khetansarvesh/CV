
# $\color{cyan}{Deep\ Reinforcement\ Learning\ (RL)}$

There are many types of Deep RL algorithms : 
1. <ins> Model Free RL Algorithms </ins> :
   - [Value Learning](https://khetansarvesh.medium.com/q-learning-deep-reinforcement-learning-ff162e9aca18) : Methods wherein we learn the policy by learning a value function
   - [Policy Learning](https://khetansarvesh.medium.com/policy-learning-deep-reinforcement-learning-83fb6e5aa025) : Methods wherein we learn the policy by learning the policy itself
   - [Actor Critic / Advantage Actor Critic(A2C)](https://khetansarvesh.medium.com/actor-critic-deep-reinforcement-learning-7632d8337d07) : combining both Value-Learning and Policy-Learning

2. <ins> Model Based RL Algorithms </ins> : In model free RL algorithms we could solve the RL problem without learning state transition probabilities <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 100">
     <text x="150" y="50" font-family="serif" font-size="30" fill="black"> P(sₜ₊₁|sₜ, aₜ) </text></svg>. But in these model based RL algorithms we will try to learn these state transition probabilities which will help  us predict correct actions for a given state. These state transition probabilities are also called transition dynamics / dynamics / models
  
3. <ins> Inverse RL Algorithms </ins> : In both the above algorithms we designed our own scalar reward structures but in these algorithms we learn the reward structure automatically 

> [!IMPORTANT]
> One of the best way to learn more about Deep RL is to use [this](https://spinningup.openai.com/en/latest/index.html) resource created by OpenAI.
> Other great resource are these [lectures](https://www.youtube.com/@rail7462/playlists) from UCB and for implementation details you can use this [github repo](https://github.com/vwxyzjn/cleanrl).
> Recently I also discovered [this](https://huggingface.co/learn/deep-rl-course/unit0/introduction) course from HF which is also a great start to Deep Reinforcement Learning


> [!CAUTION]
> Above we saw all the implementation using Neural Networks but earlier people used MDPs to model these instead of Neural Networks. Since MDPs were not scalable, Neural Networks became prominent. You can understand this scalability issue [here](https://www.youtube.com/watch?v=SgC6AZss478&list=PLs8w1Cdi-zvYviYYw_V3qe6SINReGF5M-&index=4). But if you still want to learn more about how to use RL with MDPs I would recommend watching these [IIT Madras Course](https://www.youtube.com/playlist?list=PLEAYkSg4uSQ0Hkv_1LHlJtC_wqwVu6RQX) and then watch this course by [David Silver (Google Deepmind)](https://www.davidsilver.uk/teaching/). Finally you can read this github [book](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)


# $\color{cyan}{Imitation\ Learning\ /\ Supervised\ Learning\}$
