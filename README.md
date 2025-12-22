# out_of_our_minds_RL_ABM
ABM of social interaction where agents recieve internal report information and social influence which that integrate whilst attempting to reduce dissonance. The key output of the model is the vector of topic dissonance, i.e. How much does individual A differ in their beliefs regarding how topic 1 is connected to topics 2-10, compared to their social network neighbours.

Pipeline attempt to connect the outputs from Bertopic which to the ABM via converting the metatopic vectors produced for each individual into a correlation matrix between topics. 
