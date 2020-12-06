## pongbots

A collection of pong playing bots for ESC180, developed alongside Jack Cai (https://github.com/caixunshiren/)
http://www.cs.toronto.edu/~guerzhoy/180/pong/

- Pong_AI_RL: Our final submission can be found here under the "Frankenstein" branch. Based on a small shallow RL NN implemented entirely in tensorflow. Base repo can be found [here](https://github.com/caixunshiren/Pong_AI_RL/). Note that this codebase is especially messy and hacked together; if you want to play with stuff I encourage trying another branch or another folder.
- imgRlBot: We were wondering if positional data isn't all that the bot could gain insight from. imgRlBot takes images instead of positional information about the game. Unfortunately this took too long to train so we were never able to get it going
- numeric: A collection of bots based off basic prediction: predicting the final ball positon and the best possible angle at which to hit the ball
- gdRlbot: A reimplementation of Pong_AI_RL in keras with a bunch of helpful classes to make it easier to work with. Unfortunately this was not trained due to time constraints but performance should be roughly equal in theory.

Other folders and files represent trial attempts that we felt like keeping around.

Some half-trained model .h5s are left in this git repo if I ever feel like picking this project up again.






