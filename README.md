# RL-RPT - Rocket League Replay Pre-Training
One of the most common questions we get when [showing off our reinforcement learning bots](https://www.twitch.tv/rlgym), is "Could they learn from watching humans play?".

For a long time, the answer was (essentially) no. The previous best result I'm aware of is [this](https://natebake.dev/code/rl-ai/) bot, trained on 15 000 gold/silver 1v1 replays. A fun project to be sure, but not exactly earth-shattering.

## Why haven't people done it before?
There are a couple reasons previous attempts haven't been particularly successful:

1. Replays do not contain all the player inputs directly. A lot is included, and it's theoretically possible to infer the rest algorithmically[^1] but...
2. Replays are lossy reconstructions of data from the server. Rocket League's in-game interpolation does a lot of heavy lifting.
3. Because of compounding error, a ridiculously high accuracy is needed. Choosing the wrong action at any time can lead to states the model hasn't seen before and doesn't know how to recover from.
4. Humans are very inconsistent and make very different choices in situations, there's no single right action to take.
5. Buttons are not fully independent, e.g. jump+pitch is very different from just pitch or just jump, so a fixed set of possible actions is required for the best possible performance.

In the [RLBot](https://rlbot.org/) [Botpack](https://github.com/RLBot/RLBotPack), TensorBot and Levi are actually trained on inputs/outputs of bots, 
which works alright since we can get super-accurate inputs and consistent outputs, but then again it's doubtful that method would ever outperform the bot its trained on.


## Why is this different?
The new idea here - inspired by [a paper from OpenAI](https://openai.com/blog/vpt/) - is that instead of trying to get player inputs from the replays, 
we can teach a network, called the IDM (Inverse Dynamics Model) to look at past and future states and determine which action *was* taken at a given moment in time.[^2]

For earlier versions, to get as close as possible to human gameplay, 
we recorded gameplay from [Nexto](https://github.com/Rolv-Arild/Necto) via [RLGym](https://rlgym.org/). Some adjustments were made to reduce overfitting to Nexto's outputs[^3], and some randomness was sprinkled in to cover more possible states and actions.
This also has the added benefit of restricting the space of possible actions: Nexto only picks from 90 different ones that should approximately cover everything that's possible.
When the data is collected, all we have to do is give the model a series of states[^4], and train it to predict which action was taken.[^5]

Later on, to add flexibility, the code was changed such that the network gets the series of states, and a list of actions to pick between, where only one is correct. Training is then generated with random actions from random and replay states spanning the whole continuous in-game action space. Once the network is trained it can provide a guess for the general likelihood of any action. Having recently learned about jump and dodge data being available in replays, a ton of checks were added so the IDM can only pick between actions that also match with what the replay file says. This means we can use any set of actions we choose while having a lot of the quality assurance. 

Once that's done, we can use this to get actions from human replays! 
We can then construct a new dataset, where data from the replay is used as the input[^6], and the action the IDM predicted is the target output.
Training a model on this should produce an agent that can emulate human gameplay.

## Does it work?
I think it's a big success! 
It was never intended to be anything crazy, but act more as a jumping off point for reinforcement learning, either by starting with the model, or by doing some more advanced stuff (like KL divergence) to encourage "SSL-like actions".

The agent I ended up actually looks quite competent. I only tried training on SSL replays, but I think it has quite a few moments where it looks very good, just lacking the consistency and precision.
You can also judge for yourself:

[Video with 10 minutes of varied gameplay](https://www.youtube.com/watch?v=ew_3vA7EitA)

[Imgur album](https://imgur.com/a/zqrQxcD)

At this point the models have also been further trained extensively with reinforcement learning, yielding Ripple, a high level bot with many human-like traits.


[^1]: We have values for throttle, steer, handbrake and boost, as well as data about jumps, double jumps, and dodges (including direction). We can infer pitch/yaw/roll using angular velocity data.
[^2]: The IDM also predicts some RLGym-specific stuff about the state that is not included in replays: `has_jump`, `has_flip` and `on_ground`
[^3]: Some actions will be equivalent in certain situations: E.g. If the player is demoed then all actions have the same effect. If the player is on ground then pitch/yaw/roll don't have an effect. We include several rules like that and pick evenly from assumed equivalent actions.
[^4]: 20 frames (~0.67s) are included on both sides of the frame where the action is taken, to include hidden information like torque after flipping, powerslide strength ramping up and down, etc.
[^5]: To make IDM more robust on replay data, the input is also corrupted to be more similar to the replay data by repeating player values randomly, despite originally containing perfect information
[^6]: Related to the last point, we only include frames where the current player's physics data is freshly updated.
