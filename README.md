# RL-RPT - Rocket League Replay Pre-Training
One of the most common questions we get when [showing off our reinforcement learning bots](https://www.twitch.tv/rlgym), is "Could they learn from watching humans play?".

For a long time, the answer was (essentially) no. The previous best result I'm aware of is [this](https://www.youtube.com/watch?v=-928X5gDjzc) bot, trained on 15 000 gold/silver 1v1 replays. Not exactly earth-shattering.

## Why haven't people done it before?
There are a couple reasons previous attempts haven't been particularly successful:

1. Replays do not contain all the player inputs. Some of them are included, and it's theoretically possible to infer the rest algorithmically but...
2. Replays are lossy reconstructions of data from the server, Rocket League's in-game interpolation does a lot of heavy lifting.
3. Because of compounding error, a ridiculously high accuracy is needed. Choosing the wrong action at any time can lead to states the model hasn't seen before and doesn't know how to recover from.
4. Humans are very inconsistent and make very different choices in situations, there's no single right action to take.

In the RLBot Botpack, TensorBot and Levi are actually trained on inputs/outputs of bots, 
which works alright since we can get super-accurate inputs and consistent outputs, but then again it's doubtful that method would ever outperform the bot its trained on


## Why is this different?
The new idea here - inspired by [a paper from OpenAI](https://openai.com/blog/vpt/) - is that instead of trying to get player inputs from the replays, 
we can teach a network, called the IDM (Inverse Dynamics Model) to look at past and future states and determine which action *was* taken at a given moment in time.[^1]

In principle we could do this by just spawning a car in random locations to do random actions, but to get as close as possible to human gameplay, 
we recorded gameplay from Nexto via RLGym. Some adjustments were made to reduce overfitting to Nexto[^2], and some randomness was sprinkled in to cover more possible states and actions. 
This also has the added benefit of restricting the space of possible actions: Nexto only picks from 90 different ones that should approximately cover everything that's possible.
When the data is collected, all we have to do is give the model a series of states[^3], and train it to predict which action was taken.[^4]

Once that's done, we can use the IDM to get actions from human replays! 
We can then construct a new dataset, where data from the replay is used as the input[^5], and the action the IDM predicted is the target output.
Training a model on this should produce an agent that can emulate human gameplay.

## Does it work?
I think it's a big success! 
It was never intended to be anything crazy, but act more as a jumping off point for reinforcement learning, either by starting with the model, or by doing some more advanced stuff (like KL divergence) to encourage "SSL-like actions".

The agent I ended up actually looks quite competent. I only tried training on SSL replays, but I think it has quite a few moments where it looks very good, just lacking the consistency and precision.
You can also judge for yourself:

[Video with almost 6 minutes of gameplay](https://www.youtube.com/watch?v=VXi6f0zhVrk)

[Imgur album](https://imgur.com/a/zqrQxcD)


[^1]: The IDM also predicts some RLGym-specific stuff about the state that is not included in replays: `has_jump`, `has_flip` and `on_ground`
[^2]: Some actions will be equivalent in certain situations: E.g. If the player is demoed then all actions have the same effect. If the player is on ground then pitch/yaw/roll don't have an effect. We include several rules like that and pick evenly from assumed equivalent actions.
[^3]: 20 frames (~0.67s) are included on both sides of the frame where the action is taken, to include hidden information like torque after flipping, powerslide strength ramping up and down, etc.
[^4]: To make IDM more robust on replay data, the input is also corrupted to be more similar to the replay data by repeating player values randomly, despite originally containing perfect information
[^5]: Related to the last point, we only include frames where the current player's physics data is freshly updated.
