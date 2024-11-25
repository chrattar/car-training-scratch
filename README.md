# ML Agent from Scratch

My saga to develop an ML agent has been quite intense.

Attmepting to use Box2D with pygame.

* Added logging. Need to log maximum and minimum (x,y) values to see what the reward is for what location. It seems that it is hard for the car to get out of the bottom area of the track. i obviously need to change my reward structure to compensate for this.

* Alter the reward structure for going clockwise as opposed to anti-clockwise
* Added Gates to the model for tracking
* Added nn layer 64, then bumped up to 128
* added logging for review
* Changed lr to LR = 0.1
* Revised reward structure to penalize the car from not driving long enough
* 

The learning experience regarding this has been interesting. The depth of knowledge that is required in order to implement this.

First I attempted to make this work with a png track and car diagram, but that was quickly scrapped.
Next, we moved on to runnning this inside pygame, 


This has been a terrible event and it isn't working the way I expected it to. I am curious if I need to run it for far longer?
