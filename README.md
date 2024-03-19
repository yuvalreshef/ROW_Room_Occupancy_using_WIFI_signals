# ROW: Classifier of Room Occupancy using WIFI signals

The ability to ’see through walls’ has always been somewhat of a superpower required for various
tasks and needs. Tasks like indoor positioning for smart houses, counting stationary crowds (for
example, the number of people sitting in a large room), understanding different actions people are
doing in a closed room behind a wall, etc.
Locating and counting people in outdoor conditions can be done using GPS (Global Positioning
System) signals [1]. However, this technique is less suitable for indoor conditions due to reduced
signal propagation. One could take advantage of the fact that almost every person carries a cell
phone. Localizing and counting the number of people based on the location of a cell phone is
relatively easy and can be done via different techniques [2].
But there are additional methods that do not rely on each person carrying a cell phone. These
techniques use the fact that in the modern world, most rooms are equipped with wireless network
devices that are transmitting and receiving signals constantly. These signals can be used to de-
termine the amount, nature, and distribution of interruptions. These data elements, when used
1
correctly, can serve as features to determine the number of people in a closed room. In this article,
we focus on this ability using RF (Radio Frequency) signals, specifically Wi-Fi signals, and apply
modern techniques that could be superior to other RF-based methods.
