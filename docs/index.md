---
title: Projet Dronisos
layout: default
---


## Introduction

[announcement](https://docs.google.com/document/d/1FLXtXxfzXPU8-o3bSRl9ffA2X2acWMpPDPAWWRowjEY/edit)

## Scenario

Perform a formation flight like aerobatics patrols do. For example fly in a V shaped formation along a straight line.

<!-- It feels like once this is achieved without cheating, we're maybe closer to more complex trajectorie, like the same V formation, but spinning around the horizontal axis, like an helix. -->

## Challenges

  - Operational: takeoff, landing, communications, staging, reliability, calibration...
  
  - Theoretical:
    
	- 3D+t guidance
	- online [trajectory planning](planning) (for initial rendez-vous)
    - verification that a trajectory is possible to fly (given a wind field)
	- modify the trajectory to adapt to changing wind conditions


## 2d Instance

We begin with a [planar simplification](planar_instance) of the problem as depicted on figure 1.


<!--<img src="plots/2d_traj1.gif" alt="Planar MIP simulation plot" width="640">-->
<img src="plots/2d_traj1.apng" alt="PVTOL pole simulation plot" width="640">

