---
title: Projet Dronisos
layout: default
---


## Introduction

[announcement](https://docs.google.com/document/d/1FLXtXxfzXPU8-o3bSRl9ffA2X2acWMpPDPAWWRowjEY/edit)

## Scenario

Perform a formation flight like aerobatics patrols do. For example fly in a V shaped formation along a straight line.

It feels like once this is achieved without cheating, we're maybe closer to more complex trajectorie, like the same V formation, but spinning around the horizontal axis, like and helix.

## Challenges

  - Operational: takeoff, landing, communications, staging, reliability, calibration...
  
  - Theoretical:
    
	- 3D+t guidance
	- online path planning (for initial rendez-vous)
    - verification that a trajectory is possible to fly (given a wind field) 


## 2D experiment

### State Space representation

We begin with a planar simplification of the problem defined as follow:

\\(\vect{X}\\) , \\(\vect{U}\\) and \\(\vect{W}\\) are respectively our state, input and disturbance vectors

$$
\begin{equation*}
  \vect{X} = \transp{\begin{pmatrix}x&y&\psi&\phi&v_a\end{pmatrix}} \quad \vect{U}=\transp{\begin{pmatrix}\phi_c&v_c\end{pmatrix}} \quad W = \transp{\begin{pmatrix}w_x&w_y\end{pmatrix}} 
\end{equation*}
$$


The following state space representation can be computed for the guidance dynamics of the aircraft

$$
\begin{equation}
  \dot{\vect{X}} = f(\vect{X}, \vect{U}) = \begin{pmatrix}v_a.\cos(\psi)+w_x\\v_a.\sin(\psi)+w_y\\\frac{g}{v_a}\tan(\phi)\\-\frac{1}{\tau_\phi}(\phi-\phi_c)\\-\frac{1}{\tau_v}(v-v_c)\end{pmatrix}
\end{equation}
$$


TODO: rigorously reestablish those equations.


### Differential flatness

$$
\begin{equation*}
\vect{Y} = \transp{\begin{pmatrix}x&y\end{pmatrix}}
\end{equation*}
$$

$$
\begin{equation*}
\vect{X} = \Phi_1(Y, \dot{Y}...)
\quad
\vect{U} = \Phi_2(Y, \dot{Y}...)
\end{equation*}
$$

$$
\begin{equation*}
v_{ax} = \dot{x}-w_x \quad v_{ay} = \dot{y}-w_y
\end{equation*}
$$

$$
\begin{equation}
v_a = \sqrt{v_{ax}^2 + v_{ay}^2}
\end{equation}
$$

$$
\begin{equation*}
v_a. \tan(\psi) = \frac{v_{ay}}{v_{ax}}
\end{equation*}
$$

$$
\begin{equation}
\psi = \arctan(\frac{1}{v_a}\frac{v_{ay}}{v_{ax}})
\end{equation}
$$


$$
\begin{equation*}
\phi = \arctan{ \frac{v_a \dot{\psi}}{g}} 
\end{equation*}
$$

$$
\begin{equation*}
\dot{\psi} = \frac{d}{dt} \left( \frac{1}{v_a}\frac{v_{ay}}{v_{ax}} \right) \frac{1}{1+\left( \frac{1}{v_a}\frac{v_{ay}}{v_{ax}} \right)^2}
\end{equation*}
$$
