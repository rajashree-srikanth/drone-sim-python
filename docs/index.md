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
	- modify the trajectory to adapt to changing wind conditions


## 2D experiment

### State Space representation

We begin with a planar simplification of the problem defined as follow:

\\(\vect{X}\\) , \\(\vect{U}\\) and \\(\vect{W}\\) are respectively our state, input and disturbance vectors

$$
\begin{equation*}
  \vect{X} = \transp{\begin{pmatrix}x&y&\psi&\phi&v_a\end{pmatrix}} \quad \vect{U}=\transp{\begin{pmatrix}\phi_c&v_{ac}\end{pmatrix}} \quad W = \transp{\begin{pmatrix}w_x&w_y\end{pmatrix}} 
\end{equation*}
$$


The following state space representation can be computed for the guidance dynamics of the aircraft

$$
\begin{equation} \label{eq:2d_cont_ssr}
  \dot{\vect{X}} = f(\vect{X}, \vect{U}) = \begin{pmatrix}v_a.\cos(\psi)+w_x\\v_a.\sin(\psi)+w_y\\\frac{g}{v_a}\tan(\phi)\\-\frac{1}{\tau_\phi}(\phi-\phi_c)\\-\frac{1}{\tau_v}(v_a-v_{ac})\end{pmatrix}
\end{equation}
$$


TODO: rigorously reestablish those equations, make a drawing, define notations and variable ranges.


### Differential flatness
For using \\( \vect{Y} = \transp{\begin{pmatrix}x&y\end{pmatrix}} \\) as flat input, we want to find \\(\Phi_1\\) and \\(\Phi_2\\) such as 

$$
\begin{equation*}
\vect{X} = \Phi_1(Y, \dot{Y}...)
\quad
\vect{U} = \Phi_2(Y, \dot{Y}...)
\end{equation*}
$$

Noting \\(v_{ax} = \dot{x}-w_x \\) and \\( v_{ay} = \dot{y}-w_y \\),
squaring and summing the first two lines of state space representation \eqref{eq:2d_cont_ssr}, we get

$$
\begin{equation} \label{eq:2d_df_va}
v_a = \sqrt{v_{ax}^2 + v_{ay}^2}
\end{equation}
$$


For computing \\( \psi \\), dividing line 1 and 2 of state space representation \eqref{eq:2d_cont_ssr}, we get (we're hopefull that neither \\( v_a \\) nor \\( \cos(\psi) \\) is null)

$$
\begin{equation*}
\tan(\psi) = \frac{v_{ay}}{v_{ax}}
\end{equation*}
$$

which leads to the following expression for \\(\psi\\)

$$
\begin{equation} \label{eq:2d_df_psi}
\psi = \arctan(\frac{v_{ay}}{v_{ax}})
\end{equation}
$$

\\(\dot{\psi}\\) is obtained by differentiating  \eqref{eq:2d_df_psi}

$$
\begin{equation*}
\dot{\psi} = \frac{d}{dt} \left( \frac{v_{ay}}{v_{ax}} \right) \frac{1}{1+\left( \frac{v_{ay}}{v_{ax}} \right)^2}
=
\frac{v_{ax} \dot{v}_{ay} - \dot{v}_{ax} v_{ay}}{v_{a}^2}
\end{equation*}
$$


Line 3 of  state space representation \eqref{eq:2d_cont_ssr} leads to

$$
\begin{equation} \label{eq:2d_df_phi}
\phi = \arctan{ \frac{v_a \dot{\psi}}{g}} =
\arctan{\left( \frac{v_{ax} \dot{v}_{ay} - \dot{v}_{ax} v_{ay}}{ g \sqrt{v_{ax}^2 +  v_{ay}^2} } \right)}
\end{equation}
$$

which completes the derivation of \\( \Phi_1 \\)

$$
\begin{equation}
\vect{X} = \transp{\begin{pmatrix}x&y&\psi&\phi&v_a\end{pmatrix}} = \Phi_1(Y, \dot{Y}, \ddot{Y}) = 
\begin{pmatrix}
x\\y\\
\arctan(\frac{v_{ay}}{v_{ax}}) \\
\arctan{ \frac{v_a \dot{\psi}}{g}} \\
\sqrt{v_{ax}^2 + v_{ay}^2}
\end{pmatrix}
\end{equation}
$$


\\( \dot{v}_a \\) is obtained by differentiating \eqref{eq:2d_df_va} as

$$
\begin{equation*}
\dot{v}_a = \frac{v_{ax} \dot{v}_{ax} + v_{ay} \dot{v}_{ay}}{v_a}
\end{equation*}
$$


$$
\begin{equation}
v_{ac} = \tau_v \dot{v}_a + v_a
\end{equation}
$$

\\( \dot{\phi} \\) is obtained by differentiating \eqref{eq:2d_df_phi} as

$$
\begin{equation*}
\dot{\phi} = 
\end{equation*}
$$

$$
\begin{equation}
\phi_{c} = \tau_{\phi} \dot{\phi} + \phi
\end{equation}
$$

which completes the derivation of \\( \Phi_2 \\)

$$
\begin{equation}
\vect{U}=\transp{\begin{pmatrix}\phi_c&v_{ac}\end{pmatrix}}=\Phi_2(Y, \dot{Y}, \ddot{Y}, \dddot{Y}) = 
\end{equation}
$$
