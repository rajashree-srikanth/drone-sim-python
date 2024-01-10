---
title: Projet Dronisos
layout: default
---


## Introduction

[announcement](https://docs.google.com/document/d/1FLXtXxfzXPU8-o3bSRl9ffA2X2acWMpPDPAWWRowjEY/edit)

## 2D experiment

### State Space representation

We begin with a planar simplification of the problem defined as follow:

\\(\vect{X}\\) , \\(\vect{U}\\) and \\(\vect{W}\\) are respectively our state, input and disturbance vectors

$$
\begin{equation}
  \vect{X} = \transp{\begin{pmatrix}x&y&\psi&\phi&v\end{pmatrix}} \quad \vect{U}=\transp{\begin{pmatrix}\phi_c&v_c\end{pmatrix}} \quad W = \transp{\begin{pmatrix}w_x&w_y\end{pmatrix}} 
\end{equation}
$$


The following state space representation can be computed for the guidance dynamics of the aircraft

$$
\begin{equation}
  \dot{\vect{X}} = f(\vect{X}, \vect{U}) = \begin{pmatrix}v.\cos(\psi)+w_x\\v.\sin(\psi)+w_y\\\frac{g}{v}\tan(\phi)\\-\frac{1}{\tau_\phi}(\phi-\phi_c)\\-\frac{1}{\tau_v}(v-v_c)\end{pmatrix}
\end{equation}
$$


TODO: rigorously reestablish those equations which will involve time derivatives of wind, which we can later decide to neglect.


