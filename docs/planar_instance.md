---
title: Planar Instance
layout: default
---

# 2D experiment

## State Space representation

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


## Differential flatness
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


## Guidance

Our control objective consist in following a trajectory \\( Y_r(t): C^3, \mathbb{R} -> \mathbb{R}^2 \\) while rejecting perturbation with an asymptoticaly stable dynamics.

Differential Flattness can be used to compute state \\(X_r\\)  and input \\(U_r\\) corresponding to the flat output \\( Y_r \\)

Tracking error \\( \epsilon = X - X_r \\) can be asymtoticaly stabilized used a linear feedbak, which leads to the following controller:

$$
\begin{equation*}
U = U_r - K (X-Xr)
\end{equation*}
$$

Linearizing  state space representation space representation \eqref{eq:2d_cont_ssr} around \\( (X_r, U_r) \\) leads to the following dynamics for \\( \epsilon \\)

$$
\begin{equation*}
\dot{\epsilon} = (A -BK) \epsilon \quad A = \frac{\partial{f}}{\partial{X}} \bigg\vert_{Xr, Ur} \quad  B = \frac{\partial{f}}{\partial{X}} \bigg\vert_{Xr, Ur}
\end{equation*}
$$

with

$$
\begin{equation*}
A = \begin{pmatrix}
0 & 0 & -v_a \sin\psi&0 & \cos\psi \\
0 & 0 &  v_a \cos\psi&0 & \sin\psi \\
0 & 0 & 0 & \frac{g}{v_a(1+\cos^2\phi)}  & -\frac{g}{v_a^2}\tan{\phi} \\
0 & 0 & 0 & -\frac{1}{\tau_\phi} & 0 \\
0 & 0 & 0 & 0 & -\frac{1}{\tau_v}
\end{pmatrix}
\quad
B =  \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \\ \frac{1}{\tau_\phi} & 0 \\ 0 & \frac{1}{\tau_v}
\end{pmatrix}
\end{equation*}
$$

A gain has for instance been computed using Linear Quadratic Regulation (LQR) from a set of tracking error cost \\(Q\\) and control effort cost \\( R \\)


In order to be usable in real world, the control law will in reality be

$$
\begin{equation*}
U = \text{sat}_U(U_r - K \text{sat}_\epsilon(X-Xr))
\end{equation*}
$$


\\(Sat_U \\) and \\(Sat_X \\) will need to be tuned. The feedback might be nicer if computed in body frame. Maybe it would be nicer to use a saturated linear reference model for espilon dynamic.


## Implementation


- Trajectories 

    [d2d/trajectory.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/d2d/trajectory.py)
	contains basic building blocks for trajectories, like line, circles, polynomials, etc...
	
    [d2d/trajectory_factory.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/d2d/trajectory_factory.py)
	contains the definitions of the trajectories.src/02_test_traj.py
	
    [02_test_traj.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/02_test_traj.py)
	is a tool for displaying trajectories, as plots or animations
 
- Scenarios
    
	[d2d/scenario.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/d2d/scenario.py)
	scenarios can contains several trajectories, have a wind field and initial conditions
	
    [03_test_scenario.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/03_test_scenario.py)
	is a tool for displaying scenarios, as plots or animations

- Simulation

    [d2d/dynamic.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/d2d/dynamic.py)
	aircraft dynamics
	
    [d2d/guidance.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/d2d/guidance.py)
	guidance
	
    [05_test_simulation.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/05_test_simulation.py)
	is a tool for simulating a scenario and displaying it as plots or animations

- Misc
    
	[d2d/ploting.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/d2d/ploting.py),
    [d2d/animation.py](https://github.com/poine/projet_dronisos_guidage/blob/master/src/d2d/animation.py),
	do what they're supposed to do.
