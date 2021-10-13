%Imprecise Action Selection in Substance Use Disorder: Evidence for 
%Active Learning Impairments When Solving the Explore-Exploit Dilemma 

%Drug and Alcohol Dependence

%Ryan Smith, Philipp Schwartenbeck, Jennifer L. Stewart, Rayus Kuplicki, Hamed Ekhtiari, Tulsa 1000 Investigators, and Martin P. Paulus. 

% Script for creating the MDP model for active learning task (three-armed
% bandit task)

%Ryan Smith & Philipp Schwartenbeck

% Input:

% Rprob     = true reward probabilities in blocks
% beta      = hyper-prior on precision of policy selection (higher = less precise)
% alpha     = hyper-prior on precision of action selection  (higher = more precise)
% cr        = reward sensitivity or risk-seeking (higher = stronger bias toward reward-seeking over information-seeking)
% eta_win       = learning rate (wins)
% eta_loss      = learning rate (losses)
% prior_a       = prior concentration parameter magnitude (higher = less sensitivity to information)

%example values

Rprob     = [0.2712	0.9723	0.7656];
beta      = 1;
alpha     = 8;
cr        = 2;
eta_win       = .5;
eta_loss      = .5;
prior_a       = 2;

% Output:
% mdp model containing observation model, transition probs etc
% 
% Ryan Smith & Phillip Schwartenbeck


%% Outcome probabilities: A
%==========================================================================

% Location and Reward, no uncertainty about choice, uncertainty about
% reward probabilities
%--------------------------------------------------------------------------
% States: start, left, middle, right (cols) --> outcomes: neutral, reward, no reward (rows)

probs = Rprob;

A{1} = [1 0            0            0            ; % reward neutral (starting position)
        0 probs(1)     probs(2)     probs(3)     ; % reward 
        0 (1-probs(1)) (1-probs(2)) (1-probs(3))]; % no reward

%States: start, left, middle, right (cols) --> outcomes: start, left, middle, right (rows)

A{2} = [1 0 0 0; % starting position
        0 1 0 0; % left choice 
        0 0 1 0; % middle choice
        0 0 0 1];% right choice
    
%% Beliefs about outcome (likelihood) mapping
%==========================================================================

%--------------------------------------------------------------------------
% That's where learning comes in - start with uniform prior
%--------------------------------------------------------------------------
%prior_a = 1/4;

a{1} = [1 0       0       0       ; % reward neutral (starting position)
        0 prior_a prior_a prior_a ; % reward 
        0 prior_a prior_a prior_a]; % no reward
    
% States: start, left, middle, right (cols) --> outcomes: start, left, middle, right (rows)

a{2} = [1 0 0 0; % starting position
        0 1 0 0; % left choice 
        0 0 1 0; % middle choice
        0 0 0 1];% right choice    

%% Controlled transitions: B{u}
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions of hidden states
% for each factor. Here, there are three actions taking the agent directly
% to each of the three locations.
%--------------------------------------------------------------------------
B{1}(:,:,1) = [1 1 1 1; 0 0 0 0;0 0 0 0;0 0 0 0];     % move to the starting point
B{1}(:,:,2) = [0 0 0 0; 1 1 1 1;0 0 0 0;0 0 0 0];     % choose left (and check for reward)
B{1}(:,:,3) = [0 0 0 0; 0 0 0 0;1 1 1 1;0 0 0 0];     % choose middle (and check for reward)
B{1}(:,:,4) = [0 0 0 0; 0 0 0 0;0 0 0 0;1 1 1 1];     % choose right (and check for reward)


%% Prior preferences (log probabilities over outcomes)
%==========================================================================

C{1}  = [0 cr 0]'; % preference for: [staying at starting point | reward | no reward]
C{2}  = [0 0 0 0]';
%--------------------------------------------------------------------------
%% Prior beliefs about initial state
%--------------------------------------------------------------------------
D{1}  = [1 0 0 0]'; % prior over starting point - start, left, middle, right


%% Allowable policies or sequences of actions (one action per trial).
%==========================================================================

V     = [2 3 4]; % go left, go middle, go right

%% 14. Define MDP Structure
%==========================================================================
%==========================================================================

% Set up MDP

mdp.T = 2;                      % number of timepoints in each trial (start + choice)
mdp.V = V;                      % allowable policies
mdp.A = A;                      % outcome probabilities
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = D;                      % prior over initial states

mdp.a = a;                      % beliefs about outcome probabilities                    

mdp.beta    = beta;
mdp.alpha   = alpha;
mdp.eta_win = eta_win;
mdp.eta_loss = eta_loss;


