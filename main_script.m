% Samuel Taylor and Ryan Smith, 2021

clear all
close all

% To fit a subject, specify the subject ID and directory of the subject in
% the variables below. The script will fit the subject, and if the plot
% flag is set, will plot the behavior of the subject overlayed on the
% beliefs produced by the model. 

% Before running, be sure to add spm12 and the DEM toolbox therein to your
% matlab path.

% SIM = false, FIT = true: fit a given subject (as specified by FIT_SUBJECT
% and INPUT_DIRECTORY). Will show a plot of action probabilities as
% determined by fitted parameters values, overlaid with the observations
% and responses of the true subject data.
% SIM = true, FIT = true: simulate behavior with the given parameter values
% (as specified by ALPHA, CR, ETA_WIN, ETA_LOSS, and PRIOR_A), and then fit
% to the simulated behavior. A good way to test parameter recoverability.
% Shows a plot of the original simulated behavior and accompanying actions
% probabilities, and will later show the action probabilities with the new
% fitted values as well.
% SIM = true, FIT = false: only simulate behavior. Shows a plot of
% simulated behavior and action probabilities.

SIM = true; % Generate simulated behavior (if false and FIT == true, will fit to subject file data instead)
FIT = false; % Fit example subject data 'BBBBB' or fit simulated behavior (if SIM == true)

% Specify parameters (if simulating behavior)
ALPHA    = 10.38; % Specify ACTION PRECISION (ALPHA) parameter value
CR       = 2.92; % Specify REWARD SENSITIVITY (CR) parameter value
ETA_WIN  = 0.12; % Specify WIN LEARNING RATE (ETA_W) parameter value
ETA_LOSS = 0.97; % Specify LOSS LEARNING RATE (ETA_L) parameter value
PRIOR_A  = 0.80; % Specify INFORMATION INSENSITIVITY (ALPHA_0) parameter value

PLOT_FLAG = true; % Generate plot of behavior

FIT_SUBJECT = 'TAB00';   % Subject ID
INPUT_DIRECTORY = './';  % Where the subject file is located

if SIM
    [sim_mdp, gen_data] = TAB_sim(ALPHA, CR, ETA_WIN, ETA_LOSS, PRIOR_A);
    
    if PLOT_FLAG
        gtitle = sprintf('Simulated Data (for Alpha=%.2f; CR=%.2f; Eta Win=%.2f; Eta Loss=%.2f; Prior A=%.2f)', ALPHA, CR, ETA_WIN, ETA_LOSS, PRIOR_A);
        TAB_plot(sim_mdp(1:16)', PRIOR_A, gtitle);
        shg
    end
end

if FIT
    if SIM
        fit_results = TAB_sim_fit(gen_data);
        
        if PLOT_FLAG
            figure
            gtitle = sprintf('Fit Model (to Simulation Data Generated by Alpha=%.2f; CR=%.2f; Eta Win=%.2f; Eta Loss=%.2f; Prior A=%.2f)', ALPHA, CR, ETA_WIN, ETA_LOSS, PRIOR_A);
            TAB_plot(fit_results{5}(1:16)', fit_results{3}(5), gtitle);
        end
    else
        fit_results = TAB_fit(FIT_SUBJECT, INPUT_DIRECTORY);

        if PLOT_FLAG
            figure
            gtitle = sprintf('Fit Model (to %s Subject Data, Alpha=%.2f; CR=%.2f; Eta Win=%.2f; Eta Loss=%.2f; Prior A=%.2f)', FIT_SUBJECT, fit_results{3}(1), fit_results{3}(2), fit_results{3}(3), fit_results{3}(4), fit_results{3}(5));
            TAB_plot(fit_results{5}(1:16)', fit_results{3}(5), gtitle);
        end
    end

    fprintf('Fit: \n\tAlpha =\t%.3f\n\tCR =\t%.3f\n\tEta Win =\t%.3f\n\tEta Loss =\t%.3f\n\tPrior A =\t%.3f\n', fit_results{3}(1), fit_results{3}(2), fit_results{3}(3), fit_results{3}(4), fit_results{3}(5));
end
