function fit_vba(data_file, out_file, vba_path)
% FIT_VBA  Fit the three observation-only models with the VBA toolbox.
%
%   fit_vba(DATA_FILE, OUT_FILE, VBA_PATH)
%
% Reads the dataset simulated by the notebook, fits every subject with
% VBA_NLStateSpaceModel, and writes the posterior means / variances back to
% a .mat file that the notebook reloads.
%
% The three models are observation-only ("null evolution"): there is no
% hidden state and no evolution parameter, so we pass f_fname = [] and
% dim.n = 0. VBA then reduces to a static (Variational-Laplace) regression.
%
% Model / outcome conventions used here, matching the simulation:
%   continuous  : y = b0 + b1*s + noise          (sources.type = 0)
%   binary      : p(y=1) = sigmoid(b0 + b1*s)    (sources.type = 1)
%   categorical : p(y=k) = softmax over 3 classes(sources.type = 2)
%
% IMPORTANT VBA convention: the observation function g returns the *mean of
% the outcome distribution*, i.e. a probability for binary data and a
% probability vector for categorical data -- NOT a logit. (GBM Toolbox is
% the opposite: it returns logits and applies the link itself.) The link
% function is therefore applied inside g_* below.

if nargin < 3 || isempty(vba_path)
    vba_path = '/Users/gino.diez/Documents/MATLAB/VBA-toolbox-master';
end
addpath(genpath(vba_path));

S = load(data_file);
n_subjects = double(S.n_subjects);

% VBA prints a figure per inversion by default; suppress for batch use.
opt_base = struct();
opt_base.DisplayWin = 0;
opt_base.verbose    = 0;

%% ---------------------------------------------------------------- continuous
% 2 observation parameters (intercept, slope), Gaussian noise estimated by
% VBA itself through its precision hyperparameters (a_sigma / b_sigma).
dim_c = struct('n', 0, 'n_theta', 0, 'n_phi', 2);

mu_cont    = zeros(n_subjects, 2);
var_cont   = zeros(n_subjects, 2);
sigma_cont = zeros(n_subjects, 1);   % posterior mean of the noise SD
F_cont     = zeros(n_subjects, 1);

for i = 1:n_subjects
    y = S.cont_y(i, :);              % 1 x T
    u = S.cont_u(i, :);              % 1 x T  (the regressor s)

    options = opt_base;
    options.sources.type = 0;        % Gaussian outcomes
    % Match the notebook's N(0,1) prior on the two observation parameters.
    options.priors.muPhi    = zeros(2, 1);
    options.priors.SigmaPhi = eye(2);

    [posterior, out] = VBA_NLStateSpaceModel(y, u, [], @g_linear, dim_c, options);

    mu_cont(i, :)  = posterior.muPhi(:)';
    var_cont(i, :) = diag(posterior.SigmaPhi)';
    % VBA estimates the measurement *precision* sigma with a Gamma
    % posterior: E[sigma] = a/b. Convert to an SD to compare with GBM.
    sigma_cont(i)  = sqrt(posterior.b_sigma / posterior.a_sigma);
    F_cont(i)      = out.F;
end

%% -------------------------------------------------------------------- binary
dim_b = struct('n', 0, 'n_theta', 0, 'n_phi', 2);

mu_bin  = zeros(n_subjects, 2);
var_bin = zeros(n_subjects, 2);
F_bin   = zeros(n_subjects, 1);

for i = 1:n_subjects
    y = S.bin_y(i, :);               % 1 x T, values in {0,1}
    u = S.bin_u(i, :);

    options = opt_base;
    options.sources.type = 1;        % Bernoulli outcomes
    options.priors.muPhi    = zeros(2, 1);
    options.priors.SigmaPhi = 4 * eye(2);

    [posterior, out] = VBA_NLStateSpaceModel(y, u, [], @g_logistic, dim_b, options);

    mu_bin(i, :)  = posterior.muPhi(:)';
    var_bin(i, :) = diag(posterior.SigmaPhi)';
    F_bin(i)      = out.F;
end

%% --------------------------------------------------------------- categorical
% 3 classes, class 0 is the reference (its logit is fixed to 0), so there
% are 2 free parameters per non-reference class = 4 in total.
dim_k = struct('n', 0, 'n_theta', 0, 'n_phi', 4);

mu_cat  = zeros(n_subjects, 4);
var_cat = zeros(n_subjects, 4);
F_cat   = zeros(n_subjects, 1);

for i = 1:n_subjects
    % VBA expects categorical outcomes one-hot coded as K x T.
    y = squeeze(S.cat_y(i, :, :));   % 3 x T
    u = S.cat_u(i, :);

    options = opt_base;
    options.sources.type = 2;        % categorical outcomes
    options.priors.muPhi    = zeros(4, 1);
    options.priors.SigmaPhi = 4 * eye(4);

    [posterior, out] = VBA_NLStateSpaceModel(y, u, [], @g_softmax, dim_k, options);

    mu_cat(i, :)  = posterior.muPhi(:)';
    var_cat(i, :) = diag(posterior.SigmaPhi)';
    F_cat(i)      = out.F;
end

save(out_file, 'mu_cont', 'var_cont', 'sigma_cont', 'F_cont', ...
               'mu_bin',  'var_bin',  'F_bin', ...
               'mu_cat',  'var_cat',  'F_cat', '-v7');
fprintf('VBA fits written to %s\n', out_file);

end

% ======================= observation functions ==========================
% Signature required by VBA: g(x_t, phi, u_t, in). There is no hidden
% state here, so x_t is ignored.

function gx = g_linear(~, phi, ut, ~)
    % Gaussian: g returns the predicted mean directly.
    gx = phi(1) + phi(2) * ut;
end

function gx = g_logistic(~, phi, ut, ~)
    % Binary: g must return p(y=1), so apply the sigmoid here.
    gx = 1 ./ (1 + exp(-(phi(1) + phi(2) * ut)));
end

function gx = g_softmax(~, phi, ut, ~)
    % Categorical: g must return the full probability vector (sums to 1).
    % Class 1 is the reference with logit fixed at 0.
    logits = [0; phi(1) + phi(2) * ut; phi(3) + phi(4) * ut];
    logits = logits - max(logits);          % numerical stability
    p = exp(logits);
    gx = p / sum(p);
end
