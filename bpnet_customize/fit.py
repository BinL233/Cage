import time
import numpy
import torch

from .losses import MNLLLoss
from .losses import log1pMSELoss
from .performance import pearson_corr
from .performance import calculate_performance_measures

def fit(self, training_data, optimizer, X_valid=None, X_ctl_valid=None, 
    y_valid=None, max_epochs=100, batch_size=64, validation_iter=100, 
    early_stopping=None, verbose=True):

    if X_valid is not None:
        X_valid = X_valid.cuda()
        y_valid_counts = y_valid.sum(dim=2)

    if X_ctl_valid is not None:
        X_ctl_valid = X_ctl_valid.cuda()


    iteration = 0
    early_stop_count = 0
    best_loss = float("inf")
    self.logger.start()

    for epoch in range(max_epochs):
        tic = time.time()

        for data in training_data:
            if len(data) == 3:
                X, X_ctl, y = data
                X, X_ctl, y = X.cuda(), X_ctl.cuda(), y.cuda()
            else:
                X, y = data
                X, y = X.cuda(), y.cuda()
                X_ctl = None

            # Clear the optimizer and set the model to training mode
            optimizer.zero_grad()
            self.train()

            # Run forward pass
            y_profile, y_counts = self(X, X_ctl)
            y_profile = y_profile.reshape(y_profile.shape[0], -1)
            y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
            
            y = y.reshape(y.shape[0], -1)

            # Calculate the profile and count losses
            profile_loss = MNLLLoss(y_profile, y).mean()
            count_loss = log1pMSELoss(y_counts, y.sum(dim=-1).reshape(-1, 1)).mean()

            # Extract the profile loss for logging
            profile_loss_ = profile_loss.item()
            count_loss_ = count_loss.item()

            # Mix losses together and update the model
            loss = profile_loss + self.alpha * count_loss
            loss.backward()
            optimizer.step()

            # Report measures if desired
            if verbose and iteration % validation_iter == 0:
                train_time = time.time() - tic

                with torch.no_grad():
                    self.eval()

                    tic = time.time()
                    y_profile, y_counts = self.predict(X_valid, X_ctl_valid)

                    z = y_profile.shape
                    y_profile = y_profile.reshape(y_profile.shape[0], -1)
                    y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
                    y_profile = y_profile.reshape(*z)

                    measures = calculate_performance_measures(y_profile, 
                        y_valid, y_counts, kernel_sigma=7, 
                        kernel_width=81, measures=['profile_mnll', 
                        'profile_pearson', 'count_pearson', 'count_mse'])

                    profile_corr = measures['profile_pearson']
                    count_corr = measures['count_pearson']
                    
                    valid_loss = measures['profile_mnll'].mean()
                    valid_loss += self.alpha * measures['count_mse'].mean()
                    valid_time = time.time() - tic

                    self.logger.add([epoch, iteration, train_time, 
                        valid_time, profile_loss_, count_loss_, 
                        measures['profile_mnll'].mean().item(), 
                        numpy.nan_to_num(profile_corr).mean(),
                        numpy.nan_to_num(count_corr).mean(), 
                        measures['count_mse'].mean().item(),
                        (valid_loss < best_loss).item()])

                    self.logger.save("{}.log".format(self.name))

                    if valid_loss < best_loss:
                        torch.save(self, "{}.torch".format(self.name))
                        best_loss = valid_loss
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

            if early_stopping is not None and early_stop_count >= early_stopping:
                break

            iteration += 1

        if early_stopping is not None and early_stop_count >= early_stopping:
            break

    torch.save(self, "{}.final.torch".format(self.name))