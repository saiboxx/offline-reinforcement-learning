import torch

class RadialBasisFunction(object):

    r"""
    CREDIT:
    Original code has been taken from https://github.com/rhololkeolke/lspi-python
    This version has been modified and ported to PyTorch

    Gaussian Multidimensional Radial Basis Function (RBF).
    Given a set of k means :math:`(\mu_1 , \ldots, \mu_k)` produce a feature
    vector :math:`(1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
    e^{-\gamma || s - \mu_k ||^2})` where `s` is the state vector and
    :math:`\gamma` is a free parameter. This vector will be padded with
    0's on both sides proportional to the number of possible actions
    specified.
    Parameters
    ----------
    means: list(numpy.array)
        List of numpy arrays representing :math:`(\mu_1, \ldots, \mu_k)`.
        Each :math:`\mu` is a numpy array with dimensions matching the state
        vector this basis function will be used with. If the dimensions of each
        vector are not equal than an exception will be raised. If no means are
        specified then a ValueError will be raised
    gamma: float
        Free parameter which controls the size/spread of the Gaussian "bumps".
        This parameter is best selected via tuning through cross validation.
        gamma must be > 0.
    num_actions: int
        Number of actions. Must be in range [1, :math:`\infty`] otherwise
        an exception will be raised.
    Note
    ----
    The numpy arrays specifying the means are not copied.
    """

    def __init__(self,
                 gamma: float,
                 k: int,
                 action_space: int,
                 observation_space: int,
                 device: torch.device):
        """Initialize RBF instance."""
        self.action_space = action_space
        self.observation_space = observation_space
        self.k = k
        self.means = ((-1 - 1) * torch.rand(k, observation_space) + 1).to(device)
        self.gamma = gamma
        self.device = device
        self.size = (len(self.means) + 1) * self.action_space

    def evaluate(self, state, action):
        r"""Calculate the :math:`\phi` matrix.
        Matrix will have the following form:
        :math:`[\cdots, 1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
        e^{-\gamma || s - \mu_k ||^2}, \cdots]`
        where the matrix will be padded with 0's on either side depending
        on the specified action index and the number of possible actions.
        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.
        """
        phi = torch.zeros(self.size).to(self.device)
        offset = (len(self.means) + 1) * action

        rbf = self.calc_basis_component(state)
        phi[offset] = 1.
        phi[offset + 1: offset + 1 + len(rbf)] = rbf

        return phi

    def calc_basis_component(self, state: torch.tensor) -> torch.tensor:
        mean_diff = (self.means - state)**2
        return torch.exp(-self.gamma * torch.sum(mean_diff, dim=1))
