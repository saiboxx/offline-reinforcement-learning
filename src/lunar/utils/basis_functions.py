import torch

class RadialBasisFunction(object):

    """
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
