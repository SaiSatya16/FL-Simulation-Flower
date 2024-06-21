import flwr as fl
import torch
from collections import OrderedDict
from model import Net, train, test
from typing import Dict, List, Tuple
from flwr.common import Scalar
from flwr.common.typing import NDArray, NDArrays


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_class) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = Net(num_class)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    

    
    def fit(self, parameters, config):

        #copy the parameters sent by the server into client local model
        self.set_parameters(parameters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        #do local training
        train(self.model, self.trainloader, optimizer, epochs, device=self.device)

        #return the updated model parameters to the server 
        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, device=self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloader, valloader, num_classes):
    def client_fn(cid: str):

        return FlowerClient(trainloader=trainloader[int(cid)], valloader=valloader[int(cid)], num_class=num_classes)
    
    return client_fn


