import hydra
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import pickle

import flwr as fl

@hydra.main(config_path="config", config_name="base",version_base=None)
def main(cfg: DictConfig) -> None:

    #1. parse config and get experiment output directory
    print(OmegaConf.to_yaml(cfg))

    #2. prepare your datasets
    trainloaders, validationloaders, testloaders = prepare_dataset(cfg.num_clients, cfg.batch_size)
    print(len(trainloaders), len(trainloaders[0].dataset), len(validationloaders), len(testloaders.dataset))


    #3. defiine your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)


    #4.define your strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.min_fit_clients,
        fraction_evaluate= 0.00001,
        min_evaluate_clients=cfg.min_evaluate_clients,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloaders)
    )

    #5. start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config = fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy, 
        client_resources={"num_cpus": 1, "num_gpus": 0}
    )


    #6. save results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "results.pkl"

    results = {
        "history": history,
        "config": cfg
    }

    with open(results_path, "wb") as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)








if __name__ == "__main__":
    main()