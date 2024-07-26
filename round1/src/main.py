import hydra
from omegaconf import DictConfig
import rootutils
import logging


ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIGPATH = str(ROOTPATH / "configs")

import src.data.dataset as dataset
import src.algorithm.genetic_alg as ga
import src.algorithm.fitness_func as ff
import src.algorithm.crossover_func as cf
import src.algorithm.mutation_func as mf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path=CONFIGPATH, config_name="config", version_base=None)
def main(cfg:DictConfig):
    # Load data
    data: dataset.Dataset = hydra.utils.instantiate(cfg.dataset)

    # Load fitness function
    fitness_func: ff.CustomFitnessFunction = hydra.utils.instantiate(cfg.fitness_function, dataset=data)

    # Load crossover function
    crossover_func = cf.CustomCrossoverFunc(data)
    
    # Load mutation function
    mutation_func = mf.CustomMutationFunc(data)

    # Set initial population
    initial_population = mf.generate_feasible_population(data, cfg.genetic_algorithm.sol_per_pop)

    # Load algorithm
    num_genes = (data.num_doctor1 + data.num_doctor2 + data.num_nurse) * data.num_faculties * data.num_days
    algo: ga.GeneticAlgorithm = hydra.utils.instantiate(cfg.genetic_algorithm, 
                                                        fitness_func=fitness_func, 
                                                        crossover_func=crossover_func,
                                                        mutation_func=mutation_func,
                                                        num_genes=num_genes,
                                                        dataset=data)
                                                       
    algo.setup(initial_population)
    algo.run()
    algo.plot_total_fitness(str(ROOTPATH / "figures" / "total_fitness2.png"))
    algo.plot_constraints(str(ROOTPATH / "figures" / "constraints_fitness2.png"))

if __name__ == '__main__':
    main()